import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ensure local sparse_autoencoder is importable without installation
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SAE_PATH = REPO_ROOT / "code" / "openai_sparse_autoencoder"
sys.path.insert(0, str(LOCAL_SAE_PATH))

import blobfile as bf
import sparse_autoencoder  # type: ignore
import transformer_lens


@dataclass
class Config:
    seed: int = 42
    model_name: str = "gpt2"
    sae_layer: int = 6
    sae_location: str = "resid_post_mlp"
    top_k: int = 50
    max_contexts: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: str = "results"
    plots_dir: str = "results/plots"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: Config) -> None:
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)


def load_wikitext() -> pd.DataFrame:
    dataset = load_from_disk("datasets/wikitext_2_raw")
    # Concatenate splits for context search
    all_rows = []
    for split in ["train", "validation", "test"]:
        for row in dataset[split]:
            text = row["text"]
            if text is None:
                continue
            text = text.strip()
            if not text:
                continue
            all_rows.append({"split": split, "text": text})
    return pd.DataFrame(all_rows)


def extract_contexts(df: pd.DataFrame, max_contexts: int) -> Dict[str, List[str]]:
    texts = df["text"].tolist()
    compound = []
    washing_only = []
    machine_only = []

    for text in texts:
        lower = text.lower()
        if "washing machine" in lower:
            compound.append(text)
        elif "washing" in lower and "machine" not in lower:
            washing_only.append(text)
        elif "machine" in lower and "washing" not in lower:
            machine_only.append(text)

    contexts = {
        "compound": compound[:max_contexts],
        "washing_only": washing_only[:max_contexts],
        "machine_only": machine_only[:max_contexts],
    }

    # If the dataset lacks the compound, add synthetic prompts for analysis
    if len(contexts["compound"]) == 0:
        subjects = ["The", "A", "This", "That", "My", "Our", "Their"]
        verbs = [
            "was broken",
            "stopped mid-cycle",
            "made a loud noise",
            "leaked water",
            "used less energy",
            "vibrated too much",
            "needed repairs",
            "worked flawlessly",
            "finished quickly",
            "ran overnight",
        ]
        places = [
            "in the basement",
            "in the laundry room",
            "at the apartment",
            "at the house",
            "in the garage",
        ]
        synthetic = []
        for s in subjects:
            for v in verbs:
                synthetic.append(f"{s} washing machine {v}.")
        for s in subjects:
            for p in places:
                synthetic.append(f"{s} washing machine is {p}.")
        contexts["compound"] = synthetic[:max_contexts]

    return contexts


def get_token_ids(model, token: str) -> List[int]:
    return model.tokenizer.encode(token, add_special_tokens=False)


def find_compound_positions(tokens: torch.Tensor, washing_ids: set, machine_ids: set) -> List[int]:
    # Return indices where washing_id followed by machine_id
    positions = []
    for i in range(tokens.shape[-1] - 1):
        if tokens[i].item() in washing_ids and tokens[i + 1].item() in machine_ids:
            positions.append(i)
    return positions


def find_token_positions(tokens: torch.Tensor, token_ids: set) -> List[int]:
    return [i for i in range(tokens.shape[-1]) if tokens[i].item() in token_ids]


def load_sae(layer_index: int, location: str, device: str):
    with bf.BlobFile(
        sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb"
    ) as f:
        state_dict = torch.load(f, map_location=device)
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        autoencoder.to(device)
        autoencoder.eval()
    return autoencoder


def get_cache_activations(
    model, text: str, hook_name: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    activations = cache[hook_name]
    return tokens.squeeze(0), activations


def collect_latents(
    model,
    autoencoder,
    texts: List[str],
    hook_name: str,
    washing_ids: set,
    machine_ids: set,
    condition: str,
    device: str,
) -> List[torch.Tensor]:
    latents = []
    for text in texts:
        tokens, activations = get_cache_activations(model, text, hook_name, device)
        if condition == "compound":
            positions = find_compound_positions(tokens, washing_ids, machine_ids)
        elif condition == "washing_only":
            positions = find_token_positions(tokens, washing_ids)
        else:
            positions = find_token_positions(tokens, machine_ids)

        for pos in positions:
            input_tensor = activations[pos].unsqueeze(0)
            with torch.no_grad():
                latent_activations, _ = autoencoder.encode(input_tensor)
            latents.append(latent_activations.squeeze(0).detach().cpu())
    return latents


def top_k_latents(mean_latents: torch.Tensor, k: int) -> List[int]:
    values, indices = torch.topk(mean_latents, k)
    return indices.tolist()


def jaccard(a: List[int], b: List[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_causal_patching(model, layer: int, device: str) -> Dict[str, float]:
    templates = [
        "The washing machine was",
        "A washing machine was",
        "This washing machine was",
        "That washing machine was",
        "My washing machine was",
        "The washing process was",
        "A washing process was",
        "This washing process was",
        "That washing process was",
        "My washing process was",
    ]

    washing_id = model.to_single_token(" washing")
    machine_id = model.to_single_token(" machine")

    hook_name = f"blocks.{layer}.hook_resid_post"

    def get_washing_pos(tokens: torch.Tensor) -> int:
        positions = find_token_positions(tokens, {washing_id})
        return positions[-1] if positions else -1

    def run_with_patch(source_text: str, target_text: str) -> float:
        source_tokens = model.to_tokens(source_text).to(device)
        target_tokens = model.to_tokens(target_text).to(device)

        with torch.no_grad():
            _, source_cache = model.run_with_cache(source_tokens, remove_batch_dim=True)

        source_pos = get_washing_pos(source_tokens.squeeze(0))
        target_pos = get_washing_pos(target_tokens.squeeze(0))
        if source_pos < 0 or target_pos < 0:
            return 0.0

        source_vec = source_cache[hook_name][source_pos].detach()

        def patch_hook(activations, hook):
            activations[:, target_pos, :] = source_vec
            return activations

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                target_tokens, fwd_hooks=[(hook_name, patch_hook)]
            )
        patched_logits = patched_logits.squeeze(0)

        # logit for token " machine" at position after washing
        target_next_pos = target_pos + 1
        if target_next_pos >= patched_logits.shape[0]:
            return 0.0
        logit = patched_logits[target_next_pos, machine_id].item()
        return logit

    deltas = []
    for i in range(0, len(templates), 2):
        source = templates[i]
        target = templates[i + 1]
        target_tokens = model.to_tokens(target).to(device)
        with torch.no_grad():
            base_logits = model(target_tokens)
        base_logits = base_logits.squeeze(0)
        target_pos = find_token_positions(target_tokens.squeeze(0), {washing_id})
        if not target_pos:
            continue
        target_pos = target_pos[-1]
        base_logit = base_logits[target_pos + 1, machine_id].item()

        patched_logit = run_with_patch(source, target)
        deltas.append(patched_logit - base_logit)

    if not deltas:
        return {"mean_logit_delta": 0.0, "std_logit_delta": 0.0, "n": 0}

    return {
        "mean_logit_delta": float(np.mean(deltas)),
        "std_logit_delta": float(np.std(deltas)),
        "n": int(len(deltas)),
    }


def build_bigram_dataset(df: pd.DataFrame, max_bigrams: int = 200) -> List[Tuple[str, str]]:
    text = " ".join(df["text"].tolist()).lower()
    # simple tokenization by words
    words = re.findall(r"[a-z]+", text)
    bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if len(w1) < 3 or len(w2) < 3:
            continue
        bigrams.append((w1, w2))

    # frequency
    freq = {}
    for b in bigrams:
        freq[b] = freq.get(b, 0) + 1

    # sort by frequency and keep top
    sorted_b = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    out = [b for b, _ in sorted_b[:max_bigrams]]
    if ("washing", "machine") not in out:
        out = [("washing", "machine")] + out[: max_bigrams - 1]
    return out


def get_phrase_embedding(model, phrase: str, device: str) -> torch.Tensor:
    tokens = model.to_tokens(phrase).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    # Use final layer resid_post at last token
    resid = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    return resid[-1].detach().cpu()


def run_compositionality_probe(model, bigrams: List[Tuple[str, str]], device: str) -> Dict[str, float]:
    X = []
    y = []
    for w1, w2 in bigrams:
        compound = f"The {w1} {w2}."
        w1_prompt = f"The {w1}."
        w2_prompt = f"The {w2}."

        emb_compound = get_phrase_embedding(model, compound, device)
        emb_w1 = get_phrase_embedding(model, w1_prompt, device)
        emb_w2 = get_phrase_embedding(model, w2_prompt, device)

        X.append(torch.cat([emb_w1, emb_w2]).numpy())
        y.append(emb_compound.numpy())

    X = np.stack(X)
    y = np.stack(y)

    # Train/test split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X[train_idx], y[train_idx])
    preds = model_ridge.predict(X[test_idx])

    mse = mean_squared_error(y[test_idx], preds)
    # Baseline: use only w2 embedding (copy)
    baseline_preds = X[test_idx][:, X.shape[1] // 2 :]
    baseline_mse = mean_squared_error(y[test_idx], baseline_preds)

    # Cosine similarity average
    cosines = [
        cosine_sim(preds[i], y[test_idx][i]) for i in range(len(test_idx))
    ]

    return {
        "ridge_mse": float(mse),
        "baseline_w2_mse": float(baseline_mse),
        "ridge_mean_cosine": float(np.mean(cosines)),
    }


def main() -> None:
    config = Config()
    set_seed(config.seed)
    ensure_dirs(config)

    model = transformer_lens.HookedTransformer.from_pretrained(
        config.model_name, center_writing_weights=False
    )
    model.to(config.device)
    model.eval()

    df = load_wikitext()
    contexts = extract_contexts(df, config.max_contexts)

    washing_ids = set(get_token_ids(model, " washing") + get_token_ids(model, "washing"))
    machine_ids = set(get_token_ids(model, " machine") + get_token_ids(model, "machine"))

    hook_map = {
        "mlp_post_act": f"blocks.{config.sae_layer}.mlp.hook_post",
        "resid_delta_attn": f"blocks.{config.sae_layer}.hook_attn_out",
        "resid_post_attn": f"blocks.{config.sae_layer}.hook_resid_mid",
        "resid_delta_mlp": f"blocks.{config.sae_layer}.hook_mlp_out",
        "resid_post_mlp": f"blocks.{config.sae_layer}.hook_resid_post",
    }
    hook_name = hook_map[config.sae_location]

    autoencoder = load_sae(config.sae_layer, config.sae_location, config.device)

    latents_compound = collect_latents(
        model,
        autoencoder,
        contexts["compound"],
        hook_name,
        washing_ids,
        machine_ids,
        "compound",
        config.device,
    )
    latents_washing = collect_latents(
        model,
        autoencoder,
        contexts["washing_only"],
        hook_name,
        washing_ids,
        machine_ids,
        "washing_only",
        config.device,
    )
    latents_machine = collect_latents(
        model,
        autoencoder,
        contexts["machine_only"],
        hook_name,
        washing_ids,
        machine_ids,
        "machine_only",
        config.device,
    )

    def mean_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        if not latents:
            return torch.zeros(autoencoder.encoder.weight.shape[0])
        return torch.stack(latents).mean(dim=0)

    mean_compound = mean_latents(latents_compound)
    mean_washing = mean_latents(latents_washing)
    mean_machine = mean_latents(latents_machine)

    top_compound = top_k_latents(mean_compound, config.top_k)
    top_washing = top_k_latents(mean_washing, config.top_k)
    top_machine = top_k_latents(mean_machine, config.top_k)

    compound_union = list(set(top_washing) | set(top_machine))
    compound_unique = [i for i in top_compound if i not in compound_union]

    overlap_washing = jaccard(top_compound, top_washing)
    overlap_machine = jaccard(top_compound, top_machine)
    overlap_union = jaccard(top_compound, compound_union)

    cos_washing = cosine_sim(mean_compound.numpy(), mean_washing.numpy())
    cos_machine = cosine_sim(mean_compound.numpy(), mean_machine.numpy())

    causal_patch = run_causal_patching(model, layer=config.sae_layer, device=config.device)

    bigrams = build_bigram_dataset(df, max_bigrams=200)
    probe_results = run_compositionality_probe(model, bigrams, config.device)

    metrics = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "counts": {
            "compound_contexts": len(contexts["compound"]),
            "washing_only_contexts": len(contexts["washing_only"]),
            "machine_only_contexts": len(contexts["machine_only"]),
            "compound_latent_samples": len(latents_compound),
            "washing_latent_samples": len(latents_washing),
            "machine_latent_samples": len(latents_machine),
        },
        "overlap": {
            "compound_washing_jaccard": overlap_washing,
            "compound_machine_jaccard": overlap_machine,
            "compound_union_jaccard": overlap_union,
            "compound_unique_fraction": len(compound_unique) / max(1, len(top_compound)),
        },
        "cosine": {
            "compound_washing": cos_washing,
            "compound_machine": cos_machine,
        },
        "causal_patching": causal_patch,
        "compositionality_probe": probe_results,
    }

    with open(Path(config.results_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Simple visualization
    overlap_labels = ["compound-washing", "compound-machine", "compound-union"]
    overlap_values = [
        overlap_washing,
        overlap_machine,
        overlap_union,
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(overlap_labels, overlap_values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Jaccard overlap")
    plt.title("Top-K SAE Feature Overlap")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(Path(config.plots_dir) / "sae_overlap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["logit delta"], [causal_patch["mean_logit_delta"]], color="#B279A2")
    plt.ylabel("Mean logit delta for 'machine'")
    plt.title("Causal Patching Effect")
    plt.tight_layout()
    plt.savefig(Path(config.plots_dir) / "causal_patching.png", dpi=200)
    plt.close()

    print("Saved metrics to results/metrics.json")


if __name__ == "__main__":
    main()
