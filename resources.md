# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Toy Models of Superposition | Elhage et al. | 2022 | papers/2209.10652_toy_models_of_superposition.pdf | Theory of superposition/polysemanticity |
| Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al. | 2023 | papers/2309.08600_sparse_autoencoders_interpretable_features.pdf | SAEs find sparse, interpretable features |
| Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small | Wang et al. | 2022 | papers/2211.00593_interpretability_in_the_wild.pdf | Causal circuit analysis (IOI) |
| Locating and Editing Factual Associations in GPT (ROME) | Meng et al. | 2022 | papers/2202.05262_locating_and_editing_factual_associations.pdf | Causal tracing + model editing |
| Knowledge Neurons in Pretrained Transformers | Dai et al. | 2021 | papers/2104.08696_knowledge_neurons.pdf | Neuron attribution for facts |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 (raw) | HuggingFace `wikitext` | 36,718 train | Language modeling | datasets/wikitext_2_raw_v1/ | Small, fast for activation sampling |
| LAMBADA (1k subset) | HuggingFace `lambada` | 1,000 train samples | Long-context prediction | datasets/lambada_train_1k/ | Subset for quick tests |

See `datasets/README.md` for detailed descriptions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sparse_autoencoder | github.com/openai/sparse_autoencoder | Pretrained SAEs + visualizer | code/sparse_autoencoder/ | GPT-2 SAE features |
| SAELens | github.com/decoderesearch/SAELens | SAE training/analysis toolkit | code/sae_lens/ | Tutorials + pretrained SAEs |
| ROME | github.com/kmeng01/rome | Causal tracing + model editing | code/rome/ | Includes CounterFact tooling |
| knowledge-neurons | github.com/Hunter-DDM/knowledge-neurons | Neuron attribution scripts | code/knowledge_neurons/ | Reproduces ACL 2022 work |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Manual search on arXiv and Semantic Scholar for superposition, sparse autoencoders, and mechanistic interpretability papers.
- Transformer Circuits reports for dictionary learning and mechanistic frameworks.
- HuggingFace dataset cards for corpora used in activation sampling.
- GitHub for official implementations and tooling.

### Selection Criteria
- Direct relevance to feature localization, superposition, or causal tracing.
- Methods enabling activation-level interventions.
- Availability of code and datasets for reproducibility.

### Challenges Encountered
- Paper-finder service did not respond (timed out), so manual search was used.
- Some arXiv pages prompted access challenges; metadata was gathered from alternative sources.
- OpenWebText is large; a smaller dataset was chosen instead.

### Gaps and Workarounds
- No dedicated dataset for “washing machine” compositionality; use corpus sampling + synthetic prompts.
- CounterFact dataset not downloaded; instructions provided via ROME tooling.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: WikiText-2 for quick probing; LAMBADA for long-context effects.
2. **Baseline methods**: SAE feature discovery (openai/sparse_autoencoder), neuron attribution (knowledge-neurons), causal tracing (ROME).
3. **Evaluation metrics**: Feature sparsity, causal effect size under patching, separation of atomic vs. composite prompts.
4. **Code to adapt/reuse**: SAELens for training/analysis, ROME for causal tracing, sparse_autoencoder for pretrained features.

## Research Execution Notes (2026-02-07)
- Executed experiments via `src/run_experiments.py` on GPT-2 small with SAE features at layer 6 (`resid_post_mlp`).
- Outputs saved to `results/metrics.json`, `results/examples.json`, and `results/plots/`.
- Hardware used: NVIDIA GeForce RTX 3090 (CUDA 12.8). See `results/metrics.json` for full environment details.
