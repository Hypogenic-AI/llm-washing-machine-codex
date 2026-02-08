# Cloned Repositories

## Repo 1: openai/sparse_autoencoder
- URL: github.com/openai/sparse_autoencoder
- Purpose: Trained sparse autoencoders on GPT-2 activations + viewer.
- Location: `code/openai_sparse_autoencoder/`
- Key files:
  - `sparse_autoencoder/model.py` (SAE architecture)
  - `sparse_autoencoder/train.py` (training)
  - `sae-viewer/` (visualizer)
- Notes: Provides pretrained SAEs and example code for activation encoding/decoding.

## Repo 2: TransformerLens
- URL: github.com/neelnanda-io/TransformerLens
- Purpose: Mechanistic interpretability library for transformer models (activation caching, hooks).
- Location: `code/transformer_lens/`
- Key files:
  - `transformer_lens/` (core library)
  - `demos/` (examples)
- Notes: Standard toolkit for activation patching and probing LLM internals.

## Repo 3: LM-Compositionality
- URL: github.com/nightingal3/lm-compositionality
- Purpose: Code and data utilities for compositionality experiments (Liu & Neubig, 2022).
- Location: `code/lm_compositionality/`
- Key files:
  - `src/generate_data_treebank.py` (dataset generation)
  - `src/models/` (composition probes)
  - `data/qualtrics_results/chip_dataset.csv` (CHIP dataset)
- Notes: Contains CHIP dataset and scripts for local composition probes.
