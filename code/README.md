# Cloned Repositories

## Repo 1: sparse_autoencoder
- URL: https://github.com/openai/sparse_autoencoder
- Purpose: Sparse autoencoders trained on GPT-2-small activations; feature visualization and SAE weights
- Location: code/sparse_autoencoder/
- Key files:
  - sparse_autoencoder/model.py
  - sparse_autoencoder/train.py
  - sparse_autoencoder/paths.py
  - sae-viewer/README.md
- Notes: Includes pretrained SAE weights and a web-based feature visualizer.

## Repo 2: SAELens
- URL: https://github.com/decoderesearch/SAELens
- Purpose: Training and analysis library for sparse autoencoders with TransformerLens integration
- Location: code/sae_lens/
- Key files:
  - tutorials/basic_loading_and_analysing.ipynb
  - tutorials/training_a_sparse_autoencoder.ipynb
- Notes: Supports pretrained SAEs and feature dashboards via SAE-Vis.

## Repo 3: ROME
- URL: https://github.com/kmeng01/rome
- Purpose: Rank-One Model Editing + Causal Tracing implementation and CounterFact evaluation
- Location: code/rome/
- Key files:
  - notebooks/rome.ipynb
  - notebooks/causal_trace.ipynb
  - experiments/evaluate.py
- Notes: Useful for locating and testing knowledge storage via causal tracing.

## Repo 4: knowledge-neurons
- URL: https://github.com/Hunter-DDM/knowledge-neurons
- Purpose: Reproduce "Knowledge Neurons in Pretrained Transformers" experiments
- Location: code/knowledge_neurons/
- Key files:
  - src/
  - scripts 1_run_mlm.sh ... 8_run_plot.sh
- Notes: Provides attribution scoring and neuron identification scripts.
