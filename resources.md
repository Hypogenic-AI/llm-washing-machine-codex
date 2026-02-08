# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Toy Models of Superposition | Elhage et al. | 2022 | papers/2209.10652_toy_models_of_superposition.pdf | Superposition/polysemanticity toy models |
| Are Representations Built from the Ground Up? | Liu, Neubig | 2022 | papers/2210.03575_are_representations_built_from_the_ground_up.pdf | Local compositionality probes |
| Towards Best Practices of Activation Patching | Zhang, Nanda | 2023 | papers/2309.16042_best_practices_activation_patching.pdf | Causal tracing methodology |
| The Linear Representation Hypothesis and the Geometry of LLMs | Park et al. | 2024 | papers/2311.03658_linear_representation_hypothesis.pdf | Formalizes linear concept directions |
| Scaling and Evaluating Sparse Autoencoders | Gao et al. | 2024 | papers/2406.04093_scaling_and_evaluating_sparse_autoencoders.pdf | Large-scale SAE training & metrics |
| Gemma Scope | Lieberum et al. | 2024 | papers/2408.05147_gemma_scope.pdf | Open SAEs across Gemma 2 layers |
| Residual Stream Analysis with Multi-Layer SAEs | Lawson et al. | 2025 | papers/2409.04185_residual_stream_analysis_with_multi_layer_saes.pdf | MLSAE for cross-layer features |
| Automated Interpretability Metrics... | Heap et al. | 2025 | papers/2501.17727_automated_interpretability_metrics_do_not_distinguish_trained_and_random_transformers.pdf | SAE metrics sanity checks |
| Sparse Autoencoders Do Not Find Canonical Units | Leask et al. | 2025 | papers/2502.04878_sparse_autoencoders_do_not_find_canonical_units.pdf | SAE non-atomicity |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 (raw) | HuggingFace | 7.8MB | LM corpus | datasets/wikitext_2_raw/ | Includes samples.json |

Additional documented (not downloaded): CHIP dataset from lm-compositionality repo.

See datasets/README.md for detailed descriptions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sparse_autoencoder | github.com/openai/sparse_autoencoder | SAE training + pretrained models | code/openai_sparse_autoencoder/ | GPT-2 SAEs + viewer |
| TransformerLens | github.com/neelnanda-io/TransformerLens | Mech interp toolkit | code/transformer_lens/ | Activation hooks, caching |
| lm-compositionality | github.com/nightingal3/lm-compositionality | Compositionality probes + CHIP dataset | code/lm_compositionality/ | Dataset + scripts |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Keyword-driven search on mechanistic interpretability, superposition, SAEs, and compositionality.
- Focused on recent (2022–2025) arXiv/ICLR/ICML papers with tool releases.
- Selected papers specifically tied to concept directions, SAE feature extraction, and compositionality.

### Selection Criteria
- Direct relevance to concept representation and compound concepts.
- Availability of code/weights/datasets for reproducibility.
- Mix of foundational theory (superposition, linear representation) and practical tooling (SAEs).

### Challenges Encountered
- Paper-finder service appeared unavailable; manual search was required.
- Some interpretability resources are web-only (no PDF), so not all sources are downloadable PDFs.

### Gaps and Workarounds
- Limited public datasets explicitly labeled for compound-noun concept representation; using CHIP and WikiText as practical proxies.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: CHIP (for compositionality judgments) + WikiText-2 (for activation streams).
2. **Baseline methods**: Linear probes for component words, single-layer SAE vs. MLSAE, random model baseline.
3. **Evaluation metrics**: Reconstruction loss + sparsity for SAEs, causal patching effect sizes, compositionality prediction accuracy.
4. **Code to adapt/reuse**: `TransformerLens` for hooks/patching; `openai_sparse_autoencoder` for SAE workflow; `lm-compositionality` for compositionality probes.
