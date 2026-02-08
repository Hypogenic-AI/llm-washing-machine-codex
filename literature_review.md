# Literature Review

## Research Area Overview
This project sits at the intersection of mechanistic interpretability, representation geometry, and compositionality in language models. Key threads include: (1) whether concepts correspond to linear directions (linear representation hypothesis), (2) how superposition and polysemanticity complicate feature localization, (3) sparse autoencoders (SAEs) as a practical tool for extracting features, and (4) compositionality probes that test whether phrase representations are constructed from constituent representations.

## Key Papers

### Paper 1: Toy Models of Superposition
- **Authors**: Nelson Elhage et al.
- **Year**: 2022
- **Source**: arXiv 2209.10652
- **Key Contribution**: Introduces toy models showing superposition and polysemanticity as a consequence of sparse feature representations.
- **Methodology**: Synthetic ReLU networks with sparse input features; analyze phase changes and feature geometry.
- **Datasets Used**: Synthetic data (toy models)
- **Results**: Demonstrates superposition, monosemantic vs. polysemantic features, and geometric structure of features.
- **Code Available**: Yes (paper references a repo/colab)
- **Relevance to Our Research**: Establishes why compound concepts may not correspond to single orthogonal directions.

### Paper 2: Are Representations Built from the Ground Up?
- **Authors**: Emmy Liu, Graham Neubig
- **Year**: 2022
- **Source**: arXiv 2210.03575
- **Key Contribution**: Probes whether phrase representations are predictable from constituents (local compositionality).
- **Methodology**: Affine/linear probes predicting parent phrase embeddings from child embeddings; evaluation on compositionality datasets.
- **Datasets Used**: Penn Treebank phrases, CHIP idiom dataset
- **Results**: Parent representations are predictable from constituents, but alignment with human compositionality judgments is weak.
- **Code Available**: Yes (lm-compositionality repo)
- **Relevance to Our Research**: Directly informs how compound concepts like “washing machine” might be built.

### Paper 3: Towards Best Practices of Activation Patching
- **Authors**: Fred Zhang, Neel Nanda
- **Year**: 2023
- **Source**: arXiv 2309.16042
- **Key Contribution**: Systematic evaluation of activation patching variants and metrics.
- **Methodology**: Compare corruption/patching strategies and evaluation metrics for causal tracing.
- **Datasets Used**: Standard LM tasks and prompts (various localization tasks)
- **Results**: Hyperparameter choices can change localization conclusions; provides best practices.
- **Code Available**: Yes (paper linked)
- **Relevance to Our Research**: Guides robust localization for concept features.

### Paper 4: The Linear Representation Hypothesis and the Geometry of LLMs
- **Authors**: Kiho Park, Yo Joong Choe, Victor Veitch
- **Year**: 2024
- **Source**: arXiv 2311.03658
- **Key Contribution**: Formalizes what it means for concepts to be linear directions; connects to probing and steering.
- **Methodology**: Counterfactual formalism; experiments on LLaMA-2.
- **Datasets Used**: Model-internal evaluations with counterfactual pairs
- **Results**: Demonstrates linear representations under a specific inner product.
- **Code Available**: Yes (linked in paper)
- **Relevance to Our Research**: Provides theoretical framing for “concept direction” claims.

### Paper 5: Scaling and Evaluating Sparse Autoencoders
- **Authors**: Leo Gao et al.
- **Year**: 2024
- **Source**: arXiv 2406.04093
- **Key Contribution**: Training recipe for large SAEs and evaluation metrics.
- **Methodology**: k-sparse autoencoders; scaling experiments; new quality metrics.
- **Datasets Used**: GPT-2 / GPT-4 activations (token streams)
- **Results**: Scaling laws; large SAEs recover more interpretable features.
- **Code Available**: Yes (openai/sparse_autoencoder)
- **Relevance to Our Research**: Practical path to extract “washing” and “machine” features.

### Paper 6: Gemma Scope
- **Authors**: Tom Lieberum et al.
- **Year**: 2024
- **Source**: arXiv 2408.05147
- **Key Contribution**: Releases open SAEs for Gemma 2 across layers.
- **Methodology**: JumpReLU SAEs on Gemma 2 layers; evaluation on standard SAE metrics.
- **Datasets Used**: Gemma 2 pretraining data (token streams)
- **Results**: Public SAE weights + Neuronpedia demo.
- **Code Available**: Weights + demo; paper links to HuggingFace/Neuronpedia.
- **Relevance to Our Research**: Ready-to-use SAEs for concept localization.

### Paper 7: Residual Stream Analysis with Multi-Layer SAEs
- **Authors**: Tim Lawson et al.
- **Year**: 2025
- **Source**: arXiv 2409.04185 (ICLR 2025)
- **Key Contribution**: MLSAE approach for tracking features across layers.
- **Methodology**: Train a single SAE on residual streams from all layers; analyze layer distributions.
- **Datasets Used**: Transformer activation streams
- **Results**: Latents often activate in layer-specific patterns; supports cross-layer analysis.
- **Code Available**: Yes (linked in paper)
- **Relevance to Our Research**: Helps track composition of “washing” + “machine” across depth.

### Paper 8: Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
- **Authors**: Thomas Heap et al.
- **Year**: 2025
- **Source**: arXiv 2501.17727
- **Key Contribution**: Shows SAE metrics can be misleading without random baselines.
- **Methodology**: Train SAEs on random vs. trained Pythia models; compare metrics.
- **Datasets Used**: Pythia activation streams
- **Results**: Many SAE metrics fail to discriminate trained vs. random.
- **Code Available**: Noted in paper
- **Relevance to Our Research**: Emphasizes need for baselines when testing concept features.

### Paper 9: Sparse Autoencoders Do Not Find Canonical Units of Analysis
- **Authors**: Patrick Leask et al.
- **Year**: 2025
- **Source**: arXiv 2502.04878 (ICLR 2025)
- **Key Contribution**: SAE features are not canonical/atomic; larger SAEs add novel latents.
- **Methodology**: SAE stitching + meta-SAE analysis.
- **Datasets Used**: Transformer activation streams
- **Results**: Larger SAEs can decompose latents; highlights non-atomicity of features.
- **Code Available**: Noted in paper
- **Relevance to Our Research**: Warns against assuming single “washing machine” feature exists.

## Common Methodologies
- **Sparse Autoencoders (SAE)**: Used to extract sparse features in residual stream activations.
- **Activation Patching / Causal Tracing**: Localizes where information is used in the network.
- **Compositionality Probes**: Predict parent phrase representation from child representations.
- **Linear Probing / Steering**: Tests for linear directions representing concepts.

## Standard Baselines
- Linear probes or affine composition functions
- Randomized model baselines (random initialization or shuffled activations)
- Single-layer SAEs vs. multi-layer SAEs
- Control tasks with non-compositional phrases (idioms)

## Evaluation Metrics
- Reconstruction loss / normalized MSE (SAE)
- Sparsity metrics (active latents per token)
- Auto-interpretability scores (explanation alignment)
- Compositionality reconstruction error (tree reconstruction error / cosine distance)

## Datasets in the Literature
- **Penn Treebank phrases**: Used for local composition prediction.
- **CHIP dataset**: Human-annotated idiom compositionality scores.
- **Large LM training corpora**: Token streams used for SAE training.

## Gaps and Opportunities
- Direct tests for compound noun concepts (e.g., “washing machine”) are rare.
- SAE features are not guaranteed to be canonical; need better baselines and controls.
- Few studies explicitly test whether compound concepts are compositional combinations in residual stream directions.

## Recommendations for Our Experiment
- **Recommended datasets**: CHIP for idioms/compounds, WikiText-2 for activation streams.
- **Recommended baselines**: Randomized model baseline; single-layer SAE vs. MLSAE; linear probe for “washing” and “machine” separately.
- **Recommended metrics**: SAE reconstruction + sparsity, probing accuracy for composed vs. atomic features, causal patching effect size.
- **Methodological considerations**: Use activation patching best practices; avoid over-trusting SAE auto-interpretability metrics; compare features across layers.
