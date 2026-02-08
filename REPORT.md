# REPORT: Where Is "Washing Machine" Stored in LLMs?

## 1. Executive Summary
We tested whether the compound concept "washing machine" is represented as a distinct direction in the residual stream or emerges from constituent features ("washing" + "machine") in a GPT-2 small model with sparse autoencoder (SAE) features.
Key findings: SAE top-feature overlap between compound and constituent contexts was low (Jaccard 0.11–0.14), while a compositionality probe strongly predicted compound embeddings from constituent embeddings (cosine 0.996). Causal patching from "washing machine" into "washing process" did not increase the logit for "machine" (mean Δlogit = -0.019), suggesting weak localized causal dependence at the tested layer.
Practical implication: Compound noun behavior appears more compositional in representation geometry than as a single, strong, causal feature at one layer; mechanistic interventions should consider composition across features and layers rather than assuming a single direction.

## 1.5. Research Question & Hypothesis
**Research question**: Where is the compound concept "washing machine" stored in LLMs—distinct residual direction or composition of constituent features?
**Hypothesis**: The compound does not correspond to a unique orthogonal direction; it emerges from constituent features.

## 2. Goal
**Hypothesis**: The compound concept "washing machine" is not stored as a distinct, orthogonal residual-stream direction; instead, constituent features ("washing", "machine") compose to yield compound meaning.
**Importance**: This impacts how we localize concepts, steer models, and interpret feature-level interventions in LLMs.
**Problem solved**: Tests whether a concrete compound noun maps to a single feature vs. compositional structure.
**Expected impact**: More realistic assumptions about feature localization and composition in interpretability work.

## 2.5. Literature Review Summary
- Superposition and polysemanticity (Elhage et al., 2022) imply many concepts will not have clean, orthogonal directions.
- Compositionality probes (Liu & Neubig, 2022) show phrase representations are often predictable from constituents.
- SAE work (Gao et al., 2024; Leask et al., 2025) indicates learned features are not canonical units, motivating caution when interpreting a single latent as a concept.
- Activation patching best practices (Zhang & Nanda, 2023) highlight sensitivity to localization hyperparameters.
- Linear representation formalization (Park et al., 2024) provides a framework for testing when directions are meaningful.

These findings motivate combining SAE analysis, causal tracing, and compositionality probing to test whether a compound noun behaves as an atomic concept or a composition.

## 3. Data Construction

### Dataset Description
- **Primary source**: WikiText-2 raw (`datasets/wikitext_2_raw`)
- **Size**: train 36,718; validation 3,760; test 4,358 records
- **Collection**: Pre-downloaded from HuggingFace; raw text lines
- **Bias/limitations**: Many empty lines; sparse coverage of the specific compound "washing machine" in the corpus.

### Example Samples
```text
Sample 1: "Senjō no Valkyria 3 : Unrecorded Chronicles ..."
Sample 2: "The game began development in 2010 ..."
Sample 3: "As with previous Valkyria Chronicles games, Valkyria Chronicles III is ..."
```

### Data Quality
- Missing or empty lines: train 12,951; validation 1,299; test 1,467
- Mean length: train 296.7; validation 303.8; test 295.0 characters
- Min length: 0; max length: 3,863 (train)
- Validation checks: non-empty filtering; basic length statistics saved in `results/data_stats.json`

### Preprocessing Steps
1. Loaded dataset from disk and concatenated splits.
2. Filtered empty lines and stripped whitespace for context search.
3. Extracted three context sets:
   - Compound: lines containing "washing machine"
   - Washing-only: "washing" without "machine"
   - Machine-only: "machine" without "washing"
4. Because WikiText-2 contained no "washing machine" strings, added synthetic compound sentences (templated) to ensure controlled compound contexts.

### Train/Val/Test Splits
- Used all splits for context extraction (no training in the ML sense).
- For compositionality probe, used an 80/20 random split of extracted bigram samples.

## 4. Experiment Description

### Methodology

#### High-Level Approach
Combine SAE feature analysis, causal patching, and a compositionality probe to test whether compound meanings are represented as a distinct latent or as a composition of constituent features.

#### Why This Method?
- SAE feature activations test for latent features correlated with the compound.
- Causal patching tests whether transferring compound activations causes compound-specific predictions.
- Compositionality probes quantify how well compound representations are predicted from constituents.

### Implementation Details

#### Tools and Libraries
- PyTorch 2.10.0+cu128
- TransformerLens 2.15.4
- Transformers 4.57.6
- Datasets 4.5.0
- scikit-learn 1.8.0
- blobfile 3.2.0
- matplotlib 3.10.8

#### Algorithms/Models
- **Model**: GPT-2 small (`gpt2`) loaded via TransformerLens
- **SAE**: Pretrained SAE weights from OpenAI `sparse_autoencoder` (v5 32k) for layer 6, resid_post_mlp

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| SAE layer | 6 | Prior examples in SAE repo |
| SAE location | resid_post_mlp | Standard for SAE features |
| top_k | 50 | Common feature overlap evaluation |
| max_contexts | 200 | Dataset size limit |
| ridge alpha | 1.0 | Default baseline |

#### Training Procedure or Analysis Pipeline
1. Extract contexts from WikiText-2 and synthetic compound prompts.
2. Run GPT-2 with cache; collect residual activations at layer 6.
3. Encode activations with SAE to obtain latent features.
4. Compute top-k feature overlap and cosine similarities.
5. Run causal patching at layer 6.
6. Train ridge regression probe predicting compound embeddings from constituent embeddings.

### Experimental Protocol

#### Reproducibility Information
- Runs: single deterministic run
- Random seed: 42
- Hardware: NVIDIA GeForce RTX 3090 (24GB), CUDA 12.8
- Runtime: a few minutes for full pipeline

#### Evaluation Metrics
- **Top-k Jaccard overlap**: measures shared SAE features across conditions
- **Cosine similarity**: compares mean latent vectors between conditions
- **Causal patching logit delta**: effect on predicting "machine" after patching
- **Ridge probe MSE / cosine**: compositional predictability of compound embedding

### Raw Results

#### Tables
| Metric | Value |
|--------|-------|
| Compound–washing Jaccard | 0.136 |
| Compound–machine Jaccard | 0.111 |
| Compound–union Jaccard | 0.129 |
| Compound unique fraction (top-50) | 0.68 |
| Cosine(compound, washing) | 0.578 |
| Cosine(compound, machine) | 0.041 |
| Causal patching Δlogit | -0.019 ± 0.109 (n=5) |
| Probe MSE (ridge) | 5.19 |
| Probe MSE (w2 baseline) | 12.49 |
| Probe cosine (ridge) | 0.996 |

#### Visualizations
- `results/plots/sae_overlap.png`
- `results/plots/causal_patching.png`

#### Output Locations
- Metrics JSON: `results/metrics.json`
- Data stats: `results/data_stats.json`
- Environment: `results/env.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. **Low SAE feature overlap**: Compound top-k features overlap weakly with washing or machine alone (Jaccard ~0.11–0.14), and 68% of top-50 compound features were unique relative to constituent top-50 sets.
2. **Compositionality probe success**: Ridge regression predicts compound embeddings from constituents far better than a baseline using only w2 (MSE 5.19 vs 12.49, cosine 0.996), indicating strong compositional structure.
3. **Weak causal patching effect**: Patching compound residuals into "washing process" did not increase the logit for "machine" (mean Δlogit -0.019), suggesting no strong single-layer causal trigger for "machine" at layer 6.

### Hypothesis Testing Results
- **Support**: The strong compositional probe result supports the hypothesis that compound meaning is compositional.
- **Ambiguous**: The low SAE overlap could suggest a distinct compound feature, but the synthetic contexts and SAE non-canonicality make this inconclusive.
- **Causal evidence**: Patch results do not show a clear compound-specific causal direction at the tested layer.

### Comparison to Baselines
- Probe outperforming w2 baseline suggests additive compositionality beyond simply copying the head noun.
- SAE uniqueness should be interpreted cautiously due to small and synthetic compound sample size.

### Visualizations
- See `results/plots/sae_overlap.png` for overlap comparisons.
- See `results/plots/causal_patching.png` for patching effect size.

### Surprises and Insights
- WikiText-2 contains no literal "washing machine" strings; synthetic prompts were required.
- The compositionality probe indicates high predictability even when SAE overlap is low, highlighting that SAE features may be polysemantic or non-canonical.

### Error Analysis
- Limited compound samples (synthetic only) restrict ecological validity.
- Washing-only contexts are rare in WikiText-2; only 10 usable latent samples were extracted.

### Limitations
- Single model (GPT-2 small) and single layer tested.
- SAE features are not guaranteed to be canonical units.
- Compound contexts are synthetic rather than natural corpus occurrences.
- Patching uses a small set of templates (n=5 pairs).

## 6. Conclusions

### Summary
Evidence from compositionality probing suggests compound embeddings are largely predictable from constituent embeddings, aligning with a compositional representation. SAE feature overlap and causal patching did not reveal a strong, unique compound feature at the tested layer.

### Implications
- **Practical**: Concept editing and steering should consider multi-feature and multi-layer composition instead of relying on a single direction.
- **Theoretical**: Supports the view that compound meanings are constructed rather than stored as atomic residual directions.

### Confidence in Findings
Moderate. The compositionality signal is strong, but SAE and patching evidence is limited by data scarcity and synthetic contexts.

## 7. Next Steps

### Immediate Follow-ups
1. Expand compound contexts using larger corpora (WikiText-103 or The Pile) to avoid synthetic prompts.
2. Repeat SAE analysis across multiple layers and SAE types (single-layer vs multi-layer SAEs).

### Alternative Approaches
- Use Neuronpedia or Gemma Scope SAEs on newer open models with richer corpora.
- Evaluate activation patching across multiple locations (attn_out, resid_mid, resid_post).

### Broader Extensions
- Compare compounds of varying compositionality (literal vs idiomatic) using CHIP.
- Test multiple model families to check consistency of compound representation.

### Open Questions
- Are compound-specific features more detectable in larger models or later layers?
- Do idiomatic compounds behave differently in SAE and patching analyses?

## References
- See `literature_review.md` and `resources.md` for primary papers and datasets.
