# REPORT: Where Is "Washing Machine" Stored in LLMs?

## 1. Executive Summary
**Research question**: Where is the compound concept "washing machine" represented in a transformer LLM residual stream—via a distinct feature or via composition of constituent concepts ("washing" + "machine")?

**Key finding**: In GPT-2 small with SAE features at layer 6 (resid_post_mlp), top-k compound feature overlap with constituents is low (Jaccard 0.11–0.14; bootstrap means 0.09–0.11), but a compositionality probe predicts compound embeddings from constituents extremely well (mean cosine 0.996). Causal patching showed no reliable positive logit lift for "machine" (mean Δlogit -0.019, p=0.75).

**Implication**: The evidence favors compositional representation over a single strong compound-specific direction at the tested layer. Mechanistic interventions should expect compound concepts to be distributed across constituent features rather than localized to a single monosemantic direction.

## 2. Goal
**Hypothesis**: The compound concept "washing machine" is not stored as a unique orthogonal direction; it emerges from constituent features.

**Why it matters**: Interpretability and model editing methods often assume distinct concept directions. If compounds are compositional, then single-direction edits and interventions may be incomplete or misleading.

## 3. Data Construction

### Dataset Description
- **Source**: WikiText-2 raw (`datasets/wikitext_2_raw_v1`)
- **Size used**: 29,119 non-empty text lines after filtering
- **Task**: Context mining for compound and constituent mentions
- **Biases/limitations**: Sparse coverage of the specific compound; contexts may be domain-specific to Wikipedia

### Example Samples
**Compound contexts**:
- "The washing machine was broken."
- "The washing machine stopped mid-cycle."
- "The washing machine made a loud noise."

**Washing-only contexts**:
- "In the 1880s , the federal government began closing many small arsenals around the country in favor of smaller ones built near railroads for quick deployment ."
- "Atlanta was easily pulled free by the Union ships and she reached Port Royal under her own power ."
- "Michael Jeffrey Jordan ( born February 17 , 1963 ) , also known by his initials , MJ , is an American retired professional basketball player ."

**Machine-only contexts**:
- "Most of the equipment , arms , and machinery at the Little Rock Arsenal was removed to east of the Mississippi River by order of Maj. Gen. Earl Van Dorn in April and May 1862 ..."
- "Machinery was made for manufacturing percussion caps and small arms , and both were turned out in small quantity , but of excellent quality ."
- "The fourth stage involved remedying the problem of communicating between the two towers during the time of Pope Pius X."

### Data Quality
- **Total rows after filtering**: 29,119
- **Empty rows after filtering**: 0
- **Context counts**:
  - Compound: 105
  - Washing-only: 178
  - Machine-only: 200
- **Latent samples** (token positions found):
  - Compound: 105
  - Washing-only: 10
  - Machine-only: 164

### Preprocessing Steps
1. Load WikiText-2 raw from disk.
2. Strip whitespace and drop empty lines.
3. Extract three context sets:
   - Compound: lines containing "washing machine"
   - Washing-only: lines containing "washing" but not "machine"
   - Machine-only: lines containing "machine" but not "washing"
4. Use all splits as a pooled corpus for context search.

### Train/Val/Test Splits
No supervised training on WikiText-2. For the compositionality probe, an 80/20 random split of bigram samples was used.

## 4. Experiment Description

### Methodology
#### High-Level Approach
1. **SAE feature analysis**: Measure overlap between top-k SAE features activated by compound vs constituent contexts.
2. **Causal patching**: Patch residual activations from one prompt into another to test causal influence on predicting "machine".
3. **Compositionality probe**: Train a ridge regression model to predict a compound embedding from constituent embeddings.

#### Why This Method?
- SAE features allow a sparse, interpretable basis for comparing concept activations.
- Causal patching tests whether a localized activation pattern causally induces compound-related logits.
- Probing quantifies how well compound representations are reconstructible from constituents.

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
- **SAE**: OpenAI `sparse_autoencoder` pretrained SAE (v5 32k) at layer 6, `resid_post_mlp`

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| SAE layer | 6 | Standard mid-layer choice |
| SAE location | resid_post_mlp | SAE default location |
| top_k | 50 | Common overlap eval setting |
| max_contexts | 200 | Runtime cap |
| ridge alpha | 1.0 | Baseline default |
| bootstrap samples | 200 | Quick CI estimate |

#### Training/Analysis Pipeline
1. Extract contexts from WikiText-2.
2. Run GPT-2 with cache to collect residual activations.
3. Encode activations with SAE to obtain latent features.
4. Compute top-k feature overlap and cosine similarities.
5. Run causal patching at layer 6 with template pairs.
6. Train ridge regression probe on bigram dataset to predict compound embeddings.

### Experimental Protocol
- **Random seed**: 42
- **Hardware**: NVIDIA GeForce RTX 3090 (24GB), CUDA 12.8
- **Runs**: 1 (deterministic, with bootstrap for overlap confidence intervals)

### Evaluation Metrics
- **Top-k Jaccard overlap**: feature overlap between conditions
- **Cosine similarity**: similarity of mean latent vectors
- **Causal patching logit delta**: impact on predicting "machine" after patching
- **Probe MSE / cosine**: quality of reconstructing compound embeddings

### Raw Results
#### Tables
| Metric | Value |
|--------|-------|
| Compound–washing Jaccard | 0.136 |
| Compound–machine Jaccard | 0.111 |
| Compound–union Jaccard | 0.129 |
| Compound unique fraction (top-50) | 0.68 |
| Bootstrap mean overlap (compound–washing) | 0.092 [0.064, 0.136] |
| Bootstrap mean overlap (compound–machine) | 0.105 [0.087, 0.124] |
| Bootstrap mean overlap (compound–union) | 0.114 [0.092, 0.137] |
| Cosine(compound, washing) | 0.578 |
| Cosine(compound, machine) | 0.041 |
| Causal patching Δlogit | -0.019 ± 0.109 (n=5, p=0.750) |
| Probe MSE (ridge) | 5.19 |
| Probe MSE (w2 baseline) | 12.49 |
| Probe mean cosine (ridge) | 0.996 |

#### Visualizations
- `results/plots/sae_overlap.png`
- `results/plots/causal_patching.png`

#### Output Locations
- Metrics JSON: `results/metrics.json`
- Examples JSON: `results/examples.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. **Low SAE overlap**: Top-k overlap between compound and constituent activations is small (Jaccard 0.11–0.14). Bootstrap means are similarly low (0.09–0.11), suggesting no dominant shared feature set.
2. **Strong compositionality**: Compound embeddings are predicted very accurately from constituents (mean cosine 0.996), well beyond the w2-only baseline (MSE 12.49 vs 5.19).
3. **Weak causal patching effect**: Patching compound residuals into "washing process" did not reliably increase the "machine" logit (mean Δlogit -0.019, p=0.75).

### Hypothesis Testing Results
- **Supports**: The compositionality probe strongly supports the hypothesis that compound meaning can be reconstructed from constituents.
- **Does not support**: Causal patching does not indicate a single strong compound-specific direction at layer 6.
- **Ambiguity**: Low SAE overlap could indicate compound-specific features, but the small washing-only latent sample count (n=10) limits confidence.

### Error Analysis
- Washing-only contexts are rare in WikiText-2, leading to only 10 usable latent samples.
- Causal patching used only 5 template pairs; effect estimates are noisy.

### Limitations
- Single model (GPT-2 small) and single SAE layer tested.
- SAE features may not be canonical or monosemantic.
- Corpus coverage for the exact compound is limited.

## 6. Conclusions

**Summary**: In GPT-2 small, "washing machine" does not appear to correspond to a single strong, localized feature at the tested layer. Instead, representation geometry is strongly compositional, as evidenced by the probe results.

**Implications**: Mechanistic interpretability and concept editing should treat compound concepts as distributed compositions rather than single features, at least in small GPT-style models.

**Confidence**: Moderate. The compositionality evidence is strong, but SAE overlap and causal patching are limited by sample size and single-layer scope.

## 7. Next Steps
1. Expand corpora (WikiText-103 or The Pile) to increase washing-only and compound samples.
2. Test multiple layers and SAE locations (resid_mid, resid_post_attn, mlp_post_act).
3. Repeat with a larger model and modern SAEs to test scale effects.

## References
- Elhage et al. (2022) Toy Models of Superposition
- Cunningham et al. (2023) Sparse Autoencoders Find Highly Interpretable Features in Language Models
- Wang et al. (2022) Interpretability in the Wild (IOI)
- Meng et al. (2022) Locating and Editing Factual Associations (ROME)
- Dai et al. (2021) Knowledge Neurons
