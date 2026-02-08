# Planning

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding how compound concepts (e.g., "washing machine") are represented in LLMs is central to mechanistic interpretability and reliable concept-level interventions. If compound nouns are not stored as distinct directions, then steering, editing, and safety interventions that assume atomic directions may be miscalibrated. This work also informs compositional generalization and how meaning is built across layers.

### Gap in Existing Work
Prior work studies linear directions, superposition, and phrase compositionality, but direct tests on compound-noun concepts in residual streams are rare. SAE papers show features are not canonical and can be polysemantic, while compositionality probes focus on constituent predictability rather than feature localization for concrete compounds like "washing machine".

### Our Novel Contribution
We directly test whether compound noun concepts correspond to distinct latent features or emerge from compositional interaction of constituent features, using SAE feature attribution and causal patching on real model activations.

### Experiment Justification
- Experiment 1: SAE feature activation analysis for "washing", "machine", and "washing machine" contexts to test whether a distinct compound feature exists.
- Experiment 2: Causal tracing / activation patching to measure whether intervening on constituent features reproduces the compound concept effect without a dedicated compound feature.
- Experiment 3: Compositionality probe predicting compound phrase embeddings from constituent embeddings to quantify compositional structure and compare with feature-localization evidence.

## Research Question
Where is the compound concept "washing machine" represented in LLMs: as a distinct direction/feature or as a composition of constituent features ("washing" + "machine")?

## Background and Motivation
Mechanistic interpretability increasingly relies on concept directions and feature localization. However, superposition and polysemanticity suggest that not every concept corresponds to a unique direction. Compound nouns provide a concrete testbed for whether concept representations are atomic or compositional in residual streams.

## Hypothesis Decomposition
- H1: There is no single orthogonal residual-stream direction uniquely associated with "washing machine" across contexts.
- H2: Activation patterns for "washing machine" can be approximated by combining constituent features ("washing" and "machine").
- H3: Interventions on constituent features reproduce behavioral effects similar to the compound without requiring a dedicated compound feature.

## Proposed Methodology

### Approach
Use mechanistic interpretability tools (TransformerLens + SAE features) to analyze activations and causal impact for "washing machine" contexts. Combine SAE feature inspection with causal patching and compositionality probes.

### Experimental Steps
1. Data collection: Extract contexts containing "washing", "machine", and "washing machine" from WikiText-2; create matched control contexts. Rationale: Controlled contexts reduce confounds.
2. SAE feature analysis: Use pretrained SAEs (openai/sparse_autoencoder) to obtain latent activations; identify top-activating features for each condition. Rationale: Test existence of a compound-specific latent.
3. Causal tracing: Patch activations or SAE latents from constituent contexts into compound contexts (and vice versa). Rationale: Test whether constituent features are sufficient to induce compound behavior.
4. Compositionality probe: Train linear/affine probes to predict compound phrase embedding from constituent embeddings. Rationale: Quantify compositional structure of phrase representations.

### Baselines
- Randomized model or shuffled activations baseline for SAE metrics.
- Linear probe using only constituent embedding averages.
- Single-layer SAE vs. multi-layer SAE features (if available).

### Evaluation Metrics
- SAE reconstruction loss and sparsity by condition.
- Feature overlap metrics (Jaccard of top-k features) across conditions.
- Causal effect size from patching (logit diff / probability change).
- Probe reconstruction error (cosine distance / MSE).

### Statistical Analysis Plan
- Bootstrap confidence intervals for overlap and patching effects.
- Paired t-tests or Wilcoxon signed-rank tests for condition comparisons.
- Significance level α = 0.05 with FDR correction for multiple comparisons.

## Expected Outcomes
- Support hypothesis if compound contexts do not yield unique SAE features and if constituent feature patching reproduces compound effects.
- Refute if a consistent, unique compound feature dominates and cannot be replicated by constituent features.

## Timeline and Milestones
- Resource review and setup: 0.5 hr
- Data extraction and preprocessing: 0.5 hr
- SAE analysis + patching: 1.5 hr
- Probing experiments: 1.0 hr
- Analysis + write-up: 1.0 hr

## Potential Challenges
- Pretrained SAE compatibility with target model and tokenizer.
- Limited compound examples in WikiText-2.
- Activation patching sensitivity to hyperparameters.

## Success Criteria
- Complete SAE and causal patching experiments with quantitative comparisons.
- Clear statistical conclusion on presence/absence of compound-specific feature.
- Reproducible code and documented results in REPORT.md.
