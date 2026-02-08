# Outline: Where Is "Washing Machine" Stored in LLMs?

## Abstract
- Problem: whether compound nouns are stored as distinct residual-stream features or composed from constituents.
- Approach: SAE feature overlap, causal patching, and a compositionality probe on GPT-2 small (layer 6 resid_post_mlp SAE).
- Key results: low top-k overlap (Jaccard 0.11–0.14; bootstrap means 0.09–0.11), strong probe reconstruction (mean cosine 0.996; MSE 5.19 vs 12.49 baseline), and weak patching effect (Δlogit -0.019 ± 0.109, p=0.75).
- Significance: evidence favors compositional representations at the tested layer.

## Introduction
- Hook: concept-direction assumptions drive interpretability and editing, but compounds challenge that assumption.
- Importance: impacts how we localize and intervene on concepts in LLMs.
- Gap: limited direct tests of concrete compound nouns using SAE features plus causal and probing evidence.
- Approach: analyze WikiText-2 contexts for "washing machine", "washing", and "machine"; combine SAE overlap, causal patching, and linear probe; cite method figure.
- Quantitative preview: Jaccard 0.11–0.14, probe cosine 0.996, patching Δlogit -0.019.
- Contributions (3–4 bullets): propose a concrete compound testbed, conduct SAE + patching analysis, demonstrate strong compositionality with probe, document limitations and implications.

## Related Work
- Superposition and polysemanticity (Elhage et al.).
- Sparse feature discovery with SAEs and dictionary learning (Cunningham et al.; Anthropic).
- Causal localization and editing (IOI; ROME).
- Neuron-level attribution (Knowledge Neurons).
- Positioning: we connect sparse features with compositionality tests for a concrete compound.

## Methodology
- Problem formulation: compare compound vs constituent activations in the residual stream.
- Data: WikiText-2 raw; extract compound, washing-only, and machine-only contexts; report counts.
- Model/SAE: GPT-2 small; SAE v5 32k at layer 6 resid_post_mlp.
- Metrics: top-k Jaccard, cosine of mean latents, causal patching Δlogit, probe MSE/cosine.
- Baselines: w2-only probe baseline.
- Implementation: hyperparameters, seed, hardware.

## Results
- Table: main metrics with Jaccard, bootstrap CIs, cosines, patching Δlogit, probe results.
- Figures: SAE overlap plot and causal patching plot.
- Statistical notes: bootstrap CIs and patching p-value.

## Discussion
- Interpretation: low feature overlap but near-perfect probe suggests distributed composition.
- Limitations: few washing-only latents (n=10), only one layer and model, SAE non-canonicality, small patching sample (n=5).
- Implications: concept edits should target multiple features/layers.

## Conclusion
- Summary of findings and main takeaway.
- Future work: larger corpora, multi-layer analysis, larger models.

## References
- BibTeX for cited papers.
