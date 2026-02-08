# Outline: Where Is "Washing Machine" Stored in LLMs?

## Abstract
- Problem: whether compound nouns are stored as single directions or composed from constituents.
- Approach: SAE feature overlap, causal patching, compositionality probe on GPT-2 small.
- Key results: low top-k feature overlap (Jaccard 0.11–0.14), strong compositional probe (cosine 0.996), weak patching effect (Δlogit -0.019 ± 0.109).
- Significance: supports compositional geometry over single-layer atomic feature.

## Introduction
- Hook: interpretability methods assume concept directions; compounds stress this assumption.
- Importance: affects steering/editing and concept localization in LLMs.
- Gap: few direct tests for concrete compounds with SAE + causal tracing + compositional probes.
- Approach: analyze GPT-2 small with SAE features and controlled compound contexts; include patching and probe; point to method figure.
- Quantitative preview: report Jaccard, cosine probe, patching delta.
- Contributions (3–4 bullets): propose testbed, conduct SAE+patching analysis, show strong compositional predictability, document limitations.

## Related Work
- Superposition and polysemanticity (Elhage et al.)
- Compositionality probes (Liu & Neubig)
- SAE scaling and non-canonicality (Gao et al.; Leask et al.; Heap et al.)
- Activation patching best practices (Zhang & Nanda)
- Linear representation hypothesis (Park et al.)
- Positioning: our concrete compound noun test combines these lines.

## Methodology
- Problem formulation: compare compound vs constituents in residual stream.
- Data: WikiText-2 raw; no natural "washing machine"; add synthetic templates.
- Model/SAE: GPT-2 small; SAE v5 32k at layer 6 resid_post_mlp.
- Metrics: top-k Jaccard, cosine of mean latents, patching Δlogit, probe MSE/cosine.
- Baselines: head-noun baseline (w2), constituent-only contexts.

## Results
- Table: main metrics.
- Figures: SAE overlap plot; causal patching plot.
- Statistical notes: Δlogit mean ± std, n=5.
- Comparison to baseline: probe vs w2 MSE.

## Discussion
- Interpretation: geometry is compositional despite low feature overlap.
- Limitations: synthetic contexts, small n, single layer/model, SAE non-canonical.
- Implications: concept interventions should be multi-feature/layer.

## Conclusion
- Summary and main takeaway.
- Future work: larger corpora, more layers/models, idiomatic compounds.

## References
- BibTeX for listed papers.
