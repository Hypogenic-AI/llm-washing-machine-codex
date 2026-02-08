# LLM Washing Machine

This project investigates whether the compound concept "washing machine" is represented as a distinct direction in LLM residual streams or arises from compositional features. We analyze GPT-2 small with SAE features, causal patching, and a compositionality probe.

Key findings:
- SAE top-k feature overlap between compound and constituent contexts is low (Jaccard ~0.11–0.14).
- A compositionality probe predicts compound embeddings from constituents very well (cosine 0.996).
- Causal patching at layer 6 shows no positive logit boost for "machine" (mean Δlogit -0.019).

## How To Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Run experiments:
```bash
python src/run_experiments.py
```
3. View outputs:
- Metrics: `results/metrics.json`
- Plots: `results/plots/`
- Report: `REPORT.md`

## File Structure
- `src/run_experiments.py`: end-to-end experiment pipeline
- `results/metrics.json`: metrics summary
- `results/data_stats.json`: dataset statistics
- `results/env.json`: environment versions
- `results/plots/`: figures
- `REPORT.md`: full research report

See `REPORT.md` for detailed methodology, analysis, and limitations.
