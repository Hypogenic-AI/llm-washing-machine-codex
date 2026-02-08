# LLM Washing Machine

This project tests whether the compound concept "washing machine" appears as a distinct feature in GPT-2 residual streams or emerges via composition of constituent concepts ("washing" + "machine"). We use sparse autoencoder (SAE) features, causal patching, and a compositionality probe.

Key findings:
- SAE top-k feature overlap between compound and constituent contexts is low (Jaccard 0.11–0.14).
- A compositionality probe predicts compound embeddings from constituents extremely well (mean cosine 0.996).
- Causal patching at layer 6 shows no reliable positive logit boost for "machine" (mean Δlogit -0.019, p=0.75).

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
- Examples: `results/examples.json`
- Plots: `results/plots/`
- Report: `REPORT.md`

## File Structure
- `src/run_experiments.py`: end-to-end experiment pipeline
- `results/metrics.json`: metrics summary (includes environment and data quality)
- `results/examples.json`: sample contexts
- `results/plots/`: figures
- `REPORT.md`: full research report

See `REPORT.md` for full methodology, analysis, and limitations.
