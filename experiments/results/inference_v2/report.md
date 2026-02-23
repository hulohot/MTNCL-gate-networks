# Inference v2 Sweep

Implemented improvements: richer encoding, curriculum stages, multistart training, ensembles, repeated-seed stats.

| stage | classes | feature | test mean ± std | ensemble | logic gates mean |
|---|---:|---|---:|---:|---:|
| s2_4class | 4 | hybrid2 | 43.8% ± 6.2% | 46.9% | 52 |

## Takeaways
- Hybrid/multibit features should outperform pure binary as class count grows.
- Variance across seeds remains high at low sample counts; use mean/std for comparisons.
- Ensemble voting often stabilizes small-circuit predictions.
