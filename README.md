# virgo-repro

Reproduction package for *Structure Aware Graph Augmentation for Graph Neural Networks*.

Start with **[AGENT.md](AGENT.md)**: where everything lives, the three verification levels, and how to report.
 
| place | holds |
|---|---|
| this repository | code, instructions, checksums, final result tables (`expected/`) |
| release `v1.0-data` | datasets (`input/`, `labels/`) and the feature cache |
| original SSH server (one copy) | large embeddings and generated artifacts (~72 GB), listed with SHA256 in `expected/hashes/large_artifacts.csv` |

Large embeddings and generated artifacts are preserved externally; download/restore them only if needed. Normal reproducibility verification uses the stored final CSVs, hashes, and caches and does not retrain models.

Deterministic artifacts such as datasets, splits and role-graph definitions are checked exactly where possible, while newly-trained embeddings and resulting metrics are checked using predefined numerical tolerances. 
