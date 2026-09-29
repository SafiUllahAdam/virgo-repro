# virgo-repro (private)

Complete reproduction package for *Structure Aware Graph Augmentation for Graph Neural Networks*.

Start with **[AGENT.md](AGENT.md)**: environment, datasets, run order, verification and reporting, step by step.

- Code: `virgo/` (library), `experiments/` (entry points), `notebooks/` (narrative, in study order)
- Reference results: `expected/`
- Dataset records: `data/USED.csv` (used, with checksums), `data/UNUSED.csv` (record only)
- Datasets and feature cache: release assets, not in git
- Project overview and research log: `docs/`

Deterministic artifacts such as datasets, splits and role-graph definitions are checked exactly where possible, while newly-trained embeddings and resulting metrics are checked using predefined numerical tolerances.
