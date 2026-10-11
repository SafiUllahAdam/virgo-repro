# Structure Aware Graph Augmentation for Graph Neural Networks

Code, reference results and verification tooling for the study *Structure Aware Graph Augmentation for Graph Neural
Networks*.

## Overview

Structural-role augmentation adds edges between nodes that occupy similar topological roles (for example, two hubs or
two peripheral nodes), even when no path joins them. Because a message-passing GNN only exchanges information along
existing edges, such nodes are otherwise never compared directly. The resulting **role graph** gives them a channel.

The study asks when this augmentation should be applied, and with which structural signal. It proposes a two-stage
decision framework that reads only the properties of the original graph and is applied before any training:

1. **Stage 1: augment or keep the original.** Adjusted homophily below 0.227 indicates augmentation. Among
   low-homophily graphs, average clustering of at least 0.5573 indicates keeping the original.
2. **Stage 2: choose the signal.** When Stage 1 calls for augmentation, neighbour-label statistics select between
   eigenvector centrality and degree. Identity2Vec's Ψ is the third candidate signal.

The rules are frozen in [`virgo/frozen_rules.py`](virgo/frozen_rules.py). Each threshold is the midpoint of a fitted
interval and should be read as a range.

## Main results

All results use structure only: published node features are ignored. Each role graph links every node to its K = 10
most similar nodes.

| setting | finding |
|---|---|
| GraphSAGE, 10 seeds, link prediction (19 graphs) | augment: 9 graphs · tie: 4 · keep original: 6 |
| GraphSAGE, 10 seeds, node classification (15 graphs) | augment: 0 graphs · tie: 2 · keep original: 13. No graph showed a significant gain |
| GATv2, 3 seeds, 30 graphs | augmentation helps on 13 of 30 graphs (21 of 30 under GraphSAGE); where GATv2 alone falls short, augmentation recovers 8 of 9 graphs |
| Louvain communities, 15 social graphs | community strength (modularity) is strongly anti-correlated with the link-prediction gain from augmentation, under both encoders (Spearman ρ = −0.82 for GraphSAGE and −0.81 for GATv2, both p = 0.0002). This is a descriptive pattern, not a validated rule |

Augmentation remains useful on sparse, hub-dominated, weakly clustered graphs. Role edges typically join nodes 3–6
hops apart, beyond the receptive field of a two-layer GNN. Full results and caveats are given in
[`docs/project_overview.md`](docs/project_overview.md) and the dated research log
[`docs/paper_log.md`](docs/paper_log.md).

## Method

```
input graph → structural signals → role graph (top-K, K = 10) → encoder → 64-d embeddings → evaluation
              degree, eigenvector                                GraphSAGE,                     NC: weighted F1
              centrality, Ψ, clustering                          GATv2, GIN,                    LP: AUC
                                                                 DeepWalk
```

| variant | edge set | signal |
|---|---|---|
| `original` | input graph (control) | – |
| `psi`, `degree`, `centrality` | role edges replace the original edges | Ψ, degree, eigenvector centrality |
| `hybrid`, `hybrid_degree`, `hybrid_centrality` | role edges are added to the original edges | Ψ, degree, eigenvector centrality |

- **Link prediction** uses a 70/30 edge split. The role graph is rebuilt from training edges only, so no test edge
  can leak into it.
- **Node classification** trains a logistic regression on the frozen embeddings.
- **Identity2Vec** (Oluigbo et al.) is reproduced exactly. A cached implementation produces byte-identical embeddings
  about 200× faster.
- **Datasets:** 72 graphs are trained (83 registered), spanning citation, social, web, co-purchase, biological,
  molecular and transport networks. Panels, sources and citations are listed in [`DATASETS.md`](DATASETS.md).

## Repository layout

| path | contents |
|---|---|
| `virgo/` | library: configuration (`config.py`), frozen rules, graph I/O, Identity2Vec, role-graph builder, encoders, evaluation, dataset builders |
| `experiments/` | command-line entry points, one per analysis. Each script either fits a rule or applies frozen rules, never both ([`experiments/README.md`](experiments/README.md)) |
| `notebooks/1-9` | narrative pipeline in study order, from the Identity2Vec reproduction to the community analysis |
| `expected/` | reference result tables from the original run; [`expected/PRODUCERS.csv`](expected/PRODUCERS.csv) maps each table to its script |
| `expected/hashes/` | SHA256 checksums of splits, role graphs, the feature cache and final results, plus a manifest of externally held artifacts |
| `results/` | community-analysis tables (Module 17) read by notebook 9; all other outputs are regenerated locally |
| `data/` | dataset records with checksums (`USED.csv`), every training run behind the scoreboard (`RUNS.csv`), and the restore script |
| `verify/compare.py` | verifies files against recorded hashes and result tables against `expected/` |
| `env/` | exact software environment of the original run |
| `docs/` | project overview, research log, run notes, thesis and slides |
| `third_party/struc2vec` | vendored baseline, unmodified |

## Reproducibility

Detailed instructions, rules and reporting format are in [`AGENT.md`](AGENT.md). Verification is organised in three
levels. Each level is needed only when the one below it cannot answer the question.

| level | purpose | requirements |
|---|---|---|
| 1 | verify datasets, feature cache and all final result tables (no retraining) | Python 3 with pandas, `sha256sum`, `tar` |
| 2 | restore stored embeddings or splits and re-score them without training | access to the original server copy (on request) |
| 3 | retrain from scratch | the environment below, GPU recommended |

### Level 1: standard verification

Download the assets of the GitHub release `v1.0-data` (`virgo-input-labels.tar.gz`, `virgo-feature-cache.tar.gz`,
`SHA256SUMS`) into `release/`, then run:

```bash
bash data/fetch_all.sh release
```

The script checks the archives, unpacks `input/`, `labels/` and `output/feature_cache/`, and hash-checks every dataset
file, the feature cache and every table in `expected/`. Every checked file should report `EXACT`. The fifteen label
files with no recorded hash, listed in `AGENT.md`, are regenerated on first use.

### Level 3: retraining

```bash
conda create -n i2v --file env/conda-explicit.txt
conda activate i2v
pip install -r env/pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The original environment was Linux x86_64, Python 3.12.13, PyTorch 2.12.0 (CUDA 13.0), PyTorch Geometric 2.8.0,
NumPy 1.26.4, SciPy 1.12.0, NetworkX 3.6.1, gensim 4.3.3, scikit-learn 1.9.0 and pandas 3.0.3, on one NVIDIA RTX PRO
6000 Blackwell GPU (see [`env/versions.txt`](env/versions.txt)). Training commands and seeds for every reported
number are listed in [`data/RUNS.csv`](data/RUNS.csv). The analysis order is given in `AGENT.md` §4. Then verify:

```bash
python verify/compare.py --step all
```

### Verification criteria

- **Deterministic artifacts** are compared by SHA256: datasets, splits, role graphs, the feature cache and final
  tables. One known exception: the eigenvector solver may fall back to a random start, so a Ψ role graph's hash can
  differ without a material change in any metric.
- **Newly trained embeddings** are not compared bit-for-bit, because GPU training is not bit-reproducible. Instead,
  their metrics are compared with `expected/`. Text columns (verdicts, calls, winners) must match exactly. Numeric
  columns must agree within 0.005.
- **Why 0.005:** measured run-to-run variation is about 1e-7 for an identical repeat, 7.6e-6 for a thread-count
  change and 2.7e-3 for a code-path change, against about 1e-1 across seeds.

Seeds are 42 for single runs, 42–44 for three-seed runs and 42–51 for ten-seed runs.

### Data availability

| location | contents | size |
|---|---|---|
| this repository | code, instructions, checksums, reference result tables | ~27 MB |
| GitHub release `v1.0-data` | datasets (`input/`, `labels/`) and the feature cache | ~230 MB |
| original server (single copy, available on request) | trained embeddings and generated artifacts, each listed with its SHA256 in `expected/hashes/large_artifacts.csv` | ~72 GB |

## Acknowledgements

- **Identity2Vec:** Oluigbo et al., *Learning mesoscopic structural identity representations via a Poisson
  probability metric*. Source of Ψ and the walk baseline (`virgo/identity2vec.py`, unmodified).
- **Heterophilous benchmarks:** Platonov et al., 2023. **LINKX / Facebook100:** Lim et al., 2021.
- **struc2vec:** vendored in `third_party/` and used as published.

## Citation

Citation details will be added on publication.
