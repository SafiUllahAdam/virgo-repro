# Structure Aware Graph Augmentation for Graph Neural Networks

Code, reference results and verification tools for the paper.

## Overview

We propose a **pre-training framework** that decides, before any GNN is trained, whether a graph should be augmented
with structural-role edges, and if so, which structural signal to build them from. A role graph links each node to its
K = 10 most structurally similar nodes by degree, eigenvector centrality or Identity2Vec's Ψ. This gives the encoder
direct access to similar nodes outside its receptive field. Both decisions use properties of the original graph only.

```
input graph -> Stage 1: augment? --no--> train the encoder on the original graph
                    | yes
                    v
               Stage 2: which signal? -> build the role graph -> train the encoder
```

## Decision rules

The rules are frozen in [`virgo/frozen_rules.py`](virgo/frozen_rules.py). Each threshold is the midpoint of the range
of cuts that fit the training graphs equally well.

**Stage 1: augment or keep the original graph?**

| rule | condition | threshold (range) | interpretation |
|---|---|---|---|
| 1.1 | adjusted homophily < τ ⇒ augment, otherwise keep | 0.227 (0.093 to 0.361) | when edges already follow the labels, the original graph is the better guide |
| 1.2 | if 1.1 says augment: average clustering ≥ τ ⇒ keep | 0.5573 (0.5329 to 0.5769) | triangle-rich neighbourhoods already carry the local structure |
| 1.3 | graph without labels: largest-component share > τ ⇒ augment | 0.9588 (0.9177 to 1.0) | fallback when homophily cannot be computed |

**Stage 2: which signal?** Applied only when Stage 1 says augment; requires node labels.

| rule | condition | threshold (range) | interpretation |
|---|---|---|---|
| 2.1 | adjusted neighbour-label predictability > τ ⇒ centrality | 0.0092 (0.0060 to 0.0123) | a node's class can be guessed from its neighbours' label mix better than chance |
| 2.2 | otherwise: neighbour-label entropy < τ ⇒ degree | 0.6724 (0.6296 to 0.7151) | the labels around each node are concentrated rather than mixed |
| | otherwise | | no recommendation between Ψ and degree |

**Validation.** On held-out graphs, rule 1.1 was correct on all its "keep" calls (3/3) but on only half of its
"augment" calls (2/4), which motivates rule 1.2 (4/4 decided calls on six unseen graphs). The Stage 2 rules are fitted
candidates with limited held-out evidence (2.1: 3/4; 2.2: 6/9). Rejected rules and full records are in
`virgo/frozen_rules.py` and [`docs/paper_log.md`](docs/paper_log.md).

## Main results

Structure only: published node features are ignored, so any gain comes from the added edges.

| setting | result |
|---|---|
| GraphSAGE, link prediction, 19 graphs, 10 seeds | augmentation wins on 9, ties on 4, and the original is better on 6 |
| GraphSAGE, node classification, 15 graphs, 10 seeds | no significant gain on any graph (0 wins, 2 ties) |
| GATv2, 30 graphs, 3 seeds | augmentation helps on 13 of 30 (21 of 30 with GraphSAGE) and recovers 8 of the 9 graphs where GATv2 alone fails |
| community structure, 15 social graphs | stronger communities, smaller gain from augmentation (Spearman ρ = −0.82 GraphSAGE, −0.81 GATv2; p = 0.0002) |

Augmentation remains useful on sparse, hub-dominated, weakly clustered graphs. Role edges typically join nodes 3 to 6
hops apart, beyond the reach of a two-layer GNN. Details: [`docs/project_overview.md`](docs/project_overview.md).

## Repository

| path | contents |
|---|---|
| `virgo/` | library: settings, frozen rules, role-graph builder, Identity2Vec, encoders (GraphSAGE, GATv2, GIN), evaluation |
| `experiments/` | one CLI per analysis; each script either fits or applies rules, never both ([`experiments/README.md`](experiments/README.md)) |
| `notebooks/1-9` | the study in order, from the Identity2Vec reproduction to the community analysis |
| `expected/` | reference result tables and SHA256 checksums; [`expected/PRODUCERS.csv`](expected/PRODUCERS.csv) maps each table to its script |
| `data/` | dataset records (`USED.csv`), every training run (`RUNS.csv`), restore script |
| `verify/compare.py` | checks files and result tables against `expected/` |
| `env/`, `docs/` | exact environment; research log, notes, thesis |

Datasets (72 trained graphs, from citation to transport networks) are described in [`DATASETS.md`](DATASETS.md).

## Reproducibility

Full protocol: [`AGENT.md`](AGENT.md).

| level | scope | requirements |
|---|---|---|
| 1 | verify datasets, feature cache and all result tables, no retraining | Python 3, pandas, `sha256sum`, `tar` |
| 2 | restore stored embeddings and re-score them | access to the original server copy (on request) |
| 3 | retrain from scratch | environment below, GPU recommended |

**Level 1.** Download the release `v1.0-data` assets into `release/`, then run `bash data/fetch_all.sh release`. Every
checked file should report `EXACT`.

**Level 3.**

```bash
conda create -n i2v --file env/conda-explicit.txt && conda activate i2v
pip install -r env/pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130
python verify/compare.py --step all     # after running the commands in data/RUNS.csv
```

Original environment: Python 3.12.13, PyTorch 2.12.0 (CUDA 13.0), PyTorch Geometric 2.8.0, one NVIDIA RTX PRO 6000
GPU ([`env/versions.txt`](env/versions.txt)). Seeds: 42 to 44 (3 seeds) or 42 to 51 (10 seeds).

**Checks.** Deterministic files are compared by SHA256. Retrained metrics must match `expected/` within 0.005, above
run-to-run and code-path noise (at most about 0.003) and well below seed variation (about 0.1).

## Acknowledgements

Identity2Vec (Oluigbo et al.) provides Ψ. Benchmarks from Platonov et al. (2023) and Lim et al. (2021). struc2vec is
included unchanged in `third_party/`.

## Citation

To be added on publication.
