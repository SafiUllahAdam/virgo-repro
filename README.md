# Structure Aware Graph Augmentation for Graph Neural Networks

Code, reference results and verification tools for the paper.

## Overview

We propose a **pre-training framework** that decides, before any GNN is trained, whether a graph should be augmented
with structural-role edges, and if so, which structural signal to build them from. A role graph links each node to its
K = 10 most structurally similar nodes by degree, eigenvector centrality or Identity2Vec's Ψ. This gives the encoder
direct access to similar nodes outside its receptive field. Both decisions use properties of the original graph only.

## The framework

```
graph ──► STAGE 1: augment, or keep the original?
             │                        │
       keep original               augment
             │                        ▼
             │            STAGE 2: which structural signal?
             ▼                        ▼
   train on the original      build that role graph, then train
```

Both stages read properties of the **original graph** only, so nothing is built or trained before the call. The rules are frozen in `virgo/frozen_rules.py`. Each cut is the midpoint of a fitted interval, so read it as a range rather than a constant.

### Stage 1 - augment or not?

| step | rule | reading |
|---|---|---|
| **decision** | adjusted homophily **< 0.227** ⇒ augment | When edges follow the labels, the original graph is the better guide - a reliable **veto**. Low homophily is necessary, not sufficient. |
| **exception** *(low-homophily graphs only)* | average clustering **≥ 0.5573** ⇒ keep original | Dense, triangle-rich neighbourhoods already carry the local structure. |

**When clustering matters:** only after homophily says augment. Low homophily usually means role edges will help, but not always. If the graph is also highly **clustered** (a node's neighbours are mostly linked to each other, forming many triangles), each neighbourhood already holds the local structure the encoder needs, so role edges add little and the original graph should be kept. If clustering is low, the augment call stands.

### Stage 2 - which signal? *(only when stage 1 says augment)*

Three structural signals can build the role graph: **degree** (how many neighbours a node has), **eigenvector centrality** (how influential a node is through well-connected neighbours), and **Ψ** (Identity2Vec's score comparing the shape of two nodes' neighbourhoods).

| signal | rule | reading |
|---|---|---|
| **centrality** | adjusted neighbour-label predictability **> 0.0092** | A node's neighbours' label mix predicts its class better than chance. |
| **degree** | neighbour-label entropy **< 0.6724** | The labels around each node are concentrated, not mixed. |

**The degree rule is family-dependent**: it works only on graphs similar to the ones it was learned from, that is, small to medium-sized graphs that are **unipartite** (one type of node), **sparse** (low density), **nearly connected** (one dominant connected component), **disassortative** (hubs mostly connect to low-degree nodes) and have only a few classes.

## Main results

Structure only: published node features are ignored, so any gain comes from the added edges.

| setting | result |
|---|---|
| GraphSAGE, link prediction, 19 graphs, 10 seeds | augmentation wins on 9, ties on 4, and the original is better on 6 |
| GraphSAGE, node classification, 15 graphs, 10 seeds | no significant gain on any graph (0 wins, 2 ties) |
| GATv2, 30 graphs, 3 seeds | augmentation helps on 13 of 30 (21 of 30 with GraphSAGE) and recovers 8 of the 9 graphs where GATv2 alone fails |
| community structure, 15 social graphs | stronger communities, smaller gain from augmentation (Spearman ρ = −0.82 GraphSAGE, −0.81 GATv2; p = 0.0002) |

Details: [`docs/project_overview.md`](docs/project_overview.md).

## With a stronger encoder: GATv2

Every rule was fitted under GraphSAGE. Swapping in **GATv2** (attention) with the same graphs, splits and hyperparameters - only the convolution changes - on 30 graphs, 3 seeds:

- **Augmentation still helps, on fewer graphs:** 13 of 30, against 21 of 30 under GraphSAGE.
- **Augmentation and a stronger encoder are substitutes, not complements.** Where GATv2 alone already wins, rewiring adds almost nothing; where it fails, rewiring rescues 8 of 9 graphs.
- **Stage 1 misses in one direction only.** Every disagreement is a graph where the stronger encoder made augmentation unnecessary - never one where it made augmentation newly necessary.

**Which graphs still need it** - a description, not a rule:

| still needs augmentation | does not |
|---|---|
| sparse, dominated by a few hubs | denser, evenly spread degrees |
| hubs attached to leaves | similar nodes attached to each other |
| few triangles | many triangles |

Role edges typically join nodes 3-6 hops apart, beyond the 2-hop view of a standard GNN, so each node gets direct access to structurally similar nodes it could otherwise reach only by stacking more layers, which oversmooths.

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
