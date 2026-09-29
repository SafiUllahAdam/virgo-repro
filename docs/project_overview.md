# Structure Aware Graph Augmentation for Graph Neural Networks

*A pre-training framework that decides, whether to add structural-role edges to a graph and which structural signal to build them from, before any encoder is trained*

Structural-role augmentation adds edges between nodes that play **similar topological roles** - two hubs, two bridges, two peripheral nodes - even when no path connects them. A GNN only passes messages along real edges, so these role twins never meet; a **role graph** gives them a channel.

That channel does not always help. The question is therefore not *"does rewiring work?"* but **"for this graph, should I rewire, and with which signal?"** - answered from the graph's own properties.

---

## 1 · Headline

GraphSAGE, K = 10, 10 seeds, one frozen pipeline.

| task | augment | tie | keep original |
|---|---|---|---|
| **link prediction** (19 graphs) | **9** | 4 | 6 |
| **node classification** (15 graphs) | **0** | 2 | 13 |

- **Node classification: never augment.** No graph, in any panel or at any seed count, has shown a significant gain. This is a boundary to report, not a rule to predict.
- **Link prediction genuinely splits**, so the decision is worth making, and worth predicting.

---

## 2 · The framework

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

### Stage 1 - augment or not? *(locked)*

| step | rule | reading |
|---|---|---|
| **decision** | adjusted homophily **< 0.227** ⇒ augment | When edges follow the labels, the original graph is the better guide - a reliable **veto**. Low homophily is necessary, not sufficient. |
| **exception** *(low-homophily graphs only)* | average clustering **≥ 0.5573** ⇒ keep original | Dense, triangle-rich neighbourhoods already carry the local structure. |

Homophily makes the decision; clustering is a safety check for the few low-homophily graphs where augmentation does not pay off.

### Stage 2 - which signal? *(only when stage 1 says augment)*

Three structural signals can build the role graph: **degree** (how many neighbours a node has), **eigenvector centrality** (how influential a node is through well-connected neighbours), and **Ψ** (Identity2Vec's score comparing the shape of two nodes' neighbourhoods).

| signal | rule | reading |
|---|---|---|
| **centrality** | adjusted neighbour-label predictability **> 0.0092** | A node's neighbours' label mix predicts its class better than chance. |
| **degree** | neighbour-label entropy **< 0.6724** | The labels around each node are concentrated, not mixed. |

---

## 3 · With a stronger encoder: GATv2

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

---

## 4 · Method

The variable under study is **the graph**; one fixed encoder sees every variant.

```
input graph → structural signals → role graph → encoder → embeddings → evaluation
              (degree, eigenvector  (top-K most   (GraphSAGE; (64-dim    (NC weighted F1,
               centrality, Ψ,        similar       GATv2,      per node)  LP AUC)
               clustering; cached)   nodes, K=10)  DeepWalk)
```

- **Signals** - computed once and cached; they are also the encoder's input features (an ablation shows they are necessary).
- **Ψ** - Identity2Vec's Poisson/KL score, used as published. On about a third of nodes its numerical floor reduces it to a function of degree; a normalised repair removes the floor but lowers performance, so it is reported, not adopted.
- **Role graph** - each node linked to its K most similar nodes (`virgo/virtual_graph.py`), ties broken by seeded sampling.
- **Evaluation** - NC: logistic regression on the embeddings. LP: AUC on a 70/30 split, role graph rebuilt from training edges only (no leakage).
- **Identity2Vec** - reproduced and cached: byte-identical embeddings, ~200× faster.

| variant | edges | signal |
|---|---|---|
| `original` | the input graph (the control) | - |
| `psi` · `degree` · `centrality` | role edges **replace** the original | Ψ · degree · eigenvector centrality |
| `hybrid` · `hybrid_degree` · `hybrid_centrality` | role edges **added to** the original | Ψ · degree · eigenvector centrality |

The frozen rules were fitted on the five locked variants (`original`, the three replacements, `hybrid`). Hybrids vote for the signal they add.

---

## 5 · Datasets

72 graphs trained (83 registered): citation, social, web, co-purchase, biological, molecular, transport and more, ranging from strongly homophilous to strongly heterophilous. Every graph is used **structurally only**. Published node features are ignored, so a gain cannot be credited to attributes instead of to the rewiring. Panels, sources and citations: **[DATASETS.md](DATASETS.md)**.

---

## 6 · Repository

Two code folders, one rule: **`virgo/` is imported, `experiments/` is run.** (`virgo/` is an internal package name.)

```
├── input/ labels/ splits/     # graphs, labels, saved LP splits - input/ is read-only
├── output/ results/           # embeddings; scoreboard.csv, graph_health.csv, module tables
├── notebooks/ 1-8             # the narrative: reproduce → role graphs → encoder → … → encoders
├── virgo/
│   ├── config.py              # THE settings: paths, datasets, variants, GNN params, seed
│   ├── frozen_rules.py        # THE frozen rules
│   ├── graph_io.py            # THE graph policy + the single loader
│   ├── identity2vec*.py       # original walk + the cached rewrite
│   ├── virtual_graph.py       # the top-K role-graph builder
│   ├── encoders/              # GNNEncoder base + sage · gin · gatv2 · walk (ENCODERS registry)
│   └── data/ eval/            # dataset builders; NC · LP · OGB evaluation
└── experiments/               # one argparse CLI each
    ├── run_core.py run_ogb.py                             # the frozen sweeps
    ├── characterize.py gate_rules.py strategy_select.py …   # FIT (panel-guarded)
    ├── predict_*.py score_module3.py                      # READ frozen_rules only
    └── encoder_transfer.py encoder_profile.py             # GATv2 comparison + profile
```

**Adding an encoder:** one `GNNEncoder` subclass defining `build_convs()` in `virgo/encoders/<name>.py`, plus one line in `ENCODERS`. Every driver picks it up.

---

## 7 · Setup and usage

Conda environment **`i2v`** (Python 3.12):

```bash
conda activate i2v
# from scratch:
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
pip install torch torch-geometric node2vec
```

```bash
# --- apply the framework to a new graph (predictions are written BEFORE any training) ---
python experiments/predict_gate.py     --datasets <ds> --step predict        # stage 1, original graph only
python experiments/predict_strategy.py --datasets <ds> --step predict        # only if stage 1 said augment

# --- then train and score ---
python experiments/run_core.py         --datasets <ds> --task link_prediction
python experiments/predict_gate.py     --datasets <ds> --step score
python experiments/predict_strategy.py --datasets <ds> --step score

# --- refit the analyses (panel-guarded) ---
python experiments/characterize.py --step all      # stage-1 rule screen
python experiments/stage1_pairs.py                  # the stage-1 exception screen
python experiments/strategy_select.py              # the stage-2 screen
```

Seed 42 everywhere (split, init, sampling); multi-seed runs use 42-44 or 42-51. A paper number counts as reproduced within ±0.05. The notebooks are the narrative version of the same commands; run them in order on the "Python (i2v)" kernel.

---

## Conclusion

1. **Augmentation is graph-dependent.** Role edges improve link prediction on many graphs but not all, and do not improve node classification, so the decision has to be made per graph.
2. **Stage 1 decides whether to augment, before training.** Low adjusted homophily signals that augmentation will help, with clustering as a safety check; both are read directly from the input graph.
3. **Stage 2 decides which signal to use.** Predictable neighbour labels point to centrality; concentrated neighbour labels point to degree.
4. **Augmentation matters most where the encoder alone struggles.** With a stronger encoder (GATv2), it stays decisive on sparse, hub-dominated, weakly clustered graphs, rescuing 8 of 9 graphs where the encoder alone fails.

**In one line:** role augmentation gives a GNN direct access to structurally similar nodes beyond its receptive field, and the graph's own structure tells you, before training, when to use it and with which signal.

---

## Credits

- **Identity2Vec** - Oluigbo et al., *Learning mesoscopic structural identity representations via a Poisson probability metric*. Provides Ψ and the walk baseline (`virgo/identity2vec.py`, frozen).
- **Heterophilous benchmarks** - Platonov et al., 2023. **LINKX / Facebook100** - Lim et al., 2021. **struc2vec** - vendored in `third_party/`, used as published.

This project contributes the cached Identity2Vec walker, the seven-variant role-graph builder, the GraphSAGE encoder over role graphs, and the two-stage decision framework.
