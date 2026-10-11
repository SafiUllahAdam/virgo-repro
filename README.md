# Structure Aware Graph Augmentation for Graph Neural Networks

This repository holds the code, reference results and verification tools for the study *Structure Aware Graph
Augmentation for Graph Neural Networks*.

## In short: a pre-training framework

This project is a **pre-training framework**. Before you train a GNN or any other graph encoder, it looks at your graph
and answers two questions:

1. **Should I augment this graph?** In other words, should I add edges between nodes that play similar structural roles?
2. **If yes, which structural signal should I use to build those edges?**

Both answers come from simple properties of the original graph. Nothing has to be trained first, and no new graph has
to be built first.

```
                         your input graph
                                |
                                v
                 measure a few graph properties
                (no training, no role graph needed)
                                |
                                v
          STAGE 1: augment the graph, or keep it as it is?
               |                                   |
         keep original                          augment
               |                                   |
               |                                   v
               |                 STAGE 2: which structural signal?
               |                  (centrality, degree, or no clear pick)
               |                                   |
               v                                   v
     train the encoder on              build that role graph, then
     the original graph                train the encoder on it
```

## Why this matters

A GNN only passes information along existing edges. Two nodes that play the same role (for example, two hubs in
different parts of the graph) never exchange information if no short path joins them. Structural-role augmentation adds
edges between such nodes, so they can learn from each other. We call the resulting graph a **role graph**.

The catch is that this does not always help. On some graphs it improves results clearly, and on others it makes them
worse. So the useful question is not "does augmentation work?" but "**does it work on this particular graph, and with
which signal?**" This framework answers that question before any training is done.

## Key terms

| term | plain meaning |
|---|---|
| **augmentation** | adding edges between nodes with similar structural roles |
| **role graph** | the graph built by linking each node to its K = 10 most similar nodes, by one structural signal |
| **structural signal** | the measure used to decide which nodes are similar: degree, eigenvector centrality or Ψ |
| **degree** | how many neighbours a node has |
| **eigenvector centrality** | how important a node is, counting not just its neighbours but how important they are |
| **Ψ (psi)** | the Identity2Vec score that compares the shape of two nodes' neighbourhoods |
| **link prediction (LP)** | predicting which missing edges exist; measured by AUC (0.5 is guessing, 1.0 is perfect) |
| **node classification (NC)** | predicting each node's class; measured by weighted F1 |
| **tie** | the difference between two graph versions is smaller than the variation between random seeds |
| **seed** | one repeat of an experiment with a different random start; results are averaged over seeds |
| **range of a rule** | every threshold inside it splits the training graphs equally well; the threshold used is its middle point |

## The decision rules

All rules are stored in [`virgo/frozen_rules.py`](virgo/frozen_rules.py). "Frozen" means they were fixed once and never
changed after being tested on new graphs. Every rule reads the **original graph only**.

### Stage 1: augment or keep the original?

Apply the three rules in this order.

**Rule 1.1, the homophily rule (the main decision)**

- **Rule:** if adjusted homophily **< 0.227**, augment. Otherwise, keep the original graph.
- **Range:** any threshold from 0.0926 to 0.3613 fits the training graphs equally well.
- **What the property means:** adjusted homophily measures how often edges connect nodes with the same label,
  corrected for chance. A value of 0 means edges match labels no more often than random, 1 means every edge joins two
  nodes of the same class, and a negative value means edges join different classes more often than chance.
- **Plain reading:** when edges already follow the labels, the original graph is a good guide and should be kept. When
  they do not, role edges can supply what the original edges are missing.
- **Evidence:** fitted on 7 graphs. On held-out graphs, its "keep" calls were right 3 times out of 3, but its
  "augment" calls were right only 2 times out of 4. So a "keep" answer is reliable, while an "augment" answer needs the
  check in rule 1.2.

**Rule 1.2, the clustering exception (only when rule 1.1 says augment)**

- **Rule:** if average clustering **≥ 0.5573**, keep the original graph after all. Otherwise, augment.
- **Range:** 0.5329 to 0.5769.
- **What the property means:** average clustering is how often two neighbours of a node are also connected to each
  other, that is, how many triangles the graph has. It ranges from 0 (no triangles) to 1.
- **Plain reading:** some low-homophily graphs have very tight local neighbourhoods. These neighbourhoods already carry
  the local structure, so role edges add little.
- **Evidence:** fitted on 13 graphs, where it got 8 of 10 decided calls right (rule 1.1 alone: 7 of 10). On 6 new
  graphs, 4 of which could be decided, it was right 4 times out of 4 (rule 1.1 alone: 3 of 4). This test was done after
  those graphs had already been trained, so it is supporting evidence rather than a strict pre-registered test.

**Rule 1.3, the fallback for graphs without labels**

- **Rule:** if the graph has no node labels, homophily cannot be computed. In that case, augment if the largest
  connected piece of the graph holds **more than 95.88%** of all nodes. Otherwise, keep the original.
- **Range:** 91.77% to 100%.
- **Plain reading:** role edges help most when the graph is essentially one connected piece.
- **Evidence:** fitted on the same 7 graphs as rule 1.1. It is only used when rule 1.1 cannot be applied.

### Stage 2: which structural signal?

Stage 2 runs only when Stage 1 says augment. Both rules need node labels. Apply them in this order.

**Rule 2.1, the centrality rule (check first)**

- **Rule:** if adjusted neighbour-label predictability **> 0.0092**, build the role graph from **eigenvector
  centrality**.
- **Range:** 0.0060 to 0.0123.
- **What the property means:** we try to guess each node's class from the class mix of its neighbours. The score is how
  much better this guess does than always guessing the most common class. A value of 0 means the neighbours tell you
  nothing extra. The threshold, 0.0092, is a small gain.
- **Plain reading:** if the labels around a node carry some information about its own label, even if they are
  different labels, centrality is the best signal.
- **Evidence:** fitted on 14 graphs, with no errors on the 7 that benefit from augmentation. In a leave-one-out check it
  was right 5 times out of 7, against 4 out of 7 for always picking the most common winner. On new graphs collected to
  test it, it was right 3 times out of 4, and 8 times out of 17 across every new graph it was applied to. The evidence is
  still limited. All 4 test graphs fell on the same side of the threshold, so always answering "centrality" would also
  have scored 3 out of 4.

**Rule 2.2, the degree rule (when rule 2.1 does not pick centrality)**

- **Rule:** if neighbour-label entropy **< 0.6724**, build the role graph from **degree**. Above the threshold, the
  rule makes **no recommendation**. It does not say "use Ψ".
- **Range:** 0.6296 to 0.7151.
- **What the property means:** for each node, we look at how mixed the labels of its neighbours are, on a scale from 0
  (all neighbours share one label) to 1 (labels spread evenly over all classes), then average over all nodes.
- **Plain reading:** when the labels around each node are concentrated rather than mixed, degree is the best signal.
- **Evidence:** fitted on 9 graphs with no errors (leave-one-out: 8 of 9 right). On new graphs, its "degree" calls were
  right 6 times out of 9. Across all 40 graphs that benefit from augmentation, degree wins on 59% of graphs below the
  threshold against 18% above it (p = 0.025). With the 9 fitting graphs removed, the figures are 50% against 29%, which
  is not statistically significant (p = 0.29). Treat this rule as a promising candidate, not an established result.

**If neither Stage 2 rule fires**, the framework does not choose between Ψ and degree.

### Rules tested and not kept

These are recorded in `virgo/frozen_rules.py` for transparency but are not part of the framework.

| rule | why it was dropped |
|---|---|
| **Retention gate:** augment if the role graph keeps fewer than 1.19% of the original edges | replaced on 2026-09-01 by rule 1.2; it needed the role graph to be built first, and on new graphs its only call that differed from rule 1.1 was wrong |
| **Two-sided entropy rule:** like rule 2.2, but above 0.6724 choose Ψ | right only 1 time out of 3 on new graphs, where always choosing Ψ was right 3 times out of 3; only its degree side was kept, as rule 2.2 |
| **Distinct-degree rule:** choose degree if the graph has fewer than 158.5 distinct degree values | right only 2 times out of 6 on new graphs |

## Main results

All results use graph structure only: the node features that come with the datasets are ignored on purpose, so any
gain is due to the new edges and not to extra node information.

| setting | result |
|---|---|
| GraphSAGE, 10 seeds, link prediction, 19 graphs | augmentation wins on 9 graphs, ties on 4, and the original graph is better on 6 |
| GraphSAGE, 10 seeds, node classification, 15 graphs | augmentation never gives a significant gain (0 wins, 2 ties, 13 graphs where the original is better) |
| GATv2, 3 seeds, 30 graphs | augmentation helps on 13 of 30 graphs, against 21 of 30 with GraphSAGE; on the graphs where GATv2 alone does poorly, augmentation fixes 8 of 9 |
| community structure, 15 social graphs | the stronger a graph's communities (measured by Louvain modularity), the less augmentation helps link prediction: Spearman ρ = −0.82 with GraphSAGE and −0.81 with GATv2, both p = 0.0002. This describes the 15 graphs; it is not a tested rule |

**In plain words:**

- Augmentation is useful for **link prediction** on some graphs, but not for **node classification**.
- With a stronger encoder (GATv2), augmentation is needed on fewer graphs. It still matters on **sparse graphs
  dominated by a few hubs, with few triangles**.
- Role edges usually connect nodes that are **3 to 6 steps apart**. A standard two-layer GNN only sees 2 steps, so role
  edges give each node direct access to similar nodes it would otherwise not reach.
- Graphs with **strong communities** tend to be better left as they are, because role edges cut across those
  communities.

More detail and caveats are in [`docs/project_overview.md`](docs/project_overview.md) and in the dated research log
[`docs/paper_log.md`](docs/paper_log.md).

## How the study works

```
input graph  -->  structural signals  -->  role graph  -->  encoder  -->  embeddings  -->  evaluation
                  degree, centrality,      top K = 10       GraphSAGE,    64 numbers       NC: weighted F1
                  Ψ, clustering            most similar     GATv2, GIN,   per node         LP: AUC
                                           nodes            DeepWalk
```

Each graph is tested in seven versions. The encoder and its settings stay the same, so the only thing that changes is
the graph.

| version | what the edges are | signal |
|---|---|---|
| `original` | the input graph, used as the baseline | none |
| `psi`, `degree`, `centrality` | role edges **replace** the original edges | Ψ, degree, eigenvector centrality |
| `hybrid`, `hybrid_degree`, `hybrid_centrality` | role edges are **added to** the original edges | Ψ, degree, eigenvector centrality |

- **Link prediction:** 70% of edges are used for training and 30% for testing. The role graph is built from the
  training edges only, so test edges never leak into it.
- **Node classification:** a logistic regression is trained on the node embeddings.
- **Identity2Vec:** the method of Oluigbo et al. is reproduced exactly. Our cached version gives byte-identical
  embeddings about 200 times faster.
- **Datasets:** 72 graphs were trained (83 registered), including citation, social, web, co-purchase, biological,
  molecular and transport networks. Sources, citations and their role in the study are listed in
  [`DATASETS.md`](DATASETS.md).

## Repository layout

| path | contents |
|---|---|
| `virgo/` | the library: settings (`config.py`), frozen rules, graph loading, Identity2Vec, role-graph builder, encoders, evaluation, dataset builders |
| `experiments/` | one command-line script per analysis; each script either fits a rule or applies the frozen rules, never both (see [`experiments/README.md`](experiments/README.md)) |
| `notebooks/1-9` | the full study as notebooks, in order, from the Identity2Vec reproduction to the community analysis |
| `expected/` | reference result tables from the original run; [`expected/PRODUCERS.csv`](expected/PRODUCERS.csv) lists the script behind each table |
| `expected/hashes/` | SHA256 checksums of splits, role graphs, the feature cache and final results, plus a list of the large files stored elsewhere |
| `results/` | community analysis tables (Module 17) used by notebook 9; all other outputs are rebuilt locally |
| `data/` | dataset records with checksums (`USED.csv`), every training run behind the results (`RUNS.csv`), and the restore script |
| `verify/compare.py` | checks files against the recorded hashes and result tables against `expected/` |
| `env/` | the exact software environment of the original run |
| `docs/` | project overview, research log, run notes, thesis and slides |
| `third_party/struc2vec` | the struc2vec baseline, included unchanged |

## Reproducing the results

Full instructions, rules and the report format are in [`AGENT.md`](AGENT.md). There are three levels. Go to a higher
level only when the one before it is not enough.

| level | what it does | what you need |
|---|---|---|
| 1 | checks the datasets, the feature cache and every final result table, with no retraining | Python 3 with pandas, `sha256sum`, `tar` |
| 2 | restores stored embeddings or splits and scores them again without training | access to the original server copy (on request) |
| 3 | retrains everything from scratch | the environment below; a GPU is recommended |

### Level 1: standard check

Download the files of the GitHub release `v1.0-data` (`virgo-input-labels.tar.gz`, `virgo-feature-cache.tar.gz` and
`SHA256SUMS`) into a folder named `release/`, then run:

```bash
bash data/fetch_all.sh release
```

The script checks the downloaded archives, unpacks `input/`, `labels/` and `output/feature_cache/`, and compares the
checksum of every dataset file, the feature cache and every table in `expected/`. Every file should report `EXACT`.
Fifteen label files have no recorded checksum (they are listed in `AGENT.md`); they are rebuilt automatically the
first time they are used.

### Level 3: retraining

```bash
conda create -n i2v --file env/conda-explicit.txt
conda activate i2v
pip install -r env/pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The original run used Linux x86_64, Python 3.12.13, PyTorch 2.12.0 with CUDA 13.0, PyTorch Geometric 2.8.0, NumPy
1.26.4, SciPy 1.12.0, NetworkX 3.6.1, gensim 4.3.3, scikit-learn 1.9.0 and pandas 3.0.3, on one NVIDIA RTX PRO 6000
Blackwell GPU (see [`env/versions.txt`](env/versions.txt)). The command and seeds behind every reported number are in
[`data/RUNS.csv`](data/RUNS.csv), and the order of the analysis scripts is in `AGENT.md`, section 4. When training is
done, check everything with:

```bash
python verify/compare.py --step all
```

### How results are checked

- **Files that should never change** (datasets, splits, role graphs, the feature cache and final tables) are compared
  by SHA256 checksum and must match exactly. One known exception: the eigenvector solver can sometimes start from a
  random point, so a Ψ role graph's checksum may differ without any real change in the results.
- **Newly trained embeddings** are not compared bit by bit, because GPU training is never perfectly repeatable. Instead,
  their results are compared with `expected/`. Text columns (verdicts, calls, winners) must match exactly, and numbers
  must agree within **0.005**.
- **Why 0.005:** we measured how much results move between runs. Repeating the same run moves them by about 0.0000001,
  changing the number of threads by about 0.0000076, and changing the code path by about 0.0027. Changing the random
  seed moves them by about 0.1. The 0.005 tolerance sits above normal machine noise and far below the effect of a new
  seed.

Seeds are 42 for single runs, 42 to 44 for three-seed runs, and 42 to 51 for ten-seed runs.

### Where the data lives

| location | contents | size |
|---|---|---|
| this repository | code, instructions, checksums and reference result tables | about 27 MB |
| GitHub release `v1.0-data` | datasets (`input/`, `labels/`) and the feature cache | about 230 MB |
| original server (single copy, available on request) | trained embeddings and other generated files, each listed with its checksum in `expected/hashes/large_artifacts.csv` | about 72 GB |

## Acknowledgements

- **Identity2Vec:** Oluigbo et al., *Learning mesoscopic structural identity representations via a Poisson
  probability metric*. It provides Ψ and the random-walk baseline (`virgo/identity2vec.py`, unchanged).
- **Heterophilous benchmarks:** Platonov et al., 2023. **LINKX and Facebook100:** Lim et al., 2021.
- **struc2vec:** included in `third_party/` and used as published.

## Citation

Citation details will be added on publication.
