# experiments/ - the runnable entry points

Every file here is an `argparse` CLI. They hold **no method code**: the algorithms live in `virgo/`
(imported), and these scripts only wire a dataset to a pipeline and record the result. Notebooks import
`run_ogb.py`, `characterize.py`, `strategy_select.py` and `predict_strategy.py` for their functions
instead of duplicating them.

## The one rule that matters here

**A script either FITS or PREDICTS. Never both.**

- **Fitting scripts** call `threshold()` and are **panel-guarded** - they refuse to fit on a dataset that
  is reserved as a test case, because a rule fitted on a held-out graph cannot then be tested on it.
- **Prediction scripts** only ever *read* `virgo/frozen_rules.py`. They import no fitting function. If one
  ever did, the validation would become circular and every held-out score in the paper would be void.

| file | role | fits? |
|------|------|-------|
| `run_core.py` | the frozen sweep for the non-OGB datasets: role graph → encoder → NC + LP → scoreboard row | - |
| `run_ogb.py` | the same sweep under the **official OGB protocol** (train-only embeddings, valid/test discipline) | - |
| `characterize.py` | measures graph properties + the encoder's raw inputs, relates them to the original-vs-augmented gap, and screens each property as an executable keep-vs-augment rule (each split also refitted with one dataset hidden, on its own and with the property re-chosen) | **FITS** |
| `gate_rules.py` | measures what the rewiring did to each graph, and screens those role-graph properties as a second variable inside the low-homophily zone | **FITS** |
| `strategy_select.py` | records which variant wins link prediction per dataset, and screens whether any property predicts *which structural signal* | **FITS** |
| `predict_module3.py` | pre-registers the stage-1 calls on unseen datasets, **before** training | reads only |
| `score_module3.py` | scores those calls against what training produced | reads only |
| `predict_gate.py` | build role graph → pre-register the gated stage-1 call → (train) → score | reads only |
| `predict_strategy.py` | pre-registers and scores the stage-2 signal call | reads only |
| `train.py` | Phase-1 run file: graph → I2V walks → Word2Vec → `.emb` (`--cached` uses the fast path) | - |
| `train_encoder.py` | trains ONE encoder over ONE role graph; `--arch` picks any encoder registered in `virgo.encoders` | - |
| `benchmark_baselines.py` | I2V vs DeepWalk / node2vec / struc2vec across the benchmark datasets (no flags - runs the whole loop) | - |
| `run_task.py` | small reproduction CLI: one task, one dataset, one result CSV | - |
| `plot_emb.py` | draws an embedding as a 2-D picture | - |

Run them from the repo root - data paths (`input/`, `output/`, …) are resolved against it.

## Applying the framework to a new graph

Order is part of the protocol: **build → predict → train → score.** Building the role graph is
deterministic and involves no learning, so the prediction is still made with the outcome unknown.

```bash
# 1 · build the role graph (not training)
python -m virgo.virtual_graph --input input/<ds>.edgelist --sim psi --k 10

# 2 · freeze stage 1 BEFORE any encoder runs (write-once; existing rows are never rewritten)
python experiments/predict_gate.py --datasets <ds> --step predict

# 3 · freeze stage 2 - only if stage 1 said "augment"
python experiments/predict_strategy.py --datasets <ds> --step predict

# 4 · train
python experiments/run_core.py --datasets <ds> --task link_prediction

# 5 · score the frozen calls
python experiments/predict_gate.py --datasets <ds> --step score
python experiments/predict_strategy.py --datasets <ds> --step score
```

Both predict scripts **refuse** a dataset the corresponding rule was fitted on. `predict_gate.py` has an
`--allow-panel` escape hatch for a sanity re-check only - it measures fit, never transfer.

## Refitting the analyses

```bash
python experiments/characterize.py --step all   # stage-1 rule screen  → candidate_rules.csv
python experiments/gate_rules.py                # the gate screen      → gate_candidates.csv
python experiments/strategy_select.py           # stage-2 screen       → strategy_patterns.csv
```

Each defaults to its own fitting panel and asserts on anything outside it. `--allow-refit` /
`--allow-panel` exist for deliberate exploration and write `exploratory_*.csv` so the panel tables are
never overwritten.

## Individual pieces

```bash
python experiments/run_task.py --list                              # show datasets

# Link prediction (leakage-free): split 70/30, retrain on the 70% graph, AUC
python experiments/run_task.py --task linkpred --dataset cora --retrain

# Node classification (needs labels/cora.labels)
python experiments/run_task.py --task nodeclass --dataset cora --emb output/notebook1_reproduce_i2v/cora/node_classification/i2v_s42.emb

# One encoder over one role graph (--arch = any registered encoder)
python experiments/train_encoder.py --input input/cora.edgelist --arch graphsage --sim psi --k 10 --seed 42
```

Requires `scikit-learn`. Scores land in `results/`; the master table is `results/scoreboard.csv`.

**Seed-count discipline.** `results/scoreboard.csv` is the ten-seed (42–51) board; the three-seed board it
replaced is archived at `results/scoreboard_3seed.csv`. Re-scoring a module against a different board can
move a verdict, so any number quoted from here must name its seed count.
