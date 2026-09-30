# AGENT.md - reproducing the study

You are an agent asked to verify, and if needed reproduce, the reported results of *Structure Aware Graph Augmentation for Graph Neural Networks* from this repository. Work level by level and stop at the lowest level that answers the task.

**Large embeddings and generated artifacts are preserved externally; download/restore them only if needed. Normal reproducibility verification uses the stored final CSVs, hashes, and caches and does not retrain models.**

**What "reproduced" means here.** Deterministic artifacts such as datasets, splits and role-graph definitions are checked exactly where possible, while newly-trained embeddings and resulting metrics are checked using predefined numerical tolerances. Never report anything as "identical": report what `verify/compare.py` says, line by line.

---

## 0 · Where everything lives

| place | holds | size |
|---|---|---|
| **this repository** (GitHub) | code, instructions, checksums, final result tables | ~27 MB |
| **release `v1.0-data`** (GitHub) | datasets (`input/`, `labels/`) and the feature cache | ~230 MB |
| **original SSH server** (the only copy, author access) | large embeddings and generated artifacts (`output/`, `splits/`), plus raw dataset downloads | ~72 GB recorded here |
| **your machine** | this repository and only the files the task needs | - |

The server copy is the single original. Never modify it, never copy or archive it whole: restore only the subfolder a task needs (Level 2).

| path in this repository | what |
|---|---|
| `virgo/`, `experiments/`, `notebooks/1-8` | library, entry points (one argparse CLI each), narrative pipeline in study order |
| `expected/` | the final result tables of the original run (the answer key); `expected/PRODUCERS.csv` maps each table to its script |
| `expected/hashes/artifacts.csv` | SHA256 of every split, role graph, feature-cache file and final result file |
| `expected/hashes/large_artifacts.csv` | manifest of the server-held files: path (where it belongs), size, SHA256 |
| `data/USED.csv` · `data/UNUSED.csv` · `data/RUNS.csv` | datasets used (with checksums) · downloads never used (record only) · every training run behind `expected/scoreboard.csv` |
| `data/fetch_all.sh` · `verify/compare.py` | restore datasets from the release · verify anything against the recorded hashes and tables |
| `env/` · `docs/` | exact environment of the original machine · research log (`paper_log.md`), run notes, thesis, slides |

---

## 1 · Hard rules

1. **Never refit a frozen rule.** `virgo/frozen_rules.py` is the result, not an input. Never pass `--allow-refit`.
2. **Never edit `input/`, `labels/` or restored files.** Split order depends on edgelist line order.
3. **Never modify, move, copy whole or archive the server copy.** Restore selectively, read-only.
4. **Seeds come from `data/RUNS.csv`**, never from memory (3, 6 or 10 seeds per reported number).
5. **Structural only.** Never add node attributes/features from any dataset.
6. **Do not "fix" anything to make numbers match.** A mismatch is reported, not patched.

---

## 2 · Level 1 - normal verification (default: no retraining, nothing large)

Needs Python 3 with pandas, plus `sha256sum` and `tar`.

1. Download the release `v1.0-data` assets (`virgo-input-labels.tar.gz`, `virgo-feature-cache.tar.gz`, `SHA256SUMS`) into `release/`.
2. Run:
   ```bash
   bash data/fetch_all.sh release
   ```
   It checks the archives, unpacks `input/`, `labels/` and `output/feature_cache/`, then hash-checks every dataset file, the feature cache and every final result file in `expected/` (`verify/report_data.csv`).
3. Read the reported numbers from `expected/` (the producing script of each table is in `expected/PRODUCERS.csv`).

**Pass:** every checked file `EXACT`. Fifteen label files were never recorded (`NO_REFERENCE`: actor, amazon_photo, amazon_ratings, amherst41, citeseer_linqs, cora, enzymes, johnshopkins55, lastfm_asia, minesweeper, proteins, pubmed, questions, reed98, roman_empire); their builders regenerate them on first use.

---

## 3 · Level 2 - restore large artifacts (only if needed)

Use this only when a task needs stored embeddings or splits, e.g. to inspect them or to re-score them without training.

| folder on the server | files | size | content |
|---|---|---|---|
| `output/notebook3_gnn_encoder/` | 4,825 | 58.0 GB | GNN embeddings (GraphSAGE, GATv2, GIN, feature ablations): `<task>/<dataset>/k10/<variant>/<encoder>_s<seed>.emb` |
| `output/notebook2_create_vir_graph/` | 910 (+155 role graphs) | 11.5 GB | DeepWalk embeddings on the role graphs, and the role graphs themselves |
| `output/notebook1_reproduce_i2v/` and `output/*` | 16 | 0.01 GB | Identity2Vec reproduction embeddings, run logs |
| `splits/` | 1,462 | 1.4 GB | link-prediction splits (deterministic, also rebuilt automatically when missing) |

Restore only the subfolder you need, then verify it against the manifest. Restored files are copies of the original, so they must be `EXACT`:

```bash
export VIRGO_SERVER="<user>@<host>:<path>/identity2vec"      # ask the author: the original copy lives only there
rsync -a "$VIRGO_SERVER/output/notebook3_gnn_encoder/link_prediction/cora/" output/notebook3_gnn_encoder/link_prediction/cora/
python verify/compare.py --step hashes --kinds large --prefix output/notebook3_gnn_encoder/link_prediction/cora
```

Splits and role graphs are checked with `--kinds split role_graph` (optionally with `--prefix`). On Windows, run `rsync` inside WSL.

**Re-scoring without training.** `run_core.py` reuses any embedding already on disk instead of training it (`embed()` in `experiments/run_core.py`), so after restoring a dataset's embeddings, its commands in `data/RUNS.csv` score the stored embeddings and write `results/`. Compare with `python verify/compare.py --step results`. This needs the exact environment (Level 3, step 1).

---

## 4 · Level 3 - retrain (only if needed)

1. **Environment.** Original machine (`env/versions.txt`): Linux x86_64, Python 3.12.13, torch 2.12.0+cu130, torch_geometric 2.8.0, numpy 1.26.4, scipy 1.12.0, networkx 3.6.1, gensim 4.3.3, scikit-learn 1.9.0, pandas 3.0.3, 64 CPU cores, one NVIDIA RTX PRO 6000 Blackwell GPU (CUDA 13.0).
   ```bash
   conda create -n i2v --file env/conda-explicit.txt
   conda activate i2v
   pip install -r env/pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130
   ```
   **Stop if** numpy, gensim, scipy, torch or torch_geometric differ from `env/versions.txt`; report the difference. Threads: library defaults (none set in code); record any `OMP_NUM_THREADS` you set.
2. **Datasets:** Level 1. The fallback `bash data/fetch_all.sh rebuild` re-downloads from the original sources; every file is hash-checked and any `MISMATCH` must be reported. `cora`, `citeseer_linqs`, `enzymes`, `proteins` start from the Identity2Vec author's `input.zip`.
3. **Run order** (from the repository root):

| step | what | command | produces |
|---|---|---|---|
| 1 | Identity2Vec reproduction | notebook `1-reproduce_i2v.ipynb` | Phase-1 embeddings; within ±0.05 of the paper counts as reproduced |
| 2 | splits | created on first use by `run_core.py` from input + seed | `splits/` - exact check |
| 3 | role graphs + training runs | every row of `data/RUNS.csv` (`command` column) | `results/scoreboard.csv`, `results/graph_health.csv`, embeddings |
| 4 | OGB pair | `run_ogb.py` rows of `data/RUNS.csv` (valid sweep first, then `--final`) | OGB rows of the scoreboard |
| 5 | analyses, in study order | `characterize.py --step all` → `predict_module3.py` / `score_module3.py` → `gate_rules.py` → `predict_gate.py --step all` → `strategy_select.py` → `predict_strategy.py --step score` → `stage1_pairs.py` → `tie_break.py` → `degree_rule.py` → `degree_family.py --score` → `psi_rule.py` → `encoder_transfer.py --step compare --datasets <cfg.ENCODER_PANEL>` → `encoder_profile.py` | the other tables in `expected/` |

Notes: analysis scripts default to the panels the reported numbers used. The GATv2 comparison uses the 30 graphs of `cfg.ENCODER_PANEL`, 3 seeds (42-44). Tables marked `superseded` in `expected/PRODUCERS.csv` are records; `psi_three_arm_panel.csv` and `psi_vs_rivals.csv` have no producer script (Ψ diagnostics, `docs/paper_log.md` 2026-09-08 to 09-13). The full `data/RUNS.csv` takes many GPU hours: start with `cora`, `roman_empire` and `actor`.

4. **Verify:** `python verify/compare.py --step all`.

---

## 5 · Verification reference

| check | command | status values | pass condition |
|---|---|---|---|
| hashes | `--step hashes [--kinds ...] [--prefix ...]` | `EXACT`, `MISMATCH`, `MISSING`, `NO_REFERENCE`, `KNOWN_EXCEPTION` | every checked file `EXACT`; `NO_REFERENCE` and `KNOWN_EXCEPTION` listed, not failed |
| results | `--step results` | `EXACT`, `WITHIN_TOLERANCE`, `OUT_OF_TOLERANCE`, `TEXT_MISMATCH`, `SHAPE_MISMATCH`, `MISSING` | text columns (verdicts, calls, winners) identical; numeric columns within `--tol` (default 0.005) |

Hash kinds: `dataset`, `feature_cache`, `final_result` (Level 1), `large`, `split`, `role_graph` (Level 2-3). Default: all except `large`.

**Why 0.005.** Measured noise between runs of this pipeline: repeat run ~1e-7, thread change ~7.6e-6, code-path change ~2.7e-3, new seed ~1e-1. The tolerance sits above the machine/code-path floor and far below seed-to-seed variation.

**Known exceptions.** Ψ role graphs: the eigenvector solver can fall back to a random start, so a Ψ role-graph hash may differ without changing any metric materially (`KNOWN_EXCEPTION`); the shipped feature cache avoids it. Retrained embeddings are never hash-compared (GPU training is not bit-reproducible); only their metrics are.

---

## 6 · Report

Write `verify/REPORT.md` with: the level(s) run; environment (versions, GPU, threads) and any difference from `env/versions.txt`; dataset route (release or rebuild); counts per status from the report CSVs; every non-`EXACT` hash and every `OUT_OF_TOLERANCE`, `TEXT_MISMATCH`, `SHAPE_MISMATCH` or `MISSING` table by name with its max difference; any step not run, and why.

Use this sentence and no stronger claim: *Deterministic artifacts were checked exactly where possible; newly-trained embeddings and resulting metrics were checked against predefined numerical tolerances.*

---

## 7 · Troubleshooting

| problem | fix |
|---|---|
| no access to the server | Levels 1 and 3 do not need it; ask the author for Level 2 |
| PyG host for LastFM / Twitch / Deezer dead (`graphmining.ai`) | builders already read SNAP instead (`virgo/data/make_pyg.py`) |
| OGB download slow or blocked | the release route avoids it; else `python -m virgo.data.make_ogb --dataset ogbn_arxiv` (or `ogbl_ddi`) |
| `KeyError: Unknown dataset` | name must match `virgo/config.py DATASETS` |
| fitting script refuses to run | panel-guarded by design; never pass `--allow-refit` |
| CUDA out of memory | run that dataset alone; do not change model settings |
| `data/UNUSED.csv` items (~34 GB) | never needed for any reported result; do not download unless asked |
