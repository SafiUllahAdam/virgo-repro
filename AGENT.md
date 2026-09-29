# AGENT.md - reproducing the study

You are an agent asked to reproduce every reported result of *Structure Aware Graph Augmentation for Graph Neural Networks* from this repository. Follow the steps in order. Do not skip the checks.

**What "reproduced" means here.** Deterministic artifacts such as datasets, splits and role-graph definitions are checked exactly where possible, while newly-trained embeddings and resulting metrics are checked using predefined numerical tolerances. Never report the rebuild as "identical": report what `verify/compare.py` says, line by line.

---

## 0 · What is in this repository

| path | what |
|---|---|
| `virgo/` | the library: config, frozen rules, graph loader, Identity2Vec, role-graph builder, encoders, evaluation |
| `experiments/` | every entry point (one argparse CLI each) |
| `notebooks/1-8` | the narrative version of the same pipeline, in study order |
| `env/` | exact environment: `conda-explicit.txt`, `pip-freeze.txt`, `versions.txt` (the original machine) |
| `data/USED.csv` | every dataset file the results use: builder command, size, SHA256 |
| `data/UNUSED.csv` | datasets and downloads that exist but were never used in results: kept as a record only |
| `data/RUNS.csv` | every training run behind `expected/scoreboard.csv`: dataset, encoder, task, K, seeds, variants, command |
| `data/fetch_all.sh` | restores the datasets (from the release archives, or by rebuilding) and hash-checks them |
| `expected/` | the reference results (the answer key); `expected/PRODUCERS.csv` maps each table to the script that writes it |
| `expected/hashes/artifacts.csv` | SHA256 of every split, role graph and feature-cache file of the original run |
| `verify/compare.py` | compares the rebuild against `expected/` and writes `verify/report.csv` |
| `docs/` | `paper_log.md` (every result with its context), `notes.md` (run settings and deviations), thesis, slides, Identity2Vec paper |

Not in git (rebuilt locally, all ignored): `input/`, `labels/`, `splits/`, `output/`, `results/`.

---

## 1 · Hard rules

1. **Never refit a frozen rule.** `virgo/frozen_rules.py` is the result, not an input to tune. Fitting scripts refuse to refit without `--allow-refit`; never pass it.
2. **Never edit `input/` or `labels/`** after restoring them. Split order depends on edgelist line order.
3. **Seeds come from the commands, never from memory.** Use `data/RUNS.csv` exactly; its `seeds` column is the seed set of each reported number (3, 6 or 10 seeds).
4. **Stage-1 / stage-2 predictions are written before training** (`predict_*.py --step predict` before `run_core.py`). Keep that order.
5. **Structural only.** Never add node attributes/features from any dataset.
6. **Do not "fix" anything to make numbers match.** A mismatch is reported, not patched.

---

## 2 · Environment

Original machine (`env/versions.txt`): Linux x86_64, Python 3.12.13, torch 2.12.0+cu130, torch_geometric 2.8.0, numpy 1.26.4, scipy 1.12.0, networkx 3.6.1, gensim 4.3.3, scikit-learn 1.9.0, pandas 3.0.3, 64 CPU cores, one NVIDIA RTX PRO 6000 Blackwell GPU (CUDA 13.0).

```bash
conda create -n i2v --file env/conda-explicit.txt          # exact conda packages (linux-64)
conda activate i2v
pip install -r env/pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -c "import torch, torch_geometric, numpy; print(torch.__version__, torch_geometric.__version__, numpy.__version__)"
```

**Stop if** numpy is not 1.26.4, gensim not 4.3.3, scipy not 1.12.0, or torch/torch_geometric differ from `env/versions.txt`. Report the difference instead of continuing on other versions.

Threads: the original runs used library defaults (no thread limits set in code). Do not set `OMP_NUM_THREADS` unless the machine requires it, and record it if you do.

Run everything from the repository root; library modules as `python -m virgo.<module>`, entry points as `python experiments/<script>.py`.

---

## 3 · Datasets

**Recommended: release archives (exact).** Download the release assets of this repository (`virgo-input-labels.tar.gz`, `virgo-feature-cache.tar.gz`, `SHA256SUMS`) into `release/`, then:

```bash
bash data/fetch_all.sh release
```

This verifies the archives, unpacks `input/`, `labels/` and `output/feature_cache/`, and hash-checks every file in `data/USED.csv` plus the feature cache. The feature cache holds the exact structural signals (degree, eigenvector centrality, Ψ, clustering) of the original run, which is what lets role graphs rebuild from identical inputs.

**Fallback: rebuild from sources.** `bash data/fetch_all.sh rebuild` runs every builder in `data/USED.csv`. Hosts change, so every file is hash-checked; any `MISMATCH` means that dataset's splits and results may differ, and must be listed in your report. `cora`, `citeseer_linqs`, `enzymes`, `proteins` start from the Identity2Vec author's `input.zip` (Network Repository) with labels rebuilt by `virgo/data/make_labels.py`.

**Known gap:** 15 label files (`actor`, `amazon_photo`, `amazon_ratings`, `amherst41`, `citeseer_linqs`, `cora`, `enzymes`, `johnshopkins55`, `lastfm_asia`, `minesweeper`, `proteins`, `pubmed`, `questions`, `reed98`, `roman_empire`) were not on disk when the reference was recorded, so they have no reference hash (`NO_REFERENCE`). Their builders regenerate them on first use.

`data/UNUSED.csv` lists ~34 GB of downloads that no reported result uses (e.g. GraphLand web-fraud/web-topics, attribute-driven and excluded). **Do not download them** unless explicitly asked.

---

## 4 · Run order

Each step names what it produces. Results land in `results/` (scoreboard rows are upserted by key, so re-running a step replaces its rows).

| step | what | command | produces |
|---|---|---|---|
| 1 | Identity2Vec reproduction | notebook `1-reproduce_i2v.ipynb` | Phase-1 embeddings; within ±0.05 of the paper counts as reproduced |
| 2 | splits | created on first use by `run_core.py` from input + seed | `splits/` - check exactly (step 6) |
| 3 | role graphs + all training runs | every row of `data/RUNS.csv` (`command` column) | `results/scoreboard.csv`, `results/graph_health.csv`, `output/notebook2_*`, `output/notebook3_*` |
| 4 | OGB pair | `run_ogb.py` rows of `data/RUNS.csv` (valid sweep first, then `--final`) | OGB rows of the scoreboard |
| 5 | analyses, in study order | `characterize.py --step all` → `predict_module3.py` / `score_module3.py` → `gate_rules.py` → `predict_gate.py --step all` → `strategy_select.py` → `predict_strategy.py --step score` → `stage1_pairs.py` → `tie_break.py` → `degree_rule.py` → `degree_family.py --score` → `psi_rule.py` → `encoder_transfer.py --step compare --datasets <cfg.ENCODER_PANEL>` → `encoder_profile.py` | the other tables in `expected/` (see `expected/PRODUCERS.csv`) |

Notes:
- Default arguments of every analysis script already point at the panels the reported numbers used; pass `--datasets` only when `expected/PRODUCERS.csv` or `docs/paper_log.md` says the table used a different set.
- The GATv2 comparison uses the 30 graphs of `cfg.ENCODER_PANEL`, 3 seeds (42-44).
- Tables marked `superseded` in `expected/PRODUCERS.csv` are kept for the record; a missing superseded table is not a failure.
- `psi_three_arm_panel.csv` and `psi_vs_rivals.csv` have no producer in `experiments/`; they are Ψ diagnostics described in `docs/paper_log.md` (2026-09-08 to 2026-09-13). Report them as not rebuilt.
- Runtime: the full `data/RUNS.csv` is long (many GPU hours on the original machine). For a quick check, run `cora`, `roman_empire` and `actor` first and verify them before the rest.

---

## 5 · Verification

```bash
python verify/compare.py --step all            # writes verify/report.csv
```

| check | status values | pass condition |
|---|---|---|
| hashes (datasets, splits, role graphs, feature cache) | `EXACT`, `MISMATCH`, `MISSING`, `NO_REFERENCE`, `KNOWN_EXCEPTION` | every recorded file `EXACT`; `NO_REFERENCE` and `KNOWN_EXCEPTION` listed, not failed |
| results (every CSV in `expected/`) | `EXACT`, `WITHIN_TOLERANCE`, `OUT_OF_TOLERANCE`, `TEXT_MISMATCH`, `SHAPE_MISMATCH`, `MISSING` | text columns (verdicts, calls, winners) identical; numeric columns within `--tol` (default 0.005) |

**Why 0.005.** Measured noise between runs of this pipeline: repeat run ~1e-7, thread change ~7.6e-6, code-path change ~2.7e-3, new seed ~1e-1. The tolerance sits above the machine/code-path floor and far below seed-to-seed variation, so a pass means the same computation, not a lucky one.

**Known exceptions:**
- **Ψ role graphs.** Eigenvector centrality's solver can fall back to a random start, so two Ψ builds may choose different top-K neighbours without changing any metric materially. With the shipped feature cache this does not arise; if a Ψ role-graph hash still differs it is reported as `KNOWN_EXCEPTION`.
- **GPU training is not bit-reproducible.** Embedding files are never hash-compared; only their metrics are.

---

## 6 · Report

Write `verify/REPORT.md` with:
1. environment actually used (versions, GPU, threads) and any difference from `env/versions.txt`;
2. dataset route (release or rebuild) and every non-`EXACT` hash;
3. counts per status from `verify/report.csv`, and every `OUT_OF_TOLERANCE`, `TEXT_MISMATCH`, `SHAPE_MISMATCH` or `MISSING` table by name, with its max difference;
4. any step not run, and why.

Use this sentence and no stronger claim: *Deterministic artifacts were checked exactly where possible; newly-trained embeddings and resulting metrics were checked against predefined numerical tolerances.*

---

## 7 · Troubleshooting

| problem | fix |
|---|---|
| PyG host for LastFM / Twitch / Deezer dead (`graphmining.ai`) | builders already read SNAP instead (`virgo/data/make_pyg.py`) |
| OGB download slow or blocked | `python -m virgo.data.make_ogb --dataset ogbn_arxiv` (or `ogbl_ddi`) once network allows; release route avoids it |
| `KeyError: Unknown dataset` | name must match `virgo/config.py DATASETS` |
| fitting script refuses to run | it is panel-guarded by design; do not pass `--allow-refit` |
| CUDA out of memory on large graphs | run that dataset alone; do not change batch or model settings |
