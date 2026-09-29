# ViRGo — Lab Notes

Lab notebook. Append a dated entry whenever something happens. Rules:
- Always paste the **exact command** (so it reruns).
- Always note the **seed** (project standard: `seed=42`).
- Tag surprises: `FINDING:` (result/observation), `DEVIATION:` (repo ≠ paper/spec), `TODO:` (open thread).
- One file, append-only. Newest at bottom.

> Entries dated **before 2026-06-15** are *reconstructed* from file timestamps, logs, and git — not logged live. Treat their commands as best-guess (exact flags weren't recorded). Live logging starts 2026-06-15.

---

## Environment / defaults

- Env: numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0. (torch / torch-geometric to be added for the GNN.)
- I2V `train.py` defaults: `dimensions=64`, `walk-length=40`, `num-walks=10`, `window-size=10`, `epochs=1`, `sg=1` (skipgram), `min-count=0`, `workers=1`, `e=2.7182`. Word2Vec: `alpha=0.025 → min_alpha=0.01`, `negative=5`, `sample=1e-3` (Fix 6, 2026-06-24; was `alpha=0.25` / `sample=1e-5`).
- Datasets in `input/`: cora, citeseer, dhfr, enzymes, firstmmedges, nci, politics, proteins, webkb (`.edgelist`); citeseer also has original `citeseer.txt`.
- Pretrained / trained embeddings in `output/`: `cora.emb` (author's, 2022-01-28), `webkb.emb` (trained 2026-06-13).

---

## 2026-06-12 — reconstructed

- Repo state: I2V core present — `identity2vec.py` (class `Graph`: guided walk + Poisson/KL), `train.py` (argparse CLI, `build_graph` / `learn_embeddings` / `main`).
- Created `plot_emb.py`: PCA-project an `.emb` to 2D, color nodes by degree (hubs deg>10 green, leaves deg<2 red). Sanity-check that I2V embeddings separate structural roles.
  - `python plot_emb.py output/cora.emb input/cora.edgelist cora_plot.png`
  - Outputs: `cora_plot.png`, `cora_3color.png`.
- Created `prepare_linkpred.py` — **empty stub (0 bytes).** TODO: implement 70:30 edge split for link prediction (AUC), retrain on 70% only (no leakage), per I2V Table 4.

## 2026-06-13 — reconstructed

- Trained I2V on webkb → `output/webkb.emb` (265 nodes).
  - `python train.py --input input/webkb.edgelist --output output/webkb.emb` (defaults; exact flags not logged)
- FINDING: very slow. `webkb_run.log` shows ~52 min for walk 1/10 and ~57 min for walk 2/10 on only 265 nodes → ~9 h for the full run. Confirms the I2V efficiency problem.

## 2026-06-15

- Derived `input/citeseer.edgelist` from `input/citeseer.txt` (created 11:56).
  - TODO: document the derivation step (script/command unknown — not logged). Needed for reproducibility.
- Started I2V training on citeseer (3264 nodes) → `citeseer_run.log` ("Number of Nodes: 3264").
  - `python train.py --input input/citeseer.edgelist --output output/citeseer.emb` (likely via nohup; exact flags not logged)
  - STATUS: **incomplete** — no `output/citeseer.emb` yet. Expected to be very slow at 3264 nodes given the webkb rate. TODO: confirm finished / rerun under the cached variant.

- FINDING (baseline efficiency, the project's motivation): I2V recomputes structural signal *inside the walk loop*.
  - `identity2vec.py:24` `eigenvector_centrality()` runs `nx.eigenvector_centrality(G, max_iter=1000)` over the **whole graph** on every call.
  - Called in `get_prob()` (`:92`) **and again** in `poisson_dist()` (`:117`) — i.e. per-neighbor, per-step, per-walk.
  - `node_neighbors()` (`:28`) rebuilds the full neighbor dict every `identity_walker` step (`:59`).
  - → For a static graph, degree + eigenvector centrality are constant. Computing once and **caching** is exact and removes the dominant cost. This is Deliverable #1 (cached I2V variant, embeddings identical, timing gain).

- DEVIATION: walk length. `train.py:27` sets `--walk-length` default = **40**, but the help text says "Default is 80." Repo uses 40; I2V paper says 80. Decide and pin one value for all reproductions; record which.

- Housekeeping: created `docs/` — copied `CLAUDE.md` into it (kept the root copy; **CLAUDE.md must stay in repo root to govern agentic work**), moved `Research_Proposal.pdf` in, started this `notes.md`. Also created `results/` (PNGs) and `logs/` (.log files).

## 2026-06-16

- Wrote the evaluation layer (the paper reports F1 / AUC; embeddings alone don't prove reproduction):
  - `prepare_linkpred.py` — 70:30 edge split, seed=42. Forces a spanning tree into train so the train graph stays connected; samples equal-count non-edges for train + test. Uses the largest connected component only.
  - `eval_linkpred.py` — Hadamard edge features (node2vec default) -> logistic regression -> test AUC. Operator switchable via `--op`.
  - `eval_nodeclass.py` — stratified split (default 80% train) -> logistic regression -> weighted F1.
  - `labels/` and `splits/` dirs, each with a format README.
- Ran the cora split: largest CC = 2485 nodes / 5069 edges -> train_pos=3548, test_pos=1521 (30.0%), balanced negatives. Files in `splits/`.
  - `python prepare_linkpred.py --input input/cora.edgelist --name cora`
- DECISIONS (proposal PDF is 1 page and silent on exact protocol -> chose standard I2V/node2vec conventions; revisit if the original I2V source differs):
  - Link-pred = logistic regression on Hadamard features, AUC.
  - Node-class = logistic regression, weighted F1 (per CLAUDE.md), single stratified 80/20 split, seed=42.
- DEPENDENCY: added `scikit-learn` (logreg, SVM, roc_auc, f1) — not in the base env. Install: `pip install scikit-learn`. Only the eval scripts use it; core I2V untouched.
- BLOCKER: no label files in the repo. Node classification cannot run until `labels/{cora,citeseer}.labels` exist with IDs matching the edgelists (Cora IDs 1..2708). Link prediction is unaffected.
- NOTE: a valid link-pred AUC needs an embedding retrained on `splits/{name}_train.edgelist` — the full-graph `cora.emb` would leak. Pipeline in `splits/README.md`.
- Cora train-only I2V run stopped: very slow and produced overflow warning at identity2vec.py line 128 during Poisson score calculation.

- Built a reproduction framework under `scripts/` (mirrors CLADBench's `benchmark_config.py` / `utils.py` / `results_io.py` / `runner.py` / `main.py`). **No files moved** — it wraps the existing root scripts (`train.py` via subprocess; `prepare_linkpred` / `eval_*` via import). Made the 3 eval scripts importable (callable `prepare()` / `evaluate()` core + thin `main`).
  - New dirs/files: `scripts/`, `configs/` (+ sample `cora_linkpred.json`), `notebooks/virgo_dev.ipynb`.
  - Run: `python scripts/main.py --list` / `--task linkpred --dataset cora [--retrain]`. Results -> `results/NNN.{dataset}.{task}.csv` with a JSON `#META` header (run id, seed, settings, counts).
  - DEPENDENCY: installed `scikit-learn` 1.9.0 — landed in conda env **`i2v`** (python 3.12), which is the active interpreter here. Worth confirming this is the intended env for the project.
  - Smoke test: `python scripts/main.py --task linkpred --dataset cora --emb output/cora.emb` -> AUC=0.9578, saved `results/001.cora.linkpred.csv`. This is a **plumbing check only** — uses the full-graph `cora.emb`, so it leaks; not a paper number. A real AUC needs `--retrain` (embed on `splits/cora_train.edgelist` first).

## 2026-06-17

- DECISION: **walk-length = 40 (active).** Flipped 40 -> 80 -> 40 today: set to 80 for paper-fidelity, then reverted to 40 per request. `train.py` default and `benchmark_config.I2V_PARAMS` are back to 40. The paper text says 80 (kept as a recorded deviation). 40 matches the repo default and the author's `output/cora.emb` (F1=0.6992). NOTE: on-disk `output/cora_lp.emb` is currently an **80-walk** embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number.

- AUDIT (read-only, full repo vs CLAUDE.md / proposal): seed=42 consistent across `train` / `prepare_linkpred` / `eval_*` / `utils` / `REPRO`; `input/` untouched; cached overrides verified safe (`s_path` returns length only, `node_neighbors` returns fresh copies); split filenames match between `prepare` (writer) and `eval_linkpred` (reader). Env: `python` -> conda **i2v** (`sklearn 1.9.0`, `gensim 4.3.3`, `numpy 1.26.4`) — all deps present; the bash `libtinfo` warning is the interactive shell, cosmetic only. Dataset sizes: cora 2708/5278, citeseer 3264/4536, webkb 265/479, politics 18470, enzymes 19474, dhfr 32075, nci 101924. NOTE: `input/proteins.edgelist` and `input/firstmmedges.edgelist` read as 0 nodes (empty/malformed) — not in the dataset registry, harmless for now. **[CORRECTED 2026-07-17: not malformed — both are comma-delimited, which the whitespace-default readers parse as 0 nodes. Proteins is now wired (see 2026-07-17); firstmmedges is the same fix if ever needed.]**

- FIX (accuracy repairs — NOT the cache fix):
  - `notebooks/1-reproduce_i2v.ipynb` was BROKEN: Cell 8 imported the deleted `make_planetoid_labels`; its saved outputs were a failed run (overlap 0.0028 -> STOP -> `FileNotFoundError`). Changed import to `from make_labels import make_labels` and re-ran headless. Now clean end-to-end: `edge_overlap=1.0000`, labels written, **weighted F1 = 0.6992**, saved `results/003.cora.nodeclass.csv`.
    - `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/1-reproduce_i2v.ipynb`
  - `train.py` stale help strings corrected: `--walk-length` "Default is 80." -> "40."; `--workers` "Default is 8." -> "1." (defaults were already 40 / 1; only the help text was wrong).

- FIX1 DONE — **cache wired + speedup proven** (Deliverable #1). `scripts/runner.embed` now passes `--cached` (default on) + `--seed`, so every pipeline retrain uses the fast path. Benchmark on webkb (265n/479e, num-walks=2 walk-length=10, seed=42): baseline **916.6s → cached 4.4s = 207.7×** (≈646× minus `train.py`'s fixed 3s sleep); both `.emb` md5 `e458fa5e2a360ac388803ee990afb312` → **BYTE-IDENTICAL**. Confirms the cache changes no computed value and removes the dominant cost.
  - `python train.py --input input/webkb.edgelist --output <out> --num-walks 2 --walk-length 10 --seed 42 [--cached]`

## 2026-06-18

- WebKB labels resolved. Paper ref **[16] = Network Repository**; webkb appears ONLY in the paper's Figure 2 t-SNE viz (NOT in Table 1 stats nor eval Tables 2/3/4) — so I2V never ran a labelled task on it. Source found: the author's own repo `github.com/ikenna-oluigbo/webkb-dataset` (4 universities). The **Wisconsin** subset = our `input/webkb.edgelist` (265n/479e, **proven isomorphic**, 479/479 edges).
  - PROBLEM: author renumbered nodes between his two repos, so labels do NOT transfer to `input/webkb.edgelist` by id. Isomorphism recovery is ambiguous: 252/265 nodes uniquely labelled, **13 structurally ambiguous** (automorphisms cross class). So `input/webkb.edgelist` labels are unrecoverable.
  - DECISION: shipped the author's **consistent pair** instead (100% correct, zero ambiguity): `input/webkb_wisc.edgelist` + `labels/webkb_wisc.labels` (node id = `wisconsin.content` line order, label = last field). 5 classes — student 122, course 76, faculty 35, project 22, staff 10. Reproducible via `make_labels.make_webkb()`. Registered `webkb_wisc` in `benchmark_config`; `webkb` stays labels=None (the I2V-numbered graph, used for the cache benchmark).

## 2026-06-19

- Dataset **resolver + aligned Citeseer + micro/macro/weighted F1** (implements the "one version per dataset, auto-align" design).
  - `make_labels.py`: added `make_citeseer_linqs()` — builds `input/citeseer_linqs.edgelist` + `labels/citeseer_linqs.labels` from ONE LINQS source so ids align by construction (3312 papers, 4536 cites edges, 6 classes; largest content graph = 3264 connected nodes, 0 self-loops). Added `resolve_dataset()` — author `citeseer` prints the mismatch reason then auto-switches to `citeseer_linqs`; cora/webkb_wisc pass through. `ensure_labels` routes `citeseer_linqs`.
  - `scripts/benchmark_config.py`: `citeseer` labels=None (author graph has no aligned labels); added `citeseer_linqs`; `nodeclass_train_frac` 0.80 -> **0.70**.
  - `eval_nodeclass.evaluate` now returns **{micro,macro,weighted}** F1 dict (default train_frac 0.7); updated `main`, `runner.run_nodeclass`, notebook Steps 1/4/5.
  - Notebook Step 1 calls `resolve_dataset(DATASET)` so `citeseer` auto-aligns to `citeseer_linqs` for the WHOLE run (node-class + link-pred), printing why; author files untouched.
  - VERIFIED: cora node-class @0.7 = micro **0.7036** / macro 0.6710 / weighted **0.7009** (still reproduced). resolve passthrough OK. citeseer_linqs builds; node-class + link-pred plumbing runs (checked with a throwaway walk-10 emb).
  - NOTE: citeseer_linqs embedding is slow even cached (3264 nodes); the 207x cache win was on tiny webkb/walk-10. First build is minutes — run once in the background.
- WALK-LENGTH UNIFIED at **40** everywhere: notebook config 80 -> 40 (`train.py` + `benchmark_config` already 40). Paper's 80 = recorded deviation, not used. Existing walk-80 `.emb` files (`cora_lp`, `citeseer`, `citeseer_lp`, `webkb_wisc`, `webkb_wisc_lp`) are now **stale** vs the 40 setting -> rebuild (FORCE_EMBED / delete) for a clean 40 set. `cora.emb` = author reference, keep.

## 2026-06-20

- Repro bar relaxed **±0.03 -> ±0.05** (CLAUDE.md, README, virgo_guide) per request. Walk-length stays **40** (fixed decision).
- AUDIT (senior pass, read-only) vs I2V paper Eqs 1-9 + Tables 1-4: pipeline faithful to the AUTHOR's released code (upstream remote = ikenna-oluigbo/identity2vec; `identity2vec.py` untouched since import). FINDING: released code deviates from paper Eqs 2-4 — code uses p=degree*eigcent, q=dist, norm=deg+cent; paper uses p=degree, q=cent*dist, norm=cent. **Inherited, not ours** -> document in paper, do NOT change (keeps baseline comparable). DEVIATION (acceptable): link-pred forces full MST into train, so bridge edges are never test positives (mild AUC optimism). Cache verified exact; splits leakage-free; seeds reproducible.
- **Baseline benchmark framework added** (no runs executed; code-only per request):
  - NEW `embedding_models.py`: base `EmbeddingModel` + `Identity2VecModel` (wraps train.py), `DeepWalkModel` (node2vec p=q=1), `Node2VecModel` (p=1,q=0.5 — paper doesn't fix p/q, logged default), `Struc2VecModel` (vendored CLI). `MODELS` registry + `get_model`.
  - NEW `baselines/struc2vec/src/*` — vendored official leoribeiro/struc2vec, mechanically py2->py3 ported (iteritems/xrange/cPickle, gensim4 `vector_size`/`epochs`). DEP added: `node2vec` 0.5.0, `fastdtw`. **struc2vec UNTESTED — a py2 port needs one debug run to finalize; no --seed -> not bit-reproducible.**
  - NEW `scripts/benchmark_baselines.py`: `run_benchmark` loops `BENCH_DATASETS x BENCH_MODELS` reusing `run_*_repeated` (eval untouched); `benchmark_table` -> Table 1 (node-class weighted F1) + Table 2 (link-pred AUC), datasets x methods, mean±std. Saved `results/benchmark/`.
  - EDIT `runner.py`: `run_nodeclass_repeated`/`run_linkpred_repeated` gain `model=` (default identity2vec). I2V emb filenames stay UNPREFIXED so existing cora embeddings are reused; non-I2V get a `{model}_` prefix; splits stay model-independent (shared per dataset/seed = fair). Rows carry a `model` column. `summarize_seed_results` unchanged -> existing Steps 1-4 behave identically.
  - EDIT `benchmark_config.py`: `BENCH_DATASETS`, `BENCH_MODELS`; added `enzymes_nr`/`politics_nr` aligned-fallback registry entries.
  - EDIT `make_labels.py`: `make_enzymes()` (TU `ENZYMES.node_labels`, verified-or-fallback), `make_politics()` (networkrepository, **URL/format best-guess -> verified or STOPs**), `_make_nr_labels` (verify edge-overlap >=0.90 else build self-aligned `<name>_nr`), `_ensure_nr`; registered in `_VERSIONS`/`ensure_labels`/`prepare_dataset` (citeseer pattern).
  - EDIT notebook: appended **Step 5** (markdown + code) calling `run_benchmark`+`save_benchmark`; Steps 1-4 untouched.
  - VERIFIED: `py_compile` + import of all changed files OK (NO training/eval/benchmark executed). TODO before trusting numbers: (1) one struc2vec debug run; (2) decide node2vec p/q.
  - LABEL BUILD CHECK (ran builders only, no training): **enzymes OK** -> `make_enzymes` edge_overlap=1.0000, wrote `labels/enzymes.labels` (19580 nodes, 3 classes), aligned. **politics = LINK-PRED ONLY (no NC):** `input/politics.edgelist` IS networkrepository `rt-pol` (identical 61157 edges), graph only. Original left/right labels (Conover et al. 2011, ICWSM) are at cnets.indiana.edu = **offline (404)**, no clean github/zenodo mirror, and rt-pol's renumbering can't be mapped -> label alignment UNVERIFIABLE. Per policy: **no fabrication**. `make_politics` STOPs with that reason; `benchmark_baselines.run_benchmark` falls back to LP-only so politics appears in Table 2 (AUC) but NOT Table 1 (NC). [Tried Adamic-Glance polblogs as a self-aligned substitute, then REVERTED + deleted — polblogs is a different dataset, not rt-pol.] To enable politics NC, drop in a manually verified `labels/politics.labels` (node id = our numbering).
- **struc2vec py3 port FIXED + verified working** (integrated webkb_wisc run: micro 0.475 / weighted 0.369). Fixes in `baselines/struc2vec/src/`: `collections.abc.Iterable`; `np.int`->`np.int64`; `utils.partition` coerces dict-views to list; `simulate_walks` wraps `self.G.keys()` in `list()` (multiprocessing pickling); pickles dir = `baselines/struc2vec/pickles` (one level up from src). WRAPPER (`Struc2VecModel.train`) now **clears pickles/ + random_walks.txt before each run** — struc2vec caches under FIXED filenames, so without clearing, dataset B would reuse dataset A's structural distances (silent wrong embeddings). Cosmetic only: struc2vec prints `rm: cannot remove ...pickle` (its own startup cleanup) + a benign `invalid value in scalar divide`. struc2vec has NO seed -> not bit-reproducible (variance across seeds is real). WARNING: struc2vec on enzymes (19.5k nodes) with OPT may be very slow/memory-heavy.
- **benchmark hardened:** `run_benchmark` wraps each model's NC + LP in try/except -> one model crashing no longer aborts the sweep or loses other models' rows.
- FINDING (cora, same splits/seeds/params): node2vec NC 0.8165 / LP 0.9107 >> I2V NC 0.6906 / LP 0.8011. Expected — cora is homophilous, proximity methods (node2vec/deepwalk) beat structural-identity (I2V) on NC+LP. Our node2vec matches the node2vec literature; the paper's node2vec (LP 0.7658) is under-tuned, which is why the paper shows "I2V beats all". DO NOT claim I2V beats all on homophilous NC/LP; I2V's edge is structural tasks (heterophily/webkb, roles, anomaly).
- **DECISION: dropped politics from the benchmark, swapped in `webkb_wisc`** (265 nodes, 5 classes, labels already verified/isomorphic). `BENCH_DATASETS = [cora, citeseer_linqs, enzymes, webkb_wisc]`; notebook Step 5 examples updated. Both Table 1 (NC) + Table 2 (LP) now cover **all 4** datasets. CAVEAT: webkb is heterophilous (content classes vs structural roles) -> I2V/struc2vec NC F1 may be low/near-random on it; that's expected, report as-is.

### 2026-06-23 — walk-length 80 tried, then reverted to 40 (timing-driven)

- Flipped 40 -> 80 for paper-fidelity, then **reverted to 40** after measuring walk-generation cost. Code (`train.py` default+help, `scripts/benchmark_config.I2V_PARAMS`) and docs (README, `docs/virgo_guide.md`, `CLAUDE.md`, this file: defaults + checklist) are back to **40**. Notebook inherits the config (no hardcoded value).
- TIMING — cora, **cached** path, 10 walks/node, seed=42, walk-generation only (no Word2Vec):

  | walk_length | wall time |
  |---|---|
  | 40 | 1010.8 s (~16.8 min) |
  | 80 | 1894.0 s (~31.6 min) |
  | **diff** | **+883.2 s (~14.7 min) = 1.87x slower** |

  - Measured via `scratchpad/time_walks.py`: `identity2vec_cached.Graph(cora).identity2vec_walk(10, wl)` for `wl in {40, 80}`. Each call includes one fixed 3 s sleep -> the diff is sleep-free. ~1.87x (near-linear; sub-2x because the per-graph eigenvector compute + cached BFS amortize).
  - This is the CACHED path. Uncached is far slower: the notebook's uncached cora walk-80 ran ~2:13/walk x 10 ~= 22 min for NC alone.
- PERFORMANCE (40 vs 80) — _placeholder, fill by hand:_
  - cora NC weighted F1:  40 = ____  |  80 = ____
  - cora LP AUC:          40 = ____  |  80 = ____
- DECISION: **walk-length = 40 active.** 80 = paper value, kept as a recorded deviation, not used — 1.87x slower with no confirmed metric gain (see placeholder).

## 2026-06-24 — paper-fidelity fixes (external review), Fix 8 effect, proposed sampling

Review diffed repo vs paper -> 8 suggested fixes. Status verified fresh from disk (line refs current). Fixes landed over the last few sessions; recorded here together.

- **Fix 1 — selection direction. DONE, version (b).** `identity2vec.py:92-94`. Compute Ψ of the current node, then pick the candidate minimizing `|Ψ_candidate − Ψ_current|` (least-dissimilar). Option (a) `max(pdn, key=pdn.get)` left commented as a record.
- **Fix 2 — degree distribution, not raw degree. DONE.** Added `degree_distribution()` (Δ_u = n_d/n) at `identity2vec.py:27-39`; used for the `p` signal in `get_prob`. (Raw `degree_node()` now appears only in the Fix-4 divisor.)
- **Fix 3 — p/q composition. DONE.** `get_prob:132-133`: `p = Δ`, `q = Ω·d`. Eigenvector moved out of the numerator back into the denominator; distance penalty restored. Matches paper Eqs 3-4.
- **Fix 4 — normalize by candidate, not previous node. DONE ("4A").** `identity_score:154-158`: `normalizer = degree_node[node] + eigenvector[node]` of the candidate being scored (was `bounded_curr` — constant across candidates, so it didn't discriminate). NOTE: uses RAW degree + eigenvector; review offered raw OR degree-distribution for ω -> confirm the choice.
- **Fix 5 — walk-length 80. NOT applied (intentional).** Kept 40 (timing 1.87× slower, see 2026-06-23 entry). Paper's 80 = recorded deviation.
- **Fix 6 — Word2Vec hyperparameters. DONE.** `train.py:86`: `alpha 0.25 -> 0.025`, `sample 1e-5 -> 1e-3` (min_alpha / negative / seed unchanged). Removes the 10× learning rate that only I2V used.
- **Fix 7 — cached walker in sync. DONE.** `degree_distribution` cached at `identity2vec_cached.py:29-43` (`_deg_dist`). `identity_score` (Fixes 4A+8) is inherited (not overridden) and calls the cached signals via `self` -> cached and non-cached paths stay identical.
- **Fix 8 — Poisson in log-space. DONE.** `identity2vec.py:160-167`, `from scipy.special import gammaln` (:9). `drt = max(drt, 1e-12)`; `log_poiss = k·log(drt) − drt − gammaln(k+1)`. Fixes underflow-to-0 on hubs AND the latent `factorial(k)` overflow for k>170.

- **REVERSAL of the 2026-06-20 "DO NOT change" decision.** That audit kept the released-code deviations (p=deg·eigcent, q=dist, norm=deg+cent) for baseline comparability. Per the external review + the paper-fidelity goal, Fixes 2/3/4 now move the code to the paper's Eqs 2-4. The original released-code baseline stays in git history (can be a separate comparison column if needed).

- **FINDING — Fix 8 effect (cora):** node classification improved and now matches the paper; link-pred AUC dropped from best-recorded **0.8494 -> ~0.81**.
  - Why: before Fix 8 the Poisson underflowed to 0 on high-degree nodes -> many tied scores -> selection effectively RANDOM -> the walk wandered the local neighborhood (proximity) -> accidentally boosted LP. Fix 8 made the scores meaningful -> the walk now follows structural identity as intended -> NC up, LP down.
  - Context: the paper's own cora LP = **0.8413** (Table 4); 0.81 is within the ±0.05 repro bar. So the code now matches the paper on BOTH tasks — the 0.8494 was a bug-driven over-shoot, not a real LP capability.
  - CAVEAT 1: confirm 0.8494 -> 0.81 is real, not one-seed luck — re-run 3 seeds, mean±std.
  - CAVEAT 2: Fix 8 also interacts with Fix 1b — log-space warps the `|Ψ−Ψ|` closeness geometry, so selection changes even where nothing underflowed. Regenerate `.emb`.

- **STALE RESULTS.** Fixes 1/3/4/6/8 all changed the math -> existing `.emb` and `results/` are from old code and do NOT reflect it. Regenerate embeddings (delete / FORCE_EMBED) before trusting any metric.

### 2026-06-24 — temperature sampling (Fix 8 extension) — REMOVED 2026-07-07 ❌

A non-greedy softmax next-node sampler (`--temperature`, benchmark τ=0.3) was added here — cora s42 NC 0.7486 / LP 0.8305, an improvement on both — then **removed entirely on 2026-07-07** to keep I2V paper-exact for the ViRGo comparison (the paper selects "least dissimilar" = greedy). Selection is back to `min(pdn, key=|Ψ−Ψ_curr|)` (`identity2vec.py`); `--temperature` gone from `train.py`, `embedding_models.py`, `I2V_PARAMS`. Cached path inherits the walker, so it is greedy again too. Any I2V numbers produced with τ=0.3 (incl. the cross-model benchmark tables) are stale — regenerate `.emb` greedy before quoting.

### 2026-06-25 — cross-model config standardized for fair benchmark ✅

Goal: identical Skipgram/Word2Vec **training** across all 4 models so the comparison is fair; **I2V = anchor, left untouched** (it's stable). Walk generation + method-defining knobs stay per-model.

- I2V: **UNCHANGED** (alpha 0.025, min_alpha 0.01, sample 1e-3, negative 5, hs 0). Its embeddings stay valid -> keep cora s42 NC 0.7486 / LP 0.8305; **no I2V retrain**.
- node2vec + DeepWalk (`embedding_models.py:48-50`): pinned the Skipgram params explicitly to match I2V — `alpha=0.025, min_alpha=0.01, sample=1e-3, negative=5, hs=0`. Only `min_alpha` actually differed before (gensim default 0.0001 -> 0.01); the rest already equalled I2V via gensim defaults but are now explicit/locked.
- struc2vec (`baselines/struc2vec/src/main.py:81`): switched **hierarchical-softmax -> negative sampling** (`hs=0, negative=5`) + `alpha=0.025, min_alpha=0.01, sample=1e-3` to match I2V — also matches the paper's stated protocol ("negative sampling for DeepWalk and struc2vec"). Walk-length **default 80 -> 40** (`:30`) for standalone consistency.
- walk_length = **40 for all** — benchmark already forced 40 via `I2V_PARAMS`; struc2vec standalone default now 40 too. (80 tried earlier, no gain — see 2026-06-23.)
- KEPT per-model (NOT standardized — these define each method): node2vec `p=1/q=0.5`, DeepWalk `p=q=1`, I2V Poisson/KL walk, struc2vec `OPT1/2/3` + multilayer structural walk.
- VERIFIED: `py_compile` of both edited files OK (no run executed).
- EFFECT / next: regenerate ONLY the 3 baselines' `.emb` (their config changed; delete + re-run), then re-score. struc2vec will shift (hs->neg, likely up toward its paper ~0.71 LP); node2vec/DeepWalk shift slightly (min_alpha); I2V flat. struc2vec still has NO seed -> report mean±std.

---

## TODO backlog (open threads)

- [x] `prepare_linkpred.py` (70:30 split, seed=42) — done, verified on cora.
- [x] framework under `scripts/` — done, smoke-tested (`results/001`).
- [x] Install scikit-learn — done (in env `i2v`; confirm that's the right env).
- [x] Notebook-first workflow: `notebooks/1-reproduce_i2v.ipynb` (one click-through notebook reusing the tested functions). Installed + registered `ipykernel` ("Python (i2v)" kernel) for VS Code. Removed the old `virgo_dev.ipynb`.
- [x] Made `make_planetoid_labels.py` importable (guarded its run-calls with `if __name__ == "__main__"`); node-class scorer verified with throwaway labels (the F1 path works).
- [x] Training reproducibility fixed (`train.py` only — `identity2vec.py` untouched): added `--seed` (default 42), seeded global `np.random` before the walks (covers identity2vec's `np.random.shuffle`/`choice`), and passed `seed=args.seed` to `Word2Vec`. Verified byte-identical `.emb` across two same-seed runs, and different across seeds. **Requires `--workers 1`** (the default) — gensim is nondeterministic with multiple workers even with a seed.
- [x] **Cora labels solved + node classification reproduced.** FINDING: the author's `input/cora.edgelist` numbers nodes by **order of appearance in the LINQS `cora.content`** file — that ordering reproduces the edgelist at **edge_overlap = 1.0000**. Planetoid Cora is the *same graph but a different numbering* (overlap 0.003, identical degree sequences), so its labels are WRONG here — the safety check correctly refused them. The author's `input.zip` ships **no labels** at all. Wrote `make_labels.py` (LINQS `.tgz` via plain `urllib` — bypasses PyG's flaky `fsspec` that gave `FSTimeoutError`; verifies overlap before writing); deleted the misleading `make_planetoid_labels.py`.
  - RESULT (first real reproduction): **Cora node classification weighted F1 = 0.6992** (2708 nodes, 7 classes, `train_frac=0.8`, `seed=42`, author's `output/cora.emb`). Saved `results/002.cora.nodeclass.csv`.
  - `notebooks/1-reproduce_i2v.ipynb` runs fully end-to-end (verified headless, all 8 cells pass).
  - [ ] Compare 0.6992 to the paper's **Figure 5 / Section 4.4** — that figure is likely an F1-vs-train-ratio curve, so add a train-ratio sweep (and confirm the paper's F1 averaging: weighted vs micro/macro) to match it exactly.
  - [ ] Citeseer labels: our `citeseer.edgelist` was *derived by us*, so its numbering may not match LINQS `citeseer.content` order — run `make_labels('Citeseer')` and check the overlap before trusting it.
- [ ] Retrain I2V on `splits/cora_train.edgelist` -> run `eval_linkpred.py` -> record AUC vs I2V Table 4.
- [ ] `eval_rank.py` (pairwise SVM, AUC) — only if reproducing learning-to-rank (the proposal omits it).
- [ ] Document `citeseer.edgelist` derivation from `citeseer.txt`.
- [ ] Finish/confirm citeseer embedding.
- [x] Build cached I2V variant; verify embeddings identical + measure speedup (Deliverable #1) — done 2026-06-17: webkb 207.7× faster, byte-identical; pipeline uses `--cached`.
- [x] walk-length = **40** (active; flipped 40 -> 80 -> 40 again on 2026-06-23 after timing 80 = slower, see entry below). `train.py` + `benchmark_config` = 40. Paper's 80 = recorded deviation.
- [ ] On-disk `output/cora_lp.emb` is an 80-walk embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number if wanted.
- [x] Paper-fidelity fixes (external review), done 2026-06-24: Fix 1b selection, Fix 2 Δ, Fix 3 p/q, Fix 4A normalizer, Fix 6 Word2Vec, Fix 7 cache sync, Fix 8 log-space. Fix 5 (walk-80) intentionally NOT applied.
- [ ] Regenerate all `.emb` — fixes 1/3/4/6/8 changed the math, on-disk results are stale.
- [ ] Confirm Fix 8 LP drop 0.8494 -> 0.81 is real vs seed noise (3-seed mean±std).
- [x] Confirm Fix 4 ω choice (raw degree vs degree-distribution) at review. **ANSWERED 2026-09-09: keep `deg + Ω` as shipped** — see the entry below and paper_log 2026-09-09.
- [x] Temperature sampling (2026-06-24 ablation, τ=0.3) — **removed 2026-07-07**, I2V restored to paper-exact greedy; τ TODOs (seed sweep, runner threading) dropped with it. Regenerate I2V `.emb`.
- [x] Cross-model Skipgram config standardized to I2V 2026-06-24 (node2vec/DeepWalk `min_alpha=0.01`; struc2vec hs→negative sampling; walk_length 40 all). I2V untouched; per-model walk knobs kept.
- [ ] Regenerate the 3 baselines' `.emb` (config changed) + re-run benchmark; I2V embeddings stay valid (no retrain).
- [ ] `virtual_graph.py` — top-K Ψ builder (Deliverable #2).

---

### 2026-06-26 — walk-length flipped 40 → 80 (paper value)

- DECISION: **walk-length = 80 (active)**, reverting the 2026-06-23 "40 active" decision per request. 80 = the I2V paper value; ~1.87× slower than 40 (see 2026-06-23 timing), accepted.
- Changed everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, and docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted all 42 `output/**/*.emb` (every model/dataset/seed, nc+lp) — built at 40, now stale. Next benchmark run rebuilds at 80.
- Supersedes the checklist line above (`walk-length = 40`) and the 2026-06-24 standardization note (`walk_length 40 all`).

### 2026-06-26 — walk-length reverted 80 → 40 (back to repo default)

- DECISION: **walk-length = 40 (active)**, reverting the 2026-06-25 "80 active" flip per request. 40 = repo default + the setting behind author `cora.emb`; the paper's 80 (~1.87× slower, no confirmed gain) is kept as a recorded deviation.
- Reverted everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted the 24 stale 80-walk `output/cora/*.emb` (rebuilt after the 2026-06-25 flip); next run rebuilds at 40. No author `output/cora.emb` present.
- Supersedes the 2026-06-25 "80 active" entry.

### 2026-06-37 — benchmark scoring finalized + first 3-seed cross-model run

Findings-driven cleanups (implemented one-by-one) + the first full 3-seed run.

- **Link-pred scoring — BOTH computed every seed; headline now COSINE.** `runner.run_linkpred_repeated` scores each `.emb` two ways: `auc_logreg` (Hadamard edge feature → logistic regression, node2vec protocol) and `auc_cosine` (unsupervised cosine similarity, paper-faithful, no classifier). `REPRO["linkpred_score"]` picks the headline `auc` column — now **`"cosine"`** (was `"logreg"`). `benchmark_baselines` writes `table2_linkpred_auc.csv` (=headline=cosine) + `table2_linkpred_auc_cosine.csv` (identical now); `auc_logreg` survives only in `benchmark_per_seed.csv`. Cosine for edges + logreg for labels = paper alignment. STALE COMMENTS (values correct): `benchmark_config.py:43` and `benchmark_baselines.py:79-80` still say "main = logreg". **[resolved 2026-07-20 — see that entry: both tables are now named after their scorer and can no longer be identical; the `benchmark_baselines` comments were corrected. `benchmark_config` carries no stale logreg claim.]**
- **Node-class classifier (Fix 6).** `eval_nodeclass.py:45` = `OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))` (L2 = sklearn default; `multi_class="ovr"` avoided — removed in sklearn ≥1.7). Reports micro/macro/weighted F1.
- **struc2vec OPT (Fix 4).** `Struc2VecModel.train`: `--OPT1/2/3 = False` for graphs ≤10k nodes (cora/citeseer/webkb = exact distance), `True` only for large graphs (enzymes ~19.5k → memory). struc2vec still unseeded → report mean±std.
- **node2vec vs DeepWalk kept distinct.** DeepWalk `p=q=1` (uniform), node2vec `p=1/q=0.5` (biased) → node2vec is not a DeepWalk duplicate. Paper does not fix p/q.
- **FINDING — first 3-seed cross-model run (cora, seeds 42/43/44; I2V ran with τ=0.3, temperature since removed 2026-07-07).** Snapshot captured at **walk-length 80** (06-25→06-29 window); those `.emb` were deleted in the 40-revert → **regenerate at walk-40 + greedy I2V before quoting as final.**

  | model | NC weighted F1 | LP AUC (cosine headline) |
  |---|---|---|
  | identity2vec | 0.7403 ± 0.0116 | 0.8281 ± 0.0085 |
  | deepwalk | 0.8109 ± 0.0216 | 0.9011 ± 0.0017 |
  | node2vec | 0.8166 ± 0.0090 | 0.9031 ± 0.0029 |
  | struc2vec | 0.3219 ± 0.0053 | 0.5491 ± 0.0077 |

  - Confirms the 2026-06-20 finding: on homophilous cora, proximity methods (node2vec/deepwalk) beat structural I2V on NC+LP; struc2vec weak. I2V's edge is structural/heterophilous tasks, not cora.
- **DOCS.** `docs/virgo_guide.md` full-synced to current code (notebook Steps 0–5, 70/30 split, OvR classifier, dataset registry, per-seed `.emb` naming, honest §8). Notebook walk/training progress bars restored (`embedding_models` `quiet=False`).
- TODO: regenerate the table at walk-40; run τ-sweep {0,0.1,0.3,1,3} + τ=0 vs 0.3 (delete I2V `.emb` between τ — filename ignores τ).

### 2026-06-28 — direction: Phase 1 done, focus shifts to the GNN encoder

- DIRECTION (project steer): I2V reproduction is accepted as correct. Baselines are **not** to be fine-tuned — the paper's baselines look under-tuned, but tuning them is out of scope; they stand at published/default settings. Focus moves **fully to the technical contribution**: replace the walk + Skipgram back-end with a modern GNN encoder over the virtual graph.
- NEXT: design several GNN architecture variants (GraphSAGE / GIN / GAT over the Poisson/KL virtual graph) as candidates, then compare them.
- The cross-model baseline table (cora, 3 seeds; see entry above) stands as the reference — no further baseline tuning.

### 2026-06-30 — canonical project phases fixed

Project flow locked into 5 phases (Phase 1 done):
1. **Phase 1 — reproducibility (match the I2V paper). DONE.** Cached I2V + cross-model baselines (used as-is, not fine-tuned); within ±0.05, 3-seed harness.
2. **Phase 2 — virtual-graph creation.** top-K Poisson/KL Ψ graph + degree-only / centrality-only comparison graphs (`virtual_graph.py`). ← next
3. **Phase 3 — modern GNN encoder.** GraphSAGE / GIN / GAT over the virtual graph, replacing walk + Skipgram; design + compare variants.
4. **Phase 4 — downstream tasks.** node classification, link prediction, anomaly detection (new); virtual-graph ablation (which graph best per data/task).
5. **Phase 5 — LLM context-window issue.** structural embeddings as a compact large-graph summary (stretch).
- Refines the earlier 2026-06-30 note: the immediate next is **Phase 2 (virtual graph)**, then Phase 3 (GNN) — not the GNN directly.
- Mirrored in README, CLAUDE.md §4, docs/virgo_guide.md.

### 2026-06-30 — Phase 2 framing: the virtual graph IS the study

- The central research question is **which virtual graph makes a GNN perform best** per task (node classification, link prediction, later anomaly detection). The **virtual graph — not the encoder — is the variable under study.**
- I2V's Poisson/KL similarity graph is **one generic** virtual graph; test whether it works well per task, against simpler variants (degree-only, centrality-only).
- ViRGo is **NOT** mainly "GraphSAGE vs GIN" — encoder choice is secondary.
- Phase 2 order: (1) build the virtual-graph system, (2) test virtual-graph variants; Phase 3: run different GNN encoders on them.
- Mirrored in README, CLAUDE.md §4, docs/virgo_guide.md.

### 2026-07-01 — Benchmark switched to author `citeseer` (drop `citeseer_linqs`)
- DECISION: use the **author's own `citeseer` graph** (`input/citeseer.edgelist`, derived from the paper's `citeseer.txt`) instead of `citeseer_linqs`. It is the paper's actual file → **link prediction only** (no aligned labels, node classification not run on it).
- Note: the two graphs are structurally **identical** (3264 nodes / 4536 edges / largest CC 2110); `citeseer_linqs` was the same graph renumbered + LINQS labels attached (edge overlap 16/4536 = pure relabel). So this switch does **not** change LP structure — it just uses the paper's numbering and skips the unaligned-label workaround.
- Changes: `make_labels.py` — `resolve_dataset` no longer swaps citeseer→citeseer_linqs; `_VERSIONS["citeseer"]=("citeseer","orig","citeseer")`; `prepare_dataset` skips label build for citeseer (LP-only); `__main__` no longer builds citeseer Planetoid labels (they'd be misaligned). `benchmark_config.py` — `BENCH_DATASETS = [cora, citeseer, enzymes, webkb_wisc]`.
- `citeseer_linqs` files + registry entry kept (unused by the benchmark now); can still be called explicitly if NC on Citeseer is ever needed.
- FINDING — first author-`citeseer` LP result (seed 42, walk-40, ran with τ=0.3 — temperature since removed 2026-07-07, regenerate greedy; largest CC = 2110 nodes):

  | score | AUC | paper Table 4 (I2V) | within ±0.05? |
  |---|---|---|---|
  | cosine (headline, paper-faithful) | **0.8606** | 0.8373 | yes (Δ 0.023) |
  | logreg (Hadamard, node2vec protocol) | **0.8771** | — | — |

  - emb `output/citeseer/citeseer_lp_orig_s42.emb`; splits `splits/citeseer/citeseer_lp_orig_s42_*`.
  - **Single seed (s42) — indicative, not final.** TODO: run seeds 43/44 for mean±std.
- **Why LP only, no node classification on citeseer:** the author graph (`input/citeseer.edgelist`) is the paper's own file but ships **no labels**, and its node numbering does **not** match LINQS `citeseer.content` order (edge overlap ~0.00), so LINQS labels would point at the WRONG nodes → any NC would be fake. Node classification on Citeseer needs the aligned `citeseer_linqs` build; on the paper's graph we report **link prediction only** (LP needs no labels).

### 2026-07-01 — Phase 2 started: `virtual_graph.py` (top-K structural virtual-graph builder, Deliverable #2)

- NEW `virtual_graph.py` — mirrors `train.py` shape (`argparse` / `build_graph()` / `main(args)`, core class `VirtualGraph`). Reuses `identity2vec_cached.Graph` for cached degree / Δ / eigenvector centrality (computed once).
- **DECISION (option A):** I2V's Ψ is walk-contextual (needs a reference node + shortest-path). Lifted it to an all-pairs virtual graph via a **reference-free per-node signature**, then top-K nearest signatures. For `psi` the signature = I2V's exact KL→Poisson score (Fix 4A normalizer + Fix 8 log-Poisson) with the walk shortest-path factor dropped (`q = Ω` instead of `Ω·pathlen`). Rejected option B (reuse `identity_score` literally per pair — asymmetric, slower, off-walk).
- **3 variants** (`--sim`, pluggable = add one branch): `psi` (I2V KL/Poisson), `degree` (degree-only), `centrality` (eigenvector-only). Same-K baselines answer "which virtual graph is best per data/task".
- **Constraints enforced + verified on cora (k=10):** same node set (2708, none missing) · 0 self-loops · exactly top-K per node (min_deg=K) · undirected union (symmetric) · finite weights `1/(1+dist)` ∈ (0,1], no NaN/inf · weighted `.edgelist`, NetworkX + PyG (`from_networkx` OK) readable. **Build is deterministic** (byte-identical rebuild).
- cora edge counts: psi 16251 (avg_deg 12.0), degree 26216 (19.4 — integer-degree ties inflate the union), centrality 16110 (11.9). Artifacts: `output/cora/virtual_<sim>_k10.edgelist`.
- Config: `benchmark_config.py` gains `VG_SIMS=[psi,degree,centrality]`, `VG_K=[5,10,20]`, `VG_SEEDS=[42,43,44]`.
- Virtual edgelist is a **drop-in** for the existing walk encoders (DeepWalk/node2vec read any edgelist) and for the Phase 3 GNN (same file) → variant comparison uses ONE fixed encoder, only the graph changes.
- **Next:** (2) variant-comparison harness (build sweep → one fixed encoder → existing `eval_nodeclass`/`eval_linkpred`), then Phase 3 GNN encoder over these graphs.

### 2026-07-01 — Phase 2 notebook + first variant comparison (cora, indicative)

- NEW `notebooks/2-virtual_graph_phase2.ipynb` — self-contained, runs top-to-bottom, **no terminal**: build → visualize → verify constraints → variant sweep → compare on NC + LP → save. Executes clean (nbconvert exit 0), saved with outputs + charts. I2V-repro notebook left frozen as the Phase-1 record.
- **Fixed-encoder protocol:** only the graph changes — same DeepWalk bridge (I2V_PARAMS), same K=10, same seeds [42,43,44]. LP is leak-free (virtual graph rebuilt from the 70% train edges).
- **FIRST comparison (cora, DeepWalk bridge, 3 seeds), `results/vir_graph_variants/cora_variant_comparison_K10.csv`:**

  | variant | node-class weighted-F1 | link-pred AUC |
  |---|---|---|
  | centrality | **0.381** ± 0.007 | 0.551 ± 0.008 |
  | psi (I2V KL/Poisson) | 0.234 ± 0.005 | 0.511 ± 0.006 |
  | degree | 0.152 ± 0.007 | **0.555** ± 0.002 |

- FINDING (indicative): **best virtual graph is task-dependent** — centrality wins node classification, degree wins link prediction; I2V's psi is not best on cora under the DeepWalk bridge. Directly supports the Phase-2 question "which graph per data/task".
- **CAVEATS — not final:** (a) DeepWalk bridge, NOT the Phase-3 GNN (which uses edge weights psi/centrality carry but DeepWalk ignores); (b) cora is homophily/community-labelled, so structural embeddings score modestly (F1) and cosine LP AUC sits near 0.55; (c) single dataset, K=10 only. Re-run the sweep (K=5/20, more datasets) + the GNN before drawing conclusions.
- **Notebook UX (added):** `DATASET` = single knob (Section 1); everything reads the `DATASETS` registry, NC auto-skips + LP-only when `labels is None`. `FORCE_REBUILD=False` = **reuse-if-exists at every stage** (virtual edgelists, `.emb`, LP splits); `True` = regenerate. Verified: re-run reused all (no re-embedding), byte-stable results.

### 2026-07-02 — Output layout redesign + graph-health stats

- **New canonical layout, all datasets, both phases:** everything for a dataset under ONE folder — `output/<dataset>/i2v_main/` = ALL Phase-1 reproduction embeddings (I2V no-prefix + `deepwalk_/node2vec_/struc2vec_`); `output/<dataset>/k<K>/` = Phase-2 files (`virtual_<sim>.edgelist`, `vg_{nc|lp}_<sim>_s<seed>.emb`). No flat files, no sibling `cora_10/`-style folders.
- Code updated to match (folders lead, code follows): `runner.py` `run_nodeclass_repeated`/`run_linkpred_repeated` now write `output/<safe>/i2v_main/` (Step-5 benchmark inherits automatically); Phase-2 notebook writes `output/<dataset>/k<K>/`. Existing files moved, none regenerated (filenames unchanged ⇒ reuse intact); `core_i2vorg_embeddings/` merged into `cora/i2v_main/` (24 embs) and removed.
- **Graph-health stats (notebook):** every `build_virtual()` call upserts one row — `dataset, sim, K, nodes, edges, avg_degree, components, isolates, path` — into `results/vir_graph_stats/virtual_graph_stats.csv` (one accumulating table across datasets; key = dataset×sim×K, no duplicates). For debugging bad scores (sparse/disconnected/isolates) + ablation-quality table for the paper.

### 2026-07-02 — Renamed `results/phase2/` → `results/vir_graph_variants/` (user-requested)

- **Rename:** task-score comparison CSVs (e.g. `cora_variant_comparison_K10.csv`) moved from `results/phase2/` to `results/vir_graph_variants/`. Clearer name; sits beside `results/vir_graph_stats/` so both Phase-2 result folders are self-describing.
- **Two sibling Phase-2 folders under `results/`:** `vir_graph_stats/` = per-graph health table (one graph = one row); `vir_graph_variants/` = downstream variant task-score comparison CSVs.
- Updated every reference (folders lead, code follows): notebook code cell (`Path("results") / "vir_graph_variants"`) + its markdown/stale outputs, this file, README tree, `docs/virgo_guide.md`. No Python source referenced the old path. Directory moved with `git mv` (CSV preserved), notebook JSON re-verified valid.

### 2026-07-04 — Phase 3 spine implemented (ViRGo-SAGE)

- NEW `encoder.py` — `SageEncoder` (repo style: argparse / `build_graph()` / `main(args)`, exposes `train(epochs)`). Features X = [deg, Ω, ψ, clustering] z-normalized, computed on the **original** graph (reuses `VirtualGraph.signatures` + cached core + `nx.clustering`); message passing over the **virtual** graph (2-layer mean `SAGEConv`, 4→64→64). Loss = Skipgram-analog: positives = window-10 co-occurrence on **unweighted node2vec walks over the virtual graph** (num_walks=10, len=40, workers=1, seeded — the exact Phase-2 bridge corpus), negatives ∝ deg^0.75 with Q=5 (matches Word2Vec `negative=5`), Adam lr=0.01, 50 epochs, 100k pairs/epoch (corpus capped 2M, deterministic). CPU-only + seeded torch/np/generators → reproducible. Writes word2vec-format `.emb` (existing evals unchanged).
  - `python encoder.py --input input/cora.edgelist --sim psi --k 10 --seed 42` → `output/<ds>/k<K>/sage_<sim>_s<seed>.emb`
- `benchmark_config.py` + `GNN_PARAMS` (hidden/dims 64, layers 2, mean agg, lr 0.01, epochs 50, Q=5, pairs_per_epoch 100k, max_pairs 2M).
- NEW `notebooks/3-phase3_gnn_encoder.ipynb` (15 cells) — **reads Phase-2 artifacts only, builds no graphs** (separation of concerns per request): load saved `virtual_<sim>.edgelist` → train spine per seed → `sage_nc_<sim>_s<seed>.emb` + loss curve → verify (node count, finite, KeyedVectors-readable) → NC weighted-F1 → leakage-free LP (reuses the SAME `_vglp_` splits, train-virtual rebuilt from the 70% edges, `sage_lp_*.emb`) → summary CSV `results/vir_graph_variants/<ds>_sage_spine_K<K>.csv` + bridge table displayed alongside. `2-virtual_graph_phase2.ipynb` untouched (stays the graph builder + bridge record).
- `import encoder` verified in env `i2v` (torch 2.12.0+cu130, PyG 2.8.0). Notebook NOT executed — first SAGE numbers come when it runs (README Phase-3 step 5).

### 2026-07-06 — First Phase-3 run (enzymes) + §6 head-to-head table + rename fallout

- **First spine run (user-executed, enzymes, Ψ, K=10, seed 42):** SAGE NC F1 0.5398 vs bridge 0.5051; LP AUC 0.6609 vs 0.5121 (details → `docs/paper_log.md`). Spine verified end-to-end: loss falls, evals read the `.emb`, outputs land in `output/enzymes/k10/`.
- **Phase-3 §6 rewritten into a real head-to-head:** one table with `deepwalk_bridge / virgo_sage / delta` per task (was: two differently-shaped tables side by side). Saves a second CSV `results/vir_graph_variants/<ds>_encoder_comparison_K<K>.csv`; single-seed std now 0.0 (was NaN). Fairness note added: bridge CSV must come from a Phase-2 run with the same DATASET/SIM/SEEDS (CSV does not record seeds).
- **Notebook renamed again by user:** `2-virtual_graph_phase2.ipynb` → `2-phase_2_virtual_graph.ipynb`. Patched references in README (roadmap + tree) and Phase-3 notebook (intro + load-cell assert). Historical notes entries left as-is.
- **Files missing on disk (flagged, not restored):** `results/vir_graph_variants/enzymes_variant_comparison_K10.csv` + `enzymes_sage_spine_K10.csv` were saved by the runs (per notebook outputs) but are no longer present; `results/vir_graph_stats/virtual_graph_stats.csv` lost its 9 cora rows (only the enzymes row remains). Restore = rerun Phase-2 §5/§9 and Phase-3 §6 (instant — all edgelists/`.emb` reuse from disk).
- **§6 revised again (user feedback: one table, non-technical readable):** both encoders now scored *inside* §6 directly from their saved embeddings (`vg_*` = Phase-2 bridge, `sage_*` = Phase-3) with the shared eval scripts — no dependency on the Phase-2 comparison CSV, numbers always match the Setup knobs. Single display/CSV: `task / deepwalk_phase2 / graphsage_phase3 / improvement / winner` → `results/vir_graph_variants/<ds>_encoder_comparison_K<K>.csv`. The separate `<ds>_sage_spine_K<K>.csv` save + sage-only summary table were removed from §6 (per-seed sage numbers still printed in §4/§5).
- **Accumulating scoreboard added (user request: keep every variant's results, visualize comparisons):** new `record_score()` in `scripts/results_io.py` upserts one row per dataset × encoder × sim × K × task (with seeds, mean, std) into `results/vir_graph_variants/scoreboard.csv` — same replace-never-duplicate pattern as the graph-health table. Phase-3 §6 now records both encoders there on every run; NEW §7 "Scoreboard" cell shows the full table for the current dataset + one grouped bar chart per task (bars = sim×K variants, colors = encoders, error bars = std). Old §7 "Status + next" renumbered to §8.
- **Scoreboard columns renamed for readability (user request):** `sim` → `graph_variant`, `K` → `top_K_neighbors` — in `results_io.record_score()` (params + row keys + upsert key + sort), the existing `scoreboard.csv` header (sed, data rows untouched), and Phase-3 §7 (reader cell + markdown now includes a column-meanings line). §6 calls are positional → unchanged.
- **3-seed enzymes head-to-head (user-run):** bridge NC 0.4971 / LP 0.5110±0.0297 vs SAGE NC 0.5405±0.0014 / LP 0.6632±0.0114 → paper_log entry updated (supersedes the single-seed numbers).

### 2026-07-06 — Ablation A implemented (A1 walk vs A2 edge positives)

- `encoder.py`: `corpus()` + `train()` gained `positives=` ("walk" = A1 window co-occurrence, unchanged default; "edge" = A2 = the virtual edges themselves as positive pairs, added in both directions, no walks). Same deterministic cap, negatives, loss. CLI: `--positives {walk,edge}`; CLI output auto-tags `sage_edge_` when edge.
- `scripts/benchmark_config.py`: `GNN_PARAMS["positives"] = "walk"`.
- `notebooks/3-phase3_gnn_encoder.ipynb`: Setup gained `POSITIVES` knob + derived `TAG` (`sage` / `sage_edge` file prefix) + `ENCODER` (`graphsage` / `graphsage_edge` scoreboard name); §2/§3/§4/§5 use TAG paths + pass positives; §6 compares the selected variant vs the bridge (dynamic column name, snapshot CSV suffixed `_edge` for A2), records under ENCODER; §2 markdown explains the knob; §8 status updated. A2 runs can never overwrite A1 files, snapshots, or scoreboard rows.
- **RESULTS (user-run A2 pass, enzymes ψ K10, seeds 42/43/44) → full entry in `docs/paper_log.md`:** A1 NC 0.5405±0.0014 / LP 0.6632±0.0114; **A2 NC 0.5413±0.0011 / LP 0.6909±0.0156**; bridge 0.4971 / 0.5110. FINDING: NC tie, LP A2 +0.028 (≈2σ) — walks unnecessary once similarity is explicit in the graph. DECISION: **A2 = default objective** going forward; A1 kept/reported as the bridge-comparable config. DEVIATION: Stage-1 A ran on enzymes, design doc said cora — repeat pending. Artifacts: `output/enzymes/k10/sage_edge_{nc,lp}_psi_s4{2,3,4}.emb`, `results/vir_graph_variants/enzymes_encoder_comparison_K10_edge.csv`, scoreboard rows `graphsage_edge`.
- GAP (minor, logged for later): notebook 3 reads its own `POSITIVES` knob, not `GNN_PARAMS["positives"]` — config entry currently informational only; unify to one source of truth. Notebook intro markdown still A1-worded ("same walk corpus") — stale for A2 runs.
- TODO next (Stage 1): **B** aggregation — wire `agg` (mean vs Ψ-weighted) into `SAGEConv` (currently hardcoded `aggr='mean'`, `GNN_PARAMS["agg"]` unused); then repeat A×B on cora.

### 2026-07-07 — Ablation B implemented (aggregation: mean | weighted | sum | max) + A2 default locked

- `encoder.py`: `SageEncoder(..., agg="mean")` — `mean`/`sum`/`max` = native `SAGEConv(aggr=...)`; `weighted` = `GraphConv(aggr='add')` (same root+neighbor form as SAGE but edge-weight-aware; SAGEConv ignores edge weights) fed the Ψ edge weights from the virtual edgelist, normalized per target node → Ψ-weighted mean. `forward()` passes `edge_weight` only when set. CLI `--agg {mean,weighted,sum,max}`; file tag now `sage_edge[_<agg>]_...` (mean keeps existing names → A2 rows stay the B baseline). Weights exist on both paths: saved edgelists carry `1/(1+dist)` and the LP train-virtual rebuild uses the same builder.
- **A decision applied:** `--positives` CLI default `walk` → `edge`; `GNN_PARAMS["positives"] = "edge"`; header comment + notebook §2/§8 markdown updated. Closes the "config entry informational only" note above in spirit (notebook still owns its knobs).
- `scripts/benchmark_config.py`: `GNN_PARAMS["agg"]` now live (was decorative), default `"mean"` until B decides.
- `notebooks/3-phase3_gnn_encoder.ipynb`: Setup gained `AGG` knob; `TAG`/`ENCODER` extended with `_<agg>` when non-mean (e.g. `sage_edge_weighted_*` files, `graphsage_edge_weighted` scoreboard rows — nothing overwrites); §2/§5 pass `agg=AGG`; §6 snapshot CSV suffix includes agg; §8 status updated.
- Protocol: freeze enzymes ψ K=10 seeds 42/43/44 positives=edge; `mean` already in the scoreboard (= A2 rows); run `weighted` → `sum` → `max` one by one via the knob; §7 picks the winner.
- **RESULTS (user-run, enzymes ψ K=10 seeds 42/43/44, positives=edge):** mean NC 0.5413 / LP 0.6909 · weighted 0.5399 / 0.6528 · max 0.5284 / 0.6378 · sum 0.5114 / 0.5063. Ranking **mean > weighted > max > sum on BOTH tasks**.
- **DECISION — `agg=mean` locked** (weighted lost → Ψ edge-weights uninformative beyond top-K ranking; sum degenerate = unnormalized degree blowup, LP at chance + ×4 variance; max lossy). **Finalized ViRGo-SAGE = edge positives (A2) + mean agg (B)** — already the code/`GNN_PARAMS` default, no change needed. Full table + reasoning → `docs/paper_log.md`.
- **NOT deleted:** all 4 agg rows kept in `scoreboard.csv` + the `sage_edge_{weighted,sum,max}_*` embeddings / `enzymes_encoder_comparison_K10_edge_{weighted,sum,max}.csv` snapshots — they ARE the ablation evidence for choosing mean (Deliverable #5).
- TODO next: repeat A/B on cora (confirm mean+edge hold on a 2nd graph); then C depth / E graph-variant sweep.
- **B results (user-run, enzymes ψ K10):** mean wins both tasks (NC 0.5413 / LP 0.6909) > weighted > max > sum (LP 0.5063 near-random) → `agg="mean"` stays default, encoder locked (edge + mean). Full table in `docs/paper_log.md`. No code change needed.

### 2026-07-07 — E-study sweep cell added (notebook 3 §8)

- New §8 in `notebooks/3-phase3_gnn_encoder.ipynb` (old §8 status → §9): loops `SWEEP_SIMS = [psi, degree, centrality]` × `SWEEP_KS = [10]` (add 5/20 for the full grid) under the LOCKED encoder (edge + mean). Per config/seed: reuse-or-train NC emb (saved virtual edgelist) + LP emb (train-virtual rebuilt from the shared 70% split), score with shared evals, `record_score` under `graphsage_edge`. No Phase-2 bridge dependency → degree/centrality need no Phase-2 rerun (all 9 virtual edgelists already on disk for enzymes). §7 visualizes after.
- Setup `AGG` reset "max" (user's last B run) → "mean" (B winner); knob comments now say DECIDED for A + B.
- Kept the user's §7 `DROP` display filter (hides losing B-agg rows; scoreboard untouched).
- Backend untouched (encoder.py / results_io / evals / config unchanged).

### 2026-07-07 — §8 sweep now also records the deepwalk baseline rows

- Grid target (user): {psi, degree, centrality} × {deepwalk, graphsage, graphsage_edge} × K=10 × seeds 42/43/44. Scoreboard had 5/9 cells; missing: deepwalk × {degree, centrality} (bridge `vg_*` embs already on disk from Phase 2, just never scored into the scoreboard) + graphsage(walk) × {degree, centrality} (needs training).
- `notebooks/3-phase3_gnn_encoder.ipynb` §8: after each config's encoder rows, now scores the SAVED Phase-2 bridge embeddings (`vg_nc/lp_<sim>_s<seed>.emb`) and `record_score`s them under `deepwalk`. Never trains the bridge; records only complete seed sets, else prints "run notebook 2" hint (k5/k20 have no `vg_*` files yet → skipped there).
- Fill procedure (one run): Setup `POSITIVES="walk"` → run Setup + §1 + §8 (trains 12 sage walk embs for degree/centrality, reuses psi; records graphsage + deepwalk rows) → reset `POSITIVES="edge"` → §7 for the table. graphsage_edge rows already recorded.
- Backend untouched.

### 2026-07-08 — Repository layout redesign: notebook-first zones (user-requested, applied)

- **Why.** User (beginner) found `output/<ds>/k<K>/` unnavigable — 60+ mixed files (`virtual_*`, `vg_*`, `sage_edge_weighted_lp_*`) in one folder. New rule: **every path reads as a sentence** — notebook → content/task → dataset → K → variant → `<encoder>_s<seed>.emb`.
- **New layout.** `output/notebook1_reproduce_i2v/<ds>/{node_classification|link_prediction}/<model>_s<seed>.emb` · `output/notebook2_create_vir_graph/{virtual_graphs|node_classification|link_prediction}/<ds>/k<K>/<sim>/…` (graphs named `virtual_graph.edgelist`) · `output/notebook3_gnn_encoder/{node_classification|link_prediction}/<ds>/k<K>/<sim>/<encoder>_s<seed>.emb`. Splits: `splits/link_prediction/{original_graph|virtual_graph_study}/<ds>/seed_<s>/{train.edgelist,train_neg.txt,test_pos.txt,test_neg.txt}` (fixed names, no prefixes). Results: `results/scoreboard.csv` (master), `results/graph_health.csv` (was vir_graph_stats/virtual_graph_stats.csv), `results/snapshots/` (was vir_graph_variants/ comparison CSVs), `results/notebook1_reproduce_i2v/{benchmark,cora,…}` (Phase-1 tables).
- **Renames.** Files: `vg_*` → `deepwalk_s<seed>.emb`, `sage_*` → `graphsage_walk_*`, `sage_edge_*` → `graphsage_edge_*` (agg suffixes kept); Phase-1 `<base>_nc_orig_s42.emb` → `i2v_s42.emb` (folder now carries dataset+task). Scoreboard encoder `graphsage` → `graphsage_walk` (rows edited in place; `graphsage_edge*` unchanged); graph_health `path` column refreshed.
- **Migration.** 190 items moved (idempotent script, scratchpad); splits moved with `git mv` (tracked), output/results plain moves (gitignored). Old dirs removed. No retraining — all `.emb`/edgelists byte-identical, only relocated.
- **Code follows folders** (project rule): `prepare_linkpred.prepare(input, outdir, …)` now writes the 4 fixed-name files into a seed folder (dropped `name` prefix); `eval_linkpred.evaluate(emb, split_dir, …)` reads fixed names (dropped `name`); `runner.py` uses `NB1_DIR`/`LP_SPLITS_ORIG`; `benchmark_config.py` gained zone constants (NB1/NB2/NB3_DIR, LP_SPLITS_*, SCOREBOARD_CSV, GRAPH_HEALTH_CSV, SNAPSHOTS_DIR); `results_io.record_score` default → `results/scoreboard.csv`; `main.py` saves numbered CSVs under `results/notebook1_reproduce_i2v/`; `benchmark_baselines.save_benchmark` → its `benchmark/` subfolder; `virtual_graph.py`/`encoder.py` CLI defaults → new zones (encoder CLI tag now `graphsage_walk`/`graphsage_edge`).
- **Notebooks.** All three updated to the zones; notebook 3's `TAG` removed (single `ENCODER` = filename prefix = scoreboard name) and Setup `POSITIVES` reset `"walk"` → `"edge"` (the locked A-winner; was left on walk from the grid-fill run). Notebook 1 Step 6 candidate-path search rewritten for the new zones.
- **Docs.** README (roadmap paths + tree + results path), splits/README.md (new layout + 4-file table), scripts/README.md.
- Verification: all moved files re-inventoried (18 virtual graphs, 96 embeddings, 64 split files, 8 snapshot CSVs); smoke evals on moved artifacts reproduce the recorded scores (see same-day entry below if run).

## 2026-07-09 — Notebook 3 §7 scoreboard: two display modes (presentation only, no data change)

- **Mode 1 (§7, `96eec70e`) — MAIN RESULTS:** filters the scoreboard to `deepwalk` vs `graphsage_edge` (the locked ViRGo encoder) only; two pivot tables (NC first, then LP), rows = graph_variant × K in psi→degree→centrality order, plus `improvement` (= graphsage_edge − deepwalk) and `winner` columns; completeness line on top (rows = encoders × graphs × tasks, gaps show as NaN); charts restricted to the same two encoders. Rationale (user): choose best encoder → lock it → then study graphs/K; main results = 2 encoders only.
- **Mode 2 (§7b, new cells `ed2b1ea5`+`539b2172`) — ABLATION HISTORY:** same table layout but ALL encoders in the scoreboard (graphsage_walk, B-agg variants); one knob `SHOW = "all"` or an encoder list; winner column only (no improvement). Documents how the encoder was locked (A: edge>walk, B: mean>others).
- Both modes are display-only; `results/scoreboard.csv` untouched, old DROP filter replaced by the mode split. "How to read" markdown (`0cda9c5c`) unchanged, serves both modes.
- Earlier same day: §7 rewritten from raw CSV dump to per-task pivot tables (the "17 rows" confusion = 0-based index, grid was complete); NB2 task folders restored from manual `_smoke_test` rename.

## 2026-07-09 — Ablation D plumbing (feature knob)

- `encoder.py`: `SageEncoder(..., feats="all")` + one branch in `features()`; `--features {all,degree,deg_cent,psi,random,const}` CLI flag; run tag gains `_feat_<set>` when != all. `dims` derive from `X.shape[1]`, so 1/2/4-column inputs need no further change.
- `scripts/benchmark_config.py`: `GNN_PARAMS["features"]="all"` + `D_FEATURES` registry (id → label, description).
- `notebooks/3-phase3_gnn_encoder.ipynb`: `FEATURES` knob in Setup (feeds `ENCODER` name); `feats=FEATURES` threaded into the §2, §5 and §8 training calls; new **§8b** markdown + code cell runs the D loop, records to the scoreboard and prints/saves the D table. §7 Mode 1 hides the `_feat_*` rows automatically (its `MAIN` filter); §7b `SHOW="all"` surfaces them.
- Verified: six feature sets build with dims 4/1/2/1/4/1, all finite; `const` → all-zeros X and identical embeddings (cosine LP has a `+1e-12` guard, so AUC = 0.5, no crash); `random` reproducible per seed and different across seeds; `degree` column == `all[:, 0:1]`; D0 embeddings on disk re-score to 0.5413 / 0.6909 (unchanged by the refactor).

## 2026-07-12 — Cora extended to the 5-variant E-grid + cross-dataset cell

- **Built** `original` + `hybrid` virtual graphs for cora K=10 via `virtual_graph.py` CLI (original: 2708 nodes / 5278 edges = exact copy; hybrid: 21399 edges = original ∪ Ψ top-K) → `output/notebook2_create_vir_graph/virtual_graphs/cora/k10/{original,hybrid}/`. Unblocked the §8 sweep assert (`virtual_graph.edgelist missing`).
- **DeepWalk bridge fill** for cora original/hybrid, both tasks, seeds 42/43/44 — scratchpad script with the exact notebook-2 recipe (plain edgelist → DeepWalk, I2V_PARAMS; LP from the shared 70% train splits). 12 `.emb` into notebook-2 zones; 4 scoreboard rows.
- **§8 sweep run on cora** (user): GraphSAGE NC+LP for all 5 variants recorded. Cora K=10 grid now complete: 5 variants × 2 encoders × 2 tasks = 20 rows.
- **Notebook 3 §10 added** (cells `7da74cbe` + `b9e6a690`): cross-dataset view — one pivot per task, columns = dataset × encoder, rows = graph variant; `XD_K` knob; reads scoreboard directly, independent of `DATASET`. Display only.
- Findings → paper_log.md same-day entry (headline: story inverts on cora — DeepWalk + original graph wins both tasks 0.81/0.90).

## 2026-07-13 — Notebook 3 §11: research-question view (presentation only)

- New cells `03228a47` + `f1ce4139` after §10: PRIMARY table (graph variants × dataset·task, cell = best-over-encoders score, ★ winner), ANSWER table (best graph + achieving encoder per dataset × task), SECONDARY table (Δ = graphsage_edge − deepwalk within each graph). `RQ_K` knob; reads scoreboard directly; display only.
- Framing per review: virtual graph = primary variable; encoder = secondary within-graph comparison.
- Revised same day (user): §11 reduced from 3 tables to 2 — Table 1 = best graph + winning encoder per dataset × task; Table 2 = biggest GraphSAGE gain/loss vs DeepWalk per dataset × task ("no loss" when all Δ positive). Big per-variant pivot dropped (duplicated §10).
- Revised again (user): §11 = Table 1 best NON-original graph per dataset × task + Table 2 control check (original vs best virtual, meaning column; thresholds: Δ>0.05 much / >0 slightly / <0 virtual better). `original` framed as control baseline, not ViRGo contribution.

## 2026-07-16 — D6 (features without message passing) + K=10 finalized in the notebooks

- **`encoder.py`: `layers=0` is now a documented D6 path.** No new code path was needed — with `layers=0`, `dims` degrades to `[in, out]`, `range(0)` builds no convs, and `forward()` returns the z-normed features unchanged (verified bit-identical to raw `X`). So D6 reuses the *same* feature builder as D0; the only difference between the rows is message passing. Added: class comment, `train()` assert (`layers=0` has no parameters — fails loudly instead of silently no-op'ing), `--layers` help text, CLI tag `features_only` + training skipped in `main()` when `layers=0`.
- **`scripts/benchmark_config.py`:** `D_FEATURES["none_mp"] = ("D6 features only", ...)` label.
- **Notebook 3 §8b:** code cell uncommented (was fully commented out) and D6 appended after D0–D5 behind a `RUN_D6` knob; markdown gains the D6 row, the msg-passing column, and the 2×2 decision rule. D6 writes `features_only_s<seed>.emb` (own name ⇒ never overwrites a GNN run, lands as its own scoreboard row); passes the base graph in place of the virtual graph since it is unused at `layers=0` (avoids a pointless top-K build); LP features come from the 70% train graph (verified cora: 3548 vs 5278 edges ⇒ no leakage). Smoke-tested on cora (NC 0.2700, LP 0.5077) — enzymes run not yet executed.
- **Notebook 3 §7b deleted** (user): its only feature over §7 was the `SHOW` encoder knob; §7's `MAIN` list stays hardcoded. Stale references remain in §7 markdown ("Every other encoder → Mode 2 (§7b)") and the §8b markdown pointer — noted, not yet cleaned.
- **K=10 finalized in the notebook views.** Notebook 3: `XD_K`/`RQ_K` → `[10]` (§9/§10 showed NaN for enzymes at K=5/20; §7 keeps `[5,10,20]` = cora's complete sweep). Notebook 2: single `KS = [10]` knob (was `VG_K` = `[5,10,20]`), feeding §5/§7/§8/§9 — a linear pipeline needs one K to stay coherent, unlike notebook 3 whose sections read the scoreboard independently. `VG_K` restores the full sweep. No result files deleted; `graph_health.csv` keeps all 30 K=5/10/20 rows; new summary CSV is `_K10.csv`, old `_K5-10-20.csv` retained.
- **§7 chart fix** (user-supplied): two-line variant labels, figsize 7×3.2 → 16×5, 45° right-aligned ticks — 15 bar groups were unreadable. Dropped the supplied `rot=45` (overridden by `set_xticklabels`) and `subplots_adjust` (cancelled by `tight_layout`).
- **Abandoned:** a background job to fill enzymes K=5/20 (DeepWalk bridge + GraphSAGE) was started and killed once K=10 was finalized instead. It left 8 stray `deepwalk_*.emb` under `output/notebook2_create_vir_graph/*/enzymes/k5/`; harmless (nothing at K=10 reads them) and reuse-or-create would pick them up if the sweep is ever resumed.

## 2026-07-17 — Proteins wired as a third dataset

- **Root cause of the "malformed" proteins file** (notes 2026-06-17 was wrong): `input/proteins.edgelist` is **comma-delimited** (`12,1`), while every reader in the repo uses networkx's whitespace default. Nothing was empty or corrupt.
- **`make_labels.make_proteins()`** (new): fetches nrvis `PROTEINS.zip`, writes `input/proteins_nr.edgelist` (whitespace) + `labels/proteins_nr.labels`. Rebuilds the graph from the zip's `proteins.edges` rather than normalizing the author file, so ids and labels align by construction — the `make_citeseer_linqs` pattern. Asserts edge overlap **== 1.0** against `input/proteins.edgelist` before writing (measured 1.0000; the author file is md5-identical to the zip member `1fa696dfeb062fc95b7510d0c22bb038`, so the rebuild is the same graph). Author file untouched, md5 re-verified after the build.
- **Chose the `_nr` rebuild over patching the readers**: the alternative was a comma-tolerant delimiter in `virtual_graph.build_graph`, `encoder.py`, `prepare_linkpred.py` and the notebooks — 1 derived file vs 4+ readers, and it keeps CLAUDE.md's "never modify `input/`".
- **Two parser gaps found and fixed:** `make_labels.read_edgelist()` used `line.split()` → `ValueError` on `12,1` (now `.replace(",", " ")`; `_parse_edges` already did this). And `PROTEINS.node_labels` ships `node_id,label`, whereas `ENZYMES.node_labels` ships a bare label per line — `_make_nr_labels`'s `enumerate` would have stored `"1,1"` as the class. `make_proteins` takes `ln.split(",")[-1]`, which handles both; `_make_nr_labels` left alone (enzymes-only).
- **Registries touched:** `make_labels` `_VERSIONS` (`proteins` → safe `proteins_nr`), `_ensure_nr`, `ensure_labels`, `prepare_dataset`; `scripts/benchmark_config.DATASETS` gains `proteins` + `proteins_nr`, both pointing at the rebuilt pair. `DATASET = "proteins"` now resolves in notebooks 2/3 with no further edits.
- **`BENCH_DATASETS` deliberately left as `["cora","citeseer","enzymes"]`** — that list drives the Phase-1 cross-model baseline sweep (I2V/DeepWalk/node2vec/struc2vec), which is done; proteins is for the Phase-2/3 virtual-graph study.
- Smoke test (no files written): psi/degree/centrality virtual graphs build in 1.5s/2.2s/0.9s at K=10, 0 isolated nodes. No `PowerIterationFailedConvergence` despite 1195 components. `NearestNeighbors` on the 1-D signature is a kNN tree, not O(N²) — 43k nodes is not a scale problem.

## 2026-07-17 — Notebook 2 §7/§8 progress output (opaque node2vec bar)

- **Symptom:** cell 7 on proteins printed only `Computing transition probabilities: 1%| | 493/43466 [01:09<1:37:53, 7.32it/s]` — node2vec's own tqdm bar, with no indication of dataset, variant, seed, task, or how many runs remain. A 2.5h run looked like a hang.
- **Fix (display only, no result touched):** setup cell imports `time`; `embed()` prints `reuse seed N -> file.emb` or `TRAIN seed N | nodes/edges | cost <Σdeg²>` before training and `done seed N in X.X min` after; `link_pred_auc()` announces the 70/30 split and the train-only virtual-graph build; §7/§8 loops print `[i/N] <dataset> <task> | <variant> K=10 | 3 seeds` and the mean±std per variant. node2vec's bar is **kept** (it is the only live per-node progress) but now sits under a header that says what it belongs to.
- `cost` = `sum(d*d for _, d in virtual.degree())` — node2vec precomputes 2nd-order transition tables, so runtime tracks **Σdeg², not edge count**. Flags `<- SLOW` above 1e8. This is what makes `degree` 2h27m vs `psi` 2m23s per seed on proteins (see paper_log 2026-07-17: the degree variant degenerates into a star; the runtime is the symptom, the validity problem is the point).
- **Kernel state at the time:** no ipykernel process alive — the run had already died mid-`degree` seed 43; the visible bar was a stale snapshot. `psi` s42/43/44 and `degree` s42 are on disk; nothing else.
- **Not changed:** `embedding_models._RandomWalkModel` still calls node2vec with `p=q=1`. DeepWalk is first-order uniform, so the entire 2nd-order precompute is wasted work — a plain uniform walker would cut `degree` from hours to seconds. Left alone because it would shift the RNG stream and break comparability with every scoreboard row already recorded.

## 2026-07-18 — Virtual-graph tie-break fix (`virtual_graph.py`)

- **Bug:** 1-D signature + `sklearn.NearestNeighbors` = query-independent tie-break, so every node in a degree tie class got the same K winners → disjoint stars (enzymes hub 6,724; proteins 14,644; cora 582). Full analysis + invalidated results in `docs/paper_log.md` 2026-07-18.
- **Fix:** dropped `NearestNeighbors`. Signatures quantized at `SIG_TOL = 1e-9 × spread` → exact-tie classes → each node samples K from its own class with a seeded RNG; classes smaller than K+1 widen outward to the nearest values (preserves old kNN semantics where signatures are near-unique, and the exact-K contract everywhere).
- **`VirtualGraph.__init__` gained `seed=42`** (3rd positional arg, after `e`). Callers using `VirtualGraph(G)` are unaffected; `main()` passes `--seed`, notebook 2 passes `REPRO["seed"]`. Build is deterministic per seed, and differs across seeds by design.
- **`graph_health.csv` gained `max_degree`** (notebook 2 `record_stats`) — `avg_degree` is blind to a star, which is why the defect survived 3 datasets. Existing rows lack the column until rebuilt.
- Notebook 2 `embed()` warning reworded: it blamed "degree ties make a giant hub", now points at `max_degree`.
- **`sklearn.neighbors` import removed** from `virtual_graph.py` (no longer used; sklearn is still a dep via the eval scripts).
- Verified: all 5 variants × cora/enzymes pass the section-4 constraint report; notebook-3 train-virtual path (`VirtualGraph(Gt).build`) unaffected; rebuild with the same seed is byte-identical.
- **Not done (needs a compute run, user-driven):** regenerating the virtual graphs, embeddings and scoreboard rows. Old artifacts left in place on purpose — deleting them would strand the scoreboard in a half-state.

## 2026-07-18 — LP splits no longer restricted to the largest component (`prepare_linkpred.py`)

- **Bug:** `build_graph()` returned `G.subgraph(largest_connected_component)`. Enzymes LP ran on 125/19,474 nodes (17 test edges), proteins on 620/43,466. Analysis + superseded results in `docs/paper_log.md` 2026-07-18.
- **Fix:** return the whole graph. `split_edges` needed no change — `nx.minimum_spanning_tree` yields a spanning forest on disconnected input, so per-component connectivity already held; comments/docstrings updated to say "forest".
- Verified seeds 42 on cora/enzymes/proteins: 100% node coverage, test fraction exactly 30.0%, train component count == graph component count, no train/test overlap, negatives are real non-edges.
- **All LP splits on disk are stale** (`splits/link_prediction/**`), cora included. Not deleted — regenerating is a user-driven run.
- **Not fixed (deliberate, logged):** negatives are sampled uniformly across all pairs, so on many-component graphs they are mostly cross-component and inflate AUC. Left as-is for I2V comparability.
- **Also noted, not fixed:** `split_edges` and `sample_non_edges` are both seeded with the same `seed` value — different draw types, so no direct collision, but the streams are correlated by construction.

## 2026-07-18 — LP negatives sampled within-component (`prepare_linkpred.py`)

- **Bug:** negatives drawn uniformly over all node pairs while positives are always within-component ⇒ enzymes/proteins negatives were 99.8-99.9% cross-component, so "same component?" alone separated the classes. Full analysis in `docs/paper_log.md` 2026-07-18.
- **Fix:** `sample_non_edges(..., negatives='component')` is the new default — both endpoints from one component, component chosen with cumulative weights = exact within-component non-edge capacity (`n(n-1)/2 - m`), complete/1-node components skipped, assert if capacity < k.
- **`negatives='uniform'` keeps the old protocol** (`--negatives uniform`), threaded through `prepare()`; Phase-1 numbers were made under `uniform` and must be regenerated with it.
- Also: `sample_non_edges` now seeds with `seed + 1` (was reusing `split_edges`' seed — correlated streams).
- Verified all 3 datasets both modes: 100% within-component under the default, 0 invalid pairs, 0 duplicates, deterministic.
- Added stdlib `itertools` import. No new dependency.
- **All LP splits stale again** (on top of the largest-CC fix). Expect enzymes/proteins AUC to fall — artefact removal, not regression.

## 2026-07-18 — Per-component Omega + one graph policy

- **`identity2vec_cached.Graph(..., per_component=)`**: Ω computed inside each component, clamped non-negative, rescaled to max=1. Default stays `False` = baseline-identical (Phase-1 / Deliverable #1 verified `True` after every change). `VirtualGraph` opts in via policy.
- **`graph_io.py` (new)**: `GRAPH_POLICY` (self_loops / directed / centrality / sig_tol / lp_negatives) + `I2V_BASELINE_POLICY`, `policy_of()` (rejects unknown keys), `load_graph()` (auto delimiter, self-loops per policy), `properties()`, `check()` (+`strict`).
- **Single reader**: `virtual_graph`, `encoder`, `prepare_linkpred` all call `graph_io.load_graph` — self-loop inconsistency is now structurally impossible. Verified identical graphs out of each.
- `SIG_TOL` moved out of `virtual_graph.py` into the policy; `prepare_linkpred` negatives default to the policy; `benchmark_config` re-exports the policy (`sys.path` insert of PROJECT_ROOT, no circular import — `graph_io` imports nothing from config).
- **Caught a false positive before shipping**: auto-detected directedness flagged proteins 81,044/81,044 (it counts storage format, not direction). Now declared per dataset — `politics.directed_source = True` — never inferred. `properties()` key renamed `asymmetric_edges` → `single_direction_edges` to stop implying direction.
- Notebooks NOT wired to `check()` yet (notebook 2 was mid-run).

## 2026-07-20 — Review follow-ups: single reader everywhere, deterministic splits, honest table names

Full read-only review of every module, both notebooks and the on-disk artifacts, then the fixes below. **Nothing was executed except `python -m py_compile`** — no runs, no notebooks, no re-validation of the A/B/D/K ablation decisions (user instruction: those stand as-is). Nothing here is a measured result.

**Reproducibility / correctness**
- `prepare_linkpred.split_edges` returned `list(tree) + removable[...]` where `tree` is a **set**, so the written row order — and therefore downstream node insertion order and the walk RNG — depended on set iteration order. Both returned lists are now `sorted()`. Which edge is train vs test was already seed-fixed; only row order changes.
- `sample_non_edges` accumulated into a **set** and `prepare()` slices that into train/test negatives, so *which* negatives were train vs test also came from set order. Now a list in RNG draw order — deliberately **not** sorted, since sorting would put every low-id pair in train and every high-id pair in test.
- `write_pairs` no longer rewrites a file whose content is unchanged, so a split file's mtime now genuinely means "this split changed".
- `runner.run_linkpred_repeated` reused a cached `.emb` even though it calls `prepare()` on every pass — a regenerated split plus an old embedding scores a model against a test set it never trained around. It now reuses only when `emb.mtime >= train.edgelist.mtime`, otherwise retrains and prints why. Self-healing, so no embeddings were deleted by hand.

**Phase-1 contract**
- `I2V_BASELINE_POLICY` existed but **nothing consumed it**: both `runner` LP paths called `prepare()` with no `negatives`, silently inheriting `GRAPH_POLICY`'s new `component` default and overwriting Phase-1 splits under the wrong protocol. Both calls now pass `negatives=I2V_BASELINE_POLICY["lp_negatives"]` explicitly.

**Single reader (finishing the 2026-07-18 work)**
- `train.py`, `embedding_models.py` (×2) and both notebooks (`load_base_graph`, nb3 cell 3, 4× train-graph reads) still called raw `nx.read_edgelist` — exactly the points where G enters I2V, DeepWalk/node2vec, `VirtualGraph` and `SageEncoder`. All now go through `graph_io.load_graph`, so delimiter detection and the self-loop policy apply everywhere. Latent until now (every registered dataset has 0 self-loops, and only `proteins` is comma-delimited — which is why the `proteins_nr` rebuild existed), but one new dataset would have broken it silently. `networkx` import dropped from `embedding_models.py`; kept in `train.py`.

**Honest reporting**
- `benchmark_baselines.save_benchmark` wrote `table2_linkpred_auc.csv` (= headline = cosine) and `table2_linkpred_auc_cosine.csv` — byte-identical files under two names, with a comment claiming the first was logreg. Both files are now named after the scorer that produced them (`table2_linkpred_auc_{cosine,logreg}.csv`) and the second is always the *other* scorer, so a duplicate is structurally impossible. The orphaned `table2_linkpred_auc.csv` was moved to `results/superseded_2026-07-18/`.
- `results_io.record_score` used population std (`np.std`, ddof=0) while `runner.summarize_seed_results` and pandas `.agg("std")` use sample std (ddof=1) — for 3 seeds the scoreboard's std was ~18% smaller than the benchmark's for the same numbers. Unified on ddof=1 (0.0 for a single seed). **Means unchanged; existing scoreboard `std` values remain ddof=0 until those rows are rerun.**
- `eval_nodeclass` runs LBFGS at `max_iter=300` (paper protocol, unchanged) while both notebooks call `warnings.filterwarnings("ignore")`, so a non-converged fit would have been reported as an ordinary F1. It now catches `ConvergenceWarning` locally and prints that the F1 is a lower bound. No numbers change.
- `graph_io.check()`'s isolate message claimed isolates get "no virtual edges" and are "dropped from node-classification scoring". Both wrong: isolates share the degree-0 / Ω-0 tie class and **do** receive K virtual edges; what they lack is link-prediction pairs. Text corrected.

**Artifacts**
- Moved pre-fix `cora_*` / `enzymes_*` CSVs from `results/snapshots/` into `results/superseded_2026-07-18/snapshots/`. Dated Jul 4-16, they predate all three 07-18 fixes and were the last unmarked stale results on disk. `results/snapshots/` now holds proteins only, consistent with `scoreboard.csv` and `graph_health.csv`.
- Deleted `splits/link_prediction/original_graph/webkb_wisc/` (12 tracked files): its input edgelist was deliberately deleted 2026-07-02, leaving the split orphaned. Recoverable from git history, same convention as the inputs.

**Safety cleanup (same day, user-spotted).** `encoder.features()` mixed two indexing conventions in one expression: `deg[n]`, `ev[n]` and `clus[n]` were node-keyed dict lookups while `psi[i, 0]` was a positional row index. The row order of `psi` is decided inside `VirtualGraph.signatures()` (its own `list(self.G.nodes)`); `self.nodes` was captured separately in `__init__`. Two independent captures, silently assumed equal — and `signatures()` already returns its node list, which the call discarded with `_`. Traced every path: `VirtualGraph` mutates only `V`, `identity2vec.Graph.__init__` only stores the reference, the cached subclass and `nx.clustering` only read, so `G` is never modified between the captures and networkx node dicts preserve insertion order — the two lists were in fact identical, and a divergence would most likely have raised `IndexError` rather than corrupted values. Now `psi` is converted to a `{node: value}` dict from the returned node list and looked up by node like the other three. **No numeric change; no artifact invalidated.**

**nb3 virtual-graph seed made explicit (same day).** Notebook 3's three `VirtualGraph(Gt)` calls (cells 11, 18, 20) relied on the constructor default `seed=42` while notebook 2 reads `REPRO["seed"]`. Both were 42, so nothing was inconsistent in practice — but changing the project seed would have moved nb2's virtual graphs and left nb3's on 42 silently, so nb3's link-prediction embeddings would have trained on a different train-virtual graph than nb2 built for the same config, with no error raised. All three now pass `seed=cfg.REPRO["seed"]` explicitly. **Deliberately the project seed, not the per-run loop seed:** the virtual graph is built once per (sim, K) and shared across seeds 42/43/44, so the reported ±std covers split and encoder-init variance but *not* tie-sampling variance — a limitation to state in the paper, not a bug. **No numeric change; no artifact invalidated.**

## 2026-07-21 — Direction locked (project review): LoG paper + characterization study; docs updated

- The post-fix results (original graph wins all cells) were reviewed and accepted ("expected, keep as is"). No attempt to rescue a "virtual graph wins" story.
- **Immediate focus = LoG (Learning on Graphs) conference**; thesis (~1 month out) reuses the paper content.
- **Stay purely structural** — ViRGo's features are graph-derived (degree, centrality, Ψ, clustering; they build the virtual graph and feed GraphSAGE, and ablation D showed they are necessary — random features collapse performance). Do NOT use external node attributes (OGB text, biological descriptions): they would confound the study — a gain could then come from the attributes, not the structural rewiring. Isolate structural identity.
- **Datasets: add small-to-medium OGB only** — ogbn-arxiv (node property), ogbl-collab / ogbl-ddi (link property). Skip the 100M-node graphs (too slow before the deadline).
- **OGB protocol (fair, structural-only):** on each OGB dataset both graphs get the SAME structural features (degree, centrality, Ψ, clustering); only the edges differ (original = real edges, virtual = role edges). Ignore OGB's extra attributes (text, product, biological). Only the graph structure varies, so any gap measures virtual rewiring. Do NOT compare to the OGB leaderboard (top models may use attributes); the paper states the goal is structural analysis, not leaderboard superiority.
- **Per-dataset features (verified on OGB docs, 2026-07-21):** `ogbl-ddi` = NO node features (homogeneous drug-drug graph) → drop-in structure-only LP benchmark, no change. `ogbn-arxiv` = 128-dim skip-gram text features; `ogbl-collab` = 128-dim text features → we do not load these; structural methodology unchanged. Correction: an in-conversation note that "ogbn-arxiv has no node features" was a slip for ogbl-ddi — ogbn-arxiv DOES have 128-dim features, which we ignore. (Answered Mohammed's email: ogbl-ddi is still featureless, unchanged.)
- **Characterization:** compute per-dataset properties (homophily = the primary one, flagged as such in review; plus degree spread, clustering, component fraction, label-vs-topology agreement) → predict the original-vs-best-role gap → conclude "for this class original is better, for this class this feature is better".
- **Encoder order:** finish GraphSAGE results first, THEN swap in GIN (isomorphism / WL power) to see if it helps. Learnable-alpha = future work (needs synthetic datasets), NOT in the next ~11 days.
- **Anomaly detection dropped** from the immediate plan (replaced by the characterization study).
- **Docs updated to match:** `CLAUDE.md` (goal, contributions, tasks, phases, deliverables), `README.md` (question, status, results now post-fix), `docs/virgo_guide.md` (goal/phases/datasets/deliverables). No code or notebooks changed. Research-direction entry mirrored in `docs/paper_log.md` same day.

## 2026-07-22 — OGB plumbing added (ogbn-arxiv NC + ogbl-ddi LP), official-protocol wiring

Four changes, all reusing the existing pipeline (`virtual_graph.py`, `encoder.py` untouched):
- **`make_ogb.py`** (new) — converter mirroring `make_labels.py`. `make_ogbn_arxiv()`: writes `input/ogbn_arxiv.edgelist` (full transductive graph), `labels/ogbn_arxiv.labels`, `input/ogbn_arxiv.nodes`, `splits/ogb/ogbn_arxiv_idx.npz` (official train/valid/test node ids); `data.x` (128-dim text) ignored. `make_ogbl_ddi()`: writes `input/ogbl_ddi_train.edgelist` from **training edges only**, `input/ogbl_ddi_train.nodes`, `splits/ogb/ogbl_ddi_pairs.npz` (official valid/test pos+neg). Raw OGB download cached in `output/ogb_raw/` (kept out of `input/`). `ensure_ogb(name)` builds-if-missing and returns path info.
- **`graph_io.load_graph()`** — after reading an edgelist, if a sibling `.nodes` sidecar exists, `add_nodes_from` it. Restores nodes with no (training) edge so every OGB test-pair endpoint gets an embedding. Backward compatible: core datasets have no sidecar → unchanged.
- **`scripts/benchmark_config.DATASETS`** — registered `ogbn_arxiv` (with `eval:"ogb"`, `split`, `directed_source`) and `ogbl_ddi` (with `eval:"ogb"`, `pairs`). The `eval:"ogb"` flag routes these two to the official-protocol path; the core four keep the existing path.
- **`eval_ogb.py`** (new) — `evaluate_nodeclass()` (logreg probe on official train ids → Accuracy via OGB `Evaluator`, + weighted/macro F1 on test) and `evaluate_linkpred()` (cosine-score OGB pos/neg → Hits@20 via OGB `Evaluator`, K=20). Kept separate from `eval_nodeclass.py`/`eval_linkpred.py` because OGB uses fixed splits + its own Evaluator, not the ViRGo random 70/30.

Not yet wired: notebook orchestration for the two OGB datasets (build → 5 variants × 2 encoders × K10 → `eval_ogb`) and scoreboard columns for Accuracy/Hits@20. Nothing run yet. The methodology decision behind this is in `docs/paper_log.md` (2026-07-22).

**Post-review fixes (same day).** External review confirmed the plumbing but flagged three issues; all three verified against the code and fixed:
- **Test-peeking was comment-only.** Both `eval_ogb.py` functions scored valid *and* test on every call, so variant selection would have exposed test numbers each run. Now one call scores ONE split (`split="valid"` default for selection; `split="test"` for the single final read; asserted). Secondary weighted/macro F1 now reported for whichever split is scored.
- **Scoreboard had no metric column.** `record_score` stored only task/mean/std — an arxiv Accuracy row and a cora weighted-F1 row would both read as an anonymous "mean", and valid-vs-test OGB rows would have overwritten each other. Added `metric` to the row *and* the upsert key (`results_io.py`); legacy task strings ("link prediction (AUC)", "node classification (weighted F1)") map to `auc`/`weighted_f1` via `LEGACY_METRIC`, with a backfill guard for pre-migration files. Migrated the live `results/scoreboard.csv` in place: 128 rows, `metric` inserted after `task`, no row changed value. OGB rows will pass explicit metrics like `valid_hits@20` / `test_acc`, so both splits can coexist as separate rows.
- **Features recomputed per encoder instance.** `SageEncoder.features()` recomputed degree/centrality/Ψ/clustering from the original graph on every construction — identical values 15× per dataset×task sweep (5 variants × 3 seeds); worst on ogbl-ddi, where `nx.clustering` on a ~4.3k-node/~1M-edge graph is O(Σdeg²) ≈ 10⁹ pure-Python checks per call. Added an optional disk cache: `feature_cache(input_path)` returns `output/feature_cache/<stem>_<md5[:12]>.npz`, content-hashed over the edgelist + `.nodes` sidecar bytes so a regenerated graph can never reuse stale features. The cache stores all four columns; ablation-D subsets select from it. The CLI passes it automatically; direct `SageEncoder(...)` calls without `cache=` behave exactly as before (notebooks unaffected until they opt in).

**`run_ogb.py` added (same day) — the step-6 workflow, closing the "not yet wired" gap.** One CLI runs a whole OGB dataset under the official protocol: `ensure_ogb` (converter files if absent) → `load_graph` + `check` → per variant: build-or-reuse the virtual graph (K=10, health row logged to `graph_health.csv` on build) → per encoder (`graphsage_edge` → notebook-3 zone, `deepwalk` bridge → notebook-2 zone, flat unweighted copy of V exactly as notebook 2 does) × seeds 42/43/44: train-or-reuse `.emb` → score ONE split via `eval_ogb` → `record_score` with split-carrying metric names (`valid_acc`, `valid_hits@20`, `test_acc`, `test_weighted_f1`, `test_macro_f1`, `test_hits@20`; task strings `"node classification (OGB official)"` / `"link prediction (OGB official)"`). Discipline is workflow-enforced: the default run scores **validation** and prints the validation-selected winner; `--final` is the single **test** read, reuses the saved embeddings (retrains nothing), and *refuses to run* while the scoreboard holds no validation rows for the dataset. Flags `--encoder/--sim/--seeds/--k` allow timing one seed first. Usage: `python run_ogb.py --dataset ogbl_ddi` then, after selection is locked, `python run_ogb.py --dataset ogbl_ddi --final`. Nothing run yet.

**`notebooks/4-phase4_ogb.ipynb` added (same day) — the notebook front-end, per user preference over the terminal.** Same shape as nb2/nb3: `DATASET` knob (`ogbl_ddi` | `ogbn_arxiv`), one run per dataset, minimal markdown. Thin cells import the `run_ogb.py` functions (`ensure_virtual`/`embed`/`score`/`winner`) so no logic is duplicated — `run_ogb.py` is now the module, the notebook the orchestrator (CLI `main()` kept as an alternative entry point). Sections: §1 `ensure_ogb` → §2 load+check → §3 five virtual graphs → §4 embeddings (official TRAIN graph, reuse-or-create) → §5 validation scores → §6 selection table + winner → §7 **guarded** single test read (Jupyter cannot enforce cell order, so the guard does) → §8 valid+test tables. Writes only to existing routes (nb2/nb3 zones, `scoreboard.csv`, `graph_health.csv`). Nothing run yet.

**First nb4 run tripped a torch/OGB version clash (same day, fixed).** `PygLinkPropPredDataset("ogbl-ddi")` crashed with `UnpicklingError: Weights only load failed` — PyTorch ≥ 2.6 defaults `torch.load(weights_only=True)`, and OGB loads its processed cache (pickled PyG objects, `DataEdgeAttr` et al.) with plain `torch.load`. Fixed in `make_ogb.py` by `torch.serialization.add_safe_globals(...)` — the surgical allowlist, not a blanket `weights_only=False` (files come from OGB's official host). Two rounds: the processed graph needed the PyG containers (`DataEdgeAttr`, `DataTensorAttr`, `GlobalStorage`); `get_edge_split()`'s `train.pt` then needed the numpy pickle set (`np.core.multiarray._reconstruct`, `np.ndarray`, `np.dtype`, plus every `np.dtypes.*DType` class, added generically so no third round). Download + processing had already succeeded; rerun loads the cache directly.

**Selection lock added (same day, user-requested).** The validation choice is now *persisted*, not just printed. `run_ogb.select(ds)` picks the winner from the `valid_*` scoreboard rows and writes it to **`results/ogb_selection.json`** (one entry per dataset: encoder, graph_variant, K, metric, mean±std, seeds, `locked_at`); it **refuses to run once any `test_*` row exists** for the dataset, so the winner cannot be changed after seeing test. `selection(ds)` reads the lock and is the guard every test read must pass — notebook §7 and the CLI `--final` both call it before scoring. `report(ds)` prints the headline from the **saved** lock (the winner's own test row), never re-selecting from the test table. Old `winner()` removed; notebook §6/§7/§8 now call `select` / `selection` / `report`. Training-data facts enforced by construction and worth restating: **ogbl-ddi** embeddings train on the official training links only (the edgelist contains nothing else); **ogbn-arxiv** uses the full citation graph for structure (transductive), and the only labels that touch learning are the official training papers' (probe fit), embeddings being fully unsupervised.

**Result tables reformatted (2026-07-23, presentation only).** Notebook 4's §5–§8 printed the raw scoreboard slice (one row per encoder × variant, raw ids, `metric` column). Replaced with a `graph × encoder` pivot: `run_ogb.table(ds, split, better=False)` reads the scoreboard, keeps the **primary** metric for that split (`*_acc` / `*_hits@20`; the arxiv F1 secondaries stay out), pivots to one row per variant in `VG_SIMS` order and one column per encoder (GraphSAGE first), and relabels for display via `LABELS` (`psi`→Ψ, `graphsage_edge`→GraphSAGE, …) and `UNIT` (`hits@20`→Hits@20, `acc`→Accuracy). `better=True` (used on validation) appends a **Better encoder** column — per-row argmax, `"tie"` when both encoders score identically, and `" — best overall"` on the single best cell. `select()` now prints Encoder / Graph / Validation <unit> / Std. deviation and one winner line (`Validation winner: Original graph + DeepWalk, Hits@20 = 0.0128 ± 0.0006`). `report()` prints the locked winner, both tables, the arxiv F1 secondaries when present, and a generated conclusion line; when the best test cell is < 0.05 it appends the near-floor caveat (structural-only embeddings + cosine scoring perform poorly), so the ddi numbers are never read as a meaningful ranking. Display layer only — no metric, split, selection rule or stored row changed; the scoreboard keeps the raw ids.

## 2026-07-23 — ogbl-ddi link prediction: trained MLP decoder replaces cosine

**Why.** Diagnosis of the near-zero ddi Hits@20 found two things. (1) A *metric artifact*: 156 nodes of the training graph are structural twins (degree 4, clustering exactly 1/6, eigenvector centrality equal to 9 dp), so a deterministic structural encoder gives them **identical** embeddings; 18-46 of the C(156,2) twin pairs land in the official negative set at cosine **exactly 1.000000**, and OGB's `Hits@K` compares `y_pred_pos > kth_negative` with a strict `>`. The threshold sits on cosine's ceiling, so `hits@20 = 0` by construction no matter how good the model is - confirmed by the same embeddings scoring AUC 0.66-0.76 and `hits@100 = 0.033`, and by `hybrid` (the one variant with no twin ties) being the only nonzero GraphSAGE cell. (2) A *protocol gap*: OGB's own ddi reference model (`examples/linkproppred/ddi/gnn.py`, verified) scores pairs with a trained `LinkPredictor` MLP over the elementwise product of the endpoint embeddings, not with a fixed cosine.

**Change (`eval_ogb.py`).** New `LinkDecoder` class - frozen embeddings in, `hadamard(z_u, z_v) -> Linear(dim,256) -> ReLU -> Linear(256,1)` out, trained with BCE on **training edges only** (positives) against uniformly sampled non-edges (negatives resampled every epoch, self-loops and real edges rejected via a sorted pair-key array). `evaluate_linkpred(..., scorer="mlp"|"cosine", train_edgelist=..., seed=...)`; `mlp` is the default and `run_ogb.score()` passes the training edgelist + the run's seed, so decoder variance enters the 3-seed std. `cosine` is kept so the old rows stay reproducible. Settings (`eval_ogb.DECODER`): hidden 256 and lr 0.005 mirror OGB's reference model; epochs 50, pairs_per_epoch 100k, batch 32,768 are ViRGo runtime caps (only the small MLP trains, embeddings are frozen) - a recorded deviation from OGB's 200 epochs over all 1.07M edges. Roughly 17 s per config on CPU.

**Leakage.** The decoder never sees a validation or test pair: positives come from `input/ogbl_ddi_train.edgelist` (the same edges the embeddings were built from), negatives are sampled non-edges. It is retrained deterministically per (variant, encoder, seed), identically for every graph variant, so the graph remains the only variable. Note the decoder's positives are always the **real training links**, never the virtual role edges - the task being scored is real drug-drug interaction.

**Scope.** OGB link prediction only. The core-four LP protocol (`eval_linkpred.py`, random 70/30, AUC) is untouched, so the locked post-fix results stand.

**Record keeping.** The 20 existing cosine rows for `ogbl_ddi` were **relabelled** in `results/scoreboard.csv` from `valid_hits@20`/`test_hits@20` to `valid_hits@20_cos`/`test_hits@20_cos` - nothing deleted, and `_cos` is outside `run_ogb.PRIMARY` so it no longer appears in the result tables. The `ogbl_ddi` entry in `results/ogb_selection.json` was **voided**: it locked a winner under the retired scorer, so §7 now correctly refuses until §6 re-selects under the decoder. `select()`'s freeze guard was tightened from `metric.startswith("test_")` to the PRIMARY test metrics, so the relabelled `_cos` rows do not block the new selection.

**Effect (seed 42, validation, smoke check before the full sweep):** deepwalk/original 0.0124 -> **0.0857**; graphsage/original 0.0000 -> **0.0411**; graphsage/psi 0.0000 -> **0.0228**. The tie ceiling is gone because MLP logits are unbounded, so no negative pair can sit on an unbeatable maximum. Full re-sweep (valid, select, test) still to run.

## 2026-07-23 — ogbl-ddi rerun under the trained decoder (full sweep, 3 seeds, K=10)

First complete OGB run. Embeddings reused unchanged (encoder untouched); only §5-§8 recomputed.

| graph | valid SAGE | valid DeepWalk | test SAGE | test DeepWalk |
|---|---|---|---|---|
| Ψ | 0.0223 | 0.0071 | 0.0412 | 0.0081 |
| degree | 0.0213 | 0.0071 | 0.0265 | 0.0080 |
| centrality | 0.0309 | 0.0077 | 0.0514 | 0.0100 |
| original | 0.0385 | 0.0772 | 0.0102 | 0.0378 |
| hybrid | 0.0385 | **0.1038** | 0.0117 | **0.0533** |

Locked on validation: **hybrid + DeepWalk, 0.1038 ± 0.0040** → test **0.0533 ± 0.0062** (also the best test cell, so the lock cost nothing here). Selection discipline held: valid swept, winner locked to `results/ogb_selection.json`, test read once.

**Versus the retired cosine scorer** (rows kept as `*_hits@20_cos`): every cell improved by 1-2 orders of magnitude, and the four `0.0000` GraphSAGE cells are gone — the twin-tie ceiling was the whole cause. Cosine's winner was original + DeepWalk (0.0128); the decoder's is hybrid + DeepWalk (0.1038), 8× higher.

**Two things to keep in view before adding arxiv.**
1. *Validation does not predict test on this dataset.* GraphSAGE inverts: original/hybrid lead on validation (0.0385) but collapse on test (0.0102/0.0117), while the role graphs do the opposite (centrality 0.0309 → 0.0514). ogbl-ddi's split is by protein target, so valid and test are deliberately different distributions; the OGB leaderboard shows the same valid/test decorrelation. Not a bug, but it means single-split selection is fragile here.
2. *Conclusions were scorer-dependent.* Under cosine the role graphs looked like noise on ddi; under the decoder GraphSAGE's role graphs beat GraphSAGE's original graph 4× on test. The core-four link-prediction results still use `eval_linkpred.py`'s default `score='cosine'` — a `logreg` mode (hadamard + LogisticRegression) already exists behind a flag, so the same sensitivity is testable there without new code. Worth checking before the study's LP claims are written up.

Notebook 4 is the flow under stabilization; more improvements will be tested on ddi before ogbn-arxiv is added.

## 2026-07-23 — correctness fix #1: GraphSAGE negative sampling rejects real neighbors

**Bug.** `SageEncoder.train()` drew negatives word2vec-style from a `deg^0.75` distribution over all nodes with no check against existing edges, so a sampled "negative" could be an actual neighbor of the center node - the objective then pushed apart a pair the positives pull together. Measured P(sampled negative is a real neighbor of `u`, or `u` itself) on ogbl-ddi: **original 0.320, hybrid 0.314**, role graphs 0.0035. The rate far exceeds the graph's 0.117 density because `deg^0.75` oversamples hubs and hubs are the likely neighbors - the bias roughly triples it. Consistent with the observed test scores, where GraphSAGE was worst exactly on the two dense (poisoned) graphs and best on the three sparse ones.

**Fix.** Negatives are now rejection-sampled: pairs are keyed as `min*N+max` into a sorted array of the graph's edges, colliding draws (and self-pairs) are redrawn, up to 10 rounds - collisions shrink geometrically, so sparse graphs exit on the first round and residual contamination is ~1e-5. Q stays at 5 for every variant, so the variants remain comparable. Exclusion is against **V**, the graph being trained on, not the original graph: the objective is "V-neighbors attract, non-V-neighbors repel", so a role graph's negatives exclude role edges. Deterministic (same generator, sequential draws). Runtime unchanged (ddi: ~2 s/epoch on the original graph, ~0.17 s on the role graphs).

**Scope.** A correctness fix for every dataset, not a ddi knob; on the sparse core four the collision rate is a fraction of a percent, so those results should barely move (to be confirmed later, not now).

**Rerun.** Snapshot of the pre-fix decoder rows saved to `results/snapshots/ogbl_ddi_decoder_prenegfix_2026-07-23.csv` (40 rows: decoder + cosine, valid + test) because `record_score` upserts. The 15 GraphSAGE ddi embeddings were deleted so §4 retrains them; the 15 DeepWalk files are untouched (this fix does not affect them). Validation only this round - `select()` correctly refuses while test rows exist, and the existing `test_hits@20` rows are now stale, belonging to the pre-fix embeddings; they will be cleared at pipeline-freeze time for the final valid → lock → test run.

**Falsifiable prediction for the rerun:** role graphs should barely move (0.35% → 0), original and hybrid should improve clearly. If the role graphs move too, something other than this changed.

**Result of correctness fix #1 (validation only, GraphSAGE, ogbl-ddi).** DeepWalk rows identical, confirming only the encoder changed.

| graph | before | after | delta | delta/std |
|---|---|---|---|---|
| Ψ | 0.0223 | 0.0251 | +0.0028 | 0.2 |
| degree | 0.0213 | 0.0227 | +0.0014 | 0.2 |
| centrality | 0.0309 | 0.0326 | +0.0017 | 0.3 |
| original | 0.0385 | 0.0413 | +0.0028 | 0.2 |
| hybrid | 0.0385 | 0.0554 | +0.0169 | 1.9 |

The prediction was half right: the role graphs stayed inside noise as expected, but **original did not improve** (+0.2 sigma) despite carrying the same 32% contamination as hybrid, so contamination alone does not explain the dense-graph weakness. Only hybrid moved, and 1.9 sigma on three seeds is not decisive. Training did change materially - the encoder loss fell about 0.25 on every variant (4.14 -> 3.89), so the contradictions were real and are gone; the metric simply did not follow. **The fix is kept on correctness grounds**, independent of the score: an objective must not push apart a pair its own positives pull together.

**Blocker surfaced: seed noise now matches the effect size.** Validation std/mean is 34% on original (0.0413 ± 0.0142), 51% on psi (0.0251 ± 0.0127), 40% on degree. With three seeds at that noise level the "which graph wins" comparison - the paper's core claim - is not resolvable on this dataset. This outranks further tuning. Candidate causes to check next: encoder under-convergence (50 epochs, loss still falling at the end), decoder variance stacking on encoder variance, too few seeds.

Snapshot of the pre-fix rows: `results/snapshots/ogbl_ddi_decoder_prenegfix_2026-07-23.csv`. §6 correctly refused (stale test rows from the pre-fix embeddings); §6-§8 stay unrun until the pipeline is frozen.

## 2026-07-23 — density-matched controls added (`original_k`, `random_k`)

**Why.** The original graph's density floats per dataset and even flips direction against the role graphs: avg degree 500.5 vs ~12 on ogbl-ddi (role graphs 39× sparser), but 2.8-3.9 vs 12-20 on cora/citeseer/enzymes/proteins (role graphs 3-5× denser). "Role graph vs original" therefore confounded edge meaning with edge count, in opposite directions across the datasets the study compares.

**Implementation (`virtual_graph.py`).** Two new variants using the *same* union-of-K-per-node construction as the role graphs, so counts are comparable by construction; only the neighbor source differs. `original_k` = K uniformly sampled **real** neighbors per node (semantics without volume). `random_k` = K arbitrary nodes (the null: does any sparse scaffold work as well as a role scaffold?). Both seeded, weight 1.0. Kept in `cfg.VG_CONTROLS`, deliberately **out of `VG_SIMS`**, so the Phase-2/3 study and notebooks 2-3 are unchanged; opt in with `VG_SIMS + VG_CONTROLS`.

**Validation results (ogbl-ddi, decoder scorer, post-negative-fix, 3 seeds):**

| graph | edges | GraphSAGE | DeepWalk |
|---|---|---|---|
| psi | 27,042 | 0.0251 ± 0.0127 | 0.0071 ± 0.0007 |
| degree | 27,109 | 0.0227 ± 0.0090 | 0.0071 ± 0.0011 |
| centrality | 24,886 | 0.0326 ± 0.0051 | 0.0077 ± 0.0009 |
| **original_k** | 39,588 | 0.0371 ± 0.0074 | 0.0303 ± 0.0014 |
| **random_k** | 42,632 | 0.0078 ± 0.0020 | 0.0006 ± 0.0002 |
| original | 1,067,911 | 0.0413 ± 0.0142 | 0.0772 ± 0.0082 |
| hybrid | 1,088,727 | 0.0554 ± 0.0087 | 0.1038 ± 0.0040 |

Both controls came out ~1.5-1.7× **denser** than the role graphs (similar nodes pick each other reciprocally, so the top-K union collapses; random picks rarely collide). That handicaps the role graphs, making every comparison below conservative.

**Reading.**
1. *The role edges are real signal.* GraphSAGE beats the random scaffold 2.9-4.2× (psi/degree/centrality 0.0227-0.0326 vs random_k 0.0078) and DeepWalk beats it ~12×, even though random_k has more edges. First clean evidence in the study that Ψ/degree/centrality edges carry information rather than merely providing something to message-pass over.
2. *GraphSAGE ignores 96% of the original graph.* original_k (39.6k edges) scores 0.0371 against full original's 0.0413 with 27× more edges - a 0.3 sigma difference. The original graph's GraphSAGE advantage was never about volume.
3. *DeepWalk's advantage IS volume.* original 0.0772 → original_k 0.0303, a 2.5× drop. So on ddi the encoder ranking **inverts once density is matched**: on the full graphs DeepWalk leads (0.0772 vs 0.0413), at matched density GraphSAGE leads (0.0371 vs 0.0303). The earlier "DeepWalk beats GraphSAGE on the original graph" reading was substantially a density effect.

The currently locked winner (hybrid + DeepWalk, 0.1038) sits on the full 1.09M-edge graph, so it is in the density-favoured regime.

## 2026-07-23 — stability checks, and the pipeline freeze

Two noise sources measured on ogbl-ddi validation before freezing. Everything at the standard settings (`GNN_PARAMS["epochs"]=50`, K=10, seeds 42/43/44) — no setting was changed for these checks and none was changed because of them.

**Decoder noise.** One saved embedding scored five times with five different decoder seeds (the encoder is held fixed, so this isolates the decoder's own initialisation + negative sampling).

| embedding | 5 decoder seeds | decoder std | total std (3 encoder seeds) | decoder share of variance |
|---|---|---|---|---|
| GraphSAGE / psi | 0.0249 0.0267 0.0258 0.0231 0.0268 | 0.0014 | 0.0127 | ~1% |
| GraphSAGE / original | 0.0276 0.0377 0.0384 0.0475 0.0417 | 0.0065 | 0.0142 | ~21% |
| DeepWalk / psi | 0.0069 0.0078 0.0072 0.0083 0.0091 | 0.0008 | 0.0007 | ~all |

The decoder is not the problem on the role graphs (1% of variance) but is a real contributor on the dense original graph (21%), where it has 1.07M training edges and a harder fit. DeepWalk's total spread is so small that it is entirely decoder noise — the walk encoder itself is near-deterministic across seeds here. **The GraphSAGE encoder is the dominant noise source**, so the seed spread reported for the role graphs is a property of the encoder, not the metric.

**Encoder convergence.** Training loss averaged over 10-epoch windows, single seed, two graphs at opposite densities:

| graph | ep 40-50 | 90-100 | 140-150 | 190-200 | 290-300 |
|---|---|---|---|---|---|
| psi | 4.0191 | 3.9516 | 3.9335 | 3.9190 | 3.8901 |
| original | 3.9005 | 3.8264 | 3.7614 | 3.7270 | — |

The loss is still drifting down after epoch 50 — 0.13 over the next 250 epochs on psi, 0.17 over the next 150 on original — against a total drop of ~38 from initialisation, i.e. the post-50 movement is under half a percent of the range. **Epochs stay at 50 for every graph.** The budget is equal across variants in both senses that matter: same epoch count, and same total gradient samples (50 × `pairs_per_epoch` 100,000 = 5M pairs for every variant regardless of graph size). Recorded as a limitation for the write-up: the encoder is trained to a fixed budget, not to a convergence criterion, and the dense graphs are further from their own plateau than the sparse ones at that budget.

**Pipeline frozen at this point.** No further ddi-driven changes. Frozen settings:

- graph: K=10, variants `VG_SIMS` (psi/degree/centrality/original/hybrid), controls `VG_CONTROLS` (original_k/random_k) available but out of the main sweep
- features: all four structural (degree, eigenvector centrality, Ψ, clustering), computed on the original graph, shared by every variant
- encoder: GraphSAGE mean, 2 layers, 64-dim, lr 0.01, 50 epochs, edge positives, Q=5 negatives with real-neighbour rejection; DeepWalk bridge as the second encoder
- link scorer: trained OGB-style decoder (hadamard → MLP 256), fitted on training edges only, 50 epochs
- protocol: official OGB split + Evaluator, seeds 42/43/44, winner selected on validation, test read once

Next dataset is **ogbn-arxiv**, run with this code unchanged — no per-dataset tuning, per the freeze rule.

## 2026-07-24 — scoring cache (notebook 4 §5/§7)

OGB scoring is deterministic (seeded scorer + reused embeddings), so re-running §5/§7 recomputed identical numbers — ~15 min per pass on ddi (decoder refit ×10) for no new information. Added `run_ogb.cached(ds, enc, sim, k, split)`: returns the saved scoreboard mean(s) if the config's primary `{split}` rows exist, else `None`. Cells §5/§7 now skip `score()`+`record_score` on a hit and print `reuse …`. Reuse is valid only while embeddings + scorer are unchanged; changing either means clearing the affected scoreboard rows first (same convention as deleting `.emb` files to force a retrain). Selection lock, guards, metrics, values all unchanged — verified reused means match the scoreboard byte-for-byte.

## 2026-07-24 — the core four re-run under the frozen pipeline (`run_core.py`)

**What was stale.** The core-four GraphSAGE embeddings on disk dated 2026-07-20/21, i.e. before the negative-sampling change made while validating ogbl-ddi (`SageEncoder.train` now rejects a negative that is a real neighbour of the centre node, or the node itself, and redraws). The OGB embeddings were produced after it (verified below, not assumed — the change sat uncommitted in the working tree from 07-22/23 and only entered git in the last commit, so commit dates prove nothing here). So the six-dataset table mixed two encoder versions.

**Audit first (nothing else was wrong).** Re-scoring every stored core-four embedding with the current eval code reproduced its scoreboard row exactly: 80 cells, 240/240 embeddings present, max |delta| 0.00000. The scoreboard was self-consistent — only the code that produced the embeddings had moved.

**The change is not dense-graph-only.** Measured collision rate of the deg^0.75 draws at epoch 0: cora/psi 0.552% (921/166,960), proteins/degree 0.055% (275/500,000). Small, but non-zero means the redraw fires, the generator stream diverges, and every later epoch differs. Sparse graphs were therefore affected too, just mildly.

**Re-run.** All 120 core-four GraphSAGE embeddings (4 datasets x 5 variants x 3 seeds x 2 tasks) retrained under the frozen configuration; the pre-freeze files were first moved to `output/superseded_pre_freeze_2026-07-24/` and, once the replacements were verified complete and scored, **deleted** (1.2 GB, user decision 2026-07-24 — they were produced by a sampler that treated real neighbours as negatives, so they are wrong rather than merely old; recoverable in principle since the pre-rejection encoder is still in git at `f8f502a`). DeepWalk embeddings were untouched (that code path has not changed) and their re-scored rows came back byte-identical, which is the control on the re-run itself. Result: 37 of 40 GraphSAGE rows moved, max |delta| 0.0054 (cora hybrid LP), mean 0.0011, and no cell changed its winning graph variant.

**New script `run_core.py`.** The headless equivalent of notebook 3 §8 for cora/citeseer_linqs/enzymes/proteins, mirroring `run_ogb.py`: reuse-or-create embeddings, same output zones, same scoreboard rows, defaults = the locked settings. `--train-only` writes embeddings and records nothing, so variants can train in parallel without racing on `results/scoreboard.csv`; a later plain run scores everything in one process. The whole study is now two commands (`run_core.py`, `run_ogb.py --dataset ...`) instead of a notebook pass. Feature caching (`encoder.feature_cache`) is now used on the core four as well; verified a numerical no-op — with the cache warm, five variants share one feature computation per graph.

**Reproducibility floor, measured.** Five repeats of one configuration (cora/psi/seed 42) under the current code gave weighted F1 0.2381 every time, with embeddings differing by at most 2.15e-6 — float32 aggregation order in the SAGE scatter, present even at `torch.set_num_threads(1)`. The header comment in `encoder.py` claiming bit-reproducibility is therefore too strong: the metric is exact, the tensors are not. The stored 07-20 embedding scored 0.2376 on the same evaluation, i.e. that gap is the code change, not noise.

**Verification of which code produced which file (the claim above, tested).** Dates were not trusted; each stored embedding was reproduced from source. The pre-rejection `encoder.py` was taken straight out of git (`f8f502a`) into a scratch directory and run — the repository was not modified.

| stored file | reproduced by pre-rejection code | reproduced by current code |
|---|---|---|
| cora/psi/s42 (07-20) | **1.91e-6 = noise floor**, F1 0.2376 identical | 3.43e-1, F1 0.2381 |
| ogbl_ddi psi / original / centrality / hybrid (07-23) | 5.8e-2 … 1.5e+0 | **1.0e-6 … 2.0e-5 = noise floor** |
| ogbn_arxiv psi, centrality (07-23) | 3.6e+0 | 2.7e+0 — reproduced by neither |

So: core four were pre-rejection (proved, not inferred), ogbl-ddi was already on the frozen encoder, and **ogbn-arxiv reproduces neither exactly**.

**The arxiv anomaly, characterised.** Ruled out by measurement: cached features are byte-identical to a fresh recomputation (all four columns, 0.0), node order and node count identical (169,343), `GNN_PARAMS` unchanged since 2026-07-07, the virtual-graph file predates the embeddings. What remains is perturbation amplification, which only this graph is large enough to show. Distances on arxiv/psi (mean |diff| over 169,343 x 64):

| comparison | mean \|diff\| | max \|diff\| |
|---|---|---|
| same config, repeated run | 1.46e-7 | 4.48e-5 |
| 1 thread vs 32 threads | 7.61e-6 | 1.18e-2 |
| stored vs fresh (current code) | 3.07e-3 | 2.69e+0 |
| stored vs fresh (pre-rejection code) | 3.69e-3 | 3.64e+0 |
| different seed = independent trajectory | 1.01e-1 | 4.78e+1 |

The stored file sits 33x below the independent-trajectory scale but four orders above a same-config repeat, and thread count alone moves arxiv 50x more than it moves cora. **The metrics are unaffected**: seed 42 psi scores valid/test accuracy 0.2627/0.2482 stored, 0.2620/0.2474 fresh, 0.2631/0.2480 single-threaded, 0.2619/0.2476 pre-rejection — a spread of 0.0012, inside the 3-seed std of 0.0020. Recorded as a limitation: on a graph this size the pipeline is reproducible at the reported metric precision, not at tensor level, and the write-up should claim only the former.

**Decisions taken 2026-07-24 (deliberate, not pending work).**
- **Ablation-D rows are NOT re-run.** `graphsage_edge_feat_*` and `features_only` keep their pre-freeze numbers. They were tested, verified and frozen when the configuration was chosen; they are evidence *for* the locked configuration, not results *under* it. `features_only` (D6, layers=0) never trains and is unaffected regardless. Any ablation table in the paper states this.
- **No density-matched controls on the core four.** `original_k`/`random_k` stay ogbl-ddi-only, where the original graph is denser than K. On the core four the matching would have to run the other way (role graphs rebuilt at K≈2), which is a different experiment and out of scope.
- **The link-prediction scorers differ by protocol and stay that way.** Core four: random 70/30 split, unsupervised cosine ranking, AUC. OGB: official split, trained hadamard→MLP decoder, Hits@20. Comparisons are made *within* a dataset, so the scorer cancels; no table places the two in one column.

## 2026-07-24 (later) — ablation D completed under the corrected encoder

**Code (4 places, all additive).** `encoder.py`: the feature-column map gained `"centrality": [1]` and `"clustering": [3]` (cached column order is deg, Ω, Ψ, clustering) and `--features` gained the two choices. `scripts/benchmark_config.py`: `D_FEATURES` gained D7 centrality and D8 clustering, appended so existing D-numbers keep their meaning. `run_core.py`: a `--features` passthrough plus a two-line `tag()` helper, so an ablation arm writes `graphsage_edge_feat_<set>_s<seed>.emb` and records under the same id — the same convention `encoder.main` already used; the walk encoder is skipped automatically when features vary, since DeepWalk has no inputs to ablate. `notebooks/3-phase3_gnn_encoder.ipynb` §8b: `D_SETS`/`D_DIMS` extended to match.

**Pre-flight check.** Each single-feature arm was verified to select exactly the z-normalised raw column it claims before any sweep ran (degree→0, centrality→1, Ψ→2, clustering→3, all four dims correct).

**Runs.** 168 embeddings = 7 trained arms x 4 datasets x 2 tasks x 3 seeds, psi graph, K=10, trained in three parallel waves then scored in one process. D0 = the locked `graphsage_edge` row (already post-fix, not re-run); D6 `features_only` has `layers=0` and never trains — verified byte-identical under the current code on cora and, once the Ψ instability below was accounted for, on proteins too. The 120 pre-freeze ablation embeddings were moved aside, replaced, verified, then deleted (1.2 GB).

**Effect of the re-run:** 40 re-run rows moved by at most 0.0209 (proteins D5 const LP, chance by construction), mean 0.0016; no ablation conclusion changed. Scoreboard now 228 rows (212 + 16 new D7/D8 rows), every row 3-seed at K=10.

**Why this reversed the earlier freeze decision.** Extending the table with two new arms means the new arms come from corrected code while the old arms did not; freezing five of nine rows inside a table whose purpose is comparing arms to each other would have reintroduced exactly the confound removed from the main table. Re-running all of them cost about half an hour.

## 2026-07-24 (later) — Ψ instability traced (found by the D6 verification)

`features_only` reproduced byte-identically on cora but not on proteins, and only in the Ψ column. Cause chain: 20 of proteins' 1,195 components fail power iteration, so Ω comes from the `eigenvector_centrality_numpy` (ARPACK) fallback, which starts from a random vector; Ω then differs by ~7e-12 between runs; Ψ divides by Ω inside a logarithm, so where Ω is ~1e-12 that perturbation becomes a Ψ shift of up to 52 (Ψ range −321 to −1). 336 of 43,466 proteins nodes move; enzymes moves 1 node by 1e-8; cora is exact.

Downstream: two psi-graph builds on proteins share 33.6% of edges (cora/enzymes: 100%). Node classification is unaffected (the psi graph is read from disk); link prediction rebuilds it per run, and two full rebuild+retrain passes scored 0.5614 / 0.5619 against the recorded 0.5617, inside the 0.0042 seed std.

`run_core.py` / `run_ogb.py` pin Ψ through the feature cache, so all current results share one Ψ. **Notebook 3 does not pass the cache** and would recompute it — a known gap, not fixed here. Candidate fixes when it is addressed: fixed `v0` for the ARPACK fallback, an Ω floor before the log, or making the notebook use the cache.

## 2026-07-24 (later) — ablation-D review page rewritten and republished

The shared review artifact (`claude.ai/code/artifact/c6dd674b-8db1-4cf4-b340-929588bdd815`) previously covered ablations A-E on two datasets and carried the pre-fix ablation-D table with one arm missing. It was **updated in place** — same URL, so any existing link keeps working — and narrowed to **ablation D only**, at a general-reader level of detail: two pages, plain-language explanations, no repo internals.

**Contents now:** the D question in one paragraph; the four structural inputs defined in ordinary words; the current 10-row x 4-dataset tables for node classification and link prediction; three findings (clustering is the molecular feature and random-level on citation graphs; centrality is the mirror image and below chance on citation link prediction; two features beat four on three of eight cells); the three carry-over facts (real features are necessary, message passing earns its place on NC but not molecular LP, Ψ stays weakest despite its confound); and the explicit warning that citation link prediction is a null band (0.42-0.55) where ranking arms is meaningless. Scope caveat kept on the page: Ψ graph, K=10, core four only, no OGB ablation rows, nothing per-dataset tuned.

**Presentation unchanged** — same token palette, Georgia headings, table and callout styling, and light/dark handling as the original page. Two additions in the same visual language: a filled cell marks the best arm per column, a left bar marks the two new arms. Numbers on the page are the post-fix scoreboard values, so the page and `results/scoreboard.csv` agree row for row.

Source file lives in the session scratchpad, not the repo — the page is a communication artifact, and `results/scoreboard.csv` plus `docs/paper_log.md` remain the record of truth. Republish by passing that same URL, never by publishing fresh, or the link changes.

## 2026-07-24 (later) — `characterize.py`: characterization portion 1

**New file `characterize.py`** (root, mirrors `run_core.py` / `run_ogb.py`: argparse, `main(args)`, few functions). Five functions — `graph_properties` (Family 1), `feature_properties` (Family 2 step 3), `frozen_inputs` (step 1), plus `load_cache` / `labels_of` / `gini` / `homophily` helpers. `--datasets` selects a subset; the default is all six. Pure measurement: no training, no graph building, nothing written outside `results/`.

**Reuse over reimplementation.** `graph_io.load_graph` supplies the graph (so this cannot disagree with any other stage about what the graph is) and `graph_io.properties` supplies nodes / edges / components / largest-component fraction / isolates / distinct degrees. Average clustering is read from the encoder's **feature cache** rather than recomputed — it is the same `nx.clustering` output the pipeline already stored, so it is exact and free, and it avoids an O(N·d²) recount on the 169k-node graph. Only density, degree spread (std / skew / Gini), assortativity and homophily are computed here, because nothing else in the repo computed them.

**Family 2 reads the cache, deliberately.** The cache holds the raw pre-z-normalization matrix `[degree, Ω, Ψ, clustering]` that every embedding in the study was trained on. Describing the cache therefore describes the true inputs, and it side-steps the Ψ/Ω recomputation instability recorded above — a fresh recomputation on proteins would have produced different Ψ values from the ones the results were produced with.

**Outputs (all in `results/`):** `dataset_characterization.csv` (6 rows x 22 columns), `node_feature_characterization.csv` (24 rows = 6 datasets x 4 features), `characterization_inputs.csv` (50 rows = 10 dataset x task cells x 5 variants, GraphSAGE, K=10, one primary metric per task).

**Validation gate — all passed.**

| check | result |
|---|---|
| row counts | 6 / 24 / 50 as designed |
| nodes, edges, max degree, components, avg degree vs `graph_health.csv` `original` rows | exact match on all six datasets |
| cached degree column vs a fresh `nx` degree recount | element-wise identical on cora, enzymes, arxiv, ddi |
| avg clustering read from cache vs `nx.average_clustering` | 0.2407 / 0.1447 — identical |
| ogbl-ddi label columns | NaN, `graph_scope=ogb_train_edges`, absent from step-1 NC rows |
| re-run determinism | all three CSVs byte-identical |
| edge homophily vs published values | cora 0.8100 (~0.81), citeseer 0.7377 (~0.736), arxiv 0.6542 (~0.654) |

Runtime 17 s for all six datasets.

**Two things to carry into portion 2.** (1) The `degenerate` boolean fires on molecular clustering — few distinct values because low-degree nodes admit only a few rational clustering coefficients — even though ablation D shows it is the most useful single feature there; portion 2 uses the raw columns, not the flag. (2) Family 2 currently describes the **full** graph, which is exactly what the node-classification runs consumed; the link-prediction runs consume the per-seed 70% training subgraphs, whose caches also exist on disk. Adding those rows is a scope switch, not new machinery, and is deferred until portion 2 needs it.

## 2026-07-24 (later still) — `characterize.py`: adjusted homophily column

**Change.** `homophily()` now returns four values — edge, node, **adjusted** homophily and the chance **null** — and `graph_properties` writes `homophily_null` + `homophily_adjusted` to `dataset_characterization.csv`. Three-line edit, confined to `characterize.py`; no other file touched, and no scoreboard / embedding / split changed. Re-ran: only the three characterization CSVs rewrote, all byte-stable.

**Why.** Raw edge homophily's chance floor is `Σ_c (D_c/2m)²`, which depends on class count and balance, so raw values are not comparable across datasets — enzymes (0.665, 3 classes) and arxiv (0.654, 40 classes) look identical raw but are opposite graphs. Adjusted `(h − null)/(1 − null)` removes that floor.

**Null is degree-weighted, not Σp².** Edge homophily counts edges, and edges are weighted by degree, so the null uses summed per-class degree D_c, not the node-count class fraction. They agree only on degree-regular graphs; on arxiv (Gini 0.63) they diverge — degree-weighted 0.588 vs node-count ~0.625.

| dataset | raw | null | adjusted |
|---|---|---|---|
| cora | 0.8100 | 0.1698 | 0.7711 |
| citeseer_linqs | 0.7377 | 0.1975 | 0.6731 |
| ogbn-arxiv | 0.6542 | 0.1614 | 0.5877 |
| enzymes | 0.6653 | 0.4759 | 0.3613 |
| proteins | 0.6568 | 0.4678 | 0.3552 |

Portion 2 correlates on `homophily_adjusted`; raw kept for provenance.

## 2026-07-25 — `characterize.py` portion 2 + `notebooks/5-phase5_characterization.ipynb`

**`characterize.py` gains portion 2** — four functions and a helper, portion 1 untouched: `gaps` (original vs best augmented per cell, with a 1σ seed-noise band → verdict), `feature_scores` (ablation D lift over the D4 random control, joined to the Family 2 raw spread), `correlate` (Spearman, two scopes), `rule` (prints the reading), plus `rho` shared by both scopes. New `--step measure|relate|all` (default `all`); `--step measure` reproduces the old portion-1-only behaviour.

**Three new outputs** in `results/`: `characterization_gaps.csv` (10), `feature_usefulness.csv` (74), `characterization_correlations.csv` (48). `cfg.FIGURES_DIR` added (`results/figures/`) — one line in `benchmark_config.py`, no behaviour change.

**`notebooks/5-phase5_characterization.ipynb`** — 16 cells (8 markdown, 8 code), follows the notebook-4 split: `characterize.py` does the work, the notebook orchestrates and plots. Five figures saved to `results/figures/`: variant-vs-original heatmap, gap-vs-predictor scatter, property correlation heatmap, ablation-D lift bars, spread-vs-usefulness scatter. Executed end to end on the `i2v` kernel — 0 errors, all 5 figures embedded.

**Plotting decisions worth keeping.** ogbl-ddi's relative gap is **+200%** because its original score is 0.0173 — a floor effect, not a 3× improvement; the LP panel uses a `symlog` y-axis so the five sparse datasets stay legible. Spearman is rank-based, so the floor effect does not touch ρ. The spread-vs-usefulness panels plot only **unit-free** statistics (%zero, %unique, skew) — `raw_std` is in each feature's own units and is not comparable across features.

**Scope of the ablation join.** Ablation D exists only on the `psi` graph, K=10, core four (both tasks, 3 seeds). `feature_scores` therefore returns 74 rows, not the full six datasets; arxiv and ddi have no feature-ablation arms by the settled scope decision.

Nothing was retrained. Scoreboard, embeddings and splits are unchanged.

## 2026-07-26 — Notebook 5 reorganized around the research question (presentation only)

The notebook was structured by the *internals* of `characterize.py` (§1 "build the tables", variables `fam1` / `fam2` / `inp` / `corr`), which reads as a tour of the code rather than an answer to a question. Rebuilt as a seven-section argument, user-requested: **1** the research question (which of the five graphs should GraphSAGE use, per dataset × task) · **2** what data already exists (a provenance table: scoreboard, edgelists, labels, feature caches — nothing retrained) · **3** which graph wins (new `winner` + `verdict` table beside the heatmap) · **4** what each dataset is like (new table, the five explaining properties only) · **5** why a graph won or lost (the two predictor panels + the property heatmap as evidence) · **6** which node feature helps (ablation lift + the spread-vs-usefulness test) · **7** a new final recommendation table.

**Variables renamed to what they hold**, since the old names were the thing being complained about: `fam1`→`props`, `fam2`→`dists`, `inp`→`scores`, `feat`→`ablation`; `gap` and `corr` kept. `characterize.py` is untouched — the notebook still calls the same six functions in the same order.

**Two new tables, both derived from existing frames — no new measurement.** §3 joins the variant pivot to `gaps()` and adds `winner = piv.idxmax(axis=1)`. §7 joins `gaps()` to the best single-feature arm per cell (`arm`, `lift_vs_random`) and prints `verdict` / `use_graph` / `best_variant` / `useful_feature` / `evidence`; `use_graph` is `original` for both *keep original* and *tie* (the original is the cheaper default), the augmented variant only on *augment*. `useful_feature` is **`not measured`** for ogbn-arxiv and ogbl-ddi — ablation D is core-four-only, so the column must not read as "no feature helped".

**Unchanged on purpose:** all five figure filenames (`fig1`–`fig5`), so `results/figures/` and every doc reference stay valid; every plotting body; the `show()` helper; the kernelspec (`Python (i2v)`, `language_info` restored — the working tree had dropped it). 18 cells (7 md, 11 code), up from 16.

**State:** outputs are cleared and `execution_count` is null throughout — the notebook needs one top-to-bottom run on the `i2v` kernel to re-embed the tables and the five figures. All code cells were syntax-checked (`ast.parse`); §3/§4/§7 cells have not been executed. Nothing was retrained; scoreboard, embeddings, splits and CSVs are unchanged.

## 2026-07-27 — roman_empire + tolokers wired into the frozen pipeline

Research rationale, the panel gap they fill, and the pre-registered prediction are in `docs/paper_log.md`. This entry
is the plumbing only.

- **`make_hetero.py`** (new, 3 functions) — converter mirroring `make_ogb.py`. `make_hetero(name)` pulls the PyG copy
  (`HeterophilousGraphDataset`, cache `output/hetero_raw/`) and writes `input/<name>.edgelist` + `.nodes` sidecar +
  `labels/<name>.labels`. Reuses `make_ogb._write_edges` / `_write_nodes` rather than duplicating them — that import
  also installs the `torch.load` safe-globals allowlist PyG ≥ 2.6 needs. `data.x` and the ten official train/val/test
  masks are read and discarded. `HETERO` maps the two ViRGo names to PyG's; the other three benchmarks in that class
  (amazon_ratings, minesweeper, questions) are one dict entry away if a third dataset is chosen.
- **`scripts/benchmark_config.DATASETS`** — registered `roman_empire` (`directed_source: True` — word order and
  dependency arcs are a directed relation, symmetrized upstream by PyG's `to_undirected`) and `tolokers`
  (symmetric by construction). **Neither carries an `eval` key**, which is what routes them through the core
  protocol instead of the OGB one; no `split`/`pairs` entries exist for them.
- **`run_core.CORE`** — now six datasets. Docstring/`--datasets` help updated ("all four" → "all six").
- **`run_core.embed`** — the node-classification branch called `nx.read_weighted_edgelist(...)` directly and
  **crashed if the virtual graph had not been hand-built in notebook 2**. Replaced with `run_ogb.ensure_virtual(G, ds, k, sim)`,
  which reuses an existing build or creates it and appends a `results/graph_health.csv` row. `import run_ogb` added
  (no cycle: run_ogb does not import run_core). Strictly additive for the four existing datasets — their virtual
  graphs are already on disk, so the same file is read as before, node-set restore included. Removes the only manual
  step between registering a dataset and running it.
- **`characterize.STUDY`** — two rows added (`linguistic`, `crowdsourcing`; both `NC+LP`, scope `full`). Stale
  "n is 4-5 datasets" comments in the module header, `correlate()` and the `rule()` printout replaced with
  "one point per dataset" so they stop drifting as the panel grows. `--datasets` help "six" → "eight".

**Build output (verbatim, matches the published counts):**

```
roman_empire: nodes=22662 edges=32927 classes=18 avg_degree=2.91 | data.x (300-dim) IGNORED | official masks IGNORED
tolokers:     nodes=11758 edges=519000 classes=2 avg_degree=88.28 | data.x (10-dim) IGNORED | official masks IGNORED
```

**`graph_io.check` flags to carry forward:**

- `roman_empire` — DIRECTED RELATION (declared, expected) and **COARSE DEGREE SIGNATURE: 11 distinct degrees over
  22,662 nodes** (max degree 14). The `degree` virtual-graph variant on this dataset is therefore sampled inside tie
  classes averaging ~2,000 nodes, i.e. close to a random scaffold; `psi` and `centrality` are continuous and
  unaffected. Report this as a property of the dataset, not as a bug — it is the same mechanism as the 2026-07-18
  tie-break fix, operating within its intended (post-fix) behaviour.
- `tolokers` — no flags: connected, no isolates, 754 distinct degrees.

**State.** Nothing trained or scored. `results/scoreboard.csv` unchanged (229 rows), no virtual graphs built yet for
either dataset, feature caches not yet written — so `characterize.py` will fail on these two until the encoder has
run once and cached their features.

## 2026-07-27 — questions wired into the frozen pipeline

Third heterophilous benchmark, same four registration points as roman_empire + tolokers, pipeline unchanged.

- `make_hetero.py` — `HETERO` gains `"questions": "Questions"`. No other change; the existing converter handles it.
- `scripts/benchmark_config.DATASETS` — `questions` entry, `directed_source: True` (answered-your-question is a
  directed relation, symmetrized upstream by PyG's `to_undirected`). No `"eval"` key, so it routes through the ViRGo
  core protocol exactly like the other six non-OGB datasets. 301-dim fastText features and the ten official masks
  are not loaded.
- `run_core.CORE` — appended; `--datasets` help text six → seven.
- `characterize.STUDY` — appended, `domain: "q&a interaction"`, `tasks: "NC+LP"`, `scope: "full"`.

Build output, verbatim:

    questions: nodes=48921 edges=153540 classes=2 avg_degree=6.28 | data.x (301-dim) IGNORED (structural-only) | official masks IGNORED (ViRGo core protocol)
    WROTE input/questions.edgelist + .nodes + labels/questions.labels

Measured properties match Platonov's published table exactly (adjusted homophily 0.0207 vs 0.02, avg degree 6.28,
avg clustering 0.0307), so the loader and `homophily()` are validated against a third external source.

Notebook 5 needs no edit: its dataset list is derived from `characterize.STUDY` minus the `EXCLUDE` list, so
`questions` appears automatically on the next run. After the sweep finishes, re-run `characterize.py --step all`
and then the notebook.

Carry-forward flags from `graph_io.check`: one connected component, degree median 1 against max 1,539, and a
97.0/3.0 class split — the most imbalanced label distribution in the panel.

---

## 2026-07-29 — repo restructured into `virgo/` (library) + `experiments/` (entry points)

Pure reorganization: **no method, hyperparameter, seed, path or output-zone changed**. 18 loose `.py` files in the
project root plus a `scripts/` folder were replaced by two code folders under one rule — **`virgo/` is imported,
`experiments/` is run.** Motivation: the root mixed five roles (config, graph building, encoders, evaluation, drivers)
at one level, and adding GIN would have meant a second class inside `encoder.py`.

Move map (all via `git mv`, so `git log --follow` still works):

    graph_io.py                -> virgo/graph_io.py
    identity2vec.py            -> virgo/identity2vec.py
    identity2vec_cached.py     -> virgo/identity2vec_cached.py
    virtual_graph.py           -> virgo/virtual_graph.py
    scripts/benchmark_config.py-> virgo/config.py
    scripts/utils.py           -> virgo/utils.py
    encoder.py                 -> virgo/encoders/base.py (+ sage.py; CLI -> experiments/train_encoder.py)
    embedding_models.py        -> virgo/encoders/walk.py
    make_labels/ogb/hetero.py  -> virgo/data/
    prepare_linkpred.py        -> virgo/data/prepare_linkpred.py
    eval_nodeclass.py          -> virgo/eval/nodeclass.py
    eval_linkpred.py           -> virgo/eval/linkpred.py
    eval_ogb.py                -> virgo/eval/ogb.py
    scripts/runner.py          -> virgo/eval/runner.py
    scripts/results_io.py      -> virgo/eval/results_io.py
    run_core/run_ogb/characterize/train/plot_emb.py -> experiments/
    scripts/benchmark_baselines.py -> experiments/benchmark_baselines.py
    scripts/main.py            -> experiments/run_task.py
    baselines/struc2vec/       -> third_party/struc2vec/

**The encoder split (the point of the exercise).** `encoder.SageEncoder` became `base.GNNEncoder` (features, corpus,
loss, `train()`, `save()` — architecture-free) plus `sage.SageEncoder`, whose entire body is one `build_convs()`.
`gin.GinEncoder` was added the same way and registered in `virgo/encoders/ENCODERS`. `run_core.py` / `run_ogb.py`
now dispatch on `encoder in ENCODERS` instead of `encoder == "graphsage_edge"`, so **adding an encoder needs one
new file plus one registry line and no driver edit**. `--encoder all` still means the locked `graphsage_edge +
deepwalk` pair, so a registered-but-unvalidated encoder can never join a sweep unless named on the command line.
`gin.py` is a **scaffold: no ViRGo results have been produced with it.** Its choices (2-layer MLP per conv,
`train_eps=True`, sum aggregation) follow Xu et al. 2019 and are open for review before the GIN run.

**Reproducibility check (the thing that mattered).** Old `encoder.py` (from `git show HEAD:encoder.py`) and the new
`base`+`sage` pair were run in separate processes on proteins_nr, `sim=degree`, K=10, 5 epochs, seed 42, features
pinned via the shared cache. Result: `max|Δ| = 4.8e-07`, against a same-code repeat floor of `5.4e-07` measured in
the same session. **The refactor is numerically indistinguishable from re-running the old code**, so every stored
embedding, `.emb` file and scoreboard row remains valid. (First attempt ran both classes in one process and showed
`3.1e-01` — that was the known ARPACK random-start instability in Ω, not the refactor; pinning the feature cache
removed it.)

Two behaviour changes, both deliberate:
- `feature_cache()` now returns an absolute path anchored to `config.OUTPUT_DIR` instead of the cwd-relative
  `"output/feature_cache/..."`. Same file when run from the repo root; no longer silently recomputes when not.
- `experiments/run_task.py --help` crashed with `ValueError: unsupported format character 't'` — a pre-existing
  unescaped `%` in the `--retrain` help string (`70%` -> `70%%`). Fixed; the flag itself never changed.

Verification performed: all 28 modules import; every CLI answers `--help`; all five notebook setup cells execute;
notebook JSON round-trips byte-exact so the only diff is the import lines. Command changes for the docs:
`python virtual_graph.py` -> `python -m virgo.virtual_graph`, `python encoder.py` ->
`python experiments/train_encoder.py`, `python scripts/main.py` -> `python experiments/run_task.py`, and the other
root scripts gained an `experiments/` prefix. `docs/notes.md` and `docs/paper_log.md` keep their original command
strings — they are historical records of what was run at the time, not current instructions.

## 2026-07-31 — notebook 5: Table 2 gains a screening view (presentation only)

`notebooks/5-phase5_characterization.ipynb` §5 now has three parts instead of two. The new **Part B**
(`codet2bscreen`) tabulates all seven `PREDICTORS_PRIMARY` on link prediction — impact direction, `spearman_rho`,
`fixed_gap_rho`, `lodo_sign_stable`, `loo_correct_folds`, `majority_baseline`, `n_exceptions`, `canonical`,
`passes_gates` — with a `selection_reason` generated from the first gate each property fails, so the two retained
rules and the five dropped ones are read off one table. Unfiltered: no `|rho| >= 0.7` or `credible` mask.
The previous detail table (`codet2b`) is renamed **Part C**; `mdt2` and `mda5` follow the rename.

`experiments/characterize.py` untouched — every value is read from `results/candidate_rules.csv`. No score,
threshold, rule, dataset, seed, metric or CSV changed; `git status` shows the notebook as the only modified file.

## 2026-08-04 — lastfm_asia added as a Module-3 held-out dataset

Run settings (nothing tuned; the pipeline is the frozen one, exactly as when the other held-out graphs joined):
`python -m virgo.data.make_pyg --dataset lastfm_asia` -> `input/lastfm_asia.edgelist` + `.nodes` +
`labels/lastfm_asia.labels`; `python experiments/predict_module3.py --datasets lastfm_asia` (BEFORE training);
then `python experiments/run_core.py --datasets lastfm_asia --task <t> --sim <s>` over both tasks and all five
variants, encoders `graphsage_edge` + `deepwalk`, K=10, seeds 42 43 44; `python experiments/score_module3.py`.

**Recorded deviation — raw source.** `torch_geometric.datasets.LastFMAsia` downloads from `graphmining.ai`, which is
unreachable from this environment (TLS `unexpected eof while reading`; forcing the handshake returns `404`). The raw
graph is read from SNAP instead (`https://snap.stanford.edu/data/lastfm_asia.zip`) — the same publisher's archive that
PyG repackages — and converted by the shared `_write_edges` / `_write_nodes` writers, so the edgelist format is
identical to every other dataset. The archive's inner folder is misspelled `lasftm_asia` at the source; the path
literal in `_lastfm_asia()` matches it deliberately. Node features are not loaded (structural-only), so the 128-dim
matrix PyG would have supplied is not needed. Counts match the published statistics: 7,624 nodes, 27,806 undirected
edges, 18 classes.

`make_pyg.py --dataset` default renamed `both` -> `all` (it had covered three datasets since 2026-08-02, four now);
the per-dataset choices are unchanged, and no other module calls it.

## 2026-08-05 — amazon_ratings added as a Module-3 held-out dataset

Platonov et al. (2023) heterophilous benchmark, built by the existing converter — no new loader needed, the class
already ships it. **Not** `amazon_photo`: different graph, different paper, different homophily (0.1402 vs 0.7850).
Both live in `cfg.DATASETS` under distinct names.

Commands, in this order (the order is the experiment — predict must precede train):

```
python -m virgo.data.make_hetero --dataset amazon_ratings
python experiments/predict_module3.py --datasets amazon_ratings          # write-once freeze
python experiments/run_core.py --datasets amazon_ratings --task <t> --sim <s>   # 10 chunks: 2 tasks x 5 variants
python experiments/score_module3.py
```

Build output: `nodes=24492 edges=93050 classes=5 avg_degree=7.60 | data.x (300-dim) IGNORED (structural-only) |
official masks IGNORED (ViRGo core protocol)` — matches the published statistics exactly.

Edits (registry only, pipeline untouched): `virgo/data/make_hetero.py` `HETERO` +1 entry, `virgo/config.py`
`DATASETS`, `experiments/characterize.py` `STUDY`, `virgo/frozen_rules.py` `HELDOUT`, `experiments/run_core.py`
`HELDOUT` (opt-in by name, not in the default sweep).

- **CLI rename:** `make_hetero.py --dataset both` → `--dataset all`. `both` had been wrong since `questions` joined and
  is now plainly wrong at five datasets. Named datasets (`--dataset questions`) are unaffected; nothing scripted
  called `both`.
- Ran as 10 foreground chunks of one (task, variant) each, ~4m40s per chunk — a single full sweep would exceed the
  600 s tool timeout. `run_core.py` reuses embeddings already on disk, so chunking is equivalent to one run.
- The `--sim original` NC chunk was the first run and its `graphsage_edge` score line scrolled past a `tail`; the value
  (0.2875 ± 0.0029) was read back from `results/scoreboard.csv`, which every chunk writes.

## 2026-08-05 — squirrel_filtered added as a Module-3 held-out dataset

Platonov et al.'s **de-duplicated** Squirrel. The original `WikipediaNetwork("squirrel")` has repeated nodes that leak
between train and test and is deliberately NOT used.

`HeterophilousGraphDataset` exposes only five names and squirrel_filtered is not one of them, so `make_hetero.py`
gained `_npz(name)`: a direct download of `<REPO>/<name>.npz` from
`https://github.com/yandex-research/heterophilous-graphs/raw/main/data` — **the same URL PyG's own class downloads
from** (visible in its own progress output), so the file, format and data path are identical. `HETERO` now maps a name
to either a PyG dataset name or `None`, and `None` routes to `_npz`. `chameleon_filtered` would be a one-word addition.

Commands, in order (predict must precede train):

```
python -m virgo.data.make_hetero --dataset squirrel_filtered
python experiments/predict_module3.py --datasets squirrel_filtered      # write-once freeze
python experiments/run_core.py --datasets squirrel_filtered --task node_classification
python experiments/run_core.py --datasets squirrel_filtered --task link_prediction
python experiments/score_module3.py
```

Build output: `nodes=2223 edges=46998 classes=5 avg_degree=42.28 | data.x (2089-dim) IGNORED (structural-only) |
official masks IGNORED (ViRGo core protocol)`.

- **Edge count is 46,998 undirected, NOT 23,499.** The `.npz` `edges` array has 46,998 rows and all 46,998 sorted pairs
  are distinct — no reciprocal storage, so it must not be halved. Verified the convention by checking our existing
  builds against the paper's table: roman_empire 32,927, tolokers 519,000, amazon_ratings 93,050 all match exactly,
  so Platonov reports undirected counts. (Contrast `lastfm_asia`, where PyG's advertised 55,612 *was* doubled — the
  direction of the correction is dataset-specific, always count distinct sorted pairs.)
- Small and fast: one whole task (5 variants x 2 encoders x 3 seeds) finishes well inside the tool timeout, so this
  needed 2 chunks rather than amazon_ratings' 10.
- Registry edits only, pipeline untouched: `virgo/config.py`, `experiments/characterize.py` `STUDY`,
  `virgo/frozen_rules.py` `HELDOUT`, `experiments/run_core.py` `HELDOUT` (opt-in by name).

## 2026-08-15 — Project renamed + all docs rewritten against a full accuracy audit

**Rename.** *ViRGo — Virtual Role-Graph Embedding for Structural Identity* → **Structure Aware Graph Augmentation for
Graph Neural Networks**. Docs only. The Python package stays `virgo/`: renaming the directory would break every
import, every notebook, every cached `.pyc` and every stored output path, for zero research gain. If it is ever
wanted, it is a separate mechanical change (`virgo/` → new name, ~15 source files + 7 notebooks) and must be done in
one commit with a full re-run of the smoke path.

**Files rewritten** (content, not just the name):
- `README.md` — reframed around the two-stage framework; stale tables removed (it still described 5 variants, 3 seeds,
  6 datasets, "Module 3 next", and "no virtual graph matches the original on any dataset", all superseded).
- `CLAUDE.md` — same reframe; §3 is now an explicit validation-status table, so a future session cannot restate the
  gate as validated.
- `docs/virgo_guide.md` → **`docs/project_guide.md`** (`git mv`, PDF moved with it). Rewritten; §3 is the
  "what is proven and what is not" section.
- `experiments/README.md` — the fit/predict split made explicit, the four prediction scripts added to the table.
- `notebooks/7-phase7_strategy_selection.ipynb` §4 — two false claims corrected (markdown only, no re-execution).

**Two corrections found by the audit**, both written up in full in `docs/paper_log.md` (2026-08-15):
1. `reed98` *was* trained (18:32–18:37 on 08-14, after `module5_scored.csv` was written at 17:42) and it **augments at
   2.55σ**. The gate had called `keep original`. Stage 1 is therefore **2/3**, not 2/2, and rule 1 alone would have
   been 3/3.
   **Decision (user, 2026-08-15): the scorer is NOT re-run.** `results/module5_scored.csv` keeps `pending` on that
   row as the artifact of the original scoring pass. **Notebook 7 §4 is the source of truth for reed98**, and the
   prose in `README.md`, `CLAUDE.md` and `docs/project_guide.md` was written to match it. Do not "fix" the CSV — the
   discrepancy is deliberate and documented. If a fresh table is ever needed, it is one command
   (`python experiments/predict_gate.py --step score`), but that is a decision to take, not a tidy-up.
2. Module 3's published **5/7 and 4/7 are 3-seed** numbers. The scoreboard is now the 10-seed board (3-seed archived
   at `results/scoreboard_3seed.csv`); re-scored there, they are **4/6 and 4/6**, because `lastfm_asia` moved from
   `keep original` to `tie`. No frozen cut is affected — the discovery-panel and low-homophily-zone verdicts are
   identical at both seed counts.

Everything else in the audit checked out: the frozen cuts and intervals match `results/candidate_rules.csv`,
`gate_candidates.csv` and `strategy_patterns.csv` exactly; the panel-membership assertions hold; and no prediction
script imports a fitting function.

**Known stale artifacts, deliberately left alone** (they are records of what was true when written, and rewriting
them would falsify the pre-registration trail): `results/module3_scored.csv`, `results/module5_scored.csv`. Re-score
them with the scripts above when a fresh number is needed; do not hand-edit.

## 2026-08-31 — README trimmed, dataset citations split out, docs re-pointed at the new priorities

- `README.md` 23.8 KB → 14.3 KB. Kept: the intro, the headline table, the framework, the validation record. Compressed:
  method, repository map, usage, reproducibility. Removed: the module-by-module evidence trail and the per-dataset
  role commentary — both pointed at `results/*.csv` and `docs/*.md`, which are gitignored and therefore invisible to
  anyone reading the published repo.
- New tracked `DATASETS.md` at the repo root holds the full source/citation table, the panel membership, and the
  Facebook100 missing-gender note. `docs/` is gitignored except `identity2vec.pdf`, so this could not live there.
- `README.md` §7 is now the roadmap (stage-1 stabilization → stage-2 breadth → anomaly detection → GIN/GAT), and §2
  states the known weakness of the frozen stage 1 rather than presenting it as finished.
- Dead reference removed: `docs/project_guide.md` was deleted in commit 9e6339b and no longer exists.
- `CLAUDE.md` updated in place (§1 anomaly detection back in scope + framing documents, §2 stabilization target,
  §4 tasks and encoder scope, §5 next steps, §8 deliverables). No code touched.

## 2026-08-31 — experiments/stage1_pairs.py added (Module 8 fitter)

New panel-guarded FIT script, mirroring `gate_rules.py`: `panel_cells` → `screen` (zone) → `compound` / `pairs`
(panel-level, with a leave-one-out that refits the condition in every fold) → `collinearity` → `report`. Reuses
`characterize.threshold` / `loo_threshold` / `rho` / `gaps` and `gate_rules.decisive_pair` rather than re-deriving them.
Writes `results/stage1_pair_*.csv` (`exploratory_` prefixed under `--allow-refit`). Notebook 6 gained §7 reading those
CSVs; nothing recomputes there. `virgo/frozen_rules.py` untouched — no rule was promoted.

## 2026-09-01 — stage-1 second condition swapped in code
- `virgo/frozen_rules.py`: added `FROZEN_EXCEPTION` (`avg_clustering < 0.5573`, interval (0.5329, 0.5769)); `predict_gated()`
  now calls it instead of `FROZEN_GATE`, same slot, same return arity. `FROZEN_GATE` kept and marked SUPERSEDED — Modules
  4–5 are published against it and `gate_rules.py` asserts its cut still reproduces.
- `experiments/predict_gate.py`: measures `avg_clustering` with `nx.average_clustering` on the policy-loaded original graph
  (matches the feature-cache values to 4 dp); the role-graph build is no longer a precondition for predicting —
  `original_retention` is recorded only when the `psi` edgelist already exists, else NaN. Columns `gate_pred`/`gate_correct`
  → `exception_pred`/`exception_correct`; existing `module5_predictions.csv` rows are untouched by the write-once freeze.
- Docs: README §2/§3/§7/§8 and CLAUDE.md §2/§3/§5/§8 updated; the rule is written as "homophily decides, clustering is the
  exception", never as two equal conditions.

## 2026-09-02 — DEGREE_RULE_CORPUS plumbing
- `virgo/data/make_pyg.py`: 20 corpus loaders (PyG `Airports`/`EmailEUCore`/`WikiCS`/`PolBlogs`/`Coauthor`/`Amazon`/
  `AttributedGraphDataset`/`CitationFull`/`WikipediaNetwork`, plus the SNAP musae archives for five more Twitch
  languages and GitHub). `cfg.DEGREE_RULE_CORPUS` holds the list in wave order; registry entries in `config.py`,
  domains in `characterize.STUDY`, and `run_core.RUNNABLE` extended.
- **Loader bug found and fixed before any run used it.** SNAP's `musae_FR_target.csv` has 6551 rows but only 6549
  distinct `new_id`s, so the previous `np.empty(len(t))` + scatter left two entries as uninitialised memory, which
  surfaced as a bogus 4-class label set. `_snap()` now drops duplicate ids, sizes the array by
  `max(id, edge) + 1`, and fills with -1 so unmatched nodes are written as UNLABELLED rather than as a class.
  Rebuilt every SNAP graph to confirm no drift: twitch_de, twitch_engb, twitch_es, github and deezer_europe are
  byte-identical to before; twitch_fr is now 6549 nodes / 2 classes.
- Screening runs use `--encoder graphsage_edge` only (recorded deviation, paper_log §12); DeepWalk is backfilled for any
  graph that becomes a fitting cell.


## 2026-09-03 — DEGREE_RULE_VALIDATION plumbing
- Six graphs registered as the corpus's held-out half: `wiki_attr`, `crocodile`, `cora_full`, `penn94`, `genius`,
  `twitch_gamers`. `cfg.DEGREE_RULE_VALIDATION` holds them cheapest-first, with an assert that no graph is in both it and
  `DEGREE_RULE_CORPUS`; registry entries in `config.py`, domains in `characterize.STUDY`, `run_core.RUNNABLE` extended.
- `virgo/data/make_pyg.py`: `wiki_attr` and `cora_full` join the existing `AttributedGraphDataset` / `CitationFull`
  branches; `penn94` and `genius` join `LINKX`. The `SNAP` table now stores the edge and target files as full relative
  paths plus a quantile-bin count, because Twitch Gamers ships `large_twitch_{edges,features}.csv` rather than the musae
  `_edges/_target` stems, and Crocodile ships a CONTINUOUS traffic target that is binned into 5 quantile classes (the
  same construction Geom-GCN uses to label Chameleon and Squirrel).
- Four candidates were probed and REJECTED rather than silently skipped: musae-facebook (SNAP's own `facebook_large.zip`
  returns 404 and graphmining.ai does not resolve at all), PyG's `twitch-gamers` LINKX loader (bare `AssertionError`, so
  the graph is built from SNAP instead), `AttributedGraphDataset` facebook/twitter (multi-label, 193 and 4065 columns),
  and NELL (codes unlabelled entities as class 0, 85% of nodes — a label artefact, not a majority class).
  `amazon_photo`, the obvious co-purchase sibling of the degree winner `amazon_computers`, is excluded for a different
  reason: it is a `STRATEGY_PANEL` fitting graph, so it cannot test anything stage 2 fitted.
- `experiments/degree_rule.py`: `_assert_fit` now refuses the validation names as well as the Module-9 held-out three;
  new `prereg()` + `--preregister` write every fitted cut's prediction for `--validation` to
  `results/[paired_]degree_validation_prereg.csv` (the band prefix, as for every other output) from ORIGINAL-graph properties alone, so the file exists before any of the six
  is trained. Scoring them later is the existing `--heldout` path, which needs the scoreboard rows that must not exist yet.
- The full-graph structural feature cache had to be built before the properties could be measured (`characterize.load_cache`
  asserts on it). It is written by constructing the registered encoder on the original graph — no training, and the values
  are a deterministic function of the edgelist, so a later sweep reuses the same file.

## 2026-09-04 — Module 11 run + re-search
- `run_core.py --datasets <the six> --task link_prediction --encoder graphsage_edge --seeds 42 43 44` (42 scored cells),
  then `tie_break.py` over the same six (merged into `degree_rule_corpus_verdicts.csv`, now 34 datasets × 2 bands).
  Pairing resolved 2 of 6: `crocodile` (centrality|degree|psi → psi) and `wiki_attr` (original|psi → original).
- `experiments/degree_rule.py`: `heldout()` now scores BOTH contrasts. It previously applied only the degree-vs-rest
  screen, which meant the degree-vs-Ψ candidate — the thing the validation set exists to test — was never actually
  scored by `--heldout`. degree-vs-Ψ is scored only on cells whose sole winner IS degree or Ψ; a centrality winner is
  "not applicable", not a miss, so the rule cannot be credited with a denominator it never earned.
- The re-search used `--allow-refit` on all 34 paired-verdict graphs, writing `exploratory_paired_*.csv`; the published
  `paired_*.csv` tables are untouched.
- Watcher bug worth remembering: `until ! pgrep -f "tie_break.py"; do sleep 30; done` never exits, because `pgrep -f`
  matches the waiting shell's OWN command line. `tie_break.py` had finished in minutes; the watcher span for 3.5 hours.

## 2026-09-07 — repo cleanup: generated data untracked
- The old ignore rules missed everything they were meant to catch: `splits/*.txt` and `splits/*.edgelist` are
  top-level globs, but the real paths are `splits/link_prediction/virtual_graph_study/<ds>/seed_<n>/*`. 992 split
  files were never matched and 751 more predated the rule, so `git status -uall` showed 1161 pending files.
- Now ignored: `splits/` (1.6G), `labels/` (14M), `third_party/struc2vec/pickles/` + `src/struc2vec.log` +
  `src/random_walks.txt` (162M of runtime cache, not vendored source), and the nine `input/` files over 5M (167M).
  Tracked checkout drops 227M -> 65M. Everything was removed with `git rm --cached`, so the files stay on disk.
- `third_party/struc2vec/src/utils.py` writes its cache into `../pickles/` with no `makedirs`, so the directory is
  preserved by a `.gitkeep` — dropping it outright would break struc2vec on a fresh clone.
- `input/` is deliberately NOT ignored wholesale. It is the reproducibility anchor: `split_edges()` builds the
  spanning forest with `nx.minimum_spanning_tree()` on an unweighted graph, where Kruskal breaks ties by edge
  INSERTION order = the line order of the edgelist. A re-download that renumbers nodes therefore changes the split,
  and three source hosts have already died on us (graphmining.ai for `lastfm_asia`, musae-facebook at both hosts,
  PyG's twitch-gamers loader). Committing the 125 edgelists under 5M keeps most graphs reproducible with no network.
- Verified rather than assumed: `experiments/verify_splits.py` regenerates each stored split into a temp dir and
  byte-compares. **428/428 identical, 0 differ, 0 skipped** across every dataset and seed on disk.
- `.gitignore` is itself excluded by `.git/info/exclude:27`, so the ignore rules are LOCAL ONLY and a fresh clone
  has none of them. Left as-is to match the existing convention; force-adding it is the fix if that ever bites.
- `.git` is still 882M — history retains every removed blob and origin already has those commits, so a clone is
  unchanged in size. Only `git filter-repo` + force-push would shrink it, at the cost of rewriting history.

## 2026-09-07 — split directories regenerated after the tracked-data cleanup
The same-day repo cleanup (commits `e1540ca` / `446be8c`, which added `splits/`, `results/`, `output/`, `labels/` and
`docs/*` to `.gitignore`) removed `splits/link_prediction/original_graph/` and 18 `virtual_graph_study/<ds>/` folders
from disk. Six of the seven I needed were rebuilt with `prepare_linkpred.prepare(edgelist, dir, 0.30, 42)` and are
byte-identical, as `.gitignore` documents: `actor`'s probe contrast reproduced to 8e-7 after the rebuild.
`tolokers` could NOT be rebuilt — `input/tolokers.edgelist` is one of the nine large inputs the same cleanup excluded and
deleted, so it must come back through `virgo/data/make_hetero.py` before any tolokers split or probe row can exist.

## 2026-09-08 — `psi_lambda` variant added (Ψ without the Fix-8 clamp)
- `virgo/virtual_graph.py` — `psi_signature(..., poisson=True)`; `poisson=False` returns λ directly. The `psi` path is
  algebraically unchanged (`max(..., 1e-12)` simply moved after the division), so existing `psi` embeddings stay valid.
- `virgo/config.py` — `psi_lambda` appended to `VG_SIMS`. **Deliberately NOT in `VG_SIMS_LOCKED`**: the fitting scripts
  read that list, so no frozen rule can pick the new variant up.
- `experiments/psi_clamp.py` — new. Three modes: build diagnostic (default), `--split-seed N` (measure the seed's 70%
  train graph, which is what LP cells are built from), `--paired` (score cached psi vs psi_lambda per seed).
- Runs: `run_core.py --sim psi_lambda --encoder graphsage_edge --task link_prediction --seeds 42 43 44` on cora,
  minesweeper, roman_empire, twitch_es, twitch_ptbr, ppi_rat. Results and the finding: paper_log 2026-09-08.
- DEVIATION RECORDED: `psi_lambda` drops I2V Eq. 1 entirely (no Poisson wrapper). It is a diagnostic control, not a
  replacement for `psi`.
- BLOCKER: `labels/{cora,minesweeper,roman_empire}.labels` are missing from disk, so node classification for those three
  cannot be scored (first batch crashed on it). Not regenerated — flagged, not guessed at.

## 2026-09-09 — ω sensitivity test: the 2026-06-24 open question is ANSWERED, keep `deg + Ω`
- `virgo/virtual_graph.py` — `PSI_ARMS` table + `psi_signature(..., omega=)`. Four readings of Eq. 3-4's ω:
  `deg_ev` (shipped), `deg`, `ev`, `delta_ev` (= Δ + Ω, the reviewer's alternative and the paper's own two named
  structural properties, §3.2.1). `psi` and `psi_lambda` are unchanged; the three new arms are in `VG_SIMS`, NOT in
  `VG_SIMS_LOCKED`.
- `experiments/psi_omega.py` — new. Build diagnostic (default), `--split-seed`, `--paired` (score cached arm vs psi).
- Runs: `run_core.py --sim psi_w_{deg,ev,delta} --task link_prediction --encoder graphsage_edge --seeds 42 43 44` on
  cora, twitch_ptbr, squirrel_filtered. CSVs `results/psi_omega.csv`, `results/psi_omega_paired.csv`.
- **ω is NOT a free parameter for the clamp**: ω > 0 in every arm, so `sign(λ/ω) = sign(λ)` and `clamped_frac` is
  identical across all four arms on all three graphs (verified, not assumed — the script asserts one unique value).
- Result and the decision: paper_log 2026-09-09. **No variant changes; `psi` keeps `deg + Ω`.**

## 2026-09-13 — clamp repairs implemented; `psi_pqnorm` ADOPTED
- `virgo/virtual_graph.py` — `psi_signature` split into `psi_lambda_of` (the λ, with `omega=` and `norm=`) and
  `poisson_score` (Fix 8 + the 1e-12 floor). `PSI_ARMS` is now a 4-tuple `(poisson, omega, norm, shift)`; the shift is
  applied in `signatures()` because it is a graph-level operation. `psi` verified numerically unchanged (cora signature
  min −1097.960535 / max −1.000003, matching the 2026-09-08 run).
- New arms in `cfg.VG_SIMS`, none in `VG_SIMS_LOCKED`: `psi_qnorm` (Ω normalised over N(u)), `psi_pqnorm` (Δ and Ω both),
  `psi_shift` (λ − min λ over non-isolated nodes).
- `experiments/psi_fix.py` — new; build-side measurement of the repairs, `--split-seed` supported. `psi_omega.py --paired`
  picks the new arms up automatically (it reads every `psi_*` in `VG_SIMS`).
- DECISION: **`psi_pqnorm` adopted for new work**, on defensibility not score (paper_log 2026-09-13). `psi` untouched.
- Panel re-run launched: `run_core.py --sim psi_pqnorm --task link_prediction --seeds 42 43 44` over the 18 stage-1 /
  stage-2 datasets that are on disk.
- The four datasets absent from disk (`ogbn_arxiv`, `ogbl_ddi`, `tolokers`, `cornell5`) were regenerated for the clamp
  table; `tolokers` 76.0% and `cornell5` 84.9% were measured. The two OGB graphs were DROPPED on user instruction
  (2026-09-13, "leave missing 4") and are not in the re-run.

## 2026-09-14 — DECISION REVERSED: keep old Ψ and old ω; the repairs are reported, not adopted
- User + supervisor (email, 2026-09-14): "OK to keep old Psi but add a short note on the performance drop for a
  normalized version of it. The sensitivity check is convincing also."
- `psi` and ω = degree + Ω remain THE implementation. The 2026-09-13 note adopting `psi_pqnorm` is SUPERSEDED.
- `psi_qnorm` / `psi_pqnorm` / `psi_shift` / `psi_w_deg` / `psi_w_ev` / `psi_w_delta` / `psi_lambda` stay in `VG_SIMS`
  as DIAGNOSTICS (none in `VG_SIMS_LOCKED`); their results are paper material for the Ψ section, not pipeline changes.
- Still open from the supervisor's checklist: rewrite the Ψ section to present the d = 1 choice as position-free by
  design. That is writing, not experiments.

## 2026-09-15 — GATv2 wired as a third encoder; GIN run for the first time
- New `virgo/encoders/gatv2.py`: `GATv2Encoder`, `name = "gatv2"`, one `build_convs()`. `HEADS = 4`; hidden layers
  concat (`dims[i+1] // HEADS` per head), the output layer averages, so the embedding stays 64-dim. Rejects any
  `agg` but `mean` (attention aggregates by definition, same as GIN rejects the ablation-B knob).
- Registered as `gatv2_edge` in `virgo/encoders/__init__.py`. No driver, CLI or scoreboard edit was needed — the
  `ENCODERS` extension point held exactly as documented.
- `virgo/config.py`: added `VG_SIMS_OFFICIAL` (the seven official variants) and the Module-15 panels
  `ENCODER_STAGE1` / `ENCODER_STAGE2_CENTRALITY` / `ENCODER_STAGE2_DEGREE` / `ENCODER_PANEL` / `ENCODER_SMOKE`.
  Additive only; `VG_SIMS`, `VG_SIMS_LOCKED` and every existing default are untouched.
- New `experiments/encoder_transfer.py`. Reuses `run_core.embed/score` and `tie_break.paired/verdicts` rather than
  re-implementing them; `run_core.py` is byte-identical.
- FINDING (plumbing): `strategy_select.winners()` asserts `len(s) == len(cfg.VG_SIMS)`, but `VG_SIMS` grew to 14 when
  the Ψ diagnostics were added, while 44 of 71 LP datasets have only the 7 official variants. That assert now fails on
  them. NOT fixed here (it would touch a fitting script); `encoder_transfer.py` reads `VG_SIMS_OFFICIAL` explicitly
  instead. TODO: decide whether `winners()` should read `VG_SIMS_OFFICIAL`.
- Commands (seed 42-44, K=10, LP only):
  - `python experiments/encoder_transfer.py --step run --encoder gin_edge --datasets cora --seeds 42`
  - `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets cora`
  - `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets squirrel_filtered`
  - `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets actor --seeds 42 43`
  - `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets actor --seeds 44`
  - `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets roman_empire --seeds 42` (then 43, then 44)
  - `python experiments/encoder_transfer.py --step all --encoder gatv2_edge`
- Runtime (CPU, 7 variants x 3 seeds): cora 3m09, squirrel_filtered 6m59, actor 9m07, roman_empire 15m32. Roughly
  linear in node count (the role graph has K*N edges). The full 37-graph panel will need a chunked run.
- Scoreboard after the run: `graphsage_edge` 735 rows (unchanged), `gatv2_edge` 28, `gin_edge` 7. `results/` is
  gitignored, so no tracked file was touched by the runs.
- Interpretation lives in `notebooks/8-phase8_encoder_generalization.ipynb` (written, not executed).

## 2026-09-16 — Module 15 full panel done (37 graphs, GATv2)
- All 37 x 7 variants x 3 seeds trained. `results/scoreboard.csv`: `graphsage_edge` 735 rows (unchanged throughout),
  `gatv2_edge` 259, `gin_edge` 7. `results/` is gitignored, so no tracked file moved.
- Added `--sims` to `experiments/encoder_transfer.py` so a large graph can be chunked across invocations; `--step
  compare` ignores it and reads whatever is on disk. Needed because the tool timeout is 10 min and `genius` /
  `twitch_gamers` are ~4 h each.
- Chunking is cheap to resume: `embed()` writes each variant's `.emb` as it finishes, so a killed run loses only the
  in-progress variant. Two runs were killed by session end and both resumed with no recomputation.
- Runtime ~31 h CPU over 14 chunked runs. Roughly linear in NODE count for role variants (role graph has K*N edges)
  and in EDGE count for `original`/hybrids: `genius` (422k nodes) 3.8 h, `twitch_gamers` (168k nodes, 6.8M edges) 3.9 h.
- Notebook 8 rewritten lean (14 cells -> 10) and now derives its panel from the CSVs, so it is correct at any stage of
  a partial run. Dropped: the held-constant table, the GIN section, the long limitations prose, the panel-count cell.
- Commands: `python experiments/encoder_transfer.py --step run --encoder gatv2_edge --datasets <...> [--sims <...>]`
  then `--step compare` over `cfg.ENCODER_PANEL`.

## 2026-09-17 — Module 15 reporting
- `cfg.ENCODER_PANEL` is a plain 30-graph literal; `ENCODER_STAGE1` / `ENCODER_STAGE2_CENTRALITY` /
  `ENCODER_STAGE2_DEGREE` above it record where each graph came from. One scope, no second list.
- Module-15 tables (`encoder_transfer_{perseed,fourcell,verdicts,agreement}.csv`) and the `gatv2_edge` scoreboard rows
  cover those 30. `graphsage_edge` is unchanged at 735 rows and every earlier panel is untouched.
- The 7 `gin_edge` rows are `cora`-only - GIN is a plumbing check, not a panel cell.
- Notebook 8 reads the 30 straight from the CSVs.
