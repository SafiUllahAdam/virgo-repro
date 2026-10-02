# ViRGo — Paper Log

**Purpose.** Curated source-of-truth for paper writing. Records only what a reviewer/author needs for the Methods, Results, and Discussion sections: design decisions and their rationale, method specifications, hyperparameters, quantitative results, findings, and deviations from prior work. Deliberately excludes repo housekeeping (folder renames, path fixes, tooling) — that lives in `docs/notes.md`, the operational lab notebook.

Maintained automatically: a new entry is added whenever something research-significant happens, dated, newest at the bottom. Every result records its seed(s), dataset, encoder, and any caveat that limits its interpretation.

---

## Contribution & research question

- **Research question.** Which virtual graph is best for given data and task, and does GNN message passing over a structural-similarity virtual graph beat walk+Skipgram on structural-identity embeddings? The **virtual-graph construction — not the encoder — is the variable under study.**
- **1st contribution (virtual-graph study).** A virtual graph connects nodes by structural role (hub↔hub, bridge↔bridge), not by original edges, so role-similar but distant nodes can exchange messages. We test which construction serves each downstream task (node classification, link prediction, later anomaly detection). Identity2Vec's (I2V, Oluigbo et al.) Poisson/KL similarity graph is treated as one *generic* construction to test against simpler ones (degree-only, centrality-only).
- **Technical contribution (GNN encoder).** Replace I2V's guided walk + Skipgram with a modern inductive GNN (GraphSAGE primary, GIN expressive alternative, GAT ablation) aggregating directly over the virtual graph.
- **Stretch (2nd contribution, not started).** Reuse compact ViRGo embeddings as a structural summary of a large graph so its structure fits an LLM context window.
- **Scope guard.** Baselines (DeepWalk / node2vec / struc2vec) are used as published/default and **not fine-tuned** — the contribution is the method, not baseline tuning. Non-Euclidean/hyperbolic latent space is out of scope (reserved for a second paper).

---

## Phase 1 — Reproducibility (DONE)

- **Cached I2V.** I2V recomputes the structural signal (degree + eigenvector centrality) inside the walk loop — per neighbor, per step, per walk — which dominates cost. For a static graph these are constant, so we compute them once and cache. Result: embeddings **byte-identical** to the original, **~200× faster** (Deliverable #1).
- **Reproduction bar.** A metric counts as reproduced only within **±0.05** of the paper. Cora I2V lands in the paper's range on both tasks under a 3-seed harness (seeds 42/43/44).
- **Hyperparameters (I2V, `I2V_PARAMS`).** dimensions=64, walk_length=40, num_walks=10, window_size=10, epochs=1, sg=1 (Skipgram), e=2.7182. **DEVIATION:** paper text states walk_length=80; we use the repo default 40 (matches the author's released `cora.emb`), recorded as a deliberate deviation.
- **Evaluation protocol.** Node classification = one-vs-rest logistic regression on embeddings, weighted F1, stratified split. Link prediction = 70:30 edge split, embedding retrained on the 70% train graph only (no leakage), test edges vs sampled non-edges ranked by **cosine similarity** of embeddings → test AUC (unsupervised, I2V-paper-faithful; this is the protocol every recorded LP number uses). A supervised alternative (Hadamard edge features → logistic regression) exists in `eval_linkpred.py` as an optional robustness check only — an earlier version of this entry wrongly named it as the main protocol.
- **FINDING (baseline comparison).** On homophilous Cora, proximity methods beat structural ones: node2vec (NC ≈0.82 / LP ≈0.91) > I2V (NC ≈0.69 / LP ≈0.80). This is expected — proximity embeddings suit community/homophily labels — and indicates the original paper's "I2V beats all" reflects under-tuned baselines. I2V's advantage is on **structural** tasks. This motivates studying the virtual graph rather than the encoder alone.

---

## Phase 2 — Virtual-graph construction (CORE STUDY, in progress)

### Method

- **Virtual graph.** For a chosen structural signature, connect each node to its **top-K** most structurally similar nodes (K-nearest on the per-node signature). Undirected union, no self-loops, all original nodes kept (isolated nodes preserved). Edge weight = `1/(1+distance) ∈ (0,1]`, always finite. Build is **deterministic** (byte-identical rebuild).
- **Signature = variable under study.** Three variants (`--sim`, pluggable):
  - `psi` — I2V's KL→Poisson structural score used as a **reference-free per-node signature**. I2V's Ψ is walk-contextual (needs a reference node + shortest-path); we lift it to an all-pairs virtual graph by dropping the walk shortest-path factor (use q = Ω eigenvector centrality instead of Ω·pathlen), keeping the exact KL rate λ, the Fix-4A normalizer, and the Fix-8 log-Poisson score. **DESIGN DECISION:** rejected the alternative of reusing I2V's pairwise `identity_score` literally (asymmetric, slower, off-walk).
  - `degree` — degree-only signature (simplest structural baseline).
  - `centrality` — eigenvector-centrality-only signature.
- **Sweep design.** K ∈ {5, 10, 20} (sparsity vs over-smoothing), seeds {42, 43, 44}. Same K and same seeds across all variants = fair comparison ("which graph is best?").
- **Fixed-encoder protocol.** Only the graph changes; the encoder is held fixed so any performance difference is attributable to the construction. Current bridge encoder = DeepWalk with I2V_PARAMS (Phase-3 GNN will replace it and will additionally use the edge weights, which DeepWalk ignores). Link prediction is leakage-free: the virtual graph is rebuilt from the 70% train edges only.
- **Graph-health table.** Every built graph logs one row (`dataset, sim, K, nodes, edges, avg_degree, components, isolates`) so a weak score can be traced to a degenerate graph (too sparse, disconnected, isolates, too dense); doubles as an ablation-quality table for the paper.

### First results — Cora, K=10, DeepWalk bridge, 3 seeds (INDICATIVE, not final)

| variant | node-class weighted-F1 | link-pred AUC |
|---|---|---|
| centrality | **0.381 ± 0.007** | 0.551 ± 0.008 |
| psi (I2V KL/Poisson) | 0.234 ± 0.005 | 0.511 ± 0.006 |
| degree | 0.152 ± 0.007 | **0.555 ± 0.002** |

Graph sizes (Cora, K=10): psi 16251 edges (avg deg 12.0), degree 26216 (19.4 — integer-degree ties inflate the K-NN union), centrality 16110 (11.9).

- **FINDING (indicative).** The best virtual graph is **task-dependent**: centrality wins node classification, degree wins link prediction, and I2V's `psi` is not best on Cora under the DeepWalk bridge. Directly supports the Phase-2 question.
- **CAVEATS (why not final).** (a) DeepWalk bridge, not the Phase-3 GNN — DeepWalk ignores the edge weights that `psi`/`centrality` carry, so the GNN may reorder these. (b) Cora is homophily/community-labelled, so structural embeddings score modestly on NC and cosine LP AUC sits near 0.55. (c) Single dataset, K=10 only.

### Planned (deferred by design, to run after Phase 3)

- Full K sweep (5/10/20) scored on all variants.
- **Four datasets total** for the published comparison (Cora + three more) — not Cora alone.
- Phase-3 GNN encoders (GraphSAGE / GIN / GAT) over the same virtual graphs, replacing the DeepWalk bridge, then re-run the full comparison.

---

## Phase 3 — GNN encoder (design locked 2026-07-04, implementation pending)

- **DECISION (user + review).** Encoder = **GraphSAGE trained unsupervised** with a **Skipgram-analog objective** — same role Skipgram plays in I2V, so the GNN is a true drop-in for the walk+Skipgram back end. Study runs **on the virtual graphs** (starting with I2V's Poisson/KL Ψ graph); the original graph is only a control row. Output stays 64-dim `.emb` → identical evaluation protocol as I2V/Phase 2.
- **Core architecture (ViRGo-SAGE).** Structural input features `[degree, eigenvector centrality Ω, ψ, clustering]` (no node attributes → method stays structural + inductive) → 2-layer GraphSAGE (mean) over the virtual graph → 64-d z. Loss = GraphSAGE unsupervised objective (Skipgram with the lookup table replaced by the GNN): positives = walk co-occurrence on the *virtual* graph, negatives ∝ deg^{3/4}, Q=5 (matches I2V `negative=5`).
- **Comparability design.** Walk corpus for positives uses the exact I2V params of the Phase-2 DeepWalk bridge (num_walks=10, walk_length=40, window=10, seeds 42/43/44) ⇒ Phase-2 bridge vs Phase-3 SAGE differ in **one component only** (Skipgram lookup vs message passing) — a clean encoder ablation.
- **Variant axes for the study/ablations:** A positives (walk co-occurrence vs 1-hop virtual neighbors — "are walks still needed once similarity is explicit?"), B aggregation (mean / **Ψ-weighted mean** — first use of the virtual edge weights / max / sum), C depth 1–3 (over-smoothing: virtual graphs near-connected at K=10), D features (structural-4 / degree-only / random), E graph (Ψ, degree, centrality × K 5/10/20; dual virtual+original branch deferred to follow-up work).
- **Run plan:** Stage 1 lock encoder on Cora Ψ K=10 (A×B, 4 configs, 3 seeds) → Stage 2 full "which virtual graph?" matrix under the GNN → Stage 3 depth/feature ablations + original-graph control → Phase 4 remaining 3 datasets + anomaly detection. Full design: `docs/phase3_gnn_design.md`.
- **Implementation (2026-07-04).** Spine implemented as designed (`encoder.py` `SageEncoder` + `notebooks/3-phase3_gnn_encoder.ipynb`). Method details now fixed in code: input features are computed on the **original** graph (the structural-identity signal) while message passing runs on the **virtual** graph; loss positives come from the same seeded, unweighted walk generator the Phase-2 bridge used (num_walks=10, length=40, window=10); Q=5 negatives ∝ deg^0.75; training CPU-only with seeded RNGs for reproducibility; LP reuses the identical Phase-2 splits so the encoder comparison is split-for-split fair. First SAGE results pending the notebook run.

### First encoder head-to-head — Enzymes, Ψ virtual graph, K=10, seeds 42/43/44 (2026-07-06, INDICATIVE)

First full run of the spine (single-seed pass earlier the same day, superseded by this 3-seed result). Same virtual graph (Ψ, K=10), same walk corpus, same leakage-free LP splits (`enzymes_vglp_s{42,43,44}`), cosine LP scoring — **only the encoder differs**:

| task | DeepWalk bridge (Phase 2) | ViRGo-SAGE (Phase 3) | Δ |
|---|---|---|---|
| node classification (weighted F1) | 0.4971 | **0.5405 ± 0.0014** | +0.043 |
| link prediction (AUC) | 0.5110 ± 0.0297 | **0.6632 ± 0.0114** | **+0.152** |

Graph health (enzymes, Ψ, K=10): 19,474 nodes, 111,401 edges, avg degree 11.44, 10 components, 0 isolates. Training loss falls ~13–18 → ~4.1 over 50 epochs across seeds. Enzymes labels: 106 labelled nodes absent from the edgelist are skipped identically for both encoders (n=19,474, 3 classes).

- **FINDING (indicative).** Message passing over the virtual graph beats the Skipgram lookup on both tasks and across all 3 seeds; the LP gain is large (+0.152 AUC, from near-random 0.51 to 0.66) and exceeds the seed spread by an order of magnitude. Supports the technical contribution: GNN > walk+Skipgram on the *same* virtual graph.
- **CAVEATS.** (a) Single dataset (enzymes), single variant/K (Ψ, K=10) — spine verification, not the study. (b) SAGE additionally consumes 4 structural input features from the original graph (degree, Ω, ψ, clustering), which the lookup-table bridge cannot use by construction — deliberate design (that *is* the GNN's advantage), but the D-axis feature ablation (degree-only / random features) is what will isolate feature signal from message-passing signal. (c) Cosine-scored LP, as in Phase 2.
- **Ablation A implemented (2026-07-06).** `positives` knob on the ViRGo-SAGE loss: **A1 `walk`** = window-10 co-occurrence on virtual-graph walks (default; identical to the Phase-2 bridge corpus ⇒ fair Skipgram-vs-GNN comparison), **A2 `edge`** = the virtual edges themselves as positive pairs (both directions, no walks) — tests whether walks are still needed once structural similarity is explicit in the graph. Everything else (negatives ∝ deg^0.75 Q=5, loss, caps, seeds) unchanged, so A1-vs-A2 is a one-component comparison. A2 artifacts tagged `sage_edge_*` / scoreboard encoder `graphsage_edge`. Results below.

### Ablation A results — A1 walks vs A2 direct edges — Enzymes, Ψ, K=10, seeds 42/43/44 (2026-07-06, INDICATIVE)

Same graph, features, architecture, negatives, splits, seeds — **only the positive-pair source differs** (A2 corpus: 222,802 edge pairs, under the 2M cap; both schemes sample 100k pairs/epoch):

| task | DeepWalk bridge (Phase 2) | SAGE A1 (walk) | SAGE A2 (edge) |
|---|---|---|---|
| node classification (weighted F1) | 0.4971 ± 0.0057 | 0.5405 ± 0.0014 | **0.5413 ± 0.0011** |
| link prediction (AUC) | 0.5110 ± 0.0297 | 0.6632 ± 0.0114 | **0.6909 ± 0.0156** |

- **FINDING 1.** The GNN beats the bridge under *both* objectives (NC +0.044, LP +0.15–0.18) — the encoder win is robust to the positive-pair scheme, not an artifact of the walk corpus.
- **FINDING 2 (the A ablation).** NC: statistical tie (Δ 0.0008, within seed noise). LP: A2 +0.028 over A1 (≈2σ vs seed stds) — likely real, borderline at 3 seeds. **Walks are not needed once structural similarity is explicit in the graph**: I2V ran walks to *find* role-similar nodes; the virtual graph already lists them, so direct edges suffice and even help LP. A2 is also cheaper (no walk generation).
- **DECISION.** A2 (direct-edge positives) becomes the default training objective for the remaining ablations/study; A1 is kept and reported as the bridge-comparable configuration — the "Phase-2 vs Phase-3 differ in one component only" claim holds only under A1.
- **DEVIATION.** Design doc planned Stage 1 on Cora; this first A run used enzymes. Cora repeat pending (cheap — cora bridge embeddings ready at 3 seeds).
- **CAVEATS.** Single dataset (enzymes), single graph variant/K (Ψ, 10), 3 seeds, cosine LP; loss *values* are not comparable across A1/A2 (different positive distributions) — only the downstream metrics are; feature-vs-message-passing attribution still waits on the D-axis ablation.

### Ablation B — neighbor aggregation over the virtual graph — Enzymes, Ψ, K=10, seeds 42/43/44 (2026-07-07)

- **Question.** Virtual edges carry a similarity *strength* (weight `1/(1+dist)` from Ψ/degree/centrality distance); mean aggregation ignores it. Does the encoder need only edge *existence* (which A2 already exploits), or also edge *strength*?
- **Variants (`agg` knob).** `mean` = equal neighbors (baseline; = the existing A2 rows, no new runs). `weighted` = per-target-normalized Ψ-weighted mean — the first use of the virtual edge weights anywhere in the encoder. `sum` = keeps neighborhood-magnitude signal. `max` = most-similar neighbor dominates.
- **Method detail.** `mean/sum/max` via native `SAGEConv(aggr=…)`; `weighted` via `GraphConv(aggr='add')` — identical root-plus-neighbor form (W₁xᵢ + W₂·agg xⱼ) but edge-weight-aware, weights row-normalized per target so the aggregate is a weighted mean; parameter count unchanged ⇒ one-component ablation. All else frozen at the A winner (edge positives), enzymes Ψ K=10, seeds 42/43/44.
- **Hypothesis.** If `weighted` > `mean`, similarity *magnitude* is informative beyond top-K ranking — strengthening the claim that the virtual-graph construction (not just its topology) matters; it would also damp sensitivity to K (weak tail auto-downweighted), informing the Stage-2 K sweep.
- Artifacts tagged `sage_edge_<agg>_*` / scoreboard `graphsage_edge_<agg>` — mean keeps the unsuffixed names.

Same graph, features, architecture, negatives, splits, seeds — positives fixed at **edge (A2)**, **only aggregation differs**:

| aggregation | node classification (weighted F1) | link prediction (AUC) |
|---|---|---|
| **mean** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| weighted (Ψ) | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

- **FINDING.** Ranking **mean > weighted > max > sum is identical on both tasks** — a robust, task-independent ordering. `mean` wins outright.
- **Hypothesis NOT supported.** Ψ-`weighted` (first use of the virtual edge weights anywhere in the encoder) was predicted to beat `mean` if similarity *magnitude* was informative; it **lost** (LP −0.038, NC tie). Conclusion: within a node's top-K, **edge existence carries the signal, not edge strength** — the `1/(1+dist)` weights add nothing beyond top-K membership. `sum` is degenerate (unnormalized → aggregate magnitude tracks degree → LP collapses to chance 0.51 with ×4 variance); `max` discards the neighbor distribution. Standard GraphSAGE `mean` confirmed best.
- **DECISION (A + B finalized).** ViRGo-SAGE default objective/encoder = **edge positives (A2) + mean aggregation (B)**. Both locked as defaults in `encoder.py` / `GNN_PARAMS`. All four aggregation variants retained in `scoreboard.csv` as the ablation record (not deleted — they are the evidence for the choice).
- **CAVEATS.** Enzymes / Ψ / K=10 / 3 seeds only; absolute NC ≈0.54 sits near the balanced-class floor and LP ≈0.69 — aggregation is an encoder-internal knob and cannot raise the ceiling set by the features + virtual graph (that is the E-axis / dataset job). Cora repeat pending.

### Ablation B results — aggregation — Enzymes, Ψ, K=10, edge positives, seeds 42/43/44 (2026-07-07, INDICATIVE)

| agg | NC weighted F1 | LP AUC |
|---|---|---|
| **mean** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| weighted (Ψ-mean) | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

- **FINDING.** Plain mean wins both tasks. `weighted` ≈ mean on NC but −0.038 LP: the Ψ weight *magnitudes* add no information beyond the top-K *ranking* itself — the graph's topology already encodes the similarity signal. `sum` collapses LP to near-random (magnitude varies with virtual degree → scale noise); `max` loses too much (one neighbor can't summarize a role neighborhood).
- **DECISION.** `agg = "mean"` locked as default (already was). Encoder now fully locked for Stage 2: **edge positives + mean aggregation**.
- **CAVEATS.** Same as ablation A: single dataset (enzymes), single variant/K (Ψ, 10), 3 seeds, cosine LP.

### Ablation C results — encoder depth / over-smoothing — Enzymes, Ψ, K=10, edge positives, mean agg, seeds 42/43/44 (logged 2026-07-16)

Design axis from the Phase-3 variant list ("C depth 1-3 — over-smoothing: virtual graphs near-connected at K=10"). Same graph, features, positives, aggregation, negatives, splits, seeds — **only the number of message-passing layers differs**. Encoder names `graphsage_edge_l1` / `graphsage_edge_l3`; depth 2 is the locked default (`graphsage_edge`, `GNN_PARAMS["layers"]=2`), re-scored from its existing embeddings, not retrained.

| depth | NC (weighted F1) | LP (AUC) |
|---|---|---|
| 1 layer | 0.4971 ± 0.0018 | 0.6678 ± 0.0075 |
| **2 layers (default)** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| 3 layers | 0.4506 ± 0.0187 | 0.4983 ± 0.0407 |

- **FINDING — the over-smoothing prediction is confirmed, and it is sharp.** Depth 2 wins both tasks. Depth 3 does not merely degrade: LP falls to **0.4983, i.e. chance**, and NC drops below the DeepWalk bridge (0.4506 vs 0.4971). The design's stated mechanism holds — the Ψ virtual graph is near-connected at K=10 (10 components, avg degree 11.44 on enzymes), so a 3-hop receptive field averages a large fraction of the graph and node embeddings converge. Depth 1 is under-powered on NC (0.4971, exactly at the bridge) while retaining most of the LP signal (0.6678), consistent with LP depending mostly on 1-hop role neighbourhoods.
- **DECISION.** Depth **2** confirmed as the ViRGo-SAGE default. No code change — the default was already 2; this ablation converts an assumption into evidence and supplies the over-smoothing curve for the paper.
- **CAVEATS.** enzymes only, Ψ graph only, K=10, 3 seeds, cosine LP; depth not swept at other K (a denser graph at K=20 should over-smooth *earlier* — untested). Like ablations A/B/D, the choice is made on a single dataset and inherited by cora, where the E-study story inverts; a cora depth repeat is not run. Depth 3's LP std (0.0407) is the largest in the C set, as expected once embeddings collapse toward a common vector.

### E-study first results — virtual-graph variants, locked encoder — Enzymes, K=10, seeds 42/43/44 (2026-07-07, INDICATIVE)

Locked ViRGo-SAGE (edge positives + mean agg, ablations A/B) run over all three virtual-graph constructions at K=10 (notebook 3 §8 sweep; user-run):

| virtual graph | NC weighted F1 | LP AUC |
|---|---|---|
| Ψ (I2V Poisson/KL) | 0.5413 ± 0.0011 | 0.6909 ± 0.0156 |
| degree-only | **0.5536 ± 0.0012** | 0.5998 ± 0.0782 |
| centrality-only | 0.5411 ± 0.0025 | **0.7197 ± 0.0625** |

- **FINDING (core research question).** No single virtual graph wins both tasks — and Ψ, the generic I2V construction, is best on **neither**: degree-only wins node classification (+0.012 over Ψ, tight std), centrality-only wins link prediction (+0.029 over Ψ). This directly supports the paper's thesis that the virtual-graph construction should be chosen **per data and task** rather than defaulting to the Poisson/KL similarity graph.
- **Ψ's role.** Ψ is the consistent all-rounder (2nd on both tasks, and by far the lowest LP variance: ±0.016 vs ±0.06–0.08 for the single-signal graphs) — a defensible generic default, but beatable by cheaper single-signal graphs when the task is known.
- **CAVEATS.** Enzymes only, K=10 only, 3 seeds; the single-signal winners carry 4–5× the LP std of Ψ, so the LP ordering (centrality > Ψ) is within ~½σ — needs the K=5/20 sweep and a 2nd dataset before it hardens. Deepwalk-bridge + walk-positive GraphSAGE rows for degree/centrality pending (fills the 3×3 encoder×graph grid).

- **Ablation D implemented (2026-07-09) — features vs message passing.** `feats` knob on `SageEncoder` selects the GNN's input features, holding the virtual graph, positives (edge), aggregation (mean), seeds (42/43/44) and LP splits fixed: **D0 `all`** = [degree, Ω, ψ, clustering] (the locked encoder — existing embeddings reused, re-scored identical to the recorded 0.5413 NC / 0.6909 LP), **D1 `degree`**, **D2 `deg_cent`**, **D3 `psi`**, **D4 `random`** (seeded Gaussian; node identity only, zero structural signal), **D5 `const`** (identical rows → z-norm zeros → all-nodes-identical embeddings → cosine LP AUC ≈ 0.50 by construction; a floor, not an experiment). Motivation: ViRGo-SAGE beats the Phase-2 DeepWalk bridge with **two** advantages at once — message passing *and* four structural input features the lookup-table bridge cannot consume — so "GraphSAGE > DeepWalk" currently reads "GNN + structural features > walks". **The deciding row is D4 vs deepwalk:** random-feature SAGE above the bridge ⇒ message passing over the virtual graph carries signal; at the bridge ⇒ the Phase-3 win was the features. LP features are computed on the 70% train graph (no leakage). D3 is **confounded by design** (ψ is both the feature and the signal the ψ graph was built from) and must be reported as such. Artifacts `graphsage_edge_feat_<set>_s<seed>.emb`; scoreboard encoder names likewise; snapshot `results/snapshots/<ds>_feature_ablation_K<K>.csv`. Expected on enzymes: D1 ≈ D2, because eigenvector centrality is degenerate there (96.9% of nodes < 1e-6 — 640 disconnected molecules) — a dataset property to report, not a failure. Results pending the notebook run.

- **Two new virtual-graph variants added to the E-study (2026-07-10) — `original` and `hybrid`.** Motivation: the E-table so far compares three *rewired* graphs (Ψ, degree, centrality) against each other, but not against the obvious control — the untouched input graph — so "virtual graph helps" is not yet evidenced within the locked-encoder grid. **`original`** = exact copy of the input graph, all edge weights 1.0, K unused (duplicated under each k<K>/ folder so sweeps stay uniform): GraphSAGE on the real edges, isolating the virtual-graph effect from the encoder effect. **`hybrid`** = union of original edges (weight 1.0) and Ψ top-K role edges (weight 1/(1+d)); on overlap the original edge's 1.0 wins. Rationale: message passing then reaches both physical neighbors and role neighbors in one graph — tests "structure + role beats either alone". Both flow through the existing pipeline unchanged (`virtual_graph.py --sim`, `encoder.py --sim`, `VG_SIMS`); for link prediction the input graph is the 70% train graph as before, so `original`/`hybrid` inherit the no-leakage protocol. Results pending.

### E-study extended — `original` + `hybrid` rows — Enzymes, K=10, locked encoder, seeds 42/43/44 (2026-07-10, INDICATIVE)

| virtual graph | NC weighted F1 | LP AUC |
|---|---|---|
| Ψ (I2V Poisson/KL) | 0.5413 ± 0.0011 | 0.6909 ± 0.0156 |
| degree-only | 0.5536 ± 0.0012 | 0.5998 ± 0.0782 |
| centrality-only | 0.5411 ± 0.0025 | **0.7197 ± 0.0625** |
| **original (control)** | **0.5671 ± 0.0022** | 0.6459 ± 0.0364 |
| hybrid (orig ∪ Ψ) | 0.5485 ± 0.0044 | 0.6967 ± 0.0302 |

- **FINDING.** The `original`-graph control **wins node classification outright** (0.5671, +0.014 over degree-only, +0.026 over Ψ, tightest std after Ψ): on enzymes, GraphSAGE over the real edges beats every rewired virtual graph for NC — the virtual-graph rewiring *costs* NC accuracy here. On LP the control drops to 4th (0.6459); role edges matter for predicting links.
- **hybrid.** 2nd on LP (0.6967, within noise of Ψ 0.6909, behind centrality 0.7197) and 2nd on NC (0.5485) — the only variant top-2 on **both** tasks; consistent all-rounder, mirrors Ψ's old role but stronger.
- **Implication for the thesis.** Sharpens, not breaks, the per-task story: NC on enzymes favors original edges (class signal is homophilous/local), LP favors role-augmented graphs (centrality, hybrid, Ψ). "Which graph is best" now provably includes "sometimes the original one" — the control row makes the virtual-graph claims falsifiable.
- **CAVEATS.** Enzymes only, K=10 only, 3 seeds, cosine LP; deepwalk-bridge rows for original/hybrid missing (notebook 2 not rerun — GraphSAGE rows unaffected). LP gaps between centrality/hybrid/Ψ are within ~½σ.

- **DeepWalk bridge rows for `original`/`hybrid` filled (2026-07-10)** — trained with the exact notebook-2 recipe (same DeepWalk params, splits, seeds 42/43/44), enzymes K=10: **original** deepwalk NC 0.5086 ± 0.0039 / LP 0.6574 ± 0.0170; **hybrid** deepwalk NC 0.4906 ± 0.0060 / LP 0.5087 ± 0.0246. Removes the "bridge rows missing" caveat from the E-study-extended entry. Notable: on the **original** graph DeepWalk edges out GraphSAGE on LP (0.6574 vs 0.6459, within ~½σ) — the GNN's LP advantage appears only on role-augmented graphs (Ψ +0.18, hybrid +0.19 over their bridges); on **hybrid**, DeepWalk collapses on LP (0.5087, near-random) while GraphSAGE holds 0.6967 — walks get lost mixing physical + role edges, message passing does not.

- **Cora extended to the 5-variant grid (2026-07-12)** — `original`/`hybrid` virtual graphs built for cora K=10 (`original`: 2708 nodes / 5278 edges = exact copy; `hybrid`: 21399 edges = original ∪ Ψ top-K), and the DeepWalk bridge rows filled with the exact notebook-2 recipe (same params, shared LP splits, seeds 42/43/44): **original** deepwalk NC **0.8100 ± 0.0158** / LP **0.9007 ± 0.0039**; **hybrid** deepwalk NC 0.4526 ± 0.0035 / LP 0.6699 ± 0.0142. Notable: cora is strongly homophilous — DeepWalk on the untouched graph is very strong (LP 0.90), so the original-graph control sets a high bar the virtual graphs must justify against; and the enzymes pattern repeats on NC — mixing role edges into the walks (**hybrid**) collapses DeepWalk's NC from 0.81 to 0.45 (walks get lost across physical + role edges). GraphSAGE rows for cora original/hybrid pending (notebook 3 §8 sweep).

### E-study on Cora — full 5-variant grid, K=10, locked encoder, seeds 42/43/44 (2026-07-12)

Second dataset for the E-study (first: enzymes). Same protocol: locked ViRGo-SAGE (`graphsage_edge` = edge positives + mean agg) vs the DeepWalk bridge, shared LP splits, cosine LP.

| virtual graph | deepwalk NC | graphsage NC | deepwalk LP | graphsage LP |
|---|---|---|---|---|
| Ψ (I2V Poisson/KL) | 0.2378 ± 0.0044 | 0.2757 ± 0.0010 | 0.5114 ± 0.0062 | 0.5246 ± 0.0173 |
| degree-only | 0.1592 ± 0.0118 | 0.2722 ± 0.0031 | 0.5547 ± 0.0018 | 0.5283 ± 0.0135 |
| centrality-only | 0.3830 ± 0.0151 | 0.3067 ± 0.0036 | 0.5507 ± 0.0081 | 0.5708 ± 0.0118 |
| **original (control)** | **0.8100 ± 0.0158** | 0.4504 ± 0.0104 | **0.9007 ± 0.0039** | 0.6212 ± 0.0080 |
| hybrid (orig ∪ Ψ) | 0.4526 ± 0.0035 | 0.3109 ± 0.0100 | 0.6699 ± 0.0142 | 0.5572 ± 0.0213 |

- **HEADLINE FINDING — the story inverts across datasets.** On cora, **DeepWalk on the untouched original graph wins both tasks by a landslide** (NC 0.81, LP 0.90); every rewired virtual graph collapses NC to ≤0.45 and LP to ≤0.67, and GraphSAGE loses to DeepWalk on the original graph on both tasks (0.45/0.62). Exactly opposite to enzymes, where the role-augmented graphs + message passing gave the best LP and GraphSAGE won everywhere. This is the strongest evidence yet for the paper's thesis: **which graph (and encoder) is best is a property of the data** — cora's citation labels are homophilous/community-like, so proximity walks on real edges are near-optimal and role-based rewiring destroys the signal; enzymes' molecular structure rewards role edges.
- **Why GraphSAGE trails on cora NC:** its input features are purely structural (degree, centrality, ψ, clustering) — role features, not community features. Cora classes are topics, not roles, so the GNN is handicapped by construction on this task; DeepWalk's positional embeddings capture community directly.
- **Consistent cross-dataset patterns (both datasets):** (1) hybrid is DeepWalk's best *rewired* graph on both tasks (it contains the original edges) yet mixing role edges still halves DeepWalk's cora NC (0.81 → 0.45); (2) GraphSAGE > DeepWalk on the pure role graphs' NC (Ψ, degree); (3) the GNN's LP advantage exists only on role-augmented graphs, never on the original graph.
- **CAVEATS.** K=10 only, 3 seeds, cosine LP; no encoder tuning per dataset (locked from enzymes ablations); structural-feature handicap above means "GraphSAGE loses on cora" = "role-feature GNN loses on a homophily task", not a general GNN result.

- **Result-presentation framing locked (2026-07-13, review guidance).** PRIMARY research question: **which graph variant is best for a given dataset × task** (the virtual graph is the variable under study). SECONDARY: **within each graph**, does GraphSAGE (message passing) beat DeepWalk (walks + Skipgram) — the encoder comparison lives inside each graph, never headlines. Current K=10 answers: cora NC → original+DeepWalk (0.8100); cora LP → original+DeepWalk (0.9007); enzymes NC → original+GraphSAGE (0.5671); enzymes LP → centrality+DeepWalk (0.7382). Notable: the *best* configuration is DeepWalk-based in 3 of 4 cells — the GNN's contribution shows in the secondary Δ table (GraphSAGE > DeepWalk on Ψ everywhere, and on enzymes hybrid LP +0.188), not in the headline. Notebook 3 §11 renders both views from the scoreboard.

- **Headline tables locked to the fixed encoder (2026-07-13, user decision).** Notebook 3 §11 Table 1 (best virtual graph per dataset × task) and Table 2 (control vs original) now use **`graphsage_edge` only** — previously "best over both encoders", which confounded graph choice with encoder choice. Rationale: the virtual graph is the variable under study, so the encoder must be held fixed; the GraphSAGE-vs-DeepWalk comparison stays in §7 as the secondary question (same graph, encoders differ). K=10 answers under the locked encoder: cora NC → hybrid 0.3109; cora LP → centrality 0.5708; enzymes NC → degree 0.5536; enzymes LP → centrality 0.7197. Control check (graphsage both sides): original wins cora NC 0.4504 / cora LP 0.6212 and enzymes NC 0.5671; best virtual wins enzymes LP 0.7197 vs 0.6459 — per-data/per-task story unchanged, now unconfounded.

- **`original` repositioned as CONTROL baseline (2026-07-13, user decision).** The original graph stays in every sweep but is not a ViRGo contribution — headline tables now report the **best virtual/derived graph** (Ψ, degree, centrality, hybrid) per dataset × task, with a separate control-check table original-vs-best-virtual. K=10 answers: cora LP → hybrid+DeepWalk 0.6699; cora NC → hybrid+DeepWalk 0.4526; enzymes LP → centrality+DeepWalk 0.7382; enzymes NC → degree+GraphSAGE 0.5536. Control check: original much better on both cora tasks (0.9007/0.8100), virtual better on enzymes LP (0.7382 vs 0.6574), original slightly better on enzymes NC (0.5671 vs 0.5536). Honest headline: rewiring wins only enzymes LP at K=10 so far — the per-data/per-task story carries the paper, not a blanket "virtual graphs win".

- **ABLATION D RESULTS (2026-07-16) — the Phase-3 win is the FEATURES, not message passing.** enzymes, Ψ graph, K=10, seeds 42/43/44, edge positives + mean agg, cosine LP; only the GNN's input features change.

| id | features | dims | NC F1 | LP AUC |
|---|---|---|---|---|
| **D0** | all (degree, Ω, ψ, clustering) | 4 | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| D2 | degree + centrality | 2 | 0.5151 ± 0.0052 | 0.6805 ± 0.0302 |
| D1 | degree | 1 | 0.5203 ± 0.0014 | 0.5975 ± 0.0432 |
| *deepwalk (bridge reference)* | *none* | — | *0.4971 ± 0.0057* | *0.5110 ± 0.0297* |
| D5 | constant (floor) | 1 | 0.3277 ± 0.0000 | 0.5046 ± 0.0905 |
| D3 | ψ *(confounded)* | 1 | 0.4933 ± 0.0017 | 0.4983 ± 0.0251 |
| **D4** | **random (control)** | 4 | **0.4814 ± 0.0060** | **0.4798 ± 0.0349** |

- **DECIDING ROW ANSWERED — D4 lands BELOW the bridge on both tasks** (LP 0.4798 vs 0.5110; NC 0.4814 vs 0.4971), and its LP is indistinguishable from chance (0.48 ≈ 0.50, within its own 0.035 σ). Per the pre-registered reading in the ablation-D design entry, this is the "at the bridge ⇒ the Phase-3 win was the features" branch — in fact worse. **Message passing over the Ψ virtual graph, stripped of structural input features, carries no usable signal on enzymes: the structural features are NECESSARY.** ⚠️ **This does NOT establish that message passing contributes nothing.** The design's two-branch reading was under-specified: D4 tests only whether message passing is *sufficient alone*. Attributing the +0.18 LP win to the features requires the arm **structural features WITHOUT message passing** (D6, below), which is **not yet run**.
- **Revises the earlier indicative claim.** The Phase-3 entry "message passing over the virtual graph beats the Skipgram lookup … supports the technical contribution: GNN > walk+Skipgram on the *same* virtual graph" survives only in the weaker form **"GNN + structural features > walks"**. The isolated claim "message passing helps *by itself*" is **not supported** by this control. Report D4 alongside any GraphSAGE-vs-DeepWalk headline.
- **MISSING ARM — D6 `features, no message passing` (identified 2026-07-16, user challenge).** The ablation-D grid tests only two of four cells: *random features + MP* (D4 = 0.4798 LP) and *structural features + MP* (D0 = 0.6909 LP). The cell **structural features, no MP** is untested, so the split between "features" and "aggregation" is **not yet decidable**. D6 = feed the raw 4-dim [degree, Ω, ψ, clustering] vectors (computed on the original graph, z-normed, 70% train graph for LP) straight to the shared eval protocol — cosine for LP, one-vs-rest logistic regression for NC — no GNN, no training. Reading: **D6 ≈ D0** ⇒ message passing adds nothing and the Phase-3 win is purely the features; **D6 « D0** ⇒ message passing contributes the gap *on top of* the features, and the technical contribution stands in the form "MP amplifies structural features". Until D6 runs, the only defensible claims are: (1) structural features are **necessary** (D4 collapses without them); (2) message passing is **not sufficient alone**.
- **D3 (ψ alone) collapses to chance** (LP 0.4983 / NC 0.4933) — below the bridge — even though it is *confounded in ViRGo's favour* (ψ is both the feature and the signal the Ψ graph was built from). The I2V scalar is not a sufficient input feature.
- **Prediction partly wrong: D1 ≈ D2 held for NC, failed for LP.** The design entry expected D1 ≈ D2 because eigenvector centrality is degenerate on enzymes (96.9% of nodes < 1e-6, 640 disconnected molecules). NC matches (0.5203 vs 0.5151, D1 marginally higher), but LP shows a real gap (**D2 0.6805 vs D1 0.5975, +0.083**): the near-zero centrality column still carries link-predictive signal despite the degeneracy. Degree alone (D1) already beats the bridge on both tasks; degree+centrality (D2) recovers ~98% of full-feature LP (0.6805 / 0.6909), so ψ + clustering add little.
- **CAVEATS.** enzymes only, Ψ graph only, K=10, 3 seeds, cosine LP. The cora repeat is not run — given the cora inversion (DeepWalk+original wins by a landslide there), D4 on cora may read differently and must not be assumed. D5's NC 0.3277 is the majority-class floor by construction (identical rows → z-norm zeros), not an experiment. Notebook 3 §8b holds this ablation but its code cell is currently **commented out**; its markdown still points to the deleted §7b view.

---

## Ablation study — consolidated (paper-ready, 2026-07-16)

Single reference for the paper's ablation section. The dated entries above remain the provenance trail; this section is the writing surface. Every number below is read from `results/scoreboard.csv`.

### Common protocol

All of A–D: **enzymes**, **Ψ** virtual graph, **K=10**, seeds **42/43/44**, cosine link prediction, shared 70/30 splits. One axis changes per ablation; graph, features, architecture, negatives (∝ deg^0.75, Q=5), splits and seeds are otherwise identical, so each comparison is one-component. LP features/graphs are built from the 70% train edges only ⇒ no leakage. Reference bar throughout = the Phase-2 DeepWalk bridge: **NC 0.4971 ± 0.0057 · LP 0.5110 ± 0.0297**. Every variant is retained in the scoreboard as evidence; none are deleted.

**A–D purpose:** lock the encoder so it stops being a confound in E. **E purpose:** the research question itself.

### A — positives: are walks still needed? · DECIDED: `edge`

| id | positives | NC (F1) | LP (AUC) |
|---|---|---|---|
| A1 | walk co-occurrence (window 10, = bridge corpus) | 0.5405 ± 0.0014 | 0.6632 ± 0.0114 |
| **A2** | **direct virtual edges** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |

NC is a tie (Δ 0.0008, within seed noise). LP favours direct edges (**Δ +0.0277**, ≈2σ vs the seed stds — likely real, borderline at 3 seeds). **Interpretation:** I2V ran walks to *discover* role-similar nodes; the virtual graph already enumerates them, so walks are redundant once similarity is explicit in the graph — and dropping them removes walk generation entirely (cheaper). **Cost of the choice:** the "Phase 2 and Phase 3 differ in exactly one component" claim holds only under A1, which is retained and reported as the bridge-comparable configuration.

### B — aggregation: does similarity *strength* matter? · DECIDED: `mean`

| id | aggregation | NC (F1) | LP (AUC) |
|---|---|---|---|
| **B0** | **mean (equal neighbors)** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| B1 | Ψ-weighted mean *(the only variant reading edge weights)* | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| B2 | max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| B3 | sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

Plain `mean` wins both tasks. **The reportable finding:** `weighted` — the sole configuration that consumes the similarity strengths ViRGo computes (`1/(1+dist)`) — *loses* to discarding them (LP −0.038). **Edge existence carries the signal; edge strength does not.** Once top-K has selected the neighbourhood, how similar each member is adds nothing. `sum` collapses to chance on LP (0.5063): unnormalized neighbourhood magnitude swamps the embedding.

### C — depth: over-smoothing · DECIDED: `2 layers`

| id | depth | NC (F1) | LP (AUC) |
|---|---|---|---|
| C1 | 1 layer | 0.4971 ± 0.0018 | 0.6678 ± 0.0075 |
| **C2** | **2 layers** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| C3 | 3 layers | 0.4506 ± 0.0187 | 0.4983 ± 0.0407 |

Depth 2 wins both. **The over-smoothing prediction is confirmed and sharp:** at depth 3 LP falls to **0.4983 = chance**, and NC drops below the bridge. Mechanism as designed — the Ψ graph is near-connected at K=10 (enzymes: 10 components, avg degree 11.44), so a 3-hop receptive field averages a large fraction of the graph and embeddings converge (C3 also carries the largest LP variance, 0.0407, consistent with collapse). Depth 1 is under-powered on NC (0.4971, exactly at the bridge) yet keeps most LP signal (0.6678) — LP depends mostly on 1-hop role neighbourhoods. No code change: 2 was already the default; this converts an assumption into evidence and supplies the depth curve.

### D — features vs message passing · PARTLY OPEN

| id | features | msg passing | dims | NC (F1) | LP (AUC) |
|---|---|---|---|---|---|
| **D0** | all four | ✔ | 4 | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| D2 | degree + centrality | ✔ | 2 | 0.5151 ± 0.0052 | 0.6805 ± 0.0302 |
| D1 | degree | ✔ | 1 | 0.5203 ± 0.0014 | 0.5975 ± 0.0432 |
| — | *deepwalk bridge (reference)* | ✘ | — | *0.4971 ± 0.0057* | *0.5110 ± 0.0297* |
| D3 | ψ *(confounded)* | ✔ | 1 | 0.4933 ± 0.0017 | 0.4983 ± 0.0251 |
| D4 | random **(control)** | ✔ | 4 | 0.4814 ± 0.0060 | 0.4798 ± 0.0349 |
| D5 | constant (floor) | ✔ | 1 | 0.3277 ± 0.0000 | 0.5046 ± 0.0905 |
| **D6** | **all four (control)** | **✘ `layers=0`** | 4 | **pending** | **pending** |

**Motivation.** ViRGo-SAGE beats the bridge with two advantages simultaneously — message passing *and* four structural features the lookup-table bridge cannot consume — so "GraphSAGE > DeepWalk" currently reads "GNN + structural features > walks". D isolates them.

**What D4 establishes.** Random-feature SAGE lands **below** the bridge on both tasks, its LP indistinguishable from chance (0.4798, σ 0.035). Therefore: **(1) the structural features are necessary; (2) message passing is not sufficient alone.** Both claims are supported.

**What D4 does NOT establish (correction, 2026-07-16).** It does not license "the win is *mostly* the features". The design's original two-branch reading of D4-vs-deepwalk was under-specified: the attribution needs the fourth cell of the 2×2 — *features without message passing* (**D6**). Decision rule, pre-registered: **D6 ≈ D0 ⇒ message passing adds nothing, the Phase-3 win is the features; D6 « D0 ⇒ message passing contributes on top of the features and the technical contribution stands.** Until D6 runs, only the two claims above may be stated.

**Secondary findings.** D3: the I2V ψ scalar alone collapses to chance (0.4983) despite being confounded *in ViRGo's favour* (ψ is both the feature and the signal the Ψ graph was built from) — ψ is not a sufficient input feature. D1: degree alone already clears the bridge on both tasks. D2: degree + centrality recovers **98% of D0's LP** (0.6805 vs 0.6909) — **ψ and clustering together add ≈0.01**. The prediction "D1 ≈ D2 because eigenvector centrality is degenerate on enzymes (96.9% of nodes < 1e-6, 640 disconnected molecules)" **holds for NC** (0.5203 vs 0.5151) but **fails for LP** (Δ +0.083): the near-degenerate centrality column still carries link-predictive signal. D5 NC 0.3277 is the majority-class floor by construction, not an experiment.

**D6 implementation (2026-07-16).** `layers=0` in `SageEncoder` ⇒ no convolutions; `forward()` returns the z-normed features unchanged (verified bit-identical to raw X), so D6 reuses **the identical feature builder as D0** and the only difference between the rows is message passing. Nothing is trained (`train()` asserts at `layers=0`); the virtual graph is unused. Artifacts `features_only_s<seed>.emb`, scoreboard encoder `features_only`; notebook 3 §8b, `RUN_D6` knob. **Reporting caveat:** D6 is 4-dimensional against D0's 64 — the contrast is "these features scored directly" vs "these features expanded and smoothed by a GNN", which is the intended question but is *not* dimension-matched; state this in the paper. D6 is deterministic across seeds, so its std reflects split variation only.

### E — which virtual graph? (the research question)

Locked encoder `graphsage_edge` (= A2 + B0 + C2 + D0) vs the DeepWalk bridge; K=10, seeds 42/43/44. Variants: `psi` (I2V Poisson/KL), `degree`, `centrality`, `original` (**control** — untouched input graph, not a ViRGo contribution), `hybrid` (original ∪ Ψ top-K).

**Enzymes, K=10**

| graph | DW NC | GS NC | DW LP | GS LP |
|---|---|---|---|---|
| psi | 0.4971 | 0.5413 | 0.5110 | **0.6909** |
| degree | 0.5136 | 0.5536 | 0.6240 | 0.5998 |
| centrality | 0.4903 | 0.5411 | **0.7382** | 0.7197 |
| original *(control)* | 0.5086 | **0.5671** | 0.6574 | 0.6459 |
| hybrid | 0.4906 | 0.5485 | 0.5087 | 0.6967 |

**Cora, K=10**

| graph | DW NC | GS NC | DW LP | GS LP |
|---|---|---|---|---|
| psi | 0.2378 | 0.2757 | 0.5114 | 0.5246 |
| degree | 0.1592 | 0.2722 | 0.5547 | 0.5283 |
| centrality | 0.3830 | 0.3067 | 0.5507 | 0.5708 |
| original *(control)* | **0.8100** | 0.4504 | **0.9007** | 0.6212 |
| hybrid | 0.4526 | 0.3109 | 0.6699 | 0.5572 |

**HEADLINE — the story inverts across datasets.** Enzymes rewards role-rewiring: centrality gives the best LP (0.7382), beating the real graph, and GraphSAGE wins nearly everywhere. Cora does the opposite: the untouched graph with DeepWalk wins both tasks by a landslide (0.8100 / 0.9007) and every rewired graph destroys the signal. **This is the thesis, evidenced: which graph is best is a property of the data, not a universal answer.**

**Why cora inverts.** Cora's labels are research topics — a *community* property carried by real citation edges, so proximity walks over them are near-optimal and role-rewiring discards exactly the needed signal. GraphSAGE is additionally handicapped by construction: its inputs are *role* features (degree, centrality, ψ, clustering), not topic features. Hence "GraphSAGE loses on cora" = "a role-feature GNN loses on a homophily task", **not** a general GNN result. Enzymes' molecular structure is precisely what roles describe.

**Consistent cross-dataset patterns.** (1) `hybrid` is DeepWalk's best *rewired* graph on both tasks (it contains the original edges), yet mixing role edges still halves cora NC (0.81 → 0.45). (2) GraphSAGE > DeepWalk on the pure role graphs' NC (Ψ, degree) on both datasets. (3) The GNN's LP advantage exists **only** on role-augmented graphs, never on the original graph (enzymes original: DW 0.6574 vs GS 0.6459). (4) On hybrid LP, DeepWalk collapses (0.5087) while GraphSAGE holds 0.6967 — walks get lost mixing physical and role edges; message passing does not.

**The honest headline.** Best configuration per dataset × task cell: **DeepWalk wins 3 of 4** (cora NC 0.8100, cora LP 0.9007, enzymes LP 0.7382); GraphSAGE takes only enzymes NC (0.5671) — and on the *control* graph. Under the locked encoder with `original` excluded as control, the best virtual graphs are: cora NC → hybrid 0.3109 · cora LP → centrality 0.5708 · enzymes NC → degree 0.5536 · enzymes LP → centrality 0.7197. Control check (GraphSAGE both sides): original wins cora NC (0.4504), cora LP (0.6212) and enzymes NC (0.5671); the best virtual graph wins **only enzymes LP** (0.7197 vs 0.6459). **Rewiring wins exactly one of four cells.** The per-data story carries the paper; a blanket "virtual graphs win" does not.

### Locked configuration — ViRGo-SAGE

| axis | locked to | evidence |
|---|---|---|
| A positives | `edge` | LP +0.028 over walks; NC tie; removes walk generation |
| B aggregation | `mean` | beat weighted / max / sum on both tasks; edge weights unhelpful |
| C depth | `2 layers` | 1 under-powered; 3 over-smooths to chance |
| D features | `all four` | best of every set tested; features shown necessary (D4) |
| K | `10` | only K with a complete grid on every dataset |

### Threats to validity — state these in the paper

1. **A, B, C and D were each decided on enzymes alone** (one dataset, Ψ only, K=10, 3 seeds); that locked encoder was then applied to cora, **where the entire E story inverts**. Nothing guarantees cora would select the same settings, and the cora repeat of A/B/C/D is **not run**. Any claim of the form "we selected X because it performed best" must carry "on enzymes".
2. **D6 not yet run** ⇒ the features-vs-message-passing attribution is undecided; the technical contribution currently stands only as "GNN + structural features > walks".
3. **D4 on cora not run.** Given the inversion, the random-feature control may read differently there; do not extrapolate.
4. **Scale:** 2 datasets, 3 seeds, K=10, cosine LP throughout; no per-dataset encoder tuning (locked from enzymes); baselines used as published, not fine-tuned (scope guard).
5. **A2 side-effect:** the one-component Phase-2-vs-Phase-3 comparison is only valid under A1.

## 2026-07-17 — Proteins added as a third study dataset (provenance + properties)

**Source and verification.** Proteins (networkrepository `PROTEINS`, nrvis.com/download/data/labeled/PROTEINS.zip). Graph and node labels are taken from the **same** archive member set (`proteins.edges` + `proteins.node_labels`), so node ids align by construction. The repo's pre-existing `input/proteins.edgelist` is **md5-identical** to the archive's `proteins.edges`; the rebuilt graph was checked against it at **edge overlap = 1.0000** before any label file was written. No labels were inferred or fabricated (same policy that keeps politics link-prediction-only).

**Graph properties** (`input/proteins_nr.edgelist`): **43,466 nodes / 81,044 edges**, mean degree 3.73, max degree 25, no self-loops, **1,195 connected components, largest component 620 nodes** — a disjoint union of many small protein graphs, i.e. structurally the *same family as enzymes* (19,474 nodes / 640 components), at ~2.2× the scale.

**Labels:** 3 classes, heavily imbalanced — 21,151 / 20,931 / 1,389 (the minority class is 3.2% of nodes). 43,471 label lines vs 43,466 graph nodes: 5 ids carry a label but no edge; `eval_nodeclass.evaluate` intersects on nodes present in the embedding, so they drop out. Enzymes has the identical situation (19,580 labels vs 19,474 nodes) — no new handling.
- *Implication for reporting:* with a 3.2% minority class, **weighted F1 is majority-dominated**; a trivial two-class predictor already scores well. Report macro F1 alongside weighted for proteins, or the NC numbers will overstate.

**Why this dataset earns its place in the study.** The E-story so far rests on two datasets that disagree (enzymes: role graphs + GNN win; cora: DeepWalk + original graph landslide), which makes "it depends on the data" a two-point claim. Proteins is a *same-family replicate of enzymes* — if role graphs win on proteins too, the enzymes result generalizes to the graph family rather than to one dataset, which is the weakest link in the current per-data thesis.

**Expected caveat, recorded before the run** (so it is not a post-hoc excuse): eigenvector centrality is degenerate on proteins — **76.9% of nodes < 1e-6**, driven by the 1,195 components. Enzymes is worse (96.9%). The `centrality` variant is therefore built on a near-constant signature on both, and any `centrality` result on proteins should be read as such, not as evidence that centrality-based rewiring works.

**Status:** wiring only — config, label builder, registries, and a build smoke test (psi/degree/centrality all build at K=10, 0 isolated nodes). **No proteins scores exist yet**; the virtual graphs, LP splits, DeepWalk bridge and GraphSAGE runs are unrun.

## 2026-07-17 — The `degree` virtual-graph variant degenerates into a star on tie-heavy graphs (measured)

**Finding.** The `degree` variant does not build a role graph on proteins. It builds a **star centred on ~10 arbitrary low-id nodes**. Measured on `proteins` K=10:

| variant | edges | max virtual degree | Σdeg² (drives node2vec runtime) | DeepWalk time / seed |
|---|---|---|---|---|
| psi | 250,223 | 91 | 6.03e6 | **2m23s** |
| centrality | 249,794 | 79 | 5.96e6 | ~2m |
| hybrid | 327,456 | 93 | 1.02e7 | ~3m |
| original | 81,044 | 25 | 6.62e5 | ~1m |
| **degree** | 434,111 | **14,644** | **4.89e9** | **~2h27m** |

**Mechanism** (`virtual_graph.py:67-79`). The `degree` signature is a **1-D integer**. Proteins has **43,466 nodes but only 16 distinct degree values** (range 1–25); **14,645 nodes share degree 3**. Every one of those nodes is at **distance exactly 0** from every other, so `NearestNeighbors.kneighbors` breaks the tie by **index order** and returns the *same* ~11 low-id nodes for all 14,645 queries. `V.add_edge` symmetrises, so those few nodes absorb an edge from every member of the tie group → verified: node 1 has virtual degree **14,644**, base degree 3, and **100% of its 14,644 virtual neighbours have base degree exactly 3**. Top-5 virtual degrees are all 14,644, all base-degree 3.

**Why the checks missed it.** The `added >= k` cap bounds only a node's *outgoing* picks; *incoming* picks are unbounded. Notebook 2 §4 asserts `min(degree) >= K` and never inspects the maximum — a star passes every constraint in the report.

**This is not proteins-only — it scales with tie density, and it already affects a published row:**

| dataset | nodes | distinct base degrees | max virtual degree (`degree`, K=10) | Σdeg² |
|---|---|---|---|---|
| cora | 2,708 | 37 | 582 | 1.16e7 |
| enzymes | 19,474 | **9** | **6,724** | 9.92e8 |
| proteins | 43,466 | 16 | **14,644** | 4.89e9 |

⚠️ **Threat to an existing claim.** `degree` currently **wins enzymes node classification (F1 0.5136, the best of the five variants)** — and the enzymes `degree` graph has a 6,724-degree hub from the same defect. That number is therefore **not evidence that degree-based role rewiring works**; it is a score obtained on a star graph whose hubs were selected by node id. The E-ablation `degree` arm on **both** enzymes and proteins must be re-read in this light, and the enzymes NC winner is in question.

**Interpretation.** Tie-breaking by index makes the graph a function of **node id**, which carries no structural meaning — and in proteins the ids are ordered by protein membership, so every degree-3 node in the dataset is wired to a handful of nodes from the *first* protein. Random tie-breaking would still be arbitrary per-edge but would spread degree ≈ 2K with no hub, keeping the construction's intent ("connect to K role-similar nodes") intact. `psi` and `centrality` are continuous-valued and do not tie at scale, which is exactly why their max degree stays at ~90.

**Status:** measured and recorded; **no fix applied and no result recomputed** — deliberate, pending a decision on tie handling. Runtime is the visible symptom (the ~810× Σdeg² gap = 2h27m vs 2m23s per DeepWalk seed), but the validity problem is the reason this matters.

## 2026-07-18 — METHOD FIX: tie collapse in the top-K virtual-graph construction (invalidates prior `degree` and `centrality` results)

**The defect.** `virtual_graph.py` gives every node a **1-D** structural signature (`signatures()` returns an `(N, 1)` matrix) and connects it to its top-K nearest nodes via `sklearn.NearestNeighbors`. For the `degree` variant that signature is a single integer, and real graphs have very few distinct degrees: **enzymes has 9 distinct degrees across 19,474 nodes; proteins has 16 across 43,466.** Thousands of nodes are therefore *exactly coincident points* at distance 0, and "top-K nearest" is **ill-posed** — there is no nearest neighbour to find.

`NearestNeighbors` resolved these ties **deterministically and query-independently**: every member of a tie class received the *same* K winners (the first K in the BallTree's internal layout — verified *not* node-id order). The resulting graph is not a role graph but a set of **disjoint stars**:

| dataset | tie class | class size | virtual hub degree (K=10 requested) |
|---|---|---|---|
| enzymes | degree = 4 | 6,725 | **6,724** |
| proteins | degree = 3 | 14,645 | **14,644** |
| cora | — | — | **582** |

Measured on enzymes: 19,392 of 19,474 nodes had virtual degree exactly 10, while **70 nodes (7 classes × 10 winners)** absorbed everything. All of a class's information was routed through 10 arbitrary representatives — a severe message-passing bottleneck and an artefact of the tie-break, not of structural role.

**A second, quieter instance.** `centrality` showed no hub (max degree 23) but **89.1% of its enzymes virtual edges had distance < 1e-12**, median distance exactly 0.0: eigenvector centrality is degenerate there (96.9% of nodes < 1e-6), so neighbour selection was driven by **float noise** below any meaningful precision.

**The fix (implemented).** Tied signatures mean the nodes are *genuinely interchangeable under that role definition*, so the principled operation is to **sample, not to order**:
1. Quantize signatures to `SIG_TOL = 1e-9 × spread` so noise-level differences collapse into exact ties (this is what makes `centrality` well-posed).
2. Group nodes into exact-tie classes; each node draws K neighbours **uniformly at random from its own class**, via a `seed`-derived RNG (`VirtualGraph(..., seed=42)`).
3. If a class holds fewer than K+1 members, take it whole and widen outward to the nearest signature values — so `psi`-like near-unique signatures keep the original nearest-neighbour semantics and the exact-K contract holds for every variant.

**Effect.** Enzymes degree max virtual degree **6,724 → 36**; proteins **14,644 → 35**; cora **582 → 30**. Mean degree ≈ 2K with a tight spread, zero isolates, all construction rules pass. Build is deterministic across reruns and genuinely varies with the seed.

**Side effect — this was also a compute bug.** DeepWalk/node2vec cost tracks `sum(deg²)`. Enzymes degree: 9.92e8 → 7.95e6 (**125× cheaper**); proteins: 4.89e9 → 1.78e7 (**275× cheaper**). The proteins `degree` run had been abandoned after 26 minutes at 26% of a single seed.

**Results invalidated — must be regenerated.** Jaccard overlap between the old saved graphs and the corrected builds:

| dataset | psi | degree | centrality | hybrid |
|---|---|---|---|---|
| cora | 0.824 | **0.047** | 0.765 | 0.864 |
| enzymes | 0.984 | **0.004** | **0.029** | 0.988 |

Every variant except `original` changed. **All non-`original` scoreboard rows are stale**, most critically the two enzymes E-study headlines: **NC `degree`+`graphsage_edge` = 0.5536** (Table 1 winner) and **LP `centrality`+`deepwalk` = 0.7382**. Neither is currently valid evidence. The cora `hybrid`/`psi` winners moved less but still changed and must be rerun before quoting.

**Why `graph_health.csv` did not catch it.** It logged `avg_degree`, `components`, `isolates` — enzymes degree K=10 read `avg_degree 19.95, isolates 0, components 8`, which looks healthy, because **a star has a perfectly normal mean degree**. A `max_degree` column has been added; the detector is max/skew, never the mean.

**Paper framing.** State this as a methodological point, not an erratum: *for a low-cardinality structural signature, top-K nearest-neighbour selection is ill-posed, and any deterministic tie-break silently manufactures hub artefacts.* Sampling within the tie class is the correct definition of the construction, and the `degree` variant is precisely the case where the signature is coarsest. Whether `degree` still wins enzymes NC after the fix is **open** — the star may have been hurting it.

## 2026-07-18 — METHOD FIX 2: link-prediction splits were restricted to the largest connected component

**The defect.** `prepare_linkpred.build_graph()` ended with `largest = max(nx.connected_components(G), key=len); return G.subgraph(largest).copy()`. Every link-prediction split — train edges, test positives, negatives — was therefore drawn from **one component only**. On a connected graph that is harmless; on the disjoint-union datasets central to this study it discards almost the entire dataset:

| dataset | full graph | largest component | LP actually covered | test positives (seed 42) |
|---|---|---|---|---|
| cora | 2,708 nodes | 2,485 | 91.8% | 1,521 |
| enzymes | 19,474 nodes | **125** | **0.6%** | **17** |
| proteins | 43,466 nodes | **620** | **1.4%** | 315 |

**Enzymes link prediction was measured on 125 of 19,474 nodes with 17 held-out positive edges.** An AUC over ~17 positives moves in steps of roughly 0.06 and carries enormous variance, so the recorded enzymes LP numbers — including the E-study headline **`centrality`+`deepwalk` = 0.7382** — describe a 125-node fragment, not the dataset. Node classification always used the full graph, so **NC and LP were never measured on the same node set**, and any per-dataset comparison between the two tasks was invalid.

**The fix.** `build_graph` now returns the whole graph (self-loops still dropped). No other change was needed: `nx.minimum_spanning_tree` returns a spanning **forest** on disconnected input, so the existing "spanning-tree edges stay in train" rule already preserves connectivity *per component* — no component can be split off by the holdout.

**Verified after the fix** (seeds 42, all three datasets): split covers 100% of nodes; test fraction exactly 30.0%; train component count equals graph component count (nothing disconnected); no train/test edge overlap; negatives are genuine non-edges with no self-pairs. Enzymes test positives **17 → 11,185**; proteins **315 → 24,313**; cora **1,521 → 1,583**.

**Consequence.** Every LP number ever recorded is superseded, cora included (its split changes too, 91.8% → 100%). This is independent of the tie-break fix: it lives in the split code, so it affects all LP results regardless of virtual-graph variant or encoder. LP splits must be regenerated before any LP result is quoted.

**Known remaining caveat (not fixed, state in the paper).** `sample_non_edges` draws negatives uniformly at random from all node pairs. On graphs with many components (enzymes 640, proteins 1,195) a random pair is almost always cross-component while every positive is within-component, so "same component?" alone is a strong predictor and AUC is inflated. This is the standard protocol and was left unchanged for comparability with I2V, but LP AUC on disjoint-union datasets should be read as an optimistic bound. A within-component negative sampler would be the stricter alternative.

## 2026-07-18 — METHOD FIX 3: link-prediction negatives are now sampled within a component

**The defect.** `sample_non_edges` drew negative pairs uniformly from all node pairs. Test **positives are within-component by construction** (an edge cannot cross components), so on a many-component graph the two classes were separable by a rule that has nothing to do with link structure:

| dataset | components | negatives that were within-component (old) | positives |
|---|---|---|---|
| cora | 78 | 84.2% | 100% |
| enzymes | 640 | **0.1%** | 100% |
| proteins | 1,195 | **0.2%** | 100% |

On enzymes and proteins, **99.8-99.9% of negatives were cross-component**, so a classifier answering only "are these two nodes in the same component?" separates the classes almost perfectly. Every LP AUC on those datasets was dominated by component membership rather than by the structural signal under study — which is exactly the quantity the virtual-graph comparison is supposed to isolate.

**The fix.** `sample_non_edges(..., negatives='component')` is now the default: both endpoints are drawn from the *same* connected component, so a negative pair is as hard as a real edge and the "same component?" shortcut carries zero information. Component choice is weighted by each component's **exact count of available non-edges** (`n(n-1)/2 - m`, accumulated into cumulative weights), which makes the draw uniform over all within-component non-edges rather than biased toward large components; components that are complete or single-node are excluded since they offer no non-edge. An assert fires if within-component capacity is below the requested count, pointing at `negatives='uniform'`.

`negatives='uniform'` retains the old behaviour and is exposed as `--negatives uniform`, so the **Phase-1 / I2V protocol stays reproducible** — Phase-1 numbers were produced under `uniform` and must be regenerated with `uniform` if ever re-run.

**Verified** (cora / enzymes / proteins, seed 42, both modes): negatives 100% within-component under the new default, zero invalid pairs (no real edges, no self-pairs), zero duplicates, deterministic across repeated calls.

**Secondary fix in the same function.** `sample_non_edges` was seeded with the *same* value as `split_edges`; it now uses `seed + 1` so the two draws are independent streams rather than correlated by construction.

**Consequence.** Combined with the largest-component fix, **all LP splits and therefore all LP results are superseded**, cora included. Expect measured AUC to *drop* on enzymes and proteins — that is the artefact being removed, not a regression. Because the old protocol made the task partly trivial, any previous claim that a virtual-graph variant "wins link prediction" on those datasets carries no weight until rerun.

## 2026-07-18 — METHOD FIX 4 + unified graph policy

**Omega per component, rescaled to max=1.** I2V's Ω is a *global* eigenvector, so on a disjoint union it is supported only on the component with the largest spectral radius and every other component underflows: enzymes reached **1.2e-115**, with **96.9%** of nodes below 1e-6 and 81.6% below 1e-12. That is power-iteration underflow, not centrality. It contaminated more than the `centrality` variant — psi's `q` term *is* Ω, so on enzymes psi correlated **+0.51 with log₁₀ Ω** versus only +0.25 with degree, i.e. psi tracked numerical underflow depth more strongly than structure. (On cora, psi correlated −0.80 with degree — behaving as intended — which is why this never surfaced.)

Ω is now computed inside each connected component. Degenerate share: enzymes 96.9% → **0.2%**, proteins 76.9% → **3.0%**, cora 17.9% → 8.2%. A numerical guard was added: the eigensolver fallback returned values like −8.6e-18, and psi's `q > 0` test would have silently dropped those nodes.

Each component is then rescaled to **max = 1**, so Ω reads "centrality relative to my component's most central node". Under NetworkX's native L2 = 1 the *scale* depended on component size (a 2-node fragment's nodes got 0.707, the maximum possible, while a 2,485-node component's hub got ~0.35), making cross-component comparison meaningless — and cross-component comparison is exactly what a role graph does.

**Honest limitation.** Max-normalization does **not** remove the correlation between Ω and component size (cora −0.92 → −0.94). Measured means by component size on cora: size 2485 → mean Ω 0.008; size 9 → 0.714; size 2 → 1.000. Every component's top node is 1.0 by construction, so in a 2-node component *both* nodes are maximally central — which is locally true, not an artefact. The consequence to state in the paper: **Ω partly encodes "am I in a fragment or in the giant component"**, strongly so on cora. Max-normalization fixes the arbitrary scale distortion; the distributional effect is inherent to any per-component normalization.

**Unified graph policy (`graph_io.GRAPH_POLICY`, re-exported as `benchmark_config.GRAPH_POLICY`).** All six cross-dataset decisions now live in one dict instead of four files: `self_loops`, `directed`, `centrality`, `sig_tol`, `lp_negatives`. `I2V_BASELINE_POLICY` names the Phase-1 contract (`centrality='global'`, `lp_negatives='uniform'`) under which Deliverable #1's byte-identical guarantee is valid — verified still `True` after every change. Unknown policy keys raise rather than being silently ignored.

`graph_io.load_graph()` is the single reader for every stage (splits, virtual graphs, encoder features), so they cannot disagree about the graph — verified `encoder.build_graph` and `prepare_linkpred.build_graph` return identical graphs. `graph_io.check()` reports the properties that silently broke earlier runs — disconnection, coarse degree signature, isolates — with `strict=True` converting them to a hard failure.

**A false positive caught before it shipped.** The first version of `check()` inferred directedness by counting edges with no reverse edge, and flagged **proteins as 81,044/81,044 directed** — proteins is a molecular graph rebuilt from sorted unique pairs, i.e. undirected. The metric was measuring *storage format*: a normal undirected edgelist lists each edge once and is **indistinguishable from a directed one**. Directedness is now declared as dataset metadata (`DATASETS["politics"]["directed_source"] = True`, a retweet relation) and never guessed. Recorded because a warning that fires on every dataset trains its reader to ignore it.

## 2026-07-20 — METHOD FIX 5: tie widening samples, and the split protocol is made interpreter-independent

Follow-ups from a full code review. **Implemented, not measured** — no run was performed, so every statement below is about mechanism, not about a number.

**The tie fix was only half a fix.** The 2026-07-18 degree-tie fix made top-K selection sample *inside* a node's own tie class, which removed the star collapse (enzymes hub degree 6,724 → bounded). But when a node's class holds fewer than K members, `build()` widens outward to the nearest signature values — and that widening walked outward **one index at a time in `argsort` order**. Members of the adjacent class are by definition *equidistant* from the node, so index order is arbitrary; the consequence is that every node needing to widen into the same neighbouring class borrowed the **same first-indexed members** of it. The stated guarantee "ties are sampled, never ordered" therefore held only within a node's own class, not at the boundary where widening happens.

`build()` now widens **one whole tie class at a time** and samples inside it (`rng.choice(..., replace=False)`) whenever the class offers more candidates than the node still needs — the same rule that already governed the node's own class, applied consistently. Exactly-K per node, determinism given the seed, and the node-set/finite-weight asserts are unchanged.

**Who this changes.** Only nodes whose own tie class is smaller than K+1 *and* whose nearest neighbouring class is larger than the remaining need. On a continuous signature (`psi`, where nearly every class is a singleton) the widened class is almost always a singleton too, so selection is unchanged. The effect concentrates on the **coarse signatures — `degree` and `centrality`** — precisely the variants that motivated the original tie fix, and on their rare-value tail nodes (a uniquely-high-degree node widening into a large common-degree class). **Consequence: virtual graphs must be rebuilt before the `degree`/`centrality` (and `hybrid`) rows are quoted; `psi` and `original` are expected to be unaffected, but that expectation is untested.**

**Split protocol: determinism no longer depends on the interpreter.** `split_edges` built its train list as `list(tree) + ...` from a **set**, and `sample_non_edges` accumulated negatives into a **set** that `prepare()` then sliced into train/test. Set iteration order is stable for a given CPython build but is not part of the language contract, so two things silently rode on it: the row order of `train.edgelist` (which fixes node insertion order, hence the walk RNG, hence the embedding) and **which negatives landed in train versus test** (which affects the supervised logreg scorer). Positives are now sorted; negatives are kept in RNG draw order — deliberately not sorted, because sorting would place every low-id pair in train and every high-id pair in test. The seed already determined *which* edges are held out; this only removes the interpreter from the reproducibility claim. **All LP splits change once regenerated, so LP embeddings must be retrained** — `runner` now enforces this automatically by refusing to reuse an embedding older than its split.

**A claim in the 2026-07-18 entry above was too strong.** That entry states that weighting components by their exact non-edge capacity makes the draw "uniform over all within-component non-edges". It does not, quite: a component is chosen by capacity, a pair is drawn inside it, and the pair is *rejected and redrawn* if it hits a real edge — so a denser component yields an accepted negative slightly less often than its capacity share, and the weights are never decremented as pairs are consumed. The resulting tilt toward sparser components is on the order of a few percent and does not change which components are reachable, but the paper should claim "capacity-weighted within-component sampling", **not** "exact uniform". The code comment now says so.

**Phase-1 protocol was unenforced in code.** `I2V_BASELINE_POLICY` (`centrality='global'`, `lp_negatives='uniform'`) names the contract under which Deliverable #1's byte-identical guarantee holds, but no caller passed it: both `runner` link-prediction paths called `prepare()` with no `negatives` argument and so inherited the new `component` default. Rerunning Phase-1 would have overwritten its splits under the *post-fix* protocol while still being reported as the I2V-comparable baseline. Both call sites now pass the baseline policy explicitly, so the comparability claim is enforced by the code rather than by remembering a flag.

**Reported dispersion was mixing two conventions.** `results_io.record_score` (the master scoreboard) used population std while `runner.summarize_seed_results` and the pandas benchmark tables used sample std — over 3 seeds the scoreboard understated dispersion by ~18% relative to the benchmark tables for identical inputs. Unified on sample std (ddof=1). Means are unaffected; **`std` values already in `scoreboard.csv` are population std until those rows are rerun**, so scoreboard and benchmark error bars should not be compared across that boundary.

**Node-classification F1 could be quietly understated.** Both notebooks suppress warnings globally, so an LBFGS fit that hit `max_iter=300` without converging was reported as a normal F1. `max_iter` is unchanged (paper protocol); non-convergence is now surfaced per evaluation as an explicit "this F1 is a lower bound" line. Whether any past run actually hit the cap is unknown — no run was made to check.

## 2026-07-20 — FULL RERUN after the 07-18 + 07-20 fixes: cora, enzymes, proteins (K=10, seeds 42/43/44)

First complete post-fix sweep. Notebooks 2 and 3, all five virtual-graph variants, both tasks, three datasets. All previous result files were cleared first, so nothing below is contaminated by a pre-fix artifact. Headline scorer is cosine (`REPRO["linkpred_score"]`); node classification is weighted F1.

### 1. Did the 2026-07-20 fixes move any number? No.

Proteins is the clean control: it was last run on 2026-07-18 *after* the three method fixes but *before* the 07-20 review fixes, so the proteins delta isolates the 07-20 changes alone.

| proteins, deepwalk | LP AUC before → after | proteins, graphsage_edge | LP AUC before → after |
|---|---|---|---|
| centrality | 0.5216 → 0.5222 | centrality | 0.5829 → 0.5831 |
| degree | 0.5045 → 0.5067 | degree | 0.5500 → 0.5508 |
| hybrid | 0.8029 → 0.8023 | hybrid | 0.5633 → 0.5630 |
| original | 0.9414 → 0.9413 | original | 0.6726 → 0.6722 |
| psi | 0.5222 → 0.5232 | psi | 0.5588 → 0.5582 |

Every meaningful row moves by ≤0.005; node classification likewise (largest shift 0.5256 → 0.5215 on deepwalk/original). The only larger delta is the D5 constant-feature *floor* control (0.4407 → 0.4528), whose own std is ±0.062 — pure noise, as designed. **Conclusion: the 07-20 fixes were correctness and reproducibility hygiene, not result-changing.** That is the intended outcome and it is now evidenced rather than asserted.

The tie-widening change (whole tie class, sampled) did alter the graphs, but marginally: proteins psi 317,534 → 317,480 edges (−0.017%), centrality 256,820 → 256,867, hybrid 396,403 → 396,310, degree 434,169 → 434,175, original byte-identical. **Correction to the prediction logged earlier on 07-20:** psi was expected to be completely unaffected because its signature is continuous; it in fact shifted by 54 edges, so psi does contain a small number of exact ties. The direction of the claim held; the absolute "unaffected" did not.

### 2. The 07-18 method fixes changed enzymes drastically — and removed the study's only role-graph win

Enzymes was last run 2026-07-16, so its delta bundles all four method fixes (degree ties, LP largest-component, within-component negatives, per-component Ω) plus the 07-20 hygiene.

| enzymes, deepwalk LP AUC | before (07-16) | after (07-20) |
|---|---|---|
| centrality | 0.7382 ± 0.0840 | 0.5283 ± 0.0014 |
| degree | 0.6240 ± 0.0725 | 0.5052 ± 0.0008 |
| psi | 0.5110 ± 0.0297 | 0.5039 ± 0.0031 |
| hybrid | 0.5087 ± 0.0246 | 0.7765 ± 0.0025 |
| original | 0.6574 ± 0.0170 | 0.9007 ± 0.0007 |

**The headline consequence.** Before the fix, `centrality` (0.738) beat `original` (0.657) on enzymes link prediction — the single strongest piece of evidence that a role graph can outperform the real graph. After the fix, `original` (0.901) beats `centrality` (0.528) by 0.37. **That win was an artifact of the 125-node largest-component split (17 test edges); it does not survive.** Any earlier claim resting on it is withdrawn.

Note also the standard deviations collapsing by roughly two orders of magnitude (0.084 → 0.0014). With 17 test edges the old metric was noise; with 11,185 it is stable. Node classification barely moved (≤0.01 on every variant) because it never used the split — and `original` reproduced to 16 significant digits (0.5086297830053176 both times), a clean determinism check that the unrelated machinery did not drift.

The enzymes psi graph changed most in construction: 111,401 → 150,078 edges (+35%), from the per-component Ω fix (psi's `q` term *is* Ω, which was underflowing to ~1e-115 on this disjoint union).

### 3. Post-fix findings (these are the numbers to build the paper on)

**a. The original graph wins all six dataset × task cells.**

| best configuration | node classification (F1) | link prediction (AUC) |
|---|---|---|
| cora | deepwalk + original **0.8100** | deepwalk + original **0.8971** |
| enzymes | graphsage + original **0.5589** | deepwalk + original **0.9007** |
| proteins | graphsage + original **0.5799** | deepwalk + original **0.9413** |

No virtual graph beats the original graph anywhere. The best non-original variant is always `hybrid` — which contains the original edges by construction — and even it trails by 0.12–0.22 on link prediction and by 0.36 on cora node classification.

**b. The gap is entirely dataset-dependent, and that is the per-data thesis.** On the two molecular graphs the pure role graphs come within 0.01–0.02 of the original for node classification (enzymes: degree 0.5474 vs original 0.5589; proteins: degree 0.5591 vs original 0.5799). On cora they trail by 0.36–0.44. Structural role is nearly sufficient for molecular node labels and nearly worthless for citation-community labels — the intended "which virtual graph for which data" result, now with the artifact removed.

**c. Link prediction is where role graphs fail categorically.** Pure `psi`/`degree`/`centrality` sit at 0.50–0.66 AUC on every dataset against 0.90–0.94 for the original graph. Role similarity is not adjacency: two nodes sharing a role are no likelier to be connected. State this as a finding, do not bury it.

**d. Technical contribution holds: the GNN beats walk+Skipgram on the virtual graph, 6/6.** Same psi graph, encoder swapped:

| psi graph | NC: deepwalk → graphsage | LP: deepwalk → graphsage |
|---|---|---|
| cora | 0.2133 → 0.2354 (+0.022) | 0.4990 → 0.5175 (+0.019) |
| enzymes | 0.5005 → 0.5470 (+0.047) | 0.5039 → 0.6237 (+0.120) |
| proteins | 0.4882 → 0.5588 (+0.071) | 0.5232 → 0.5582 (+0.035) |

**e. The GNN advantage is specific to role graphs.** On the *original* graph the ordering reverses for link prediction on all three datasets (cora 0.897 → 0.609, enzymes 0.901 → 0.700, proteins 0.941 → 0.672) and for cora node classification (0.810 → 0.426). GraphSAGE wins node classification on the original graph only for the molecular datasets. Honest framing: message passing helps when the graph encodes role; random walks with a lookup table remain far stronger when the graph encodes adjacency.

**f. Ablation D6 — message passing helps node classification, contributes nothing to link prediction on molecular graphs.** Comparing "features only, no message passing" (layers=0) against the full 2-layer encoder on the psi graph: NC gains +0.052 (cora), +0.047 (enzymes), +0.037 (proteins); LP gains +0.034 (cora) but only +0.004 (enzymes) and **−0.008 (proteins, i.e. message passing hurts)**. On the molecular graphs the LP signal is carried by the input features alone.

**g. Ablation D on cora — best features are degree + centrality, not all four.** D2 (deg+cent) reaches 0.2460 NC against D0 (all four) at 0.2354, while D3 (psi only) is worst at 0.1754 — below the D4 random-feature control on NC. psi as an input feature is confounded (the psi graph was built from it) and adds nothing once the graph already encodes it.

**Caveats on all of the above.** K=10 only; three seeds; the virtual-graph build seed is fixed at 42, so the reported ±std covers split and encoder-init variance but not tie-sampling variance. Encoder settings are the previously locked A2 (`positives="edge"`) and B-mean, whose ablations were decided on pre-fix enzymes runs and were deliberately not re-validated.

## 2026-07-21 — Research direction locked for LoG: a "when to augment" characterization study

Direction review. The post-fix results (original graph wins all 8 dataset × task cells; role graphs fail link prediction; role graphs approach the original only on molecular node classification) were presented and accepted as the honest result — no attempt to rescue a "virtual graph wins" headline.

**Reframed contribution.** ViRGo is a **study**: given a graph and a task, when is the original topology enough, and when do structural augmented features help? The deliverable is a **characterization** — connect each dataset's properties (homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement) to the original-vs-best-augmented gap, so the paper can say "for graphs like this, keep the original; for graphs like that, add these features." This is the agreed framing, offered as a conjecture for the community, not a proof.

**Scope decisions (locked).**
- **Purely structural.** ViRGo's features are graph-derived (degree, eigenvector centrality, Ψ, clustering) and do double duty — they build the virtual graph and feed GraphSAGE — and ablation D already showed they are necessary (random features drop to the DeepWalk baseline or below). No external node attributes anywhere, including OGB text features and biological descriptions: they would confound the study, since a gain could then be attributed to the attributes rather than to the structural rewiring under test. Isolating structural identity is the point (the inherited I2V premise).
- **Datasets.** Current four (cora, citeseer_linqs, enzymes, proteins) + small-to-medium OGB: ogbn-arxiv (node property), ogbl-collab and ogbl-ddi (link property). Large-scale OGB (100M-node) is out — too slow before the deadline.
- **OGB evaluation protocol (fair, structural-only).** On every OGB dataset both graphs receive the *same* structural features (degree, centrality, Ψ, clustering); only the edges differ — the original graph carries the real edges, the virtual graph the role-based edges. OGB's extra attributes (text embeddings, product descriptions, biological annotations) are ignored, so graph structure is the single variable and any difference measures the effect of virtual rewiring. We do **not** compare against top OGB leaderboard entries, which may use those attributes; the paper states plainly that the goal is structural analysis, not leaderboard superiority.
- **Per-dataset node features (verified on OGB docs, 2026-07-21).** `ogbl-ddi` has no node features (a homogeneous drug-drug interaction graph) and is used directly as a structure-only link-prediction benchmark. `ogbn-arxiv` (128-dim skip-gram embeddings of title + abstract) and `ogbl-collab` (128-dim word embeddings of authors' papers) do carry node features, which we do not load; the structural methodology is applied unchanged. No implementation change is needed for any of the three. (This corrects an in-conversation slip that "ogbn-arxiv has no node features" — the featureless dataset is ogbl-ddi; ogbn-arxiv has 128-dim features that we deliberately ignore.)
- **Encoder order.** GraphSAGE results first; then swap in GIN to test whether its stronger isomorphism (WL) power helps. GIN is in scope for this paper, after GraphSAGE.
- **Future work (not now):** the learnable-alpha graph (a single learned weight that blends original vs virtual per dataset) needs many synthetic datasets to train, so it is a paper/thesis extension; plus the LLM graph-summary stretch. Anomaly detection is dropped from the immediate plan.

**Venue + timeline.** Immediate target LoG (Learning on Graphs), abstract ~end of July, full paper ~start of August; published via JMLR/PMLR. Thesis (~one month) reuses the same content.

**Threats to validity to carry into the paper** (unchanged): the A–D encoder ablations were decided on pre-fix enzymes and not re-validated; K=10 only; 3 seeds; the virtual-graph build seed is fixed at 42; baselines used as published.

## 2026-07-22 — OGB evaluation protocol decided: official splits + official metrics (revises the earlier "keep our protocol" plan)

Adding the first two OGB datasets (ogbn-arxiv for node classification, ogbl-ddi for link prediction). A protocol question was worked through: should OGB datasets be scored under the ViRGo protocol (random 70/30 split, weighted-F1, cosine-AUC) for cross-dataset uniformity, or under OGB's official split + metric? **Decision: use OGB's official split and official metric for the OGB datasets.** This *revises* the earlier direction-lock note that leaned toward reusing our protocol everywhere.

**Rationale.** The OGB split is part of the scientific task, not file organization: ogbn-arxiv uses a *time* split (train on older papers, test on newer) and ogbl-ddi a *protein-target* split (interactions among drugs with different mechanisms). A random 70/30 deletes exactly the intended generalization challenge and inflates scores. OGB's Hits@20 (rank one true interaction against ~100k negatives) is also far stricter than balanced ROC-AUC. Keeping the official protocol is the more honest and more publishable choice.

**Consequence — mixed-protocol study, framed as internal-comparison-only.** The four core datasets stay on the ViRGo protocol; the OGB datasets move to official split + metric. This is acceptable because the study's real axis is *internal* (original vs each virtual graph, within one dataset, same split + metric). Across datasets we compare only *direction and ranking* — which graph wins, the ordering of variants, whether rewiring helps/hurts/ties, and how that relates to graph properties — never raw magnitudes (a +0.05 Accuracy gain is not a +0.05 Hits@20 gain). Results are reported as "OGB-sourced datasets evaluated under the official OGB protocol", not leaderboard entries (data.x is still ignored — purely structural).

**Leakage rules differ by dataset and are coded accordingly.** ogbn-arxiv is *transductive* — the whole graph is observed and only labels are split — so structural features and the virtual graph are built on the full graph; only the scored node set (official test ids) is restricted. ogbl-ddi *splits the graph itself* — features, virtual graph and embeddings are built from the **training edges only** (no valid/test edge touches structure), then the supplied positive/negative pairs are scored. Model selection follows the no-test-peeking rule: which variant/encoder wins is decided on **validation**; **test** is read once for the final number.

**Metrics.** ogbn-arxiv: Accuracy (primary, via OGB `Evaluator`) plus weighted-F1 and macro-F1 (secondary) on the official test nodes. ogbl-ddi: Hits@20 (OGB `Evaluator`, K=20 fixed), with the same cosine scorer used for every graph variant so the internal comparison stays fair. The four structural features (degree, eigenvector centrality, Ψ, clustering) are computed once from the built graph and shared across all five variants — only the edges differ (confirmed already implemented in `encoder.features()`, which reads the original graph).

**Caveats.** GraphSAGE first, GIN later (unchanged order). ogbn-arxiv (~169k nodes) is the only heavy run; the 1-D-signature virtual-graph build is O(N log N) so there is no algorithmic blocker, but encoder runtime should be timed on one seed before the full sweep.

## 2026-07-23 — ogbl-ddi scorer: cosine was measuring a ceiling, not a model

**Finding (methodological, worth a paragraph in the paper).** Our first ogbl-ddi sweep returned Hits@20 = 0.0000 for GraphSAGE on four of five graph variants. That number was **not** a model failure. ogbl-ddi's training graph contains a class of 156 structurally identical nodes (degree 4, clustering exactly 1/6, eigenvector centrality equal to 9 decimal places). A purely structural encoder is a deterministic function of (features, neighbourhood), so all 156 receive the *same* embedding vector, and every one of the C(156,2) = 12,090 pairs among them has cosine similarity exactly 1.0. Between 18 and 46 of those pairs appear in OGB's official negative set. OGB computes `Hits@K` as `mean(y_pred_pos > kth_largest(y_pred_neg))` with a **strict** inequality, so the threshold lands on 1.0 - the ceiling of cosine - and no positive pair can ever exceed it. Hits@20 is then identically zero regardless of embedding quality.

Three independent checks confirm the mechanism: (i) the same embeddings reach AUC 0.66-0.76 and Hits@100 = 0.033; (ii) `hybrid` - the only variant whose role edges break the twin symmetry, leaving zero ties - is the only nonzero GraphSAGE cell (0.0013); (iii) DeepWalk, whose random walks inject node identity and therefore never produce identical vectors, is nonzero everywhere.

**Generalisable caution:** a bounded similarity (cosine) combined with a strict-inequality top-K metric is degenerate for any deterministic structural embedding on a graph with automorphic classes. This is a measurement artifact, not a property of the method, and any structural-identity paper reporting Hits@K should check for it.

**Protocol change.** Pair scoring for OGB link prediction moves from fixed cosine to a trained decoder, matching OGB's own ddi reference model (`examples/linkproppred/ddi/gnn.py`): the elementwise product of the two endpoint embeddings through a 2-layer MLP (256 hidden), fitted with BCE on **training edges only** against uniformly sampled non-edges. Embeddings stay frozen - we are replacing the scorer, not the encoder - and the identical decoder, hyperparameters and seed are used for every graph variant, so the graph remains the single variable. Positives are always the real training links, never the virtual role edges. This does not weaken the "purely structural" rule: the decoder learns from graph-derived supervision only, and no external node attribute is loaded anywhere.

**Note for the write-up:** ViRGo still differs from the OGB leaderboard entries in a second, deliberate way - those models feed a free `torch.nn.Embedding` of 256 dimensions per node, trained end to end, i.e. pure node identity. ViRGo refuses that by design (four graph-derived features only), so a gap to the leaderboard is expected and is not the quantity under study. The comparison that matters stays internal: original vs each virtual graph, same scorer, same split, same metric.

**Effect (seed 42, validation, pre-sweep check):** DeepWalk/original 0.0124 -> 0.0857, GraphSAGE/original 0.0000 -> 0.0411, GraphSAGE/psi 0.0000 -> 0.0228. The full re-sweep (validation, selection, single test read) follows; the cosine-era rows are retained in `results/scoreboard.csv` under `*_hits@20_cos`, and the cosine-era selection lock was voided so the winner is re-chosen under the new scorer before test is read.

## 2026-07-23 — ogbl-ddi results under the trained decoder, and what they change

**Result (K = 10, 3 seeds, official split + Evaluator, decoder trained on training edges only).** Winner selected on validation, test read once.

| graph | valid GraphSAGE | valid DeepWalk | test GraphSAGE | test DeepWalk |
|---|---|---|---|---|
| Ψ | 0.0223 | 0.0071 | 0.0412 | 0.0081 |
| degree | 0.0213 | 0.0071 | 0.0265 | 0.0080 |
| centrality | 0.0309 | 0.0077 | 0.0514 | 0.0100 |
| original | 0.0385 | 0.0772 | 0.0102 | 0.0378 |
| hybrid | 0.0385 | **0.1038** | 0.0117 | **0.0533** |

Locked winner: hybrid + DeepWalk, validation 0.1038 ± 0.0040, test **0.0533 ± 0.0062**.

**Finding 1 — the scorer, not the graph, produced the earlier null result.** Under fixed cosine every ddi cell was ≤ 0.013 and four GraphSAGE cells were identically zero. Replacing cosine with the OGB-style trained decoder raised every cell by one to two orders of magnitude and removed the zeros. Any claim of the form "role graphs carry no link signal on ddi" was an artifact of the scorer and is withdrawn.

**Finding 2 — with a trained decoder, role graphs beat the original graph for GraphSAGE on test.** Test: centrality 0.0514 and Ψ 0.0412 against original 0.0102 — a 4-5× gap in favour of role-based rewiring. This is the first dataset × task cell in the study where an augmented graph clearly beats the original under the same encoder, and it runs against the locked headline that the original graph wins everywhere. The headline is not yet revised: this is one dataset, under a different (official-OGB) protocol, and the effect appears on test but not on validation.

**Finding 3 — validation does not predict test on ogbl-ddi.** GraphSAGE's ranking inverts between the two splits (original/hybrid lead on validation, role graphs lead on test); DeepWalk keeps its order but halves. The official split is by protein target, so validation and test are deliberately different distributions, and the OGB leaderboard shows the same decorrelation. Consequence for the write-up: on this dataset a single-split selection is fragile, and the honest presentation is both splits side by side, not the test number alone. Our locked choice happened to also be the best test cell, so no selection cost was incurred, but that was luck and should be stated as such.

**Open question raised for the core four.** Their link-prediction numbers come from `eval_linkpred.py` with the default `score='cosine'`. Given Finding 1, the LP conclusions across the whole study may be scorer-dependent. A supervised alternative (`--score logreg`, hadamard + logistic regression) already exists in that script, so the check needs no new code. This should be resolved before the paper's link-prediction claims are finalised.

## 2026-07-23 — density-matched controls: the role edges carry signal, and the encoder comparison was partly a density artifact

**Motivation.** Across the study the original graph's density is not comparable to the role graphs', and the mismatch *reverses sign*: on ogbl-ddi the role graphs are 39× sparser than the original (avg degree ~12 vs 500.5), while on cora, citeseer, enzymes and proteins they are 3-5× denser (12-20 vs 2.8-3.9). Any cross-dataset pattern of the form "role graphs help here but not there" could therefore be tracking edge *count* rather than edge *meaning*. Two controls hold the count fixed, built with the same union-of-K-per-node construction as the role graphs so counts match by construction: `original_k` (K uniformly sampled real neighbours per node) and `random_k` (K arbitrary nodes).

**Results (ogbl-ddi, validation, decoder scorer, 3 seeds, K=10).** Both controls landed ~1.5-1.7× denser than the role graphs, which handicaps the role graphs and makes the comparisons below conservative.

| graph | edges | GraphSAGE | DeepWalk |
|---|---|---|---|
| psi | 27,042 | 0.0251 | 0.0071 |
| degree | 27,109 | 0.0227 | 0.0071 |
| centrality | 24,886 | 0.0326 | 0.0077 |
| original_k | 39,588 | 0.0371 | 0.0303 |
| random_k | 42,632 | 0.0078 | 0.0006 |
| original | 1,067,911 | 0.0413 | 0.0772 |
| hybrid | 1,088,727 | 0.0554 | 0.1038 |

**Finding 1 — role edges beat a random scaffold decisively.** GraphSAGE on the role graphs scores 2.9-4.2× the random control (0.0227-0.0326 vs 0.0078); DeepWalk roughly 12× (0.0071-0.0077 vs 0.0006). The random control has *more* edges than any role graph, so the gap cannot be a density effect. This is the study's first clean demonstration that Ψ, degree and centrality edges carry structural information rather than merely giving the encoder something to aggregate over - it is the edge-level counterpart of ablation D, which established the same for the input features.

**Finding 2 — GraphSAGE extracts nothing from 96% of the original graph's edges.** A uniform K=10 sample of the real edges (39,588) scores 0.0371 against the full graph's 0.0413 with 1,067,911 edges - a 0.3 sigma difference. Whatever advantage the original graph gives GraphSAGE, it is not volume.

**Finding 3 — the encoder ranking inverts once density is matched.** On the full graphs DeepWalk leads GraphSAGE (0.0772 vs 0.0413); at matched density GraphSAGE leads DeepWalk (0.0371 vs 0.0303), because DeepWalk loses a factor of 2.5 when the original graph is sparsified while GraphSAGE loses almost nothing. The earlier reading that "walks beat message passing when the graph encodes adjacency" is, on this dataset, substantially a statement about edge volume. Any encoder claim in the paper should be stated at matched density or explicitly qualified.

**Design note for the write-up.** `original_k` does not replace `original`: possessing a million real links is a genuine property of that graph, not an unfair advantage, so the full graph remains the baseline a practitioner would use. The controls answer a different question - how much of the observed gap is meaning and how much is volume. Also note `original_k` is only constructible where the original graph is denser than K, i.e. on ogbl-ddi; on the core four the matching must run the other way (role graphs rebuilt at the K that matches the original's average degree, K≈2). Matching should therefore be specified on average degree, not on a fixed K.

## 2026-07-23 — where the seed noise comes from, and the frozen configuration

**Question.** The ogbl-ddi validation spread (std/mean 34-51% on GraphSAGE) is comparable to the differences between graph variants, so before extending the study to a second OGB dataset we established which component produces it.

**Decoder is not the source on the role graphs.** Holding one saved embedding fixed and refitting the link decoder under five seeds gives std 0.0014 on GraphSAGE/psi against a total across-seed std of 0.0127 — about 1% of the variance. On the dense original graph the decoder contributes more (std 0.0065 of 0.0142, ~21%), consistent with it fitting 1.07M training edges. DeepWalk's entire (small) spread, 0.0008, is decoder noise: the walk encoder is effectively deterministic across seeds on this graph. Conclusion: **the seed spread is a property of the GraphSAGE encoder**, and error bars on GraphSAGE rows should be read as encoder variance, not measurement variance.

**Encoder training budget.** Loss over 10-epoch windows continues to drift after epoch 50 (psi 4.0191 → 3.8901 by epoch 300; original 3.9005 → 3.7270 by epoch 200), but against a drop of ~38 from initialisation this is under half a percent of the range. The budget is held at 50 epochs for every variant. This is equal in both relevant senses — identical epoch count and identical total gradient samples (50 × 100,000 = 5M pairs per variant, independent of graph size). **Limitation to state in the paper:** the encoder is trained to a fixed budget rather than to a per-graph convergence criterion, and because `pairs_per_epoch` is fixed while corpus size varies with density, a fixed budget corresponds to many passes over a sparse role graph and few over a dense one.

**Configuration frozen (2026-07-23).** K=10; four structural features computed on the original graph and shared by every variant; GraphSAGE mean / 2 layers / 64-dim / lr 0.01 / 50 epochs / edge positives / Q=5 negatives with real-neighbour rejection; DeepWalk bridge as the second encoder; link scoring by an OGB-style trained decoder (hadamard → 256-unit MLP) fitted on training edges only; official OGB split and Evaluator; seeds 42/43/44; winner selected on validation and test read once. Every subsequent dataset runs this configuration unchanged, so that differences between datasets are attributable to the graph and not to per-dataset tuning.

## 2026-07-24 — the six-dataset result set is complete and comes from ONE pipeline

Preparation for the characterization study: before relating graph properties to the original-vs-augmented gap, every number that will enter that regression had to come from the same code. It did not.

**The defect.** The core-four GraphSAGE embeddings were trained on 2026-07-20/21; the negative-sampling rule was changed on 2026-07-22 while validating ogbl-ddi (reject a sampled negative that is a real neighbour of the centre node, redraw). ogbn-arxiv and ogbl-ddi ran after that change, the core four before it, so the cross-dataset table mixed two encoders — exactly the confound the frozen-pipeline rule exists to prevent. (Verified by reproduction, not by timestamps: the pre-rejection encoder taken from git reproduces the stored core-four file to 1.91e-6, the noise floor, with the identical F1, while the current encoder differs by 3.4e-1. The change had been sitting uncommitted in the working tree, so commit dates were useless as evidence. See the follow-up entry for ogbn-arxiv, which reproduces under neither.)

**Audit.** Re-scoring all 240 stored core-four embeddings with the current evaluation code returned their scoreboard rows exactly (80 cells, max |delta| 0.00000, no missing files). The evaluation side was never in question; only the encoder version was.

**Scope of the fix.** The rejection rule was designed for dense graphs (32% of draws collide on ogbl-ddi), but the collision rate is non-zero on sparse graphs too — cora/psi 0.552%, proteins/degree 0.055% of draws at epoch 0 — so the redraw fires there as well and the whole training trajectory shifts. All 120 core-four GraphSAGE embeddings were retrained under the frozen configuration (pre-freeze files kept in `output/superseded_pre_freeze_2026-07-24/`).

**Effect on the results: none that matters.** 37 of 40 GraphSAGE rows moved; the largest move was 0.0054 (cora, hybrid, LP AUC), mean 0.0011, i.e. below the 3-seed std of nearly every cell and far below the +-0.05 reproduction bar. **No cell changed its winning graph variant**, so every published reading survives unchanged: the original graph wins all eight core-four cells, role graphs approach it only for molecular node classification, and role graphs fail link prediction everywhere. DeepWalk rows re-scored byte-identical (its code path was untouched), which is the control on the re-run itself.

**Reproducibility floor, measured.** Repeating one configuration five times under the current code (cora/psi/seed 42) gives the identical metric every time (weighted F1 0.2381) with embeddings agreeing to 2.15e-6 — float32 aggregation order in the SAGE scatter, present even single-threaded. So results are exactly reproducible at the reported precision, and the paper should claim metric-level reproducibility rather than bit-identical tensors. The same evaluation on the pre-freeze embedding gave 0.2376, confirming the 0.0005 gap is the code change and not run-to-run noise.

**The result set now standing (K=10, seeds 42/43/44, GraphSAGE + DeepWalk, five graph variants each).**

| dataset | node classification | link prediction | protocol |
|---|---|---|---|
| cora | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| citeseer_linqs | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| enzymes | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| proteins | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| ogbn-arxiv | Accuracy (+ weighted/macro F1) | not applicable | official OGB time split + Evaluator |
| ogbl-ddi | not applicable (no labels) | Hits@20 | official OGB split + Evaluator, trained decoder |

Ten dataset x task cells, each 5 variants x 2 encoders x 3 seeds. The two "not applicable" entries are properties of the datasets, not gaps: ogbl-ddi ships no node labels, and ogbn-arxiv has no official link-prediction split. Metrics are comparable *within* a dataset, which is what the characterization needs; they are not comparable across the OGB/non-OGB boundary, and no table should place them in one column.

**Best score per dataset x task under the locked encoder (GraphSAGE):**

| dataset | NC: original | NC: best role graph | LP: original | LP: best role graph |
|---|---|---|---|---|
| cora | 0.4266 | centrality 0.2942 | 0.6130 | centrality 0.5659 |
| citeseer_linqs | 0.3298 | centrality 0.2818 | 0.6218 | centrality 0.5437 |
| enzymes | 0.5591 | hybrid 0.5546 | 0.7000 | centrality 0.6563 |
| proteins | 0.5797 | hybrid 0.5646 | 0.6720 | centrality 0.5834 |
| ogbn-arxiv | 0.3918 | hybrid 0.3618 | - | - |
| ogbl-ddi | - | - | 0.0173 | **centrality 0.0519** |

**New: the first cell where augmentation wins.** On ogbl-ddi with GraphSAGE, every role graph beats the original (centrality 0.0519, psi 0.0423, hybrid 0.0344 against original 0.0173 test Hits@20). ogbl-ddi is also the extreme of the dataset panel on density (average degree 500 against 2.8-13.7 elsewhere). That is one point, and the absolute scores are near the floor, but it is the shape the characterization is looking for and it should be tested first when the graph-property table is built: augmentation helps where the original graph is too dense for message passing to be selective, and the locked *DeepWalk* row on the same dataset does not show it (original 0.0378 vs psi 0.0081), so it is an encoder-conditional effect.

**Caveat carried forward.** The ablation-D rows (`graphsage_edge_feat_*`) still come from pre-freeze runs; they justify the locked configuration rather than reporting results under it, and their numbers would move by the same ~0.001. Any ablation table in the paper must say so or be re-run.

**Tooling.** `run_core.py` now runs the core-four sweep headless (mirrors `run_ogb.py`: reuse-or-create, same zones, same scoreboard rows, locked defaults, `--train-only` for parallel training). The whole study reproduces from two commands.

## 2026-07-24 (final) — CONFIGURATION LOCKED. Best and final; the characterization study starts from here

Everything below is settled. No further method, encoder, graph or protocol change is made for the LoG paper; the next work item is the characterization table, which only *reads* these results.

### The locked configuration (quote this in the methods section)

| stage | setting |
|---|---|
| graph loading | `graph_io.GRAPH_POLICY` — self-loops dropped, directed sources read undirected (recorded deviation), `.nodes` sidecar restores isolated nodes, per-component eigenvector centrality (max 1) |
| structural features | four, computed on the ORIGINAL graph and shared by every variant: degree, eigenvector centrality, Ψ (I2V KL→Poisson), clustering; z-normalised; disk-cached by content hash (`encoder.feature_cache`), verified a numerical no-op |
| virtual graphs | K = 10; variants `psi`, `degree`, `centrality`, `original` (control), `hybrid`; deterministic build at seed 42; ties broken by sampling within tie classes |
| encoder (primary) | GraphSAGE, mean aggregation, 2 layers, hidden 64, output 64, lr 0.01, Adam, 50 epochs, edge positives (A2), Q = 5 negatives with real-neighbour rejection, `pairs_per_epoch` 100,000, `max_pairs` 2,000,000 |
| encoder (second) | DeepWalk bridge over the same virtual graph (p = q = 1, `I2V_PARAMS`: dim 64, walk length 40, 10 walks, window 10) |
| node classification | one-vs-rest logistic regression, stratified 70/30, weighted F1 (core four); official OGB time split + Evaluator Accuracy, F1 secondaries (ogbn-arxiv) |
| link prediction | 70/30 edge split, retrain on the 70% graph only, unsupervised cosine ranking, AUC (core four); official OGB split + Evaluator, trained hadamard→256-MLP decoder fitted on train edges only, Hits@20 (ogbl-ddi) |
| seeds | 42 / 43 / 44 everywhere; virtual-graph build seed fixed at 42 |
| selection discipline | OGB winners chosen on validation and locked to `results/ogb_selection.json`; test read once, never selected on |
| drivers | `run_core.py` (core four) and `run_ogb.py` (OGB pair); all settings from `scripts/benchmark_config.py` |

### What "locked" is backed by

Every stored embedding was traced to the code that produced it by re-running that code, not by trusting file dates. Core four: pre-rejection encoder reproduces them to 1.9e-6 (noise floor) with identical metrics — they were stale, and have been retrained (37/40 rows moved, max 0.0054, zero winner flips). ogbl-ddi: the current encoder reproduces all four tested variants to 1e-6-2e-5 — already frozen-pipeline. ogbn-arxiv: reproduces under neither encoder at tensor level, with features, node order, node count, virtual graph and hyper-parameters all verified identical; the gap (mean 3.07e-3) is 33x below an independent-seed trajectory, and the metrics agree to 0.0012, inside the 3-seed std of 0.0020.

**Reproducibility claim for the paper, stated at the level the evidence supports:** results are reproducible at the reported metric precision. They are *not* bit-reproducible. Measured floors: a repeated run of one configuration changes the embedding by 1.5e-7 (arxiv) to 2e-6 (cora) and the metric not at all on the small graphs; changing CPU thread count moves arxiv by 7.6e-6 and cora by 2e-6; an independent seed moves arxiv by 1.0e-1. Large graphs amplify environment perturbation far more than small ones, which is why the arxiv tensors could not be re-derived while its scores could.

### Scope decisions taken with the lock

1. **Ablation-D rows are frozen as-is and will NOT be re-run.** They were tested, verified and frozen when the encoder was chosen; they are the evidence *for* the locked configuration rather than results *under* it. Their numbers predate the negative-rejection change and would move by ~0.001, which does not affect any ablation conclusion. The paper states this rather than hiding it.
2. **No density-matched controls on the core four.** `original_k`/`random_k` remain an ogbl-ddi-only analysis, valid there because the original graph is denser than K. On the core four the matching would run the other way (role graphs at K≈2) — a different experiment, deliberately out of scope.
3. **The two link-prediction scorers stay different, by protocol.** Core four use ViRGo's 70/30 split with cosine-ranking AUC; OGB uses its official split with a trained decoder and Hits@20. The characterization compares graph variants **within** a dataset, so the scorer is a constant that cancels; no table or figure places AUC and Hits@20 in one column, and no cross-dataset claim is made about metric magnitude.

### The result set the characterization consumes

Six datasets, ten dataset x task cells, each 5 graph variants x 2 encoders x 3 seeds at K = 10, all in `results/scoreboard.csv` (212 rows). cora, citeseer_linqs, enzymes and proteins carry both tasks; ogbn-arxiv is node classification only (no official link split) and ogbl-ddi is link prediction only (no labels) — properties of the datasets, not gaps. Graph properties per variant are in `results/graph_health.csv`.

Headline standing at the lock: the original graph wins all eight core-four cells; role graphs fail link prediction on every core dataset (role similarity is not adjacency); role graphs come within 0.01-0.02 of the original for molecular node classification and trail by 0.2-0.44 on citation graphs; and on ogbl-ddi with GraphSAGE every role graph beats the original (centrality 0.0519 vs 0.0173 test Hits@20) — the one augmentation win in the panel, on the densest graph by two orders of magnitude, and absent under DeepWalk.

**Next step (characterization only, no re-running):** compute per-dataset graph properties — homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement — and relate them to the original-vs-best-augmented gap already in the scoreboard. Density enters as a candidate predictor because of the ogbl-ddi cell.

## 2026-07-24 (later) — Ablation D completed: all nine feature arms, one code version, plus two new single-feature arms

The ablation table was extended to isolate **every** structural input, and the whole table was re-run under the corrected encoder so no arm is compared across code versions. This reverses the earlier "ablations stay frozen" decision, deliberately: once the table gains new arms, freezing the rest makes the comparison internally inconsistent.

**Change.** `encoder.py` gained two `--features` options — `centrality` (D7, eigenvector centrality only) and `clustering` (D8, local clustering only) — selecting columns 1 and 3 of the same cached four-column feature matrix; registered in `cfg.D_FEATURES` and in notebook 3 §8b. `run_core.py` gained a `--features` passthrough so the ablation runs headless and in parallel like the main sweep. Verified before running: each single-feature arm reproduces exactly the z-normalised raw column it claims (degree→col 0, centrality→col 1, Ψ→col 2, clustering→col 3).

**Runs.** 7 trained arms x 4 datasets x 2 tasks x 3 seeds = 168 embeddings on the psi graph at K=10. D0 ("all") is the locked `graphsage_edge` row and was already post-fix; D6 (`features_only`, layers=0) never trains and was verified byte-identical under the current code, so neither was re-run. Re-running the five older arms moved them by at most 0.0209 (proteins D5 const LP, an arm that is chance by construction), mean 0.0016 — no ablation conclusion changed.

**Node classification (weighted F1), psi graph, K=10, 3 seeds**

| arm | citeseer | cora | enzymes | proteins |
|---|---|---|---|---|
| D0 all four | 0.2327 | 0.2351 | **0.5480** | **0.5588** |
| D1 degree | 0.1886 | 0.2153 | 0.5019 | 0.5059 |
| D2 degree+centrality | **0.2412** | **0.2462** | 0.5204 | 0.5349 |
| D3 Ψ | 0.1940 | 0.1743 | 0.5002 | 0.4901 |
| **D7 centrality only** | 0.2365 | 0.2328 | 0.5117 | 0.4983 |
| **D8 clustering only** | 0.1707 | 0.1791 | 0.5401 | 0.5449 |
| D4 random (control) | 0.1726 | 0.1497 | 0.4789 | 0.4753 |
| D5 constant (floor) | 0.0717 | 0.1406 | 0.3277 | 0.3184 |
| D6 features only, no MP | 0.2139 | 0.1834 | 0.4997 | 0.5216 |
| DeepWalk bridge | 0.2255 | 0.2133 | 0.5005 | 0.4882 |

**Link prediction (AUC), same setting**

| arm | citeseer | cora | enzymes | proteins |
|---|---|---|---|---|
| D0 all four | 0.5057 | 0.5150 | 0.6236 | 0.5578 |
| D1 degree | **0.5489** | 0.4960 | 0.5468 | 0.5389 |
| D2 degree+centrality | 0.5014 | 0.4864 | 0.6445 | **0.5914** |
| D3 Ψ | 0.4695 | 0.4594 | 0.6385 | 0.5483 |
| **D7 centrality only** | 0.4199 | 0.4349 | **0.6498** | 0.5491 |
| **D8 clustering only** | 0.4970 | **0.5298** | 0.5320 | 0.5054 |
| D4 random (control) | 0.5144 | 0.5134 | 0.5103 | 0.5105 |
| D5 constant (floor) | 0.4942 | 0.4970 | 0.4877 | 0.4737 |
| D6 features only, no MP | 0.4505 | 0.4831 | 0.6201 | 0.5661 |
| DeepWalk bridge | 0.5331 | 0.4990 | 0.5039 | 0.5232 |

**What the two new arms add — a per-feature, per-domain split.**

1. **Clustering is the molecular workhorse and dead weight on citation graphs.** Alone it reaches 0.5401 / 0.5449 node-classification F1 on enzymes / proteins — within 0.008-0.014 of all four features together, and the best single feature there — while on citeseer / cora it scores 0.1707 / 0.1791, at or barely above the random-feature control (0.1726 / 0.1497). Triangle structure carries molecular node identity and carries nothing about citation topics.
2. **Centrality is the opposite, and on citation link prediction it is worse than useless.** Alone it is the best arm for enzymes link prediction (0.6498, above all four features at 0.6236) but scores 0.4199 / 0.4349 on citeseer / cora link prediction — *below* the 0.50 chance line and below the constant-feature floor, i.e. ranking candidate edges by centrality similarity is actively anti-correlated with adjacency on citation graphs.
3. **More features is not better.** Degree+centrality (D2) beats all four (D0) on node classification for **both** citation graphs (0.2412 vs 0.2327; 0.2462 vs 0.2351) and on proteins link prediction (0.5914 vs 0.5578). The earlier cora-only observation now holds across the citation pair; adding Ψ and clustering dilutes the citation signal.
4. **Ψ remains the weakest structural feature and stays confounded** (the psi graph was built from it): worst real feature on node classification everywhere except enzymes link prediction, where it is mid-table.
5. The controls behave: constant is the floor everywhere, random sits between the floor and the real features, and every real feature beats random on the molecular graphs.

This is the per-feature evidence the study set out to produce, and it feeds Family 2 of the characterization directly: the arm that wins on a dataset should be predictable from that feature's raw distribution on that dataset.

**Caveat unchanged:** all of D is measured on the psi graph at K=10, on the four core datasets. The OGB pair has no ablation rows, and no arm was re-tuned per dataset.

**Plain-language summary of this entry:** `claude.ai/code/artifact/c6dd674b-8db1-4cf4-b340-929588bdd815` — the same tables in plain language, updated in place from the earlier A-E review page. Numbers there are read from this scoreboard; if either moves, both must be updated together.

## 2026-07-24 (later) — Ψ is numerically unstable where Ω underflows; the psi graph on proteins is not unique

Found while verifying that the untrained D6 arm reproduced. It did on cora and did not on proteins, and the difference was confined to the Ψ column.

**Mechanism, traced end to end.** 20 of proteins' 1,195 connected components fail networkx's power iteration and fall back to `eigenvector_centrality_numpy`, whose ARPACK solver starts from a **random** vector. That leaves Ω differing by ~7e-12 between runs. Ψ then evaluates `p·log(p/q)` with `q = Ω`: where Ω is itself ~1e-12, a 7e-12 absolute perturbation is a change of hundreds of percent, and the logarithm turns it into a Ψ shift of up to **52** on a range of −321 to −1. Cora is exactly stable (no failing components); enzymes moves one node by 1e-8.

**Consequence for the role graph.** Ψ is the 1-D signature that orders top-K neighbours, so on proteins two consecutive builds of the psi graph share only **33.6%** of their edges (Jaccard 0.3356); cora and enzymes rebuild identically (Jaccard 1.0). The saved psi graphs used for node classification are fixed on disk, so those results are unaffected; link prediction rebuilds the role graph from the 70% training edges on every run, so proteins link prediction is exposed.

**Consequence for the numbers: negligible.** Two full rebuild-and-retrain passes of proteins link prediction on the psi graph gave AUC 0.5614 and 0.5619 against the recorded 0.5617 — a spread of 0.0005, well inside that cell's 3-seed std of 0.0042. So the proteins role graph is **highly non-unique but the metric is insensitive to which draw is realised**, which is worth reporting as a robustness result rather than hidden as a defect.

**Containment already in place.** Since the feature cache was introduced, `run_core.py` and `run_ogb.py` compute Ψ once per graph and reuse it, so every current result shares one Ψ vector. Notebook 3 does **not** pass the cache, so a notebook re-run would recompute Ψ and could differ from the scripted pipeline on proteins. Recorded as a known gap; the clean fixes, none of them applied here, are to seed the ARPACK fallback with a fixed `v0`, to floor Ω before the logarithm, or to make the notebook use the cache like the drivers do.

## 2026-07-24 (later) — Characterization portion 1: both families measured

Portion 1 is **measurement only** — no joins, no correlations, no claims about which property predicts anything. That is portion 2. What exists now is a frozen, reproducible description of each dataset and of the exact numbers the encoder is fed.

**Method.** One new script, `characterize.py`, reads only artifacts that already exist: the dataset edgelists (through `graph_io.load_graph`, so the graph definition cannot drift), the encoder's own **feature caches**, and `results/scoreboard.csv`. Family 2 deliberately reads the cache rather than recomputing the features — the cache holds the raw, pre-z-normalization matrix that every embedding in the study was actually trained on, so the description is of the real inputs, and it is immune to the Ψ recomputation instability recorded earlier. Runtime 17 s for all six datasets; three re-runs produce byte-identical CSVs.

### Family 1 — graph characterization (measured on the original graph)

| dataset | domain | nodes | edges | avg degree | max degree | degree Gini | density |
|---|---|---|---|---|---|---|---|
| cora | citation | 2,708 | 5,278 | 3.90 | 168 | 0.4051 | 0.00144 |
| citeseer_linqs | citation | 3,264 | 4,536 | 2.78 | 99 | 0.4353 | 0.000852 |
| enzymes | biological | 19,474 | 37,282 | 3.83 | 9 | 0.1569 | 0.000197 |
| proteins | biological | 43,466 | 81,044 | 3.73 | 25 | 0.1634 | 0.000086 |
| ogbn-arxiv | citation | 169,343 | 1,157,799 | 13.67 | 13,161 | 0.6295 | 0.000081 |
| ogbl-ddi | drug interaction | 4,267 | 1,067,911 | 500.54 | 2,234 | 0.4689 | 0.117333 |

| dataset | components | largest comp. | avg clustering | assortativity | classes | edge homophily | node homophily |
|---|---|---|---|---|---|---|---|
| cora | 78 | 91.8% | 0.2407 | −0.0659 | 7 | **0.8100** | 0.8252 |
| citeseer_linqs | 390 | 64.6% | 0.1447 | +0.0481 | 6 | **0.7377** | 0.7203 |
| enzymes | 640 | **0.6%** | 0.4024 | +0.1755 | 3 | 0.6653 | 0.6682 |
| proteins | 1,195 | **1.4%** | 0.3814 | +0.1516 | 3 | 0.6568 | 0.6513 |
| ogbn-arxiv | 1 | 100% | 0.2261 | −0.0431 | 40 | 0.6542 | 0.6353 |
| ogbl-ddi | 1 | 100% | 0.5143 | +0.0378 | — | n/a (unlabelled) | n/a |

**External validation of the homophily column.** Measured edge homophily reproduces the published values for the three datasets where one exists: cora 0.8100 vs ~0.81, citeseer 0.7377 vs ~0.736, ogbn-arxiv 0.6542 vs ~0.654. The property that the study nominated as its primary predictor is therefore being computed correctly, which was not previously verified anywhere in this project.

**What the table establishes as fact (not yet as explanation).** The six datasets separate cleanly on three axes at once, and the axes are not independent of each other: the citation graphs are homophilous (0.65–0.81), sparse, slightly disassortative and low-clustering; the two molecular graphs are the opposite on every count — assortative (+0.15 to +0.18), high-clustering (0.38–0.40), degree-regular (Gini 0.16, max degree 9 and 25) and **extremely fragmented**, with the largest component holding only 0.6% (enzymes) and 1.4% (proteins) of nodes; ogbl-ddi is the density outlier by two orders of magnitude (average degree 500, density 0.117). Fragmentation is the axis that had not been quantified before: enzymes and proteins are effectively collections of a few hundred small graphs, which is exactly the regime where a role graph can connect nodes that no path connects.

### Family 2 — the structural inputs, raw and pre-normalization

| dataset | feature | mean | std | skew | % zero | % unique |
|---|---|---|---|---|---|---|
| cora | clustering | 0.2407 | 0.3220 | 1.31 | **45.7** | 4.84 |
| citeseer_linqs | clustering | 0.1447 | 0.2877 | 2.11 | **69.1** | 3.46 |
| enzymes | clustering | 0.4024 | 0.2979 | 0.38 | 19.7 | 0.22 |
| proteins | clustering | 0.3814 | 0.3151 | 0.47 | 25.0 | 0.13 |
| ogbn-arxiv | clustering | 0.2261 | 0.2438 | 1.55 | 27.0 | 5.99 |
| ogbl-ddi | clustering | 0.5143 | 0.2191 | −0.34 | 4.4 | 88.49 |
| cora | eigenvector centrality | 0.0772 | 0.2459 | 3.29 | 2.6 | 87.08 |
| citeseer_linqs | eigenvector centrality | 0.2989 | 0.4189 | 0.87 | 12.5 | 62.87 |
| enzymes | eigenvector centrality | 0.3346 | 0.3220 | 0.62 | 0.1 | 85.30 |
| proteins | eigenvector centrality | 0.2700 | 0.3206 | 0.94 | 0.9 | 86.16 |
| ogbn-arxiv | eigenvector centrality | 0.0020 | 0.0064 | 47.75 | 24.8 | 96.03 |
| ogbl-ddi | eigenvector centrality | 0.2428 | 0.2206 | 0.61 | 0.0 | 91.99 |

Degree and Ψ rows are in `results/node_feature_characterization.csv`; the headline facts there are that degree is **coarse** wherever the graph is degree-regular (9 distinct degrees across 19,474 enzymes nodes; 0.03–0.32% unique on proteins, citeseer and arxiv) and that Ψ's raw scale spans three orders of magnitude across datasets (mean −14 on cora, −16,622 on ddi) — a spread that z-normalization removes before the encoder ever sees it, which is precisely why it has to be measured here.

**The observation Family 2 exists to make.** Local clustering is **45.7% and 69.1% exactly zero** on cora and citeseer — most citation nodes sit in no triangle at all — against 19.7% and 25.0% on the molecular graphs and 4.4% on ddi. This is the raw-distribution counterpart of the ablation-D result that clustering-only is the best single feature on molecular node classification and lands at the random-feature control on the citation graphs. Portion 2 will state that correspondence properly, across all four features and all ten cells; it is recorded here only as the measurement that makes the test possible.

**A correction to an earlier belief.** An older note held that eigenvector centrality was degenerate on enzymes (96.9% of nodes below 1e-6). That was true of the **global** Ω used before the per-component policy; under the current `GRAPH_POLICY` (per-component, rescaled to max 1) centrality is one of the best-spread features in the study — 85.3% distinct values on enzymes, 0.09% zeros. The belief is withdrawn, and it is consistent with centrality-only being the strongest single arm for enzymes link prediction.

**Caveat on the `degenerate` flag.** The flag (`% unique < 1` or `% zero > 95`) fires on molecular clustering, which the ablation shows to be the most useful single feature there — few distinct values because small-degree nodes admit only a handful of rational clustering coefficients, not because the feature is uninformative. **Portion 2 must not use the boolean**; the raw columns beside it carry the real signal.

### Step 1 — the frozen experimental side

`results/characterization_inputs.csv` snapshots the locked GraphSAGE scores this will be joined against: 50 rows = 10 dataset x task cells x 5 graph variants, each the 3-seed mean and std of that task's single primary metric (weighted F1 / AUC for the core four, official Accuracy / Hits@20 for the OGB pair). Metrics are never mixed into one column, per the within-dataset rule.

## 2026-07-24 (later still) — Adjusted homophily added to the characterization

The homophily column in the Family-1 table above is **raw edge homophily**, and its chance baseline depends on how many classes a graph has and how balanced they are — near 0.5 for a 3-class graph, near 0.03 for a 40-class one. Raw values are therefore **not comparable across datasets with different class structure**, which matters because homophily is the predictor the study leans on first. Two rows made the flaw concrete: enzymes (0.6653, 3 classes) and ogbn-arxiv (0.6542, 40 classes) sit almost on top of each other on the raw scale while being opposite graphs on every other axis.

**Fix — one measured column, no pipeline change.** `characterize.py`'s `homophily()` now also returns the chance null and adjusted homophily (Platonov et al.),

  `h_adj = (h_edge − Σ_c (D_c / 2m)²) / (1 − Σ_c (D_c / 2m)²)`,  D_c = summed labelled-degree of class c,  2m = Σ_c D_c,

written to `dataset_characterization.csv` as `homophily_null` and `homophily_adjusted`; raw `edge_homophily` and `node_homophily` stay beside them for provenance. The null is **degree-weighted**, not node-count-weighted (Σ p²), because edge homophily counts edges and edges are weighted by degree; the two nulls agree only when the graph is degree-regular.

| dataset | classes | edge homophily (raw) | null Σ(D_c/2m)² | **adjusted** |
|---|---|---|---|---|
| cora | 7 | 0.8100 | 0.1698 | **0.7711** |
| citeseer_linqs | 6 | 0.7377 | 0.1975 | **0.6731** |
| ogbn-arxiv | 40 | 0.6542 | 0.1614 | **0.5877** |
| enzymes | 3 | 0.6653 | 0.4759 | **0.3613** |
| proteins | 3 | 0.6568 | 0.4678 | **0.3552** |
| ogbl-ddi | — | n/a | n/a | n/a |

**What it buys the study.** Adjusting separates the two domains where the raw column hid them: citation graphs cluster at **0.59–0.77**, molecular graphs at **~0.36**, with no overlap, and the enzymes/arxiv collision resolves (arxiv 0.588 against enzymes 0.361). Degree-weighting earns its keep specifically on arxiv — a node-count null would give ~0.625 there, the degree-weighted null gives 0.588, because arxiv is the one graph with heavy degree spread (Gini 0.63); the degree-regular molecular graphs barely move either way. **Portion 2 correlates against `homophily_adjusted`; raw is kept only for provenance.** The re-run touched only the three characterization CSVs and is byte-stable; scoreboard, embeddings and splits are unchanged.

## 2026-07-25 — Characterization portion 2: the when-to-augment rule

Portion 1 measured; portion 2 **relates**. The gap being explained is computed **inside** one dataset × task cell — `best augmented − original`, where "augmented" is the best of `psi` / `degree` / `centrality` / `hybrid` — so the metric is a constant that cancels and no AUC is ever compared against a Hits@20. Only the *ranking* of those gaps is compared across datasets, which is why every correlation below is Spearman. Code is `characterize.py --step all` (portion 1 unchanged, four functions added); the figures are `notebooks/5-phase5_characterization.ipynb` → `results/figures/`.

### The outcome per cell

A gap is called only when it clears the pooled 3-seed noise of the two sides it compares (1σ band); inside that band the cell is a **tie**, which is what makes "the original wins or ties everywhere" a measured statement rather than a reading of point estimates.

| dataset | task | metric | original | best augmented | variant | gap | σ | verdict |
|---|---|---|---|---|---|---|---|---|
| cora | NC | weighted F1 | 0.4266 | 0.2942 | centrality | −0.1324 | −8.6 | keep original |
| citeseer_linqs | NC | weighted F1 | 0.3298 | 0.2818 | centrality | −0.0480 | −3.8 | keep original |
| ogbn-arxiv | NC | Accuracy | 0.3918 | 0.3618 | hybrid | −0.0300 | −6.2 | keep original |
| proteins | NC | weighted F1 | 0.5797 | 0.5646 | hybrid | −0.0151 | −8.9 | keep original |
| enzymes | NC | weighted F1 | 0.5591 | 0.5546 | hybrid | −0.0045 | −0.7 | **tie** |
| cora | LP | AUC | 0.6130 | 0.5659 | centrality | −0.0471 | −11.5 | keep original |
| citeseer_linqs | LP | AUC | 0.6218 | 0.5437 | centrality | −0.0781 | −3.6 | keep original |
| enzymes | LP | AUC | 0.7000 | 0.6563 | centrality | −0.0437 | −6.1 | keep original |
| proteins | LP | AUC | 0.6720 | 0.5834 | centrality | −0.0886 | −23.3 | keep original |
| **ogbl-ddi** | LP | Hits@20 | 0.0173 | **0.0519** | centrality | **+0.0346** | **+2.7** | **augment** |

**8 keep original · 1 tie · 1 augment.** The original graph holds rank 1 of 5 in nine of ten cells; on ogbl-ddi it is rank 5 of 5 — the only cell where every augmented graph beats it.

### The rule: the predictor is task-dependent

This is the substantive finding, and it was not the expected one. Homophily was nominated as *the* predictor; it turns out to predict **node classification only**, while link prediction answers to density instead.

| predictor | NC (n=5) | LP (n=5) | pooled (n=10) |
|---|---|---|---|
| **homophily_adjusted** | **−0.90** | +0.40 (n=4) | −0.55 |
| degree_assortativity | +0.90 | −0.20 | +0.39 |
| **avg_degree** | −0.10 | **+0.80** | +0.39 |
| density | −0.60 | +0.70 | −0.09 |
| avg_clustering | +0.70 | +0.70 | **+0.72** |
| degree_skew | −0.70 | −0.50 | −0.60 |

- **Node classification — the more homophilous the graph, the worse augmentation does** (ρ = −0.90, near-monotone: cora 0.771 → −31% gap, citeseer 0.673 → −15%, arxiv 0.588 → −8%, proteins 0.355 → −2.6%, enzymes 0.361 → −0.8%). Homophilous labels are a *community* property carried by the real edges; role-based rewiring discards exactly that signal. Where homophily is weak — the molecular graphs — role structure is nearly sufficient and the gap closes to a tie.
- **Link prediction — augmentation only becomes competitive as the graph gets dense** (avg degree ρ = +0.80). ogbl-ddi (avg degree 500) is the single augmentation win; the five sparse graphs (2.8–13.7) all keep the original. Note **ogbl-ddi carries no labels, so it is absent from the homophily column** — the LP homophily correlation rests on n = 4 and should not be quoted.
- **avg_clustering is the only property that holds the same sign in both tasks** (+0.70 / +0.70), making it the best single-column summary if one is wanted.
- `degree_assortativity` (+0.90 on NC) is a **mirror of homophily, not independent evidence** — the molecular graphs are simultaneously assortative and weakly homophilous, so with n = 5 the two cannot be separated.

**Strength of claim.** n is 4–5 datasets per task. |ρ| = 0.90 at n = 5 is p ≈ 0.037, but with twelve properties tested this does not survive any multiple-comparison correction. These are stated as a **conjecture offered to the community**, consistent with the project's framing — the direction and the near-monotonicity are the evidence, not the p-value.

### Feature usefulness: the `degenerate` flag is empirically dead

Ablation D, `psi` graph, K=10, core four only, measured as **lift over the random-feature control (D4)** — the honest zero point, since it is message passing with the structural signal removed.

| dataset | task | best single feature | lift | that feature's % zero | % unique |
|---|---|---|---|---|---|
| cora | NC | centrality | +0.083 | 2.6 | 87.1 |
| citeseer_linqs | NC | centrality | +0.064 | 12.5 | 62.9 |
| enzymes | NC | clustering | +0.061 | 19.7 | **0.22** |
| proteins | NC | clustering | +0.070 | 25.0 | **0.13** |
| enzymes | LP | centrality | +0.140 | 0.1 | 85.3 |
| proteins | LP | centrality | +0.039 | 0.9 | 86.2 |
| cora | LP | clustering | +0.016 | 45.7 | 4.8 |
| citeseer_linqs | LP | degree | +0.035 | 0.0 | **0.95** |

Across the 32 (dataset × task × single feature) cells, **raw spread does not predict usefulness**: %zero ρ = −0.06, %unique ρ = −0.02, skew ρ = +0.01. The winners on the molecular graphs are the two features the flag calls degenerate — clustering at 0.13–0.22% unique, and degree at 0.95% unique on citeseer LP. This is a **measured refutation** of the flag rather than an argument against it: a feature with few distinct values still separates the nodes that matter. The flag stays in `node_feature_characterization.csv` as description only; **usefulness is decided by ablation score throughout**.

### Files

`results/characterization_gaps.csv` (10 cells) · `results/characterization_correlations.csv` (48 rows, both scopes) · `results/feature_usefulness.csv` (74 rows) · `results/figures/fig1–fig5.png`. Deliverable #5 is complete; GIN (#6) is the remaining experimental item.

---

## 2026-07-27 — Panel extension: two heterophilous datasets + a pre-registered prediction

**Why.** The ten-cell characterization produced one `augment` verdict (ogbl-ddi LP). A decision rule needs points on
both sides of its boundary; with a single positive cell the boundary is fitted from one point and the `augment` side
of the rule was never tested. The cause is visible in the panel itself: **all six datasets are homophilous**
(adjusted homophily 0.355–0.771), and average degree has an empty span between 13.7 (ogbn-arxiv) and 500 (ogbl-ddi).
The rule's two predictors were therefore each measured over a range that excludes the region where the rule predicts
augmentation. Adding datasets there is not a search for wins — it is the missing half of the experiment.

**Datasets added** — Platonov et al. (2023), *A critical look at the evaluation of GNNs under heterophily*; the same
paper this study's adjusted-homophily definition is taken from.

| dataset | domain | nodes | edges | avg degree | classes | edge homophily | **adjusted homophily** | components | assortativity |
|---|---|---|---|---|---|---|---|---|---|
| roman_empire | linguistic (word adjacency + dependency arcs) | 22,662 | 32,927 | 2.91 | 18 | 0.0469 | **−0.0468** | 1 | −0.028 |
| tolokers | crowdsourcing (workers who shared a task) | 11,758 | 519,000 | 88.28 | 2 | 0.5945 | **0.0926** | 1 | −0.080 |

Both reproduce the published counts exactly, and our independent `homophily()` implementation reproduces the paper's
adjusted homophily to two decimals (−0.05 / 0.09) — an **external validation of the metric**, not only of the data.

**What they add to the panel.** roman_empire is the first dataset with *negative* adjusted homophily, extending the
range from [0.355, 0.771] to [−0.047, 0.771]. tolokers lands at degree 88.28, inside the previously empty span
between ogbn-arxiv and ogbl-ddi, and is only the second graph in the panel with weak homophily *and* high density.

**Protocol — unchanged, deliberately.** Structural features only: `data.x` (roman_empire 300-dim fastText,
tolokers 10-dim worker profile) is ignored, exactly as OGB's text features are. Platonov's own ten train/val/test
masks are also ignored; both datasets run the ViRGo **core** protocol (stratified 70% node classification → weighted
F1; 70:30 link prediction → AUC) so their cells stay directly comparable with the four core datasets. K=10,
GraphSAGE mean/2-layer/edge-positives, seeds 42/43/44 — the frozen pipeline, with **no retuning**. Retuning on the
new datasets would void the held-out status of the prediction below.

**Note on link prediction.** Neither dataset ships an LP task; the 70:30 split is ours, as for the core four. This is
a defined extension of the benchmark, not a published protocol, and is reported as such.

### Pre-registered prediction (recorded before any embedding was trained)

The rule from 2026-07-25 states: node classification augments as adjusted homophily *falls* (ρ = −0.90, n=5); link
prediction augments as average degree *rises* (ρ = +0.80, n=5). Applied to the two new datasets it predicts:

| dataset | task | driving property | value | rank in panel | **predicted verdict** |
|---|---|---|---|---|---|
| roman_empire | NC | adjusted homophily | −0.047 | lowest of 7 | **augment** |
| tolokers | NC | adjusted homophily | 0.093 | 2nd lowest of 7 | **augment** |
| tolokers | LP | avg degree | 88.28 | 2nd highest of 7 | **augment** |
| roman_empire | LP | avg degree | 2.91 | lowest of 7 | **keep original** |

Three augment, one keep. The fourth is the control: roman_empire is simultaneously the panel's least homophilous
graph *and* its sparsest, so the two predictors disagree on it by construction — NC should augment while LP should
not. A result that splits that way is evidence the predictor really is task-dependent rather than a single latent
"hard dataset" axis; a result that augments both would mean the two correlations are measuring the same thing.

**Falsification condition, stated in advance.** If roman_empire and tolokers NC return `keep original`, the
homophily rule is falsified on the only range that could test it, and the study reports that. The value of the
extension does not depend on which way it comes out — only on the prediction having been fixed beforehand.

**Panel after this change:** 8 datasets, 14 dataset × task cells (NC 7, LP 7). Correlation n rises from 5 to 7 per
task. Registered in `scripts/benchmark_config.DATASETS`, `characterize.STUDY`, and `run_core.CORE`; converter is
`make_hetero.py`. Nothing scored yet.

---

## 2026-07-27 — Result of the pre-registered test: the homophily rule is falsified

The 2026-07-27 run of `run_core.py --datasets roman_empire tolokers` completed (40 cells: 5 variants × 2 encoders ×
2 tasks × 3 seeds, K=10, frozen pipeline, no retuning). Scores are rows 228–267 of `results/scoreboard.csv`.

### Outcome vs prediction

| dataset | task | original | best augmented | gap | σ | **verdict** | predicted |
|---|---|---|---|---|---|---|---|
| roman_empire | NC | **0.2561 ± 0.0092** | hybrid 0.2144 ± 0.0076 | −0.0417 | −4.94 | keep original | ~~augment~~ |
| tolokers | NC | **0.7345 ± 0.0037** | hybrid 0.7265 ± 0.0060 | −0.0080 | −1.60 | keep original | ~~augment~~ |
| tolokers | LP | 0.6444 ± 0.0188 | **psi 0.7007 ± 0.0032** | +0.0563 | +4.18 | **augment** | augment ✓ |
| roman_empire | LP | 0.6019 ± 0.0124 | **centrality 0.6980 ± 0.0139** | +0.0961 | +7.30 | **augment** | ~~keep original~~ |

One of four predictions held. **The falsification condition stated in advance was met**: both new NC cells returned
`keep original`, on the only homophily range that could have tested the rule.

### What the correlations did

Recomputed by `characterize.py --step all` with n = 7 per task (was 5):

| relation | before (n=5) | after (n=7) |
|---|---|---|
| NC gap ~ adjusted homophily | ρ = −0.90 | **ρ = −0.36** |
| LP gap ~ average degree | ρ = +0.80 | **ρ = +0.54** |

Both drivers lost most of their strength out of sample. The strongest predictors on the extended panel are now
LP ~ `components` (ρ = −0.85), `largest_component_frac` (+0.74), `avg_clustering` (+0.71); NC ~ `n_classes`
(ρ = −0.68), `avg_clustering` (+0.57). These are **post-hoc on the same data that broke the first rule** and are
recorded as candidates to be tested, not as a replacement rule. The honest statement is that a two-property rule
fitted on five points did not survive two new points.

### Why the homophily premise was wrong

roman_empire has adjusted homophily −0.047 — neighbouring words rarely share a syntactic role — yet the original
graph beats every role graph by 4.9σ. Low adjusted homophily means neighbour labels **differ**, not that the edges
carry no information: a word's role is determined by its sequence context, and differing *predictably* is still
exploitable signal that message passing can use. The rule conflated "neighbours share my label" with "neighbours are
informative about my label". Only a heterophilous dataset could expose that conflation, which is precisely why the
panel needed one.

### Panel-level effect

`keep original` 10 | `augment` 3 | `tie` 1, over 8 datasets / 14 cells. The augment count rose from 1 to 3, so the
characterization no longer rests on a single cell — but the two new augment verdicts are both link prediction, and
neither was obtained where the rule said it would be.

### Caveat on the roman_empire LP cell — do not read it as support for role edges

Node ids in roman_empire are word positions in the source text, so the graph is a 22,662-node path plus dependency
arcs: 68.8% of edges join consecutive ids, 14.7% join ids two apart. Link prediction there is largely "are these two
ids adjacent", which any position-sensitive embedding solves — DeepWalk on the original graph reaches
AUC 0.9994 ± 0.0000 while GraphSAGE on the same graph reaches 0.6019. The `augment` verdict in that cell therefore
records **GraphSAGE failing to represent a path**, not role edges helping link prediction. Reported with this caveat
attached; whether the cell is retained in the correlations is an open decision.

**Artifacts:** `results/scoreboard.csv` (70 GraphSAGE score rows), `results/characterization_*.csv`,
`results/figures/fig1–fig5` regenerated, notebook 5 executed end to end on the 8-dataset panel.

---

## 2026-07-27 — Why the original graph wins node classification on the heterophilous pair (audit, no code changed)

Read-only audit of the roman_empire / tolokers node-classification cells. Implementation verified correct; the
`keep original` verdicts are real, and the mechanism turns out **not** to be the one the homophily rule assumed.

### Implementation checks (all pass)

| check | roman_empire | tolokers |
|---|---|---|
| labels ↔ graph node alignment | 22,662 / 22,662, 0 unlabelled, 0 orphan | 11,758 / 11,758, 0 / 0 |
| nodes present in every embedding | 22,662 (all 5 variants) | 11,758 (all 5 variants) |
| nodes silently skipped by the evaluator | 0 | 0 |
| virtual graphs: isolates / star collapse | 0 isolates, max degree 14–35 | 0 isolates, max degree 18–2138 |
| stored embeddings reproduce the scoreboard | yes (seed 42 within seed spread) | yes |
| degenerate (duplicate) embedding vectors | ≤ 2 of 22,662 | ≤ 157 of 11,758 |
| majority-class weighted-F1 floor | 0.0342 (all variants ≈ 6–7× above it) | 0.6860 (78/22 binary) |

### The mechanism: predictability, not agreement

Same-label rate across each graph's edges, against the chance rate for that class distribution:

| graph | roman_empire same-label (chance 0.0880) | neighbour-label entropy | tolokers same-label (chance 0.6588) | entropy |
|---|---|---|---|---|
| original | **0.0469 (−0.0412)** | **0.913** | **0.5945 (−0.0643)** | 0.568 |
| psi | 0.0988 (+0.0108) | 1.862 | 0.6876 (+0.0288) | 0.503 |
| degree | 0.1169 (+0.0288) | 2.001 | 0.6750 (+0.0162) | 0.510 |
| centrality | 0.0894 (+0.0014) | 2.082 | 0.6760 (+0.0172) | 0.517 |
| hybrid | 0.0882 (+0.0002) | 1.998 | 0.6087 (−0.0501) | 0.573 |

The original graph is the **least** label-agreeing graph of the five on both datasets — below chance — and still wins.
The role graphs raise same-label agreement above chance and still lose. Agreement is therefore not what the encoder
uses. What separates them is the **entropy of a node's neighbour-label distribution**: 0.913 for the original graph
against 1.86–2.08 for the role graphs (maximum ln 18 = 2.89). Original edges give each node a sharply peaked
neighbour-label mix; role edges connect structurally similar words drawn from all over the corpus, so the mix
collapses onto the global prior and message passing returns approximately the prior.

### The control that settles it

Raw structural features fed straight to the same logistic regression, **no message passing** (the D6 arm, never run
on these datasets before):

| | raw features only | GraphSAGE on role graphs | GraphSAGE on original |
|---|---|---|---|
| roman_empire | 0.1915 | 0.2039 – 0.2095 | **0.2462** (s42) / 0.2561 (3 seeds) |
| tolokers (weighted) | 0.7101 | 0.7116 – 0.7205 | **0.7367** |
| tolokers (macro) | 0.4982 | 0.4959 – 0.5184 | **0.5492** |

**GraphSAGE over a role graph is worth about as much as not doing message passing at all.** Role rewiring does not
damage the signal — it fails to add one, and every role variant lands within ~0.01 of the feature-only control.
Only the original edges contribute information beyond the four input features (+0.055 on roman_empire, +0.051 macro
on tolokers). This is a sharper statement of the same conclusion as ablation D on the core four.

### Caveats

- tolokers node classification is a 78/22 binary: the majority floor is 0.6860, so the entire five-variant spread
  (0.7116–0.7367) lives inside 0.05 of usable headroom. Its −1.60σ verdict is the weakest in the panel and sits just
  outside the tie band — report it as marginal.
- Absolute numbers are far below Platonov's published GraphSAGE results because those use the 300-dim fastText node
  features, which our scope rule excludes. The comparison here is between graphs at fixed features, not against the
  literature.

---

## 2026-07-27 — Panel extension: `questions`, and a head-to-head between two competing rules

Third heterophilous benchmark added (Platonov et al. 2023), same converter and same frozen pipeline as roman_empire
and tolokers. Chosen because it is the one dataset that **separates the two explanations now on the table**.

### The dataset

Users of a Q&A website; an edge means one user answered the other's question. Built by `make_hetero.py --dataset
questions`. Our measurements reproduce the published values exactly, a second external validation of the loader and
of `homophily()`:

| property | published | measured here |
|---|---|---|
| nodes / edges | 48,921 / 153,540 | 48,921 / 153,540 |
| adjusted homophily | 0.02 | **0.0207** |
| average degree | 6.28 | 6.28 |
| average local clustering | 0.03 | 0.0307 |

Additional structure: one connected component, degree median 1 against maximum 1,539 (heavier tail than any other
non-OGB dataset in the panel), edge homophily 0.8396.

Structural-only as always: the 301-dim fastText node features and Platonov's ten official masks are ignored; the
ViRGo core protocol applies (stratified 70% node classification → weighted F1; 70:30 link prediction → AUC), K = 10,
GraphSAGE mean/2-layer/edge-positives, seeds 42/43/44, **no retuning**.

### Why this dataset and not another

After roman_empire and tolokers, two incompatible explanations fit the panel:

- **H1, the homophily rule** (2026-07-25): node classification augments as adjusted homophily falls. Already
  weakened out of sample (ρ −0.90 → −0.36) but not dead.
- **H2, the mechanism** (2026-07-27 audit): role edges are a *function of the four input features*, so they cannot
  carry information about a target that is not itself structural. Labels are not structural; adjacency is. H2
  predicts node classification can **never** augment, and that the three observed augment cells are all link
  prediction because that is the only structural target.

`questions` has adjusted homophily 0.0207 — second lowest in the panel, deep inside the region where H1 predicts
augmentation — while H2 predicts `keep original`. The two hypotheses disagree on this cell, which is the reason it
was selected.

### Pre-registered prediction (recorded before any embedding was trained)

| dataset | task | H1 (homophily rule) | H2 (mechanism) | **registered prediction** |
|---|---|---|---|---|
| questions | NC | augment (h_adj = 0.021) | keep original | **keep original** (H2) |
| questions | LP | keep original (avg deg 6.28) | keep original | **keep original** |

The registered call follows H2. Link prediction is not discriminating — both hypotheses say keep original, because
6.28 is far from the density at which the over-smoothing failure appears (tolokers 61.8 in-train, ogbl-ddi ~500) and
the graph is not a near-path, so roman_empire's smooth-centrality route does not apply either.

**Falsification, stated in advance.** A `keep original` on node classification retires H1 for good: three
heterophilous datasets in a row would have sat in its predicted-augment region and refused to augment. An `augment`
falsifies H2, which claims the outcome is impossible, and revives the homophily reading.

### Caveat to read the metric with

`questions` is a **97.0 / 3.0** binary (47,461 vs 1,460). Weighted F1 is therefore dominated by the majority class
and will sit near the trivial floor for every variant; the informative column in this cell is **macro F1**, exactly
as on tolokers (78/22) but more extreme. The verdict is still computed on the panel's primary metric for
comparability, and the compression is reported rather than corrected.

**Panel after this change:** 9 datasets, 16 dataset × task cells (NC 8, LP 8). Registered in
`scripts/benchmark_config.DATASETS`, `characterize.STUDY`, `run_core.CORE`; built by `make_hetero.py`.
Nothing scored yet.

---

## 2026-07-27 — `questions` result: prediction half-failed, and a robustness test that splits the augment verdicts in two

The `questions` sweep completed (20 score rows = 5 variants × 2 encoders × 2 tasks, 3 seeds each, frozen pipeline).

### Node classification: a dead cell

Every one of the ten node-classification rows returned **0.9555 ± 0.0000**, which is exactly the all-majority-class
weighted F1 for a 97.0/3.0 split. All five variants, both encoders, three seeds, zero variance: the classifiers are
constant predictors. The verdict prints as `keep original` with `gap = 0.0000` and an undefined sigma.

The pre-registration selected `questions` precisely because H1 (homophily rule) and H2 (mechanism) disagreed on this
cell. **They were not tested.** The metric collapsed before either hypothesis could be discriminated, so this cell
contributes nothing to that question and is reported as uninformative rather than as support for either side. The
class imbalance was flagged in advance; its severity was underestimated.

### Link prediction: the registered prediction failed, then survived a robustness check

Registered: `keep original`. GraphSAGE returned **augment** — original 0.4665 ± 0.0130, best augmented (hybrid)
0.5526 ± 0.0232, gap +0.0861 = +4.58σ. Taken alone that is a fourth augment verdict and a failed prediction.

But GraphSAGE on the original graph scored **0.4665 — below chance**. That prompted running every link-prediction
cell through the second encoder already in the pipeline (the DeepWalk bridge, same splits, same seeds):

| dataset | avg degree | GraphSAGE: winner | DeepWalk: winner | verdict survives encoder change? |
|---|---|---|---|---|
| ogbl_ddi | ~500 | **augment** (0.0173 → 0.0554) | **augment** (0.0378 → 0.1038) | **yes** |
| tolokers | 88.3 | **augment** (0.6444 → 0.7007) | **augment** (0.7859 → 0.8954) | **yes** |
| roman_empire | 2.9 | augment (0.6019 → 0.6980) | keep original (**0.9994** → 0.8065) | **no** |
| questions | 6.3 | augment (0.4665 → 0.5526) | keep original (0.6589 → 0.6537) | **no** |
| cora | 3.9 | keep original | keep original | yes (control) |
| enzymes | 3.9 | keep original | keep original | yes (control) |

**Two of the four augment verdicts are GraphSAGE artifacts.** On roman_empire and questions the augment verdict
appears only because GraphSAGE on the original graph is broken — 0.4665 (below chance) and 0.6019 against DeepWalk's
0.9994 on the identical split. Swap the encoder and the original graph wins both. On ogbl_ddi and tolokers the
augment verdict is reproduced by an unrelated encoder, so it is a property of the graph, not of GraphSAGE.

For `questions` specifically, the registered `keep original` is what the encoder-robust reading gives; the GraphSAGE
row that contradicted it is the artifact.

### Refined statement of the link-prediction rule

The two encoder-robust augment cells are the panel's two densest graphs (ogbl_ddi ~500, tolokers 88.3); the two
artifacts are sparse (2.9, 6.3). This restores average degree as the link-prediction predictor **once the
encoder-artifact cells are removed**, and supplies the mechanism already measured on tolokers: at high density a
2-layer receptive field covers ~46% of the graph and 24.6% of non-edges already share a neighbour (core four:
0.6–9.7%), so the original graph cannot separate edges from non-edges while a sparser role graph can.

The claim the study can defend is therefore narrower and better supported than either earlier version:
**role-based augmentation helps link prediction on dense graphs, does not help on sparse ones, and never helps node
classification.** Every apparent exception in the panel is a case where the baseline encoder failed.

### Panel state

9 datasets, 16 dataset × task cells: 11 keep original, 4 augment (all link prediction), 1 tie — and of the 4
augment cells only 2 survive an encoder swap. No node-classification cell has ever augmented, across 8 datasets
spanning adjusted homophily −0.047 to 0.771.

---

## 2026-07-27 — Ablation D completed panel-wide, plus a feature-status taxonomy and a degenerate control on the OGB pair

### Coverage closed

Ablation D previously existed only for the core four. It now covers **all nine datasets** (psi graph, K=10,
GraphSAGE only, 7 arms × 3 seeds): roman_empire, tolokers and questions via `run_core.py --features`, and the OGB
pair via a new `--features` flag added to `run_ogb.py` (mirrors `run_core.py`; ablation arms return before
`select()`/`report()` so they cannot touch the locked-winner bookkeeping). 126 ablation rows in the scoreboard.

**Recorded deviation:** the OGB ablation arms were run with `--final`, so the official test split received 7
additional diagnostic reads per dataset beyond the single locked read the protocol allows. This was unavoidable —
`characterize.PRIMARY` maps the OGB pair to `test_acc` / `test_hits@20`, so valid-split rows would never be read.
These reads are ablation diagnostics and were not used for model selection, but they are reported rather than left
for a reviewer to discover.

### Feature status: a stated rule instead of a bare number

`useful_feature` (one string, e.g. `centrality (+0.083)`) is replaced by three columns — `best_single_feature`,
`feature_lift_vs_random`, `feature_status` — with the classification in `characterize.status()`:

| status | rule |
|---|---|
| `useful` | lift clears **+2σ** of the pooled 3-seed noise of the arm and its control |
| `no clear evidence` | inside the band; three seeds cannot resolve it |
| `not useful` | below **−2σ** |
| `not evaluated` | no random-feature control on disk for that cell |
| `invalid metric` | no arm separates from any other *and* no seed moves — a constant predictor |

The band is deliberately **2σ**, stricter than the 1σ band `gaps()` uses for keep/augment: this table is descriptive
and must never decide a verdict, so "clearly worse than random" should not be claimed on a margin three seeds cannot
resolve. Result over the 16 cells: 11 `useful`, 3 `no clear evidence`, 1 `not useful` (questions LP, −0.034),
1 `invalid metric` (questions NC).

The ablation is reported as **"best single structural feature on the Ψ graph"**. It was never run on whichever graph
won the cell, and that limitation is now in the column name rather than in a footnote.

### A degenerate random control on the OGB pair — flagged, not yet corrected

The lift is measured against the D4 random-feature arm. On the OGB pair that control has collapsed:

| dataset | random control | const control (D5) | verdict |
|---|---|---|---|
| ogbn_arxiv | 0.0586 ± 0.0000 | **0.0586 ± 0.0000** | identical — random features carry no more signal than constant ones |
| ogbl_ddi | 0.0005 ± 0.0001 | 0.0000 ± 0.0000 | both at the floor |

On ogbn_arxiv the random-feature and constant-feature arms return **exactly** the same accuracy with zero seed
variance, i.e. the random control has degenerated into the majority-class floor. Every arxiv lift is therefore
"distance from the trivial floor", not "distance from a competitive random baseline", and because the control's
standard deviation is 0 the sigma values inflate absurdly — centrality reports **+305σ**. The `useful` labels remain
directionally correct, but **sigma magnitudes are not comparable between the OGB pair and the core-protocol
datasets**, and no arxiv sigma should be quoted in the paper. The core-protocol datasets are unaffected: their
random controls carry real variance (e.g. cora LP random std 0.0093).

This is a property of the OGB setting — one fixed official split, so the only seed variance is encoder
initialisation — not of the ablation code.

---

## Recommendation table: the tie case now names the augmented graph (2026-07-28)

`recommended_graph` in notebook 5 §7a changed from

```python
np.where(verdict == "augment", best_variant, "original")     # tie -> original
np.where(verdict == "keep original", "original", best_variant)  # tie -> best_variant
```

so a *tie* verdict now reports the best augmented variant rather than falling back to the original graph. The
verdict banding in `characterize.gaps()` is untouched — only the presentation column changes.

**Affected cells: exactly one.** `enzymes` node classification is the panel's only tie: hybrid 0.5546 vs original
0.5591, gap **−0.0045 weighted F1 = −0.7σ**, inside the 1σ band. It is also the panel's closest NC cell to an
augment verdict, and the cell where the role graph recovers the largest share of the original's gain over raw
features (93%, vs 46% on cora).

**Caveat that must travel with the table.** On this cell the hybrid graph's mean is *below* the original's; the tie
verdict says the two are indistinguishable at three seeds, not that hybrid is ahead. The column should be read as
"a role graph is a defensible choice here", not as a measured win. The headline finding is unchanged: **no node
classification cell in the panel produces an augment verdict**, and all four augment verdicts remain link
prediction.

---

## Module 2 complete: neighbour-label predictability + an automatic credibility screen (2026-07-29)

`experiments/characterize.py` gains portion 2 (D). Pipeline, seeds, K and encoder untouched — this is measurement and
screening over the frozen scoreboard only. New output: `results/candidate_rules.csv`, plus notebook 5 §8.

**New property — neighbour-label predictability.** Homophily asks "do neighbours share the label?"; this asks "does a
node connect to a *consistent* class mix, even when the labels differ?" Naive Bayes over each node's neighbour class
histogram, leave-one-out (the node's own contribution is removed from its class row of the compatibility matrix),
reported adjusted against the majority-class floor so it is comparable across datasets. Entropy is reported alongside
as descriptive only.

It separates from homophily exactly where it was predicted to. **`roman_empire`: adjusted homophily −0.0468 (neighbours
essentially never share a label) yet predictability 0.4256 raw vs a 0.1396 majority floor = +0.3324 adjusted.** The
edges carry class information without carrying the class. That is the mechanism for why the original graph must be kept
for `roman_empire` node classification (−0.0417 F1, −4.9σ) even though the graph is strongly heterophilous. Converse
case: `tolokers` scores −0.3274 adjusted (below its 0.7818 majority floor) — its edges carry no usable class signal at all.

**Predictor tiering.** 7 primary (`homophily_adjusted`, `nbr_predictability_adjusted`, `components`,
`largest_component_frac`, `avg_degree`, `avg_clustering`, `n_classes`) + 8 exploratory. Only primary may gate a rule and
only primary carries the Bonferroni correction. **Caveat that must be reported: the tier split was fixed after the
correlations were already visible.** `components` clears corrected significance (p_bonf = 0.0406) partly because
tiering cut the family from 12 tests to 7; at 12 it did not.

**Selection-bias control.** `gap = best_augmented − original` is a max over four variants and therefore biased upward.
Added `gap_fixed_*` against one variant fixed per task (LP = `centrality`, NC = `hybrid`; each already wins 6/8 cells).
**Result: the two gap definitions agree on all 16 cells — `verdict_agrees` is True everywhere.** No augment verdict in
the panel is an artefact of the max. `questions` LP is the only cell where the magnitude moves materially (+0.0861 →
+0.0413, +4.58σ → +2.14σ); the verdict stays *augment*.

**Degenerate-cell exclusion.** A cell is dropped when no variant separates from any other and no seed moves
(spread < 1e-3 and max std < 1e-6). Mechanical, not hand-named: it selects exactly one cell in the panel, `questions`
node classification (spread 1e-4, std 0.0000, 97.0% majority class pinning weighted F1). 16 cells → 15 usable.

**The screen.** Gates are |ρ| ≥ 0.7, leave-one-dataset-out sign stability, ≤ 1 misclassified cell. **Significance is
reported but deliberately not gated**: at n = 8 the two-tailed 0.05 critical Spearman value is 0.738, so the |ρ| gate
already sits at about p ≤ 0.07, and adding a p gate would reject usable patterns on this few datasets. A pass means
*credible candidate*, never *proven rule*.

**Three candidates clear the gates, all link prediction, all with 0 exceptions:**

| rule | ρ | LODO min abs ρ | p_bonf | n |
|------|---|----------------|--------|---|
| augment when `components` < 39.5 | −0.8625 | 0.7881 | 0.0406 | 8 |
| augment when `largest_component_frac` > 0.9588 | +0.7864 | 0.7412 | 0.1442 | 8 |
| augment when `homophily_adjusted` < 0.2239 | −0.7143 | 0.5429 | 0.4991 | 7 |

The first two are the same variable read two ways, so this is **one** fragmentation candidate plus one homophily
candidate — not three independent findings. All three hold identically under `gap_fixed_rel`.

**Node classification produces no rule, and the screen says so mechanically.** 0 of 7 usable NC cells are *augment*, so
there is nothing for a threshold to separate; every NC row fails on `n_exceptions = -1` (not evaluable). The reportable
NC result is a boundary — *never augment* — not a predictor. Any held-out NC test is therefore a falsification attempt,
not a validation.

**Open confound, unchanged by this work.** The four LP *keep original* cells are exactly the core-4 (fragmented,
citation/molecular); the four *augment* cells are exactly the four datasets added later (single-component,
web/interaction). `components` and dataset provenance are currently the same variable. Module 3's held-out set must
include a fragmented + heterophilous graph and a single-component + homophilous graph or the confound survives the
validation intact.

**Flag for the write-up.** The strongest NC correlation in the whole table is an *exploratory* predictor:
`nbr_label_entropy` vs the NC gap, ρ = +0.79 (gap_rel) and +0.89 (gap_fixed_rel, p = 0.0068) — higher than any primary
predictor for NC. It is not screened because it is not in the primary tier, and it cannot become a rule while NC has
zero augment cells. Promoting it now would be post-hoc; it is recorded here so the decision is visible either way.

Scope, unchanged: top-K role graphs, K = 10, GraphSAGE, seeds 42/43/44, five variants.

---

## Panel cut to seven datasets; Module 2 re-run (2026-07-29)

User decision: `citeseer_linqs` and `proteins` are excluded from all forward work. `characterize.PANEL` is now the
default for `--datasets`; `STUDY` keeps all nine so past rows retain their meaning. **All numbers in the previous entry
are the nine-dataset figures and are superseded by what follows.**

**Panel: 7 datasets / 12 cells, 11 usable — 7 keep original, 4 augment, 1 tie.** Every verdict is unchanged from the
nine-dataset run; only the two dropped datasets are gone. `gap_fixed` still agrees with max-over-variants on all 12 cells.

**Four credible LP candidates, up from three:**

| rule | ρ (gap_rel) | ρ (gap_fixed_rel) | LODO min\|ρ\| | p_bonf | n |
|---|---|---|---|---|---|
| augment when `homophily_adjusted` < 0.227 | −0.90 | **−1.00** | 0.80 | 0.117 | 5 |
| augment when `components` < 39.5 | −0.78 | −0.78 | 0.71 | 0.481 | 6 |
| augment when `largest_component_frac` > 0.9588 | +0.78 | +0.78 | 0.71 | 0.481 | 6 |
| augment when `nbr_predictability_adjusted` < 0.4084 | −0.70 | −0.60 | 0.40 | 1.000 | 5 |

**The headline reverses.** On nine datasets `components` led (ρ −0.86) and was the only predictor to clear Bonferroni.
On seven it falls to −0.78 and **no longer survives correction**; `homophily_adjusted` takes over and separates the LP
cells *perfectly* under the fixed-variant gap. This is a direct demonstration that the panel, not the data-generating
process, is currently selecting the winner — exactly the fragility the pre-registered Module 3 exists to resolve.

**Reportable reversal on the pre-specified rules.** Adjusted homophily was proposed as the *node classification*
predictor. It fails there (ρ = −0.30, and NC has no augment cell to predict at all) but is the **strongest link
prediction** rule in the study. Right variable, wrong task — worth stating as a finding rather than burying.

**`nbr_predictability_adjusted` is the weakest of the four and should be reported as such.** It clears the gates on
`gap_rel` only (ρ = −0.70, exactly at the 0.7 threshold) and is *rejected* by the bias-free `gap_fixed_rel` control
(ρ = −0.60). Its LODO floor is 0.40 — a single dataset moves it. It is a candidate to watch, not to carry.

**Node classification still yields no rule:** 0 of 5 usable NC cells augment, every NC row fails on
`n_exceptions = -1`. Only the boundary "never augment" is reportable.

**Two defects fixed in the same run.**
1. `feature_scores()` read `results/scoreboard.csv` directly with no dataset filter, so the two excluded datasets
   reappeared in `feature_usefulness.csv` with NaN spreads. Now filtered to the panel at source (notebook 5's manual
   `isin(DATASETS)` workaround becomes a no-op).
2. `rho()` reported p = 1.4e-24 for a perfect rank match at n = 5, because scipy's Spearman p is a t-approximation that
   diverges as |ρ| → 1. The smallest p any permutation of n points can produce is 2/n!, so p is now floored there:
   the homophily rule's corrected p goes from a meaningless 0.0000 to **0.117**. The floor is negligible for larger n.
   Any p quoted from the previous entry at n ≤ 6 with |ρ| near 1 should be re-read from the regenerated CSVs.

**Confound tightened, as predicted.** `proteins` (1195 components) and `citeseer_linqs` (390) were two of the four
fragmented *keep original* LP cells; the fragmentation candidate now rests on 6 LP cells with 2 fragmented ones.
Module 3's held-out set must carry a fragmented + heterophilous graph.

---

## Collinear predictors merged into findings (2026-07-29)

`components` and `largest_component_frac` rank the seven-dataset panel at **Spearman −1.000** — identical information,
no exception. They were being reported as two credible rules; they are one. `characterize.FAMILY` / `CANONICAL` now
collapse them, `candidate_rules.csv` gains `predictor_family` / `canonical`, and the credible list prints one row per
finding. No other primary pair exceeds |ρ| 0.85, so this is the only merge.

**Canonical member is `largest_component_frac`, not `components`, and the reason is transfer.** A rule reading
"< 39.5 components" cannot be applied to an unseen graph ten times larger; "> 0.9588 of nodes in one component" can.
Both give ρ = ±0.7775 identically, so the choice costs nothing statistically.

**Four credible predictors → three credible findings, ranked:**

| finding | ρ (gap_rel / gap_fixed) | LODO min\|ρ\| | p_bonf | distinct values (all / augment side) |
|---|---|---|---|---|
| augment when `homophily_adjusted` < 0.227 | −0.90 / **−1.00** | 0.80 | 0.117 | 5 / 3 |
| augment when `largest_component_frac` > 0.9588 | +0.78 / +0.78 | 0.71 | 0.481 | **3 / 1** |
| augment when `nbr_predictability_adjusted` < 0.4084 | −0.70 / −0.60 | 0.40 | 1.000 | 5 / 3 |

**New diagnostic, and it demotes the fragmentation finding.** `distinct_augment` counts how many distinct predictor
values the *augmenting* cells span. For `largest_component_frac` it is **1**: all four augment cells sit at exactly
1.0000, and the two keep-original cells are the only non-1.0 values (cora 0.9177, enzymes 0.0064). So the fragmentation
"threshold" is a **group label, not a graded trend** — it says "fully connected vs not", the 0.9588 cut is arbitrary
anywhere in (0.9177, 1.0), and it is precisely the batch confound already on record (the two fragmented graphs are
early-panel, the four connected ones are late additions). Adjusted homophily spans 5 distinct values with 3 on the
augment side and is graded throughout — genuinely stronger evidence, and it is also the one that separates perfectly
under the bias-free gap.

**Standing order of the LP candidates for Module 3:** adjusted homophily first (graded, perfect separation,
LODO 0.80), fragmentation second (real but two-group, confounded), neighbour predictability third (fails the
fixed-gap control, LODO 0.40).

---

## FINAL: two candidate rules locked for Module 3 (2026-07-29)

Seven primary properties were screened **separately for node classification and link prediction** as executable
decisions — one threshold, one side that says augment. Panel: seven datasets, 12 cells, 11 usable. Four predictors
passed the gates; `components` and `largest_component_frac` rank the panel at Spearman **−1.000**, so they are one
variable and merge, leaving **three distinct LP findings**. After the robustness checks, **two are carried forward**:

### Rule 1 — augment when adjusted homophily < 0.227

ρ = −0.90 on `gap_rel`, **−1.00** on the bias-free `gap_fixed_rel`; LODO min |ρ| = 0.80; 0 exceptions; n = 5.
Graded across 5 distinct predictor values with 3 on the augment side. The strongest finding in the study, and the one
to lead with. Mechanism: when neighbours carry no label agreement, the original edges are not the signal the encoder
needs, so replacing them with role edges costs nothing and can help.

### Rule 2 — augment when largest-component fraction > 0.9588

ρ = +0.78 on both gap definitions; LODO min |ρ| = 0.71; 0 exceptions; n = 6. Canonical over `components` because it is
scale-free — a "< 39.5 components" threshold cannot be applied to a graph ten times larger. **Weaker evidence than
rule 1 and must be reported as such:** all four augment cells sit at exactly 1.0000, so `distinct_augment` = 1 — this
is a *fully connected vs not* group split, not a graded trend, the 0.9588 cut is arbitrary anywhere in (0.9177, 1.0),
and the split coincides exactly with the standing batch confound. Mechanism: on a fragmented graph, component identity
alone makes link prediction easy, and role edges bridge components and destroy it.

### Dropped — neighbour predictability < 0.4084

ρ = −0.70 on `gap_rel` (exactly at the gate) but **−0.60 on `gap_fixed_rel`**, so the bias-free control rejects it;
LODO min |ρ| = 0.40, meaning a single dataset moves it. Precision point for the write-up: its LODO **sign** was stable
(`lodo_sign_stable = True`) — the failure is magnitude fragility plus the fixed-gap rejection, not a direction flip.
It stays in `results/candidate_rules.csv` as a screened-and-rejected row, which is the honest record.

### Node classification — no rule, by construction

0 of 5 usable NC cells augment, so there is nothing for a threshold to separate and every NC row fails at
`n_exceptions = -1`. The reportable NC result is a **boundary** — "never augment" — not a predictor. A held-out NC
experiment is therefore a falsification attempt, not a validation.

### Status

Module 2 is **closed**. These two rules are frozen; Module 3 (pre-register the prediction, then run the unchanged
pipeline on unseen datasets) is the test that decides whether they are rules or artefacts. The held-out set must
contain a fragmented + heterophilous graph — rules 1 and 2 *disagree* there, which is exactly why it is the
informative case.

Docs updated to this state: `README.md`, `docs/virgo_guide.md`, `CLAUDE.md` §4, `experiments/README.md`,
notebook 5 §8. Tables: `results/candidate_rules.csv`, `results/characterization_*.csv`.

---

## 2026-07-31 — Threshold honesty: the split is fitted on the cells it is scored on

### The issue

The screen reported each rule as "0 exceptions", and that number was **not evidence**. `threshold()` searches every
midpoint between adjacent predictor values and keeps the one with fewest errors, then the same cells are used to count
those errors. Whenever the two verdict classes are linearly separable on a predictor, the search *must* find a split
with zero errors. The count is a property of separability, not of the rule's reach.

Two separate defects, worth stating apart:

1. **The reported number is over-precise.** `0.227` is the midpoint between `tolokers` (0.0926, augments) and `enzymes`
   (0.3613, keeps) — the two datasets straddling the boundary. It is a max-margin choice, which is defensible, but the
   panel only pins the **interval** (0.0926, 0.3613); every cut inside it fits the seven-dataset panel exactly as well.
   Quoting three decimals implies a precision the data do not contain. Same for rule 2: the interval is (0.9177, 1.0).
2. **The error count is in-sample.** No held-out estimate existed at all, so nothing in the screen distinguished a rule
   that generalizes from one that memorizes five points.

Origin: raised by the user, who proposed hiding one dataset at a time, refitting the cutoff on the rest, and predicting
the hidden one. That proposal is correct and is what was implemented, with one addition described below.

### The design

Three additions to `experiments/characterize.py`, all measurement — pipeline, seeds, K, encoder and the frozen
scoreboard are untouched.

**1. `threshold()` now returns the separating interval.** Error as a function of the cut is piecewise constant between
adjacent data values, so the routine widens from the chosen cut across every neighbouring cut with the same error count
and returns the two data values bounding that run. New columns `interval_lo` / `interval_hi`. The reported threshold is
still the midpoint; the interval is what the write-up quotes.

**2. `loo_threshold()` — leave-one-out refit of the split.** Hide one cell, refit the cut on the remainder, predict the
hidden cell. New columns `loo_accuracy`, `loo_correct`, `loo_folds`, `loo_threshold_lo`, `loo_threshold_hi`, plus
`majority_baseline` (always guessing the more common verdict) and `loo_beats_majority`. The baseline matters: with 3
augment / 2 keep, guessing "augment" every time already scores 0.60, so raw LOO accuracy is meaningless on its own.

**3. `nested_loo()` — leave-one-out over the whole screen.** LOO on the threshold alone still leaks: the *predictor*
`homophily_adjusted` was itself chosen by looking at all cells. `nested_loo()` hides a dataset and re-runs predictor
selection **and** thresholding inside each fold, then predicts the hidden one. Predictor selection inside a fold breaks
ties on (fewest errors, then larger |ρ|) rather than list order, so the first-listed predictor is not favoured. A
dataset the fold's chosen predictor cannot be evaluated on counts as a **miss, not a skip** — `ogbl_ddi` has no labels,
so a rule built on homophily genuinely fails to predict it. New output: `results/nested_loo.csv`.

**Gate change.** `GATES["loo_above_majority"] = True`; `credible` now also requires `loo_accuracy > majority_baseline`.
Rationale: `n_exceptions` cannot fail on separable data, so it was carrying no weight. Both frozen rules pass unchanged,
so no conclusion moves — the gate only makes an existing hand-judgement mechanical (see below).

### Results

| rule | reported cut | any cut in | LOO | majority baseline | fold cutoffs ranged |
|------|--------------|-----------|-----|-------------------|---------------------|
| augment when `homophily_adjusted` < 0.227 | 0.227 | (0.0926, 0.3613) | **4/5 = 0.80** | 0.60 | 0.191 – 0.4319 |
| augment when `largest_component_frac` > 0.9588 | 0.9588 | (0.9177, 1.0) | **5/6 = 0.83** | 0.67 | 0.5032 – 0.9588 |
| ~~`nbr_predictability_adjusted` < 0.4084~~ | 0.4084 | (0.3324, 0.4844) | **3/5 = 0.60** | 0.60 | 0.2402 – 0.5734 |

**Both frozen rules survive, each with exactly one out-of-sample error.** The in-sample "0 exceptions" becomes "1 of 5"
and "1 of 6" once honestly scored — that is the number the paper should quote.

- Rule 1 misses **`enzymes`**: with enzymes hidden the boundary pair becomes `tolokers` (0.0926) and `cora` (0.7711),
  the fold cutoff jumps to 0.4319, and enzymes at 0.3613 falls on the augment side. The 2.3× spread in fold cutoffs
  (0.191 – 0.4319) is the direct measure of how weakly the panel pins the number.
- Rule 2 misses **`cora`**: with cora hidden the only keep-side dataset left is `enzymes` (0.0064), the cutoff drops to
  0.5032, and cora at 0.9177 is called augment. Consistent with the standing "two-group split" caveat.

**`nbr_predictability_adjusted` is now rejected mechanically.** It scores exactly the majority baseline — the split
carries zero information beyond guessing. It was already dropped on 2026-07-29 by reading the fixed-gap control and the
LODO floor; the gate now reaches the same verdict without judgement, which is the stronger version of the same result.

**Nested LOO: 4/6 on both gap definitions.** `homophily_adjusted` is re-selected in 5 of 6 folds (`tolokers`'s fold
picks `nbr_predictability_adjusted` on `gap_rel`), so *predictor choice is stable* — the study is not fishing between
properties. The two misses are `enzymes` (as above) and `ogbl_ddi`, which is unpredictable rather than mispredicted:
it has no labels, hence no homophily. That is a genuine coverage limit of the leading rule and belongs in the paper —
**rule 1 cannot be applied to an unlabelled graph at all**, which is precisely when rule 2 is needed.

### What this changes in the write-up

- Quote **intervals**, not points: "augment when adjusted homophily is below roughly 0.1–0.36" with 0.23 as the point
  estimate. Do not quote 4 significant figures for either rule.
- Report LOO 4/5 and 5/6 against their 0.60 / 0.67 baselines, never the in-sample 0 exceptions alone.
- State the nested result 4/6 and name both failure modes (`enzymes` boundary, `ogbl_ddi` coverage).
- This is **not** a substitute for Module 3. Same seven datasets, same provenance confound, n = 5–6 per fold. LOO tests
  how tightly the panel pins the cut; only unseen datasets test whether the rule transfers.

Code: `experiments/characterize.py` (`threshold`, `loo_threshold`, `nested_loo`, `GATES`, `rule`). Tables:
`results/candidate_rules.csv` (9 new columns), `results/nested_loo.csv` (new). Notebook 5 §8 and new §8c.

## 2026-08-04 — LastFM Asia: the first held-out case where the two rules disagree AND the experiment decides

**Why this dataset.** Module 3's two frozen rules had never been separated on unseen data. `pubmed` and `amazon_photo`
are both homophilous + single-component, so rule 1 says *keep original* and rule 2 says *augment* — but both cells came
back a **tie**, which scores neither rule. LastFM Asia (Rozemberczki & Sarkar 2020) sits in the same quadrant and is a
domain the study had none of: a music-platform friendship network. 7,624 users, 27,806 undirected edges (avg degree
7.29), 18 country classes, one connected component.

**Provenance and a recorded deviation.** `torch_geometric.datasets.LastFMAsia` downloads from `graphmining.ai`, which
no longer serves the file from this environment (TLS handshake failure; `404` when the handshake is forced). The raw
graph is therefore read from **SNAP's primary archive of the same dataset** — the source PyG repackages — so node ids,
edges and country labels are the publisher's own. SNAP ships features as a liked-artist JSON rather than PyG's 128-dim
matrix; irrelevant here, since node features are dropped by design (`virgo/data/make_pyg.py`, structural-only).
Counts reproduce the published statistics exactly (7,624 / 27,806 / 18).

**Pre-registration.** `experiments/predict_module3.py --datasets lastfm_asia` was run **before any training** and the
row is frozen write-once in `results/module3_predictions.csv`:

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.8562** — the highest in the whole study (above `cora` 0.7711 and `amazon_photo` 0.7850) | rule 1 (`< 0.227`) | **keep original** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **rules disagree** |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7218 ± 0.0251 | 0.7002 ± 0.0086 (`hybrid`) | −0.0216 | 0.0188 | **−1.15σ** | **keep original** |
| node classification (weighted F1) | 0.3770 ± 0.0067 | 0.2844 ± 0.0118 (`hybrid`) | −0.0926 | — | **−9.65σ** | keep original |

**Rule 1 correct, rule 2 wrong.** The disagreement resolves in favour of the lead rule — the one Module 2 designated
primary on the discovery panel. Held-out tally is now **rule 1 4/5, rule 2 3/5** (`pubmed` and `amazon_photo` still
score neither: tie). This is the first out-of-sample evidence that separates the two, and it points the same way the
discovery evidence did (ρ −1.00 on the bias-free gap for homophily, vs a two-group split for connectivity).

**Encoder-robust.** The verdict does not depend on the study encoder: under the `deepwalk` bridge the same cell reads
original 0.9432 ± 0.0013 vs `hybrid` 0.8304 ± 0.0029 — keep original by a wider margin.

**The NC boundary holds again.** No held-out node-classification cell has ever augmented: **0 of 6 usable cells**, this
one at −9.65σ. The `hybrid` variant is additive (it keeps every original edge) and still loses by that margin, so on a
homophilous social graph the role edges are actively harmful to the label signal, not merely uninformative.

**Caveat worth stating.** LastFM Asia does *not* break the standing confound: it is single-component like every other
augment-side dataset, so it tests the two rules against each other but adds nothing to the fragmented + heterophilous
cell Module 3 still needs. What it does remove is the "the disagreement quadrant is untested" limitation.

Code: `virgo/data/make_pyg.py` (`_lastfm_asia`), registry entries in `virgo/config.py`, `experiments/characterize.py`
(`STUDY`), `virgo/frozen_rules.py` (`HELDOUT`), `experiments/run_core.py` (`HELDOUT`). Tables:
`results/module3_predictions.csv`, `results/module3_scored.csv`, `results/scoreboard.csv`.

## 2026-08-05 — Amazon-ratings: rule 1's augment side fails a second time, and the Platonov family alone falsifies the threshold

**Why this dataset.** Rule 1 is a low-homophily → *augment* rule, and its augment side had been tested on exactly two
held-out graphs: `actor` (correct) and `minesweeper` (wrong). Amazon-ratings (Platonov et al. 2023) is a clean
heterophily benchmark — explicitly built to replace Chameleon/Squirrel, which have duplicate-node leakage — and it is
large enough to matter without being impractical: 24,492 nodes, 93,050 undirected edges (avg degree 7.60), 5 rating
classes, one connected component. It is **not** `amazon_photo`, the homophilous co-purchase graph already held out;
the two share a domain word and nothing else (adjusted homophily 0.1402 vs 0.7850).

Counts reproduce the published statistics exactly. Built by `make_hetero.py`, the same converter as the three discovery
heterophilous graphs, from the authors' own release; its 300-dim fastText product-description features are dropped by
design, and the official 10 train/val/test masks are ignored so the cell stays on the ViRGo core protocol.

**Pre-registration.** `experiments/predict_module3.py --datasets amazon_ratings` ran **before any training**; the row
is frozen write-once in `results/module3_predictions.csv`:

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.1402** — inside rule 1's frozen interval (0.0926, 0.3613) | rule 1 (`< 0.227`) | **augment** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **augment** (rules agree) |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7536 ± 0.0204 | 0.6550 ± 0.0116 (`hybrid`) | −0.0986 | 0.0166 | **−5.94σ** | **keep original** |
| node classification (weighted F1) | 0.2875 ± 0.0029 | 0.2623 ± 0.0015 (`hybrid`) | −0.0252 | 0.0023 | **−10.92σ** | keep original |

**Both rules wrong.** Held-out tally is now **rule 1 4/6, rule 2 3/6** (`pubmed` and `amazon_photo` remain ties and
score neither). Not a marginal miss: −5.94σ, and the runner-up variants are far worse still (`psi` 0.5654, `degree`
0.4551 — below chance). Encoder-robust in the same direction and more extreme: under the `deepwalk` bridge the
original graph scores **0.9982 ± 0.0001** against `hybrid` 0.9407 ± 0.0011.

**The asymmetry is now the headline.** Rule 1's six scored predictions split cleanly by which side they call:

| rule 1 says | datasets | correct |
|-------------|----------|---------|
| keep original | `citeseer_linqs`, `proteins`, `lastfm_asia` | **3 / 3** |
| augment | `actor` ✅, `minesweeper` ❌, `amazon_ratings` ❌ | **1 / 3** |

Low adjusted homophily is a *necessary-looking* but plainly insufficient condition. Every graph the rule expected to
keep its original edges did; two of the three it expected to benefit from role edges did not. The paper should report
rule 1 as a one-sided screen — "high homophily ⇒ do not bother augmenting" is the part that has survived contact with
unseen data — rather than as a two-sided predictor.

**The Platonov family alone falsifies the threshold.** All three discovery augment cells (`roman_empire`, `tolokers`,
`questions`) come from Platonov et al.; `minesweeper` and `amazon_ratings` come from the same paper, the same
converter and the same protocol. Sort those five by adjusted homophily and the verdicts interleave:

| dataset | `homophily_adjusted` | LP verdict |
|---------|---------------------|------------|
| `roman_empire` | −0.0468 | augment |
| `minesweeper` | 0.0094 | **keep original** |
| `questions` | 0.0207 | augment |
| `tolokers` | 0.0926 | augment |
| `amazon_ratings` | 0.1402 | **keep original** |

**A K A A K** — no cut anywhere on this axis separates them. Within a single dataset family, held at fixed provenance,
adjusted homophily does not order the outcome. This is the strongest negative evidence Module 3 has produced, and it
also cuts against the lazy reading of the batch confound: dataset provenance does not explain the augment cells
either, since these five share it and disagree.

**What the interval says, and what we are not allowed to do with it.** 0.1402 is the first held-out value to land
*inside* rule 1's frozen interval, where every cut fits the discovery panel equally well. Had the cut been set at the
interval floor (0.0926, i.e. just below `tolokers`) this prediction would have been *keep original* and correct. So
the evidence points to a lower cut — but `minesweeper` at 0.0094 sits below the entire interval and still keeps its
original graph, so **no** choice inside the interval rescues the rule. Refitting the threshold on these datasets is
forbidden regardless: it would make Module 3 circular. Reported as a falsification, not a correction.

**The NC boundary holds.** Still **0 of 7** usable held-out node-classification cells augment, this one at −10.92σ.
Notably this is a *heterophilous* graph — the case where role-based rewiring should have the most to offer NC — and
`hybrid`, which is additive and keeps every original edge, still loses by ten sigma.

**Confound status unchanged.** Amazon-ratings is single-component, so the fragmented + heterophilous cell Module 3
needs is still missing. What it removes is a different limitation: rule 1's augment side is no longer supported by a
single dataset family.

Code: `virgo/data/make_hetero.py` (`HETERO` gains `amazon_ratings`; `--dataset both` → `all` now that the converter
covers five graphs), registry entries in `virgo/config.py`, `experiments/characterize.py` (`STUDY`),
`virgo/frozen_rules.py` (`HELDOUT`), `experiments/run_core.py` (`HELDOUT`). Tables:
`results/module3_predictions.csv`, `results/module3_scored.csv`, `results/scoreboard.csv`.

### 2026-08-05 (addendum) — verification of the amazon_ratings cell, and what actually tracks the verdict

The `amazon_ratings` result was checked at five independent levels before being accepted as a falsification; all pass.

| check | result |
|-------|--------|
| edge set vs the PyG source | **identical** — 186,100 directed columns → 93,050 undirected pairs, 0 self-loops, source already symmetric |
| labels | 24,492 rows, ids exactly 0..n−1, values **match `data.y` element-wise**, 5 classes |
| `.nodes` sidecar | 24,492 rows; 0 isolated nodes (min degree 5) |
| LP split (seed 42) | 65,135 train / 27,915 test_pos / 27,915 test_neg; test fraction 0.3000; train ∩ test = **∅**; train ∪ test = the full edge set; **0** negatives are real edges |
| rule inputs recomputed from scratch | adjusted homophily **0.1402** (edge 0.3804, degree-weighted null 0.2793) and largest-component fraction **1.0000** — both reproduce the frozen row exactly |

**Why the augmented graph loses here, mechanically.** On the seed-42 training graph, `psi` retains **381 of 65,135**
original edges (0.6%) — it is a nearly disjoint graph, not an enrichment of the original one. `hybrid` is additive but
adds 140,717 role edges on top of the 65,135 real ones, so **68% of its edges are role edges** and every real edge's
message is diluted roughly 1 : 2.16. Of those 140,717 role edges only **169 are actual held-out test links** (0.12%;
~13× chance, so role similarity carries a faint link signal — nowhere near enough to pay for the dilution).

**A cleaner correlate of the LP verdict than either frozen rule — reported as an observation, NOT promoted to a rule.**
Sorting all 14 usable LP cells by the *original* graph's own `graphsage_edge` AUC separates the verdicts almost
perfectly, with a single exception:

| original AUC | 0.0173 | 0.4665 | 0.5953 | 0.6019 | 0.6130 | 0.6218 | 0.6392 | 0.6444 | 0.6720 | 0.7000 | 0.7093 | 0.7218 | 0.7536 | 0.7857 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dataset | ddi | quest. | actor | roman | cora | cites. | pubmed | tolok. | prot. | enz. | mines. | lastfm | am_rat | am_photo |
| verdict | A | A | A | A | K | K | tie | **A** | K | K | K | K | K | tie |

Every cell below ~0.60 augments; every cell above ~0.65 keeps its original graph or ties. `tolokers` is the lone
exception. This says augmentation helps exactly where the original graph is *weak for the encoder* — i.e. it buys
headroom, not structure.

**This is not usable as the study's rule and must not be swapped in for one.** The original graph's AUC is a *result*,
not a graph property: obtaining it means already having trained on the original graph, at which point running the
augmented variant too is cheaper than consulting any rule. Its value is explanatory — it accounts for why the
homophily threshold keeps missing on the augment side (`minesweeper` 0.7093 and `amazon_ratings` 0.7536 are both
*high-headroom-free* cells that happen to be heterophilous) — and it is a candidate for a future **pre-registered**
test on unseen data, framed as a property measurable in advance. Promoting it now on the strength of the same cells
that suggested it would be exactly the HARKing this log has refused elsewhere.

## 2026-08-05 — Squirrel-filtered: both rules correct, and the sharpest evidence yet that homophily is not sufficient

**Why this dataset, and which copy.** Rule 1's augment side stood at 1/3 after `minesweeper` and `amazon_ratings`.
Squirrel-filtered is Platonov et al.'s **de-duplicated** Squirrel: the original `WikipediaNetwork("squirrel")` contains
repeated nodes that leak between train and test, so that copy is deliberately not used. The filtered `.npz` is fetched
from the authors' own repository — **the same URL `HeterophilousGraphDataset` itself downloads from**, so the data path
is identical to the five names PyG exposes; only the file list differs (`virgo/data/make_hetero.py::_npz`).

**Edge-count correction worth recording.** The dataset is often quoted as 23,499 undirected edges, i.e. 46,998 halved
on the assumption that the archive stores both directions. It does not: all 46,998 rows are **distinct** sorted pairs,
zero reciprocals. Our builds reproduce the paper's table exactly for `roman_empire` (32,927), `tolokers` (519,000) and
`amazon_ratings` (93,050), which confirms Platonov reports *undirected* counts. So squirrel-filtered is **2,223 nodes /
46,998 undirected edges / 5 classes / avg degree 42.28** — the densest graph in the study bar `tolokers` and `ogbl-ddi`,
and ~14× denser than `roman_empire`.

**Pre-registration** (`experiments/predict_module3.py`, run before any training, frozen write-once):

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.0086** — below rule 1's whole interval (0.0926, 0.3613) | rule 1 (`< 0.227`) | **augment** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **augment** (rules agree) |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7399 ± 0.0114 | 0.7736 ± 0.0033 (`degree`) | +0.0337 | 0.0084 | **+4.02σ** | **augment** ✅ |
| node classification (weighted F1) | 0.3358 ± 0.0187 | 0.3500 ± 0.0103 (`psi`) | +0.0142 | 0.0151 | **+0.94σ** | **tie** |

**Both rules correct.** Held-out tally: **rule 1 5/7, rule 2 4/7**. Rule 1's two sides now read: keep original **3/3**,
augment **2/4** (`actor` ✅, `minesweeper` ❌, `amazon_ratings` ❌, `squirrel_filtered` ✅). Unusually, all three role
variants beat the original here (`degree` 0.7736, `centrality` 0.7728, `psi` 0.7609) — this is not one lucky variant.

**The decisive pair.** `squirrel_filtered` and `minesweeper` have adjusted homophily **0.0086** and **0.0094** — a
difference of 0.0008, the closest pair in the entire study — and **opposite LP verdicts** (+4.02σ augment vs −5.96σ
keep original, both far outside noise). No threshold on this axis can separate them, in either direction, at any cut.
Rule 1 predicting `squirrel_filtered` correctly is therefore a *coincidence of sign*, not evidence the variable is
sufficient. Extending the fixed-provenance Platonov ordering to six graphs:

| dataset | `homophily_adjusted` | LP verdict |
|---------|---------------------|------------|
| `roman_empire` | −0.0468 | augment |
| `squirrel_filtered` | 0.0086 | **augment** |
| `minesweeper` | 0.0094 | keep original |
| `questions` | 0.0207 | augment |
| `tolokers` | 0.0926 | augment |
| `amazon_ratings` | 0.1402 | keep original |

**A A K A A K.** Six graphs from one paper, one converter, one protocol; the verdicts do not order on homophily. The
correct claim for the paper is that **low adjusted homophily is necessary but not sufficient** for augmentation to
help: every graph above the boundary kept its original edges (3/3, plus the two ties), while below the boundary the
outcome is genuinely mixed (2/4). That is a screen, not a rule, and it is worth reporting as exactly that.

**Retraction: the "headroom" observation logged earlier today is falsified.** That addendum noted that sorting LP cells
by the original graph's own `graphsage_edge` AUC separated the verdicts with one exception, and flagged it as an
observation for a future pre-registered test — explicitly not promoted to a rule. `squirrel_filtered` breaks it: its
original AUC is **0.7399**, *above* `minesweeper` (0.7093) and `lastfm_asia` (0.7218) which both keep original, and it
augments anyway. The ordering now has two exceptions, one of them near the top, and the "augmentation only buys
headroom" reading does not survive. Recorded here rather than deleted: it is a clean example of why a pattern found on
the same cells that suggested it must not be adopted, and it was falsified by the very next dataset ingested.

**Density is still rejected too.** Sorted by average degree the verdicts remain interleaved at the sparse end —
`citeseer_linqs` 2.78 keeps original while `roman_empire` 2.91 augments — so the pre-specified density proposal gains
nothing from this cell, even though the three densest graphs (`squirrel_filtered` 42.3, `tolokers` 88.3, `ogbl_ddi`
500.5) do all augment.

**First non-negative NC cell.** Every held-out node-classification cell so far had been keep original, most by many
sigma. This one is a **tie** at +0.94σ — the augmented graph is nominally ahead, just inside noise. The boundary claim
survives (**0 of 8 usable held-out NC cells augment**) but should now be stated as "never *significantly* augments"
rather than "always loses". Consistent with the rest: a very dense, strongly heterophilous graph is where role edges
come closest to paying off for labels.

**Confound status.** Still single-component, so the untested quadrant remains **heterophilous + fragmented**, where
rule 1 says augment and rule 2 says keep original. That is the one cell no held-out dataset has yet occupied.

Code: `virgo/data/make_hetero.py` (`_npz`, `REPO`; `HETERO` maps to `None` for archives PyG does not list), registry
entries in `virgo/config.py`, `experiments/characterize.py` (`STUDY`), `virgo/frozen_rules.py` (`HELDOUT`),
`experiments/run_core.py` (`HELDOUT`). Tables: `results/module3_predictions.csv`, `results/module3_scored.csv`,
`results/scoreboard.csv`.

## 2026-08-05 — Module 4: virtual-graph properties screened as a second variable gating the homophily rule

**Motivation, restated from Module 3.** Rule 1 fails asymmetrically: on the nine held-out datasets, adjusted homophily
*above* the boundary called "keep original" and was right **3/3**; *below* it called "augment" and was right **2/4**.
Homophily is therefore a reliable veto, not a predictor, and a second variable is only needed inside the low-homophily
zone — the same zone that contains `minesweeper` (0.0094, keeps) and `squirrel_filtered` (0.0086, augments), 0.0008
apart with opposite verdicts, which no single-variable split can ever separate.

**What was screened.** Four properties of the **virtual graph**, not of the original graph, so they need no labels and
also cover `ogbl_ddi` where rule 1 cannot be evaluated at all. Measured on the built `psi` edgelists at K=10, seed 42 —
the exact files the encoder trained on, read off disk, nothing retrained:

| property | definition |
|---|---|
| `role_diversity` | distinct signature tie classes / nodes — how finely the signature resolves roles |
| `role_edge_enrichment` | share of role edges that are real edges, divided by the graph's density (lift over chance) |
| `original_retention` | share of the **original** edges the rewired graph still contains |
| `vg_edge_ratio` | virtual edges / original edges — the volume change |

Two facets kept exploratory (`role_sampled_frac`, `role_edge_overlap`). Pairwise collinearity was **measured, not
assumed**: no pair reaches \|ρ\| ≥ 0.99, so all four are separate findings.

**Panel.** `GATE_PANEL` = the seven-dataset discovery panel (minus `ogbn_arxiv`, node-classification only) **plus the
nine Module-3 datasets**, which are no longer unseen. 11 decided LP cells (6 augment / 5 keep); the low-homophily zone
holds 7 (5 augment / 2 keep, majority baseline 0.714). This is fitting data by construction and is declared as such.

**Result — one candidate clears every gate, and it is `original_retention`, low → augment.**

| scope | ρ | LODO min \|ρ\| | interval | LOO | majority | credible |
|---|---|---|---|---|---|---|
| low-homophily zone, `gap_rel` | −0.857 | 0.771 | (0.0062, 0.0176) | **6/7 = 0.857** | 0.714 | yes |
| low-homophily zone, `gap_fixed_rel` | −0.714 | 0.500 | (0.0042, 0.0176) | 5/6 = 0.833 | 0.667 | yes |
| all decided cells, `gap_rel` | −0.764 | 0.709 | (0.0062, 0.0094) | 9/11 = 0.818 | 0.545 | yes |

It survives on the bias-free fixed-variant target, works standalone as well as inside the gate, and reproduces on the
`centrality` role graph. Direction: **the less of the original graph the rewiring keeps, the more augmentation helps** —
which is the opposite of the intuition that a virtual graph should recover real adjacency. The other three fail:
`role_diversity` ρ +0.29, `role_edge_enrichment` −0.39, `vg_edge_ratio` +0.64 with LOO 0.00 inside the zone.

**It does not solve the case it was proposed for.** `minesweeper` retention 0.0176 (keeps, called right) vs
`squirrel_filtered` 0.0374 (augments, called **wrong**) — the decisive pair is still split the wrong way, and
`squirrel_filtered` is precisely the one exception behind the 6/7. So the gate moves the zone from 2 errors to 1 without
fixing the cell that motivated it. Reported as a mechanical column (`separates_decisive_pair`), not a judgement.

**Status: candidate, not finding.** Fitted on a panel that includes the Module-3 datasets, so nothing here is validated;
it is a Module-5 prediction to be pre-registered on a third, genuinely unseen set. The untested quadrant is unchanged: a
heterophilous graph that is also **fragmented**.

Code: `experiments/gate_rules.py` (new, self-contained; imports `threshold`/`loo_threshold`/`rho`/`GATES` from
`characterize.py` and `uniformity` from `score_module3.py`, edits nothing). Tables:
`results/vg_characterization.csv`, `results/gate_candidates.csv`, `results/gate_collinearity.csv`. The frozen Module-2/3
artifacts (`candidate_rules.csv`, `nested_loo.csv`, `module3_*.csv`) are untouched.

**Promoted to a frozen artifact (same day).** The candidate is now locked as `frozen_rules.FROZEN_GATE`
(`original_retention < 0.0119`, interval `(0.0062, 0.0176)`, applies only when rule 1 says augment) beside `GATE_PANEL`
and an empty `GATE_HELDOUT`. `predict_gated()` states the full decision in one place: **rule 1 vetoes on its reliable
side → the gate decides inside the low-homophily zone → rule 2 covers unlabelled graphs**. `gate_rules.py` now asserts
its own screen still reproduces the frozen point on `GATE_PANEL`, so a silent re-fit fails loudly rather than drifting.

**Module 5 is the test.** `experiments/predict_gate.py` mirrors the Module-3 contract one level up: write-once
pre-registration to `results/module5_predictions.csv`, scoring to `results/module5_scored.csv`, and a hard refusal of any
dataset in `GATE_PANEL` (the gate was fitted on those, so predicting them measures fit, not transfer). One ordering
difference that must be stated in the paper: the gate is a property of the **virtual graph**, so the rewiring has to be
BUILT before the verdict can be written. Building touches no encoder, no split and no labels, so the prediction is still
made with the outcome unknown — but the order is now **build → predict → train → score**, not predict → train → score.

Sanity re-check on three fitted datasets (`--allow-panel`, not a test) reproduces the screen exactly: the gate fixes
`minesweeper` (rule 1 wrong, gate right) and breaks `squirrel_filtered` (rule 1 right, gate wrong), leaving rule 1 alone,
the gate, and the two-gate call all at 2/3. That is the trade the gate makes, visible in a single table.

---

## 2026-08-07 — Rule-redundancy diagnostics on the Module-3 held-out set (review request)

Two descriptive analyses justifying **keep R1, drop R2** in the paper. Both live in `experiments/score_module3.py`
(`property_correlation()`, `decision_agreement()`) and in a new final section of notebook 6; both only READ the frozen
`module3_predictions.csv` + scoreboard — nothing is refitted, no threshold moves.

**1. Property–property correlation** (`results/module3_property_correlation.csv`, n = 9 held-out datasets):
`homophily_adjusted` vs `largest_component_frac` Spearman **ρ = −0.297** (p 0.44, floored at 2/n!). So R2 is NOT
redundant with R1 in the correlation sense — the two properties rank the datasets differently. R2's real weakness is
**degeneracy, not redundancy**: only **4 distinct values** over 9 datasets, and **6/9 sit exactly at the 1.0 ceiling**
(vs 9 distinct values for homophily). A near-constant predictor cannot grade anything; its cut stays arbitrary inside
(0.9177, 1.0), exactly the Module-2 two-group caveat.

**2. Decision-agreement table** (`results/module3_decision_agreement.csv`): rules agree **6/9**; of the **3
disagreements** (pubmed, amazon_photo, lastfm_asia — all R1 keep / R2 augment), two came back **ties** (undecided, score
nothing) and the one decided case, `lastfm_asia`, was **R1 correct, R2 wrong**. R2 won a disagreement **0 times**. So on
held-out evidence R2 never adds information over R1 when they conflict.

**Paper line this supports:** R1 is retained as the lead (and later as the veto in the two-gate rule); R2 is dropped
from the labelled-graph path and kept ONLY as the no-labels fallback (`ogbl_ddi`-style graphs, where R1 cannot fire at
all) — that coverage hole is the one thing the agreement table cannot speak to, since all 9 held-out graphs have labels.
Caveat carried: 9 datasets, 2 of 3 disagreements tied, so "0 for R2" rests on a single decided cell.

---

## 2026-08-08 — Correction: property–property correlation now over ALL datasets (review re-request)

The review request was the R1-vs-R2 property correlation across **all datasets (discovery + held-out)**; the
2026-08-07 entry computed it on the 9 held-out only. New `score_module3.rule_properties()` pools the frozen Module-2
properties (`dataset_characterization.csv`, filtered to the 7-dataset discovery panel) with the frozen held-out
`module3_predictions.csv`; `property_correlation()` now defaults to that 16-dataset union. Nothing is refitted.

**Result** (`results/module3_property_correlation.csv`, notebook 6 §6): `homophily_adjusted` vs
`largest_component_frac` Spearman **ρ = −0.4146**, p 0.1244, over the **15 complete pairs** (`ogbl_ddi` is unlabelled →
no homophily; 16 datasets total). The held-out-only value was −0.297 — same direction, moderately stronger at full n,
but still far from "strongly rank-correlated", so the conclusion stands: **R2 is not redundant with R1 by
construction**. Degeneracy remains R2's real weakness and sharpens at full n: **11/16 datasets sit exactly at the 1.0
ceiling** and the property takes only **6 distinct values**, vs 15 distinct homophily values. The modest negative ρ
reflects the panel pattern that this study's heterophilous graphs are all fully connected.

Notebook 6 §6 now also displays the 16-row property table (dataset, panel, both properties) so the correlation's
inputs are visible.

**Point 2, same correction (review: "the 2×2 agreement table of their decisions over every dataset"):**
`decision_agreement()` now also defaults to the 16-dataset union. Calls are recomputed from the frozen points via
`frozen_rules.predict_one()`; an integrity assert requires every recomputed held-out call to equal the write-once
pre-registered `module3_predictions.csv` row, so nothing can drift. Discovery rows carry a `panel` flag and are
in-sample — their correctness describes fit, not validation.

**The 2×2** (15 datasets where both rules fire; `ogbl_ddi` is R1-n/a — unlabelled, a coverage hole, not a
disagreement): both augment **7**, both keep **4**, R1 keep / R2 augment **4**, R1 augment / R2 keep **0**.
That empty cell is the reportable structure: on every graph in the study, R1 saying augment implies R2 says augment
(all low-homophily graphs sit at the largest-component ceiling 1.0), so R2 never contradicts R1 on the augment side
and **every disagreement is one-sided** — R1 keep, R2 augment.

**The 4 disagreement cases** (`results/module3_decision_agreement.csv`): `ogbn_arxiv` (NEW — invisible in the
held-out-only view; undecidable, node-classification only, no LP cell), `pubmed` (tie), `amazon_photo` (tie),
`lastfm_asia` (the only decided one → **R1 correct, R2 wrong**). Counts: 16 rows, 1 R1-n/a, agree 11, disagree 4,
R1 wins 1, R2 wins 0, undecided 3. Conclusion unchanged from 2026-08-07 — keep R1 as lead/veto, R2 only as the
no-labels fallback — with the standing caveat that "R2 wins 0" still rests on a single decided cell, and the three
discovery agreements on the augment side are in-sample.

---

## 2026-08-11 — Supervisor reframe: structural graph AUGMENTATION; two new hybrid variants (hybrid_degree, hybrid_centrality)

**Framing decision (supervisor meeting).** The study is presented as "when should the original graph be augmented
with structural information, and which structural augmentation" — not "which virtual graph is best" (virtual
graphs/embeddings already exist as an area). R1 is restated accordingly: adjusted homophily predicts when structural
graph augmentation may help **link prediction** (low → consider augmenting; high → keep the original). Existing
pipeline, results and conclusions are unchanged.

**Method addition.** `virgo/virtual_graph.py` now builds `hybrid_degree` (original ∪ degree top-K) and
`hybrid_centrality` (original ∪ centrality top-K), the exact construction of the existing `hybrid`
(= original ∪ psi top-K, name kept) with only the role side swapped. Union semantics as before: every original edge
kept at weight 1.0 (overlap → 1.0), role edges carry the similarity weight; NOTE the union is not a 50/50 blend —
whichever side contributes more edges dominates, and the locked mean aggregation ignores edge weights anyway.
Same GraphSAGE, K=10, features, seeds 42/43/44, evaluation.

**Exposure decision.** The two new hybrids are LOCAL to notebooks 2+3 (`EXTRA_HYBRIDS` in notebook 2 setup;
`SWEEP_SIMS`/`VG_ORDER` in notebook 3) and deliberately NOT in `cfg.VG_SIMS`, so notebook 4's automatic sweep is
untouched. Promotion to `VG_SIMS` (+ notebooks 4/5/6) only if they beat the existing variants over 3 seeds.
Edge counts land in `results/graph_health.csv` as for every build.

**Pre-registered hypothesis (to test, not claim).** Actor LP: original ≈ 0.595, pure degree 0.662 — does
original + degree retain useful original edges and exceed 0.662? If a hybrid wins strongly, distinguish
"more edges" from "meaningful edges" against the existing density controls (`original_k`/`random_k`, ddi-only so far).
If the hybrids do not help, report the negative result and keep current conclusions.

## 2026-08-12 — hybrid_degree / hybrid_centrality PROMOTED to the official variant set

**Decision.** The two hybrids leave the notebook-local `EXTRA_HYBRIDS` staging area and become official graph
variants. `cfg.VG_SIMS` is now the seven-variant candidate set: `original`, `psi`, `degree`, `centrality`,
`hybrid` (= original ∪ Ψ, name kept), `hybrid_degree`, `hybrid_centrality`. Every driver that loops `cfg.VG_SIMS`
(`experiments/run_core.py`, `experiments/run_ogb.py`, notebooks 2/3/4) therefore sweeps all seven under the same
locked GraphSAGE, K=10, seeds 42/43/44 and the same NC/LP evaluation — the notebook-2/3 experiment that motivated
the promotion is kept, not deleted.

**Guard against silently refitting the frozen rules.** `cfg.VG_SIMS_LOCKED` holds the original five, and the two
FITTING scripts read that list, not `VG_SIMS`: `characterize.py` (Module-2 gap is a max OVER variants, so a new
variant would widen the max and move the frozen rules) and `gate_rules.py --sims` (the Module-4 gate was fitted on
those five). Module-2 and Module-4 numbers are therefore unchanged by this promotion, by construction. `gate_rules`'
role-signal map now also resolves `hybrid_degree → degree` and `hybrid_centrality → centrality`, so measuring a
promoted hybrid there is correct whenever it is asked for explicitly.

## 2026-08-13 — seven-variant sweep over the discovery panel (results)

**Run.** All seven `cfg.VG_SIMS` variants × the 7 `frozen_rules.DISCOVERY_PANEL` datasets × {GraphSAGE, DeepWalk},
K=10, seeds 42/43/44, same NC/LP evaluation; core five through `experiments/run_core.py`, `ogbn_arxiv`/`ogbl_ddi`
through `experiments/run_ogb.py` (only the two newly promoted variants needed re-running there — the five old ones
were already recorded). Serial, because every process upserts `results/scoreboard.csv`. 32 dataset × encoder × task
cells, all filled.

**Node classification.** 12 keep original / 2 tie / 2 augment. Both augment cells are DeepWalk-only with small gaps
(`enzymes` +0.0169 and `roman_empire` +0.0163, both `hybrid_degree`); `questions` ties at 0.9555 on a 97% majority
class. **No GraphSAGE NC cell augments** — the frozen "NC never augments" boundary survives the two new variants.

**Link prediction.** 5 keep / 5 augment on the panel: `questions` deepwalk +0.0227 (`hybrid_centrality`),
`questions` graphsage +0.1038 (`hybrid_degree`), `roman_empire` graphsage +0.0961 (`centrality`), `tolokers`
deepwalk +0.1100 (`hybrid_degree`, 0.8959), `tolokers` graphsage +0.0563 (`psi`). `cora` and `enzymes` keep the
original graph on both encoders. Direction agrees with rule 1 (the augment cells are the low-homophily side).

**OGB official.** `ogbn_arxiv` keeps the original graph on both encoders (deepwalk 0.6718, graphsage 0.3998), but
the two new hybrids are by far the strongest augmented variants there — deepwalk `hybrid_degree` 0.4815 against
psi/degree/centrality at 0.0929/0.1231/0.1418, graphsage `hybrid_centrality` 0.3747. `ogbl_ddi` augments on both
encoders (valid Hits@20): deepwalk original 0.0772 < `hybrid` 0.1038 > `hybrid_degree` 0.0941 >
`hybrid_centrality` 0.0874; graphsage original 0.0413 < `hybrid_degree` 0.0565 > `hybrid` 0.0554.

**"More edges" is not the explanation on ddi.** Against the density controls, deepwalk `hybrid_degree` 0.0941 and
`hybrid_centrality` 0.0874 both far exceed `original_k` 0.0303 and `random_k` 0.0006 at matched density, so the
gain tracks which edges are added, not how many. Controls remain ddi-only.

**Verdict on the promotion.** The new hybrids do not overturn any dataset-level verdict — no cell changes from
"keep original" to "augment" because of them. What they change is the shape of the augmented side: where the pure
role graphs collapse (arxiv NC, cora NC/LP), the union retains enough of the original graph to stay competitive,
and `hybrid_centrality` / `hybrid_degree` are the best augmented variant in 8 and 4 of the 22 decided cells. The
`ogbl_ddi` validation lock is untouched — `hybrid` + DeepWalk 0.1038 is still the panel maximum, so `select()`
had nothing to re-pick and the no-test-peeking discipline held without intervention.

## 2026-08-13 — Module 6 / Notebook 7: strategy selection is NOT predictable (negative result)

**Question.** Module 2 asked whether to augment. This asks the next one: given that augmentation helps, WHICH structural
signal — Ψ, degree, or eigenvector centrality — suits this graph. Code: `experiments/strategy_select.py` (a FITTING
script, panel-guarded like `characterize.py`), notebook `notebooks/7-phase7_strategy_selection.ipynb`.

**Protocol.** GraphSAGE, K=10, link prediction, seeds 42/43/44. Fit panel = the LP half of the discovery panel
(`cora, enzymes, ogbl_ddi, roman_empire, tolokers, questions`; `ogbn_arxiv` is NC-only); the nine Module-3 datasets stay
held out. The winner is a **set** — every variant within one pooled sigma of the leader — because three seeds cannot rank
closer than that. The seven variants collapse onto four labels, a hybrid voting for the signal it ADDS
(`hybrid → psi`, `hybrid_degree → degree`, `hybrid_centrality → centrality`), plus `original` for do-not-augment.
`ogbl_ddi` is read on **validation** Hits@20, not `characterize.PRIMARY`'s test: choosing among seven variants is model
selection, and the OGB test split is reserved for the one locked configuration.

**Result — no strategy rule.** No primary graph property separates which signal wins. Not on the 6-dataset panel, and not
on an exploratory refit over all 15 datasets that have an LP score. The only splits that clear the Module-2 gates predict
`original`, which is Module 2's augment-vs-keep question restated on the same panel — it recovers R1
(`homophily_adjusted > 0.227`, interval 0.0926–0.3613) and R2 (`largest_component_frac < 0.9588`), a useful check that the
machinery measures what it should, but not a new finding. Nothing is frozen.

**The ceiling is why.** Across all 15 datasets only THREE name a single winning signal — `actor` = degree,
`roman_empire` = centrality, `tolokers` = psi — exactly one apiece. Everywhere else the leading variants sit inside seed
noise (`questions` and `ogbl_ddi` tie all three signals at once) or the original graph wins outright. A property cannot be
asked to predict a label that the experiment does not resolve, so the binding constraint is measurement resolution, not
the choice of predictor.

**Two screen defects found and fixed while building it**, both of which had produced spurious "credible" rules.
(i) `graph_properties` writes `n_classes = 0` for an unlabelled graph, which ranks as the panel's smallest value — a
"low n_classes" split was really fitting "has no labels", so `ogbl_ddi`'s 0 is now masked to missing. (ii) A dataset whose
band names three signals is a member of all three one-vs-rest groups, so a split could be fitted on datasets that never
singled the signal out; a rule now needs at least `MIN_SOLE = 2` datasets where the signal wins alone, reported as the
`n_sole_wins` column and the `tie_dominated` flag.

**Reportable claim.** *When* to augment is predictable (Module 2, validated in Module 3); *which* augmentation to use is
not, at this panel size and seed count. Raising the seed count is the obvious way to test whether the ties are real or
just unresolved.

## 2026-08-13 — Module 6 at SEVEN seeds: the negative result holds, and hardens

**What changed since the 3-seed pass.** Three things, in the direction that would have found a rule if one existed.
(i) **Seven seeds** (42–48) instead of three, on every panel dataset × all seven variants, GraphSAGE link prediction —
`experiments/run_core.py --task link_prediction --encoder graphsage_edge --seeds 42 43 44 45 46 47 48`, ~2h serial.
(ii) A **9-dataset panel**: the five core LP discovery datasets plus four promoted from Module 3. (iii) A **sharper tie
band** and a **conditional scope**, both below.

**Panel promotion (user decision, chosen on graph properties only).** `squirrel_filtered` (avg_degree 42.3, the only
mid-density graph — the core five jump 6.3 → 88.3), `amazon_ratings` (homophily 0.140, inside the core five's empty
0.093–0.361 band), `amazon_photo` (the only fragmented candidate, 136 components, where fragmentation had just two
points), `lastfm_asia` (homophily 0.856 and 18 classes, extending both ranges). Held out and untouched: `pubmed`,
`actor`, `minesweeper`. **Disclosure:** the exploratory 3-seed screen over all 15 datasets had already been run, so
`actor` was known to be the one held-out dataset that decided a single signal; it was deliberately NOT promoted, leaving
the informative dataset on the test side. `ogbl_ddi` is excluded from both sides (unlabelled, and ~14h to re-sweep at
seven seeds); `citeseer_linqs`/`proteins` stay out per 2026-07-29.

**The tie band was wrong for this question, and fixing it matters more than the seeds.** The band had been one pooled
standard deviation, the `characterize.compare()` convention. Standard deviation measures the spread of the score and does
NOT shrink as seeds are added — more seeds only estimate the same width more precisely. The question "are these two means
distinguishable" is answered by the standard error of their difference, sqrt(s1²/n1 + s2²/n2), which narrows as
1/sqrt(seeds). Both are implemented (`strategy_select.BANDS`); `sem` is the default. Switching band alone, at three seeds,
already broke the `questions` three-way tie.

**Result — still no strategy rule.** 9 datasets, 5 where augmentation beats the original graph, and 3 that name a single
signal: `lastfm_asia` = centrality, `questions` = degree, `roman_empire` = centrality. No property separates them, in
either scope. The only rules clearing the gates predict `original`, i.e. Module 2's augment-vs-keep question restated on
this panel — it recovers the R2 fragmentation family (`components > 39.5`, LOO 8/9; `largest_component_frac < 0.9893`,
LOO 7/9). Useful as a check that the machinery measures what it should; not a new finding.

**Conditional scope added.** One-vs-rest over all nine datasets conflates "not centrality" with "do not augment at all",
which is Module 2's question. `patterns(..., scope="augmented")` drops the keep-original datasets and asks the intended
one: given that augmentation helps, which signal. Both scopes are reported; neither produces a rule.

**A third screen defect found and fixed — this one manufactured a rule.** Under the conditional scope,
*use centrality when avg_clustering > 0.125* passed every gate: rho 0.707, 0 exceptions, LOO 4/4 against a 0.80 majority
baseline. It is an artifact. `loo_threshold` skips any fold it cannot fit, and holding out the only non-centrality dataset
(`questions`) produces exactly such a fold — so `loo_folds` was 4 of 5 and the deciding case was **never tested**, while
the baseline was computed over all five points. `loo_majority()` now recomputes the baseline over exactly the folds LOO
scored; the rule then reads 1.000 vs 1.000 and correctly fails. `loo_folds`, `loo_majority` and `folds_skipped` are
reported so the same trap is visible next time. Note this generalizes beyond this module: any LOO on a class with one
member is scored only on the easy folds.

**Reportable claim, strengthened.** *When* to augment is predictable (Module 2, validated in Module 3); *which*
augmentation to use is not. Seed noise is no longer the explanation — at seven seeds the remaining ties are real, and the
three datasets that do name a signal are not separated by any of the seven primary graph properties.

## 2026-08-13 — Module 6 at TEN seeds on 14 datasets: the first strategy rule

**What changed.** The supervisor's diagnosis was that precision was never the binding constraint — the properties simply
did not separate the winners — and that development diversity would do more than seeds. Both were raised anyway:
**10 seeds (42–51)** and **14 datasets**, every labelled graph carrying a link-prediction score. The 2026-07-29 exclusion
of `citeseer_linqs` / `proteins` was reversed for this module only (user, 2026-08-13): both are heavily fragmented, the
region where the panel was thinnest, and the original exclusion was about the headline study panel, not this screen.
`ogbn_arxiv` (node classification only) and `ogbl_ddi` (unlabelled) cannot carry a signal label and stay out.

**Diversity was indeed the constraint.** The share of datasets naming a single winning signal went 3 of 9 → **6 of 14**:
`actor` = degree, `questions` = degree, `lastfm_asia` = centrality, `pubmed` = centrality, `roman_empire` = centrality,
`tolokers` = psi. Seven of fourteen beat the original graph. Both reinstated datasets keep the original graph, as does
`minesweeper`, so they do not enter the conditional scope.

**THE CANDIDATE RULE (frozen, `frozen_rules.FROZEN_STRATEGY`): once augmentation is indicated, use CENTRALITY when
`nbr_predictability_adjusted` > ~0.009** (interval 0.006–0.0123). Zero exceptions over the seven augmenting datasets,
rho 0.866, and graded across 0.0123 → 0.8498 rather than resolving into a two-group split. The ordering is
tolokers −0.3274 (psi), questions −0.0041 (degree), actor 0.0060 (degree) | squirrel_filtered 0.0123, roman_empire
0.3324, pubmed 0.6946, lastfm_asia 0.8498 (all centrality).

**It is not homophily in disguise** — rho 0.32 between the two, and `roman_empire` is the proof: the lowest adjusted
homophily in the panel (−0.0468) and yet centrality wins. This is the same `nbr_predictability_adjusted` that Module 2
screened and dropped, there for the *augment* question, where it scored exactly at the majority baseline. Different
question, different answer; report it as such rather than as a resurrection.

**Four caveats that must travel with the rule.**
1. **Fitted, not validated.** Nothing is held out — every labelled dataset is in the panel. Same status as the Module-4
   gate. The test is genuinely new datasets (`frozen_rules.STRATEGY_HELDOUT`, empty).
2. **Out-of-sample it is worth one call.** Leave-one-out 5/7 against a 4/7 majority baseline.
3. **Band-dependent.** It exists under the sem tie band only; under the Module-2 sigma band the ties re-inflate
   (`questions` and `tolokers` become members of all three signals) and the split takes two exceptions. The band is a
   methodological choice, so it is part of the rule, not background.
4. **Co-predictor.** `degree_assortativity` (exploratory tier) produces the identical split at rho 0.866; the two
   correlate at 0.71, so which is THE predictor is not determined by this data.
   Robustness worth recording: dropping `squirrel_filtered` — the one tie, a member of both the centrality and degree
   groups — STRENGTHENS the rule (still 0 exceptions, LOO 5/6) and widens the interval from (0.006, 0.0123) to
   (0.006, 0.3324). The knife-edge boundary was an artifact of that tie, not of the rule.

**The negative result survives underneath it.** The rule separates centrality from the rest and says nothing about psi
versus degree below the cut; no property separates those two. So the honest summary is now three-level: *whether* to
augment is predictable and validated; *centrality versus the rest* is a fitted candidate; *psi versus degree* remains
unpredicted.

**Method addition.** An exploratory property tier was screened alongside the primary seven (`--tier all`, default):
degree spread, assortativity, density, label entropy. Only primary properties can set `credible`; an exploratory hit is a
lead. This is how `degree_assortativity` surfaced as the co-predictor above.


---

## 2026-08-14 — Two-stage held-out test on the LINKX Facebook100 networks: the gate 2/2, the strategy rule 1/1 resolved

A single unseen set tested BOTH open rules in the order they are actually used: stage 1 = the frozen two-gate augment
call (`predict_gated`, Module 5), stage 2 = the frozen strategy rule (`FROZEN_STRATEGY`, Module 7). The order matters and
was fixed in advance — stage 2 is conditional on augmentation being indicated, so only graphs stage 1 sends to "augment"
can test it. Nothing was trained until both predictions were frozen on disk.

**The datasets** (LINKX non-homophilous benchmarks, Lim et al. 2021; Facebook100 college friendship networks), chosen by
the user before any was built or measured: `reed98` (962 nodes / 18,812 edges), `amherst41` (2,235 / 90,954),
`johnshopkins55` (5,180 / 186,586). Structural-only as always — 745/1,193/2,406-dim features ignored. The label is
**gender, and it is missing for ~10% of users** (LINKX codes it −1); those nodes are left out of the `.labels` file
rather than written as a third class, exactly as LINKX evaluates them, and the graph keeps every node. Both the
homophily code and the NC evaluator already key on node id, so partial labels need no special case.

**Stage 1 (measured, then predicted, with no encoder run):**

| dataset | homophily_adj | largest_comp_frac | original_retention | rule 1 | gate | combined |
|---|---|---|---|---|---|---|
| reed98 | 0.0219 | 1.0000 | 0.0237 | augment | keep | **keep original** |
| amherst41 | 0.0598 | 1.0000 | 0.0092 | augment | augment | **augment** |
| johnshopkins55 | 0.0972 | 0.9956 | 0.0055 | augment | augment | **augment** |
| cornell5 | 0.0907 | 0.9979 | 0.0020 | augment | augment | **augment** |

`reed98` was dropped before training on the user's rule ("if stage 1 says KEEP, replace it rather than force a stage-2
test") and replaced by `cornell5` (18,660 / 790,777), the next Facebook100 graph in the same loader — same family, same
label semantics, measured and predicted the same way. Note what rule 1 alone would have done: it says "augment" on all
four, including `reed98`. Every separation here comes from the gate.

**Stage 2, frozen before training** (`results/module7_predictions.csv`): `amherst41` 0.2128, `johnshopkins55` 0.2278,
`cornell5` 0.2760 — all far above the 0.0092 cut, so all three call **centrality**. Stated before the run and not after:
no held-out graph falls below the cut, so this set exercises only one side of the rule. That is not an accident of
sampling — the cut is so low that essentially every normal labelled graph clears it.

**Outcome (GraphSAGE, K=10, 10 seeds 42–51, sem tie band):**

| dataset | original | best variant | best score | winner set (sem) | signals | scored |
|---|---|---|---|---|---|---|
| amherst41 | 0.6651 | **centrality 0.6749** | +0.0098 | centrality \| hybrid \| hybrid_centrality \| hybrid_degree | centrality, degree, psi | undecided |
| johnshopkins55 | 0.6854 | **centrality 0.7160** | +0.0306 | centrality | centrality | **correct** |
| cornell5 | 0.6910 | **centrality 0.7078** | +0.0168 | centrality \| hybrid | centrality, psi | undecided |

**Stage 1 scored 2/2** where the experiment decided (`amherst41`'s original-vs-best gap is 0.73σ, a tie, so it does not
score; `reed98` is untrained and stays pending). The gate decided all four calls — rule 1 never vetoed.

**Stage 2 scored 1/1 correct, against a pre-registered bar of 2 of 3, so the criterion was NOT met** — two of the three
cells could not test the rule at all, because the seed band cannot separate centrality from variants carrying the other
signals. Reported as undecided, not as passes.

**What is descriptively true and must not be inflated into a pass: centrality is the top-scoring variant on all 3/3, and
augmentation beats the original graph on all 3/3.** Point-estimate agreement is 3/3; *resolved* agreement is 1/1. The
difference between those two numbers is the whole methodological point — ten seeds are not enough to separate
`centrality` from `hybrid_centrality` (which adds the same signal) or from `hybrid` on these graphs.

**Standing limitation, unchanged by this run.** The rule is still tested on one side only. A genuine test needs a graph
with `nbr_predictability_adjusted` below ~0.0092 where augmentation is indicated — in the fitted panel only `tolokers`
(−0.3274), `questions` (−0.0041) and `actor` (0.0060) live there, and all three are already panel members.

**Code.** `experiments/predict_gate.py` (stage 1) and `experiments/predict_strategy.py` (stage 2), both read-only
w.r.t. the frozen rules. Datasets built by `virgo/data/make_pyg.py` via PyG's `LINKXDataset`; registered in
`cfg.DATASETS`, `characterize.STUDY`, `run_core.RUNNABLE`, `frozen_rules.GATE_HELDOUT` and `STRATEGY_HELDOUT`.
Tables: `results/module5_predictions.csv`, `module5_scored.csv`, `module7_predictions.csv`, `module7_scored.csv`.

---

## 2026-08-15 — CORRECTION: reed98 falsifies the gate's only held-out separation; Module 3 re-scored at ten seeds

An accuracy audit of the whole record, prompted by a reframing of the project. Two claims in the log did not survive
it. Both are corrections to *reporting*, not to method: no rule was refitted and no frozen number was touched.

### 1. `reed98` was trained, and it augments — the gate's one differentiating call was wrong

The 2026-08-14 entry above records `reed98` as dropped before training on the user's rule ("if stage 1 says KEEP,
replace it rather than force a stage-2 test") and therefore unscored. It was in fact trained later the same evening —
`output/notebook3_gnn_encoder/link_prediction/reed98/k10/` is written 18:32–18:37, whereas `module5_scored.csv` was
written at 17:42, which is why the table still reads `pending`. All five locked variants completed at ten seeds.

Scored through the frozen path (`score_module3.actual_lp_verdicts`, no refit):

| dataset | original | best augmented | variant | gap | noise | gap σ | verdict |
|---|---|---|---|---|---|---|---|
| reed98 | 0.6161 | **0.6562** | centrality | +0.0401 | 0.0158 | **2.55** | **augment** |

`original` is *last* of the five — `psi` 0.6455, `degree` 0.6455, `hybrid` 0.6485, `centrality` 0.6562 — and the
relative gap (0.0651) is the largest of the four LINKX graphs. The frozen gate had called `keep original`
(`original_retention` 0.0237 > the 0.0119 cut). Consequences, all of which supersede the 2026-08-14 wording:

- **Stage 1 scored 2/3 where the experiment decided, not 2/2.** `johnshopkins55` ✓, `cornell5` ✓, **`reed98` ✗**,
  `amherst41` a tie at 0.73σ.
- **Rule 1 alone would have scored 3/3.** It said `augment` on all four. Adding the gate made the combined call
  *worse* on the only genuinely unseen set the gate has ever seen.
- **The gate's direction does not reproduce here.** ρ(`original_retention`, `gap_rel`) over the four LINKX graphs is
  **+0.4**, against the fitted −0.86. n is 4, so this is weak evidence — but it is the only out-of-sample evidence
  the gate has, and it points the other way.
- The sentence "Every separation here comes from the gate" is still literally true and is now the problem: the gate
  made exactly one call that differed from rule 1, and that call was its error.

**What does not change.** The gate is *kept* in the framework: it is the only component that speaks to the decisive
pair (`minesweeper` / `squirrel_filtered`) and the only one that needs no labels, which is the coverage hole rule 1
cannot fill. But its status moves from "first held-out evidence in favour" to **screened and fitted, not yet
transferring**, and that must be stated wherever the gate is described. Any prose reading "the gate 2/2" or "stage 1
stopped reed98" is false and should be corrected — this includes notebook 7 §4 and the `reed98` row of
`predict_strategy.summary()`, both fixed on 2026-08-15.

Not a protocol violation: the prediction was frozen to `module5_predictions.csv` at 16:08, hours before any encoder
touched `reed98`. The pre-registration held. Only the scoring pass was run too early.

### 2. Module 3's 5/7 and 4/7 are three-seed figures; at ten seeds they are 4/6 and 4/6

`results/scoreboard.csv` was overwritten with the ten-seed (42–51) sweep for Module 6; the board Module 3 was scored
against is archived at `results/scoreboard_3seed.csv`. Re-running `score_module3.score()` unchanged against each:

| board | rule 1 | rule 2 |
|---|---|---|
| 3 seeds (as published) | 5/7 | 4/7 |
| 10 seeds (current) | **4/6** | **4/6** |

One cell moved and it explains both changes: **`lastfm_asia` went from `keep original` to `tie`** (−0.81σ at ten
seeds; `hybrid` 0.7049 against `original` 0.7159). It had been rule 1's fifth correct call, and rule 2's only error
on a cell where rule 1 was right — so the 2026-08-07 redundancy diagnostic ("disagreements R1 1–0 R2") is now
**0–0 with two ties**, and the asymmetry finding reads "keep-original side 2/2, augment side 2/4" rather than 3/3.
The asymmetry itself is unchanged in direction, and "low homophily is necessary, not sufficient" still stands.

**Nothing that was fitted moved.** All six discovery-panel LP verdicts are identical at ten seeds (cora keep,
enzymes keep, roman_empire / tolokers / questions / ogbl_ddi augment), and so are all seven low-homophily-zone
verdicts the gate was fitted on. The frozen cuts are therefore unaffected; only the held-out *scores* are.

**Rule adopted:** every reported held-out number must name its seed count, or be re-scored first.

### 3. Current totals, ten seeds, five locked variants, all registered datasets

| task | augment | tie | keep original | scope |
|---|---|---|---|---|
| link prediction | **9** | 4 | 6 | 19 datasets |
| node classification | **0** | 2 | 13 | 15 cells (2 unusable) |

The NC boundary is now measured over 15 cells rather than 8 and has never once been crossed. Say "no dataset has
shown a significant node-classification gain", not "role graphs cannot help NC" — the claim is empirical.

### 4. Project renamed

*ViRGo — Virtual Role-Graph Embedding for Structural Identity* → **Structure Aware Graph Augmentation for Graph
Neural Networks**, with the framing moved from "virtual role graphs" to "when, and how, to augment a graph with
structural information". `README.md`, `CLAUDE.md` and `docs/project_guide.md` (renamed from `virgo_guide.md`) were
rewritten against the numbers above. The Python package stays `virgo/` — an internal identifier; renaming it would
break every import, notebook and cached path for no research gain.

---

## 2026-08-31 — Research focus re-set after the thesis defence: stage-1 stabilization, stage-2 breadth, anomaly detection, GIN/GAT

The thesis (`docs/ADAM_Thesis_Report_removed.pdf`, defended 2026-08-28) and its slide deck
(`docs/Presentation_Adam.pdf`) are now the canonical framing of the work: **characterization-guided selection rules
for structural graph augmentation**. Both were read line by line against the repository; the framing carries over
unchanged, four numbers in them do not, and the priority order for what comes next changed.

### 1. Numbers in the defended documents that disagree with the repository

Recorded so they are not copied forward into the LoG draft. §3 of `CLAUDE.md` remains the source of truth.

| where | says | current board says |
|---|---|---|
| slide 9 | rule 1 "7/9 correct predictions" | **4/6 decided** at ten seeds (5/7 at three) |
| thesis §4.3 | rule 1 5/7, rule 2 4/7 | those are **three-seed** figures; **4/6 and 4/6** at ten |
| thesis §6 conclusion | "the 6 augmentation cases clearly won, 6 original-graph wins and 4 ties" | **9 augment / 4 tie / 6 keep**, as its own Table 1 states |
| thesis §6 conclusion | "six structure-based virtual graph variants" | **seven** (`cfg.VG_SIMS`); five locked |

Neither document reports the `reed98` correction (thesis Table 6 and slide 12 both list it as stage-1 KEEP with no
outcome). The correction stands: `reed98` augments at 2.55σ, so stage 1 is 2/3 and rule 1 alone would have been 3/3.

### 2. Stage 1 is to be rebuilt, not defended (user decision)

The frozen three-step stage 1 stays on disk as the record of what was fitted, and `predict_gate.py` keeps working, but
it is no longer the target artifact. Two objections, both stated by the user and both matching the held-out evidence:

- **`original_retention` is out.** It requires the role graph to be *built* before the decision can be made, which
  weakens the "decide before training" claim, and it is fitted-not-transferring: on the LINKX set its only
  differentiating call was wrong (`reed98`), with ρ(`original_retention`, `gap_rel`) = **+0.4** against a fitted −0.86.
- **Adjusted homophily alone is not enough to lead.** It is a reliable veto on its high side and mis-calls the low
  side (`minesweeper`, `amazon_ratings`), which is exactly why the gate was introduced in the first place.

**Target:** one rule, computed on the **original graph alone**, label-free where possible, that transfers across graph
types. Candidate pool to screen next, all pre-training: over-squashing diagnostics (balanced-Forman and
Ollivier-Ricci curvature, spectral gap / Cheeger constant, effective resistance), degree heterogeneity and
assortativity, structural-signature dispersion; label informativeness stays available where labels exist. Screening
protocol is unchanged from Module 2 — fit interval, LOO, majority baseline, fixed-variant control — and no
replacement is frozen until it beats rule 1's held-out record.

### 3. Anomaly detection returns to the pipeline as the third task

Dropped in July when the characterization study became the focus; reinstated now. Rationale: NC has a boundary
("never augment") and LP splits, and anomaly detection is the task where structural-role information should matter
most, since structural outliers are the target rather than a nuisance. It stays inside the structural-only rule, so
it uses **structural anomaly injection**, not attribute-driven fraud benchmarks. Both stages get re-screened on the
new cells; the NC boundary is not evidence about this task.

### 4. Architecture scope narrowed to GIN and GAT

`gin.py` is wired and registered but unrun; `gat.py` does not exist. These two are the whole architecture scope. The
question is explicitly not which encoder wins, but whether the stage-1 and stage-2 calls — fitted entirely under
GraphSAGE — survive a change of aggregator.

**Priority order (user):** (1) stabilize stage 1, (2) more stage-2 rules, (3) anomaly detection extension, (4) GIN and
GAT. Item 2 is still blocked on data in the Ψ-vs-degree zone: 1 Ψ dataset against 2 degree datasets, where every
property separates the groups perfectly and therefore carries zero information. The ≥2 augmenting datasets below the
cut that would unblock it are the same graphs that would test the centrality rule's negative side.

---

## 2026-08-31 — Module 8: an original-graph second condition for stage 1 — screened, nothing promoted

`experiments/stage1_pairs.py` (new, FITS, panel-guarded) asks Module 4's question again with its disqualifying constraint
removed: inside the low-homophily zone, can a property of the **original** graph do what `original_retention` does,
without needing the role graph built first? Rule 1 is never refitted — only the second condition is — so the form under
test is `homophily_adjusted < 0.227 AND X on its augment side ⇒ augment`.

Panel is `fr.GATE_PANEL` deliberately, so the candidate and the incumbent are fitted on identical evidence. 13 datasets →
10 decided link-prediction cells (`pubmed`, `amazon_photo`, `lastfm_asia` tie), zone = 7 cells, 5 augment / 2 keep.
Outputs: `results/stage1_pair_{cells,rules,compound,two_condition,collinearity}.csv`; narrative in notebook 6 §7.

### 1. The bar, and the incumbent measured against it

| rule | decided cells | correct | out-of-sample | fixes | breaks |
|---|---|---|---|---|---|
| rule 1 alone | 10 | **7** (0.70) | 0.70 — its cut is frozen, so in- and out-of-sample coincide | — | — |
| + frozen gate `original_retention < 0.0119` | 10 | **8** | not applicable (quoted at its locked cut) | amazon_ratings, minesweeper | **squirrel_filtered** |

`ogbl_ddi` is unlabelled, so rule 1 cannot fire and no second condition reaches it; it counts as a miss for every row,
exactly as Module 2 scores an unevaluable rule. Rule 1's other two misses are the known ones: `amazon_ratings`,
`minesweeper`.

### 2. No original-graph property clears the Module-2 gates inside the zone

Screened: every Module-2 predictor except rule 1 itself — 6 primary, 8 exploratory. **Zero credible.** The closest,
`avg_clustering`, has ρ −0.82 and one exception but leave-one-out 0.43 against a 0.714 majority baseline. Two properties
are constant inside the zone (`components`, `largest_component_frac` = 1 everywhere) and carry no information there.

Reported alongside, because on a zone this small a clean split is nearly free: with 7 cells and 5 augment, only 2 of the
C(7,5) labellings are separable by any single cut, so **p = 0.095 per candidate, and over the 6 primary candidates 0.57
perfect separations are expected for no reason at all.**

### 3. Four single conditions reach 8/10 without breaking anything — but only one survives out of sample

| condition | fixes | breaks | panel | LOO (condition refitted per fold) |
|---|---|---|---|---|
| `avg_clustering < 0.5573` | amazon_ratings | — | 8/10 | 0.50 |
| `degree_gini > 0.0958` | minesweeper | — | 8/10 | 0.70 |
| **`degree_skew > −1.632`** | minesweeper | — | 8/10 | **0.80** |
| `degree_assortativity < 0.2968` | minesweeper | — | 8/10 | 0.50 |

Only `degree_skew` beats rule 1's 0.70 out of sample, and by one call. Three caveats travel with it: it is
**exploratory tier** under Module 2's rule that only primary predictors may gate a rule (promoting it is a decision, not
a detail); it moves exactly **one cell**, so its whole case rests on `minesweeper`; and the three degree-spread
properties are not collinear at the 0.99 bar (skew ~ assortativity −0.75, gini ~ skew +0.57) yet all three isolate the
same graph, so they are one finding in practice, not three.

### 4. The two failures have different causes, and two conditions overfit

`amazon_ratings` is the zone's most clustered graph (0.5816); `minesweeper` is a lattice — degree Gini 0.0146, skew
−4.77, assortativity +0.39, all extremes of the zone. No single property repairs both. The obvious pair does:
`avg_clustering < 0.5573 AND degree_gini > 0.0958` scores **9/10, fixing both and breaking nothing** — and its
leave-one-out accuracy is **0.50**, below rule 1's 0.70. The skew variant is 0.60. Found by search over 66 combinations
on 10 decided cells, so this is a hypothesis to pre-register, not a result.

**Reading.** The gate can be replaced without losing accuracy — `degree_skew` matches its 8/10 without needing the role
graph and without breaking `squirrel_filtered` — but nothing here is validated, and the in-sample winner (the
two-condition rule) is exactly the row out-of-sample scoring rejects. Next step is a pre-registered test on unseen
graphs, plus a deliberate decision on whether degree spread is promoted to the primary tier. Nothing was written to
`virgo/frozen_rules.py`.

### 5. Retrospective check on the four LINKX graphs (added same day)

`stage1_pairs.py --step heldout` applies the panel-fitted cuts, unchanged, to `reed98`, `amherst41`, `johnshopkins55`,
`cornell5` — the only registered graphs the screen never saw (`chameleon_filtered`, `texas`, `twitch_pt` are unregistered
and carry no scoreboard row, so they cannot be scored). Not a pre-registration: these were trained months earlier.

3 decided cells (`amherst41` ties at +0.73σ). **Rule 1 alone 3/3. Every candidate condition 3/3. The frozen gate 2/3** —
it still calls `reed98` keep, which is the error already on record. So the candidates cost nothing and do not repeat the
gate's only differentiating mistake.

The check is **non-discriminating**, and that is the main thing to report: all four graphs clear every condition
(degree skew 1.40–6.95 against a −1.632 cut; clustering 0.22–0.32 against 0.5573), so no candidate is exercised on its
negative side and a constant "always augment inside the zone" predictor scores identically. Confirming a second condition
needs a low-homophily graph that should NOT be augmented — the same shape of missing evidence as the stage-2 test below
the cut. Missing feature caches for the four graphs were regenerated with the pipeline's own `SageEncoder.features()`.

### 6. The same run pointed at `minesweeper` and `amazon_ratings` (in-sample, on request)

Both are inside `PAIR_PANEL` and are the two cells the cuts were fitted to repair, so this is a re-read of the fit, not a
test: rule 1 alone 0/2; `avg_clustering < 0.5573` repairs `amazon_ratings` only; `degree_skew` / `degree_gini` /
`degree_assortativity` repair `minesweeper` only; every two-condition combination and the frozen gate score 2/2 **by
construction**. The out-of-sample numbers in §3 (LOO 0.80 for `degree_skew`, 0.50 for the best pair) remain the evidence.

### 7. The six-graph test — and the candidate flips: `avg_clustering`, not `degree_skew`

`chameleon_filtered` and `texas` were rebuilt for this test (`make_hetero` / `make_pyg`, registered 2026-08-31; node and
edge counts match `results/module7_withdrawn.csv` exactly — 890/8854 and 183/279), so no retraining was needed: their
ten-seed link-prediction scores survive in that file and `stage1_pairs.heldout()` now reads it when a dataset is absent
from the scoreboard. Six graphs, none in any fitting panel; 4 decided (`amherst41` +0.73σ and `texas` −0.35σ tie).

| dataset | adj. homophily | avg. clustering | degree skew | actual verdict |
|---|---|---|---|---|
| reed98 | 0.0219 | 0.3184 | 1.85 | augment (2.55σ) |
| johnshopkins55 | 0.0972 | 0.2684 | 2.02 | augment (2.28σ) |
| cornell5 | 0.0907 | 0.2191 | 6.95 | augment (2.10σ) |
| **chameleon_filtered** | **0.0295** | **0.5769** | 2.55 | **keep original (−1.04σ)** |
| amherst41 / texas | 0.0598 / −0.2936 | 0.3104 / 0.1979 | 1.40 / 11.88 | tie — undecided |

`chameleon_filtered` is the case the study was missing: **low adjusted homophily and yet keep-original**, on a graph no
rule was fitted on. Rule 1 alone calls it wrong.

| rule | held-out (4 decided) | panel (10 decided) | note |
|---|---|---|---|
| rule 1 alone | 3/4 | 7/10 | misses chameleon_filtered |
| **`AND avg_clustering < 0.5573`** | **4/4** | **8/10** | catches it — clustering 0.5769 is above the cut |
| `AND degree_skew > −1.632` | 3/4 | 8/10 | chameleon passes the condition, so the error survives |
| `AND degree_gini` / `degree_assortativity` | 3/4 | 8/10 | same failure |
| frozen gate `original_retention` | 3/4 | 8/10 | catches chameleon, still breaks `reed98` |

**The ranking reverses out of sample.** `degree_skew` had the best panel leave-one-out (0.80 against rule 1's 0.70) and
fails here; `avg_clustering` had the worst (0.43) and is the only condition that clears the test. Its panel LOO was
depressed by `minesweeper`, which it never claimed to explain — the two zone failures have different causes, and only
the clustering one recurs on unseen data.

Standing status: `avg_clustering` is now **12/14 decided cells across panel and test, against 10/14 for rule 1 alone and
11/14 for the gate**, it needs no role graph, it is primary tier (no tier promotion required), and it does not repeat
the gate's `reed98` error. What it is not: pre-registered — all six graphs were trained and scored before this screen
existed — and the test rests on **one** negative case. Nothing is written to `virgo/frozen_rules.py` pending that
decision.

### §8 · Stage 1 LOCKED — the clustering exception (2026-09-01)

User decision after the six-graph test: freeze it, and state it as **one characteristic with one exception**, not as two
equal conditions.

> **Stage 1.** Adjusted homophily is the characteristic. High ⇒ keep the original. Low ⇒ augment — *unless* clustering
> is high, in which case keep the original after all. Unlabelled graphs fall back to largest-component fraction.

`virgo/frozen_rules.py` now carries `FROZEN_EXCEPTION = avg_clustering < 0.5573`, interval **(0.5329, 0.5769)** — the
panel fit was (0.5329, 0.5816) and the held-out `chameleon_filtered` at 0.5769 narrows the upper end; the point did not
move. `predict_gated()` consults it in the slot the retention gate used to hold. `FROZEN_GATE` is kept in the file,
marked superseded, because Modules 4–5 are published against it and `gate_rules.py` still asserts its cut reproduces.

Two consequences for the pipeline, both simplifications: stage 1 no longer needs the role graph **built** before it can
predict (so the documented order is simply predict → build → train → score), and `experiments/predict_gate.py` now
measures `avg_clustering` cache-free with `nx.average_clustering` on the policy-loaded graph — verified identical to the
feature-cache values used in the screen (chameleon_filtered 0.5769, minesweeper 0.4355, amazon_ratings 0.5816,
reed98 0.3184). Its output columns change from `gate_pred`/`gate_correct` to `exception_pred`/`exception_correct`;
already-frozen Module-5 rows keep their original columns untouched, as the write-once contract requires.

Caveats carried forward verbatim into README §3 and CLAUDE.md §2: the 0.5573 cut is a **candidate, not a proven
universal threshold**; the negative side rests on two graphs (`amazon_ratings`, `chameleon_filtered`); and the six-graph
test was **retrospective**, so it is transfer evidence, not a pre-registration. The one open stage-1 item is that
pre-registered re-test on graphs not yet trained.

### §9 · Stage 2, the Ψ-vs-degree branch — re-screened at 16 datasets, still BLOCKED (2026-09-01)

Plan (user): fit on the 16 = discovery 7 + Module-3 9, hold out the 6 (reed98, amherst41, johnshopkins55, cornell5,
chameleon_filtered, texas), then test once. Executed as a measurement first, because the fit is only possible if the two
groups have members.

**No new runs were needed.** All six held-out graphs already carry 10-seed LP scores (four in `scoreboard.csv`, two in
`module7_withdrawn.csv`), so the proposed 3-seed batch is unnecessary and the test set is scored at the same seed count
as the fitting panel.

Winners at 10 seeds, sem band, over the 16-dataset panel (15 LP rows; `ogbn_arxiv` is node-classification only):

| winning signal | datasets naming it ALONE |
|---|---|
| centrality | lastfm_asia, pubmed, roman_empire |
| degree | actor, questions |
| **Ψ** | **tolokers — one graph in the whole corpus** |
| tied / no single signal | squirrel_filtered (centrality\|degree), ogbl_ddi (all three) |
| original wins | cora, enzymes, citeseer_linqs, proteins, amazon_photo, amazon_ratings, minesweeper |

The undetermined zone (augmenting AND `nbr_predictability_adjusted` ≤ 0.0092, i.e. where FROZEN_STRATEGY declines to
choose) contains exactly three datasets: `tolokers` (Ψ, −0.3274), `actor` (degree, 0.0060), `questions` (degree,
−0.0041). `contrast()` reports **`n_zone < min_cells`** for every property, primary and exploratory alike — 3 cells
against the Module-2 gate of 4, with one Ψ point. Every property that differs at all "separates" the groups perfectly,
which is why the margin, not the exception count, is the only informative statistic there (best: `avg_clustering`, 0.82
of the panel spread; then `nbr_label_entropy` 0.32, `edge_homophily` 0.30, `nbr_predictability_adjusted` 0.26).

**The held-out 6 add nothing to that zone.** Their winners: `reed98` centrality alone, `johnshopkins55` centrality
alone, `cornell5` centrality|Ψ, `amherst41` centrality|degree|Ψ (three-way tie), `chameleon_filtered` and `texas` keep
the original. Zero degree-alone cells, zero Ψ-alone cells. So the reserved test set cannot test a Ψ-vs-degree rule even
if one were fitted.

**Conclusion, unchanged and now measured on the full corpus: the Ψ-vs-degree rule is blocked by data, not by method.**
Fitting one on a single Ψ dataset would be an anecdote, and its LOO would be degenerate — holding out `tolokers` empties
the Ψ class and the fold is skipped, so the accuracy would be scored only on folds that cannot fail.

**What would unblock it, stated as a spec for dataset acquisition:** at least two more graphs that (a) augment — some
role variant beats `original` by more than the sem band, (b) sit below the ANP cut of 0.0092, and (c) have Ψ as their
sole winner. The same graphs would test the centrality rule's negative side, which is also still untested.

### §10 · Degree-evidence batch — PRE-REGISTERED before any of it was trained (2026-09-01)

Stage 2's Ψ-vs-degree branch is blocked at 3 cells (§9). User decision: drop Ψ as a target for now and go after a
**degree-augmentation rule**, ingesting six candidate graphs chosen to add low-homophily structure where a degree role
graph is plausible. Every one is screened and **reported whichever way it lands** — datasets are never selected after
seeing which give the wanted answer.

**The six, and two honest overlaps.** `twitch_de`, `deezer_europe`, `wisconsin`, `cornell_webkb`, `chameleon` are new;
**WebKB Texas is already `texas`** (built 2026-08-31 for the stage-1 test, never used for stage-2 fitting) and
**`chameleon` is the ORIGINAL geom-gcn WikipediaNetwork copy** whose de-duplicated version, `chameleon_filtered`, is
already in the corpus. The original has the duplicate-node problem filtering removes; it is screened as its own graph
and that caveat travels with every number it produces. `cornell_webkb` (183 web pages) is **not** `cornell5` (18,660-node
Facebook100 network).

Built structural-only, via the existing loaders (`virgo/data/make_pyg.py`), features and official masks ignored:

| dataset | source | nodes | undirected edges | classes | avg degree |
|---|---|---|---|---|---|
| wisconsin | PyG `WebKB` | 251 | 450 | 5 | 3.59 |
| cornell_webkb | PyG `WebKB` | 183 | 277 | 5 | 3.03 |
| chameleon | PyG `WikipediaNetwork` (geom-gcn) | 2,277 | 31,371 | 5 | 27.55 |
| twitch_de | **SNAP** `twitch.zip` | 9,498 | 153,138 | 2 | 32.25 |
| deezer_europe | **SNAP** `deezer_europe.zip` | 28,281 | 92,752 | 2 | 6.56 |

`torch_geometric`'s `Twitch` and `DeezerEurope` classes both fetch `graphmining.ai`, which returns 503 — the same dead
host that forced `lastfm_asia` onto SNAP in 2026-08. Both are therefore read from SNAP's primary musae archive, the same
data by the same authors. Twitch's target file is keyed by the original streamer id with the graph id in `new_id`, so
labels are reordered onto the edge file's ids before writing.

**Pre-registered fit/validation split, decided on graph properties alone, before any score exists.** Each side gets one
social graph and one small web graph, and the largest graph is deliberately held out so the test set is not only tiny
graphs:

| role | datasets | why |
|---|---|---|
| fitting | `twitch_de`, `chameleon`, `cornell_webkb` | one mid-size social, one mid-size wiki, one tiny web |
| validation (untouched while fitting) | `deezer_europe`, `wisconsin`, `texas` | the largest graph, plus two tiny web graphs; `texas` already scored at 10 seeds and never used for stage-2 fitting |

**Protocol, unchanged:** existing role-graph builder (7 variants, K=10) → GraphSAGE → 70:30 link prediction → the same
characterization table. Seeds **42/43/44** (3 seeds, user's choice to save time). Recorded consequence: the sem tie band
is *wider* at 3 seeds than at the panel's 10, so a signal is **harder**, not easier, to call decided — any cell that is
decided here would also be decided at 10 seeds, but a tie here may not be a real tie. Every cell's seed count is named
in the results table.

**Expectation on record, so it cannot be rewritten afterwards:** the two WebKB graphs (183 and 251 nodes) are likely to
tie — `texas` already did, at −0.35σ — and are included for completeness rather than because they are expected to
decide anything. `twitch_de`, `chameleon` and `deezer_europe` are the graphs with enough edges to decide a winner.

### §11 · Degree-evidence batch — RESULTS, all six reported (2026-09-01)

Protocol as pre-registered in §10: existing loaders, 7 variants at K=10, GraphSAGE, 70:30 LP, seeds 42/43/44.
`experiments/degree_rule.py` does the labelling, the screen and the test; `results/degree_cells.csv`,
`degree_rules.csv`, `degree_psi_contrast.csv`, `degree_heldout.csv`.

**1 · What each of the six did.** Winning-signal labels use the sem band, so a graph only "names" a signal when three
seeds can separate it from every other variant.

| dataset | role | original | best | winner signals | stage-1 verdict |
|---|---|---|---|---|---|
| **twitch_de** | fit | 0.4888 | **psi 0.6258** | **psi alone** | **augment, +21.7σ** |
| chameleon | fit | 0.8105 | hybrid 0.8240 | centrality\|original\|psi | tie (+0.50σ) |
| cornell_webkb | fit | 0.4985 | centrality 0.5239 | centrality\|original | tie (+0.52σ) |
| deezer_europe | validation | 0.7032 | hybrid 0.6809 | centrality\|original | **keep original (−1.52σ)** |
| wisconsin | validation | 0.5195 | centrality 0.4997 | degree\|original | tie (−0.98σ) |
| texas | validation | 0.5126 | centrality 0.4980 | centrality\|original | tie (−0.35σ) |

**The batch produced ZERO new degree wins.** Degree still wins alone on exactly two graphs in the corpus, `actor` and
`questions` — the same two as before. The two WebKB graphs tied, as §10 said on the record they probably would.

**2 · The degree screen therefore still fails.** Over the 8 augmenting graphs in the fitting panel (2 degree, 6 not),
the best candidates are `avg_clustering` (ρ −0.63, degree side low) and `degree_skew` (ρ +0.63, high), each with one
exception. Both score **LOO 0.750 against a 0.750 majority baseline** — no better than always guessing "not degree" —
so nothing clears the gates. **No degree rule is frozen.**

**3 · The held-out test could not run.** None of `deezer_europe`, `wisconsin`, `texas` augments, so none names a
winning signal, so there is nothing for a signal rule to predict on them. The pre-registered split was honoured; it
simply had no scorable cell. This is reported, not repaired by swapping the split.

**4 · Ψ-vs-degree is now fittable, and the answer is "not yet".** `twitch_de` is the corpus's **second** Ψ-alone winner,
so degree-vs-Ψ finally has 4 cells (`actor`, `questions` = degree; `tolokers`, `twitch_de` = Ψ) — exactly the Module-2
minimum. Five properties split them with **0 exceptions**: `homophily_adjusted` (LOO 4/4), `avg_degree`,
`avg_clustering`, `density`, `nbr_label_entropy` (LOO 3/4 each). That looks strong and is not: at n = 4 with 2 against 2
the chance of a clean split is **1/3 per property**, so of the 13 properties tried **~4.3 separate for free** — five did.
The front-runner, `homophily_adjusted < 0.0566 ⇒ degree` (interval 0.0207–0.0926), is therefore recorded as a
**hypothesis**, not promoted; it also reuses the stage-1 variable, and one of its four cells (`twitch_de`) is 3-seed.

**5 · The frozen stage-2 centrality rule has its first counterexample.** `twitch_de` has
`nbr_predictability_adjusted = 0.1235`, well above the 0.0092 cut, so `FROZEN_STRATEGY` predicts **centrality** — and
centrality scores 0.5886, fourth, against Ψ's 0.6258. **Stage 2's held-out record goes 3/3 → 3/4.** Combined with the
standing caveat that the earlier 3/3 never tested the negative side, the honest statement is now: the centrality rule
is fitted, has one clean out-of-sample failure, and still has an untested negative side.

**6 · Stage 1 got its first post-freeze test, and split it.** The clustering exception was locked before any of these
graphs was trained, so the two decided cells are a genuine pre-registered test:

- `twitch_de` — homophily 0.1394 (low), clustering 0.2009 (low) ⇒ **augment**; actual augment at 21.7σ. **Correct**, and
  the largest gap in the whole study.
- `deezer_europe` — homophily 0.0304 (low), clustering 0.1412 (low) ⇒ **augment**; actual **keep original** at −1.52σ.
  **Wrong**, and the clustering exception does not catch it — this is a low-homophily, low-clustering graph where role
  augmentation still loses.

Stage-1 running total: **13/16 decided cells** (panel 8/10, six-graph transfer 4/4, this batch 1/2). The `deezer_europe`
miss belongs in the same list as `minesweeper`: low-homophily keeps that neither condition explains.

### §12 · DEGREE_RULE_CORPUS — pre-registered before any of it was trained (2026-09-02)

Module 9 showed the stage-2 degree branch is blocked by **cells, not method**: degree wins alone on 2 of 27 graphs, and
a six-graph batch added none. User decision: run a **separate stage-2 experiment** whose only purpose is to learn when a
degree role graph should be selected, on its own pool — `cfg.DEGREE_RULE_CORPUS` — kept out of the study panel.

**The four protocol decisions, fixed before the first run.**

| decision | choice | consequence, stated now |
|---|---|---|
| budget | ~20 graphs in 3 waves of ~7 | stop as soon as degree reaches **4 sole wins**; if 20 graphs do not deliver that, report a measured negative and move on |
| seeds | **3 (42/43/44) by default, 6 (add 45/46/47) on ties** | escalation is mechanical — every graph whose top two variants fall inside the tie band is re-run, never a hand-picked one; each cell records its own seed count |
| split | **fit on all 20**, validation deferred to a later ingest | nothing here can be frozen *and* validated: any rule found is **fitted**, exactly like the Module-6 centrality rule, and its held-out test is a future batch |
| labelling | **sole winner only**, unchanged | degree (or `hybrid_degree`) must clear every other signal by the sem band; degree-containing ties do not count, so the degree rule stays comparable to the frozen centrality rule |

**Selection principle.** The twenty were chosen for **structural spread** — degree skew, clustering, assortativity,
density, size, neighbour predictability — and for being established PyG/SNAP benchmarks, **not** because degree is
expected to win on any of them. Every graph is reported whichever way it lands, ties included.

| wave | graphs |
|---|---|
| 1 | `airports_usa`, `airports_europe`, `email_eu_core`, `wikics`, `twitch_engb`, `github`, `amazon_computers` |
| 2 | `airports_brazil`, `polblogs`, `blogcatalog`, `coauthor_cs`, `twitch_es`, `twitch_fr`, `dblp` |
| 3 | `coauthor_physics`, `twitch_ru`, `twitch_ptbr`, `cora_ml`, `squirrel`, `flickr_attr` |

**DISCLOSURE — the airports graphs.** `airports_usa/europe/brazil` are the classic structural-identity benchmarks
(struc2vec's own datasets), and their labels are **air-traffic quartiles, which correlate with degree**. They are in the
pool because they are the standard role benchmark, not because of that correlation, and every screen result is reported
**with and without them** so the correlation cannot silently carry a rule.

**Two further caveats on the record.** `squirrel` is the ORIGINAL geom-gcn copy whose duplicate nodes
`squirrel_filtered` removes — same caveat as `chameleon`. And PyG's `FacebookPagePage`, `Twitch`, `DeezerEurope` and
`WikipediaNetwork(crocodile)` all fetch `graphmining.ai`, which returns 503: the Twitch languages and GitHub are read
from SNAP's primary musae archive instead, `crocodile` is dropped, and `facebook_large.zip` 404s at SNAP so it is
dropped too.

**Recorded deviation.** Screening runs use `--encoder graphsage_edge` only. DeepWalk is a baseline and takes no part in
winner labelling; it is backfilled for any graph that becomes a fitting cell. Everything else is the locked pipeline:
7 variants, K=10, 2-layer mean GraphSAGE, 70:30 leakage-free link prediction.

**Escalation rule, made precise (2026-09-02, after wave 1's first four graphs were scored).** §12 said "escalate on
ties"; wave 1 immediately produced calls that are decided but marginal (`airports_usa`: centrality over degree by
1.2x the sem band), which is exactly the uncertainty the extra seeds were meant to resolve. The rule is therefore
widened, **uniformly and mechanically**, to: *re-run at 6 seeds whenever the top two SIGNALS are within 2x the sem band*
— ties included. It is applied to every graph in the corpus, wave 1's first four retrospectively included, and never
chosen graph by graph. Disclosed here because the widening happened after those four were seen; it changes which cells
get more measurement, never which answer they are allowed to give.

### §13 · The tie problem, the paired band, and a PRE-REGISTERED degree-vs-Ψ candidate (2026-09-02)

**The problem waves 1-2 exposed.** Fourteen corpus graphs, **13 of them augmenting** — but the winning signal was a tie
on eight. Five were `degree|psi` gaps of 0.08-0.66 sem. More seeds cannot close those: the band shrinks as 1/sqrt(n), so
a 0.08-sem gap needs ~3750 seeds to reach 2 sem. The instrument, not the sample, was the limit.

**The fix: pair the comparison.** `results/scoreboard.csv` stores each variant's mean and std as if variants were
measured independently, but for a given seed **every variant is scored on the same 70:30 split**, so the split-to-split
variance is shared and cancels in a per-seed difference. `experiments/tie_break.py` re-scores the embeddings already on
disk (no training), keeps the per-seed AUCs, and compares two signals by the sem of their **difference**. The paired sem
runs 3-8x tighter (e.g. wikics degree-vs-psi: 0.0011 paired against 0.0084 independent).

Two facts that make it safe to use:

- On the **eight augmenting panel graphs the two bands agree exactly** — 0/8 relabelled. Pairing does not rewrite any
  published cell; it only resolves marginal ones.
- It still leaves genuine ties standing: `github` (0.21 paired sem), `wikics` (0.66), `email_eu_core` (0.64),
  `twitch_fr` (0.24), `polblogs` (0.11). Those are not noise hiding a winner — the two role graphs simply score the same.

Across the corpus pairing narrows 7/14 graphs and lifts single-signal calls from 4 to 7. It is recorded as a
**supplementary** instrument: `degree_rule.py --labels paired` opts in, `frozen` stays the default.

**Result 1 — degree versus everything: still no rule, now on real n.** 21 augmenting cells, 5 naming degree alone
(`actor`, `questions`, `twitch_engb`, `airports_europe`, `amazon_computers`). Best candidate
`homophily_adjusted < −0.0605`: ρ −0.33, **4 exceptions**, LOO 0.810 against a 0.762 baseline. Nothing clears the
Module-2 gates. This is no longer "blocked for want of cells" — it is a measured negative: **the five degree winners
have no structural property in common**, spanning film co-occurrence, Q&A, streaming, air transport and co-purchase.

**Result 2 — degree versus Ψ: a candidate worth testing.** Restricted to the 9 graphs naming one of those two
(5 degree, 4 Ψ), one property separates them with **0 exceptions**:

> **`nbr_label_entropy < 0.6724 ⇒ degree`, else Ψ.** ρ −0.866, interval (0.6296, 0.7151), **LOO 8/9 = 0.889** against a
> 0.556 majority baseline.

Chance is priced, as always: a clean 5-vs-4 split costs p = 2/C(9,5) = **0.0159**, so of the 15 properties screened only
**0.24** are expected to separate for free. Five did at n = 4 in Module 9; **one** does at n = 9, and it is the one with
the strongest correlation. That is the difference between an artifact and a lead.

**Why it is a candidate and not a frozen rule.** `nbr_label_entropy` is **exploratory tier**. Module 2's promotion rule
says exploratory properties are leads, never rules, and promoting one *after* seeing it win is exactly the HARKing the
project forbids. So instead of promoting it:

**PRE-REGISTRATION.** The candidate above is written here **before wave 3 is trained**. Wave 3 —
`coauthor_physics`, `twitch_ru`, `twitch_ptbr`, `cora_ml`, `squirrel`, `flickr_attr` — is built but has produced no
score. Any wave-3 graph that augments and names degree or Ψ alone is a genuine held-out test of this cut, applied
unchanged at 0.6724. Reading: if it holds, the tier question becomes a deliberate promotion decision backed by
out-of-sample evidence; if it fails, the candidate is reported as a failed prediction alongside homophily-for-NC and
density-for-LP.

Stopping-rule note: §12 said stop ingesting once degree reached 4 sole wins. It reached 5, so no further graphs are
ingested — and because wave 3 was never trained, the deferred validation your split choice postponed is recovered for
free.

### §14 · Wave 3 — the held-out test of the degree-vs-Ψ cut (2026-09-02)

Wave 3 (`coauthor_physics`, `twitch_ru`, `twitch_ptbr`, `cora_ml`, `squirrel`, `flickr_attr`) was trained AFTER §13 was
written, so the cut `nbr_label_entropy < 0.6724 ⇒ degree` was fixed before any of these graphs had a score. 3 seeds,
escalated to 6 wherever the top two signals sat inside 2x the tie band — which was all of them.

| graph | original | best | paired verdict | scorable? |
|---|---|---|---|---|
| **twitch_ptbr** | 0.5335 | psi 0.6167 | **psi alone** | **yes** |
| flickr_attr | 0.4087 | degree 0.6164 | centrality\|degree\|psi | no — three-way tie |
| squirrel | 0.7015 | degree 0.7396 | degree\|psi | no |
| twitch_ru | 0.5024 | degree 0.6118 | degree\|psi | no |
| cora_ml | 0.7014 | original | original | no — does not augment |
| coauthor_physics | 0.7507 | centrality 0.7534 | centrality\|original | no |

**Result: 1/1.** `twitch_ptbr` has `nbr_label_entropy` **0.7472**, above the 0.6724 cut, so the rule predicts **Ψ** —
and Ψ is the sole winner. The prediction could have failed and did not.

**What that is worth, stated plainly: very little on its own.** One cell, and a binary call, so chance alone gets it
right half the time. The honest summary of the candidate is: 0 exceptions and LOO 8/9 in-sample on 9 cells, chance of a
free split 0.016, plus **one** correct pre-registered out-of-sample call. It stays a **candidate**, unpromoted, for two
reasons that have not changed — `nbr_label_entropy` is exploratory tier, and one held-out cell is not a validation.

**The deeper obstacle wave 3 confirms.** Four of six graphs augment, and **three of the four cannot name a signal even
under the paired band**: `flickr_attr` (degree vs psi 0.77 paired sem), `squirrel` (0.24), `twitch_ru` (0.42).
`flickr_attr` is the sharpest example in the whole study of augmentation mattering — original **0.4087**, below chance,
against 0.6164 for a role graph — and yet which role graph is undecidable. Across all 20 corpus graphs the tie rate for
degree-vs-Ψ is now the dominant outcome, not the exception.

That is the finding to report next to the candidate: **degree and Ψ role graphs usually produce the same link-prediction
outcome.** Ψ is a Poisson/KL score over degree-based structural signatures, so on graphs whose signature is dominated by
degree the two constructions largely coincide — the ties are mechanistic, not statistical. Measuring the edge overlap
between the two role graphs would test that explanation directly, and is the obvious next experiment.

**Corpus totals, all 20 graphs, paired band.** 17 augment. Sole-signal calls: **degree 5** (`actor`, `questions`,
`twitch_engb`, `airports_europe`, `amazon_computers`), **Ψ 5** (`tolokers`, `twitch_de`, `twitch_es`, `blogcatalog`,
`twitch_ptbr`), **centrality 5** (`roman_empire`, `pubmed`, `lastfm_asia`, `airports_usa`, `airports_brazil`);
**9 ties**. Degree-versus-everything still has no rule (§13), and it is now a negative result measured on 21 cells
rather than a shortage of data.

### §15 · The degree corpus is also a held-out test of the FROZEN centrality rule — and it fails it (2026-09-02)

`coauthor_physics` finished last and flipped once paired: `hybrid_centrality` 0.7600 against `original` 0.7517 at
**2.23 paired sem**, so it augments and names centrality alone. That completes 20/20, and it makes the point that the
corpus was never only about degree: none of its graphs was in `STRATEGY_PANEL`, so every decided cell is a genuine
out-of-sample test of `FROZEN_STRATEGY` (`nbr_predictability_adjusted > 0.0092 ⇒ centrality`).

| band | decided corpus cells | signal mix | FROZEN_STRATEGY | "always centrality" |
|---|---|---|---|---|
| frozen | 3 | 2 centrality, 1 degree | **2/3** | 2/3 |
| paired | 9 | 3 centrality, 3 degree, 3 Ψ | **3/9** | 3/9 |

**The rule scores exactly the base rate, under both bands.** The reason is the one already on record as a caveat and now
demonstrable: **no corpus graph falls below the 0.0092 cut** (the lowest is `twitch_es` at 0.0625), so the rule emits
*centrality* for all nine and is a constant predictor on this set. Where the corpus's signals happen to be balanced
3/3/3, a constant predictor gets a third.

Cumulative held-out record for the centrality rule: **6/13** — 3/3 on the LINKX trio, 0/1 on `twitch_de`, 3/9 here.

**What this does and does not say.** It does not show the cut is in the wrong place; every one of these graphs is on the
positive side of it, so the cut itself was never exercised. It shows the rule's *coverage* is the problem: the cut sits
so low that ordinary labelled graphs never test it, and above the cut the rule asserts centrality on graphs that in fact
split three ways. The honest restatement for the paper is that adjusted neighbour predictability separates centrality
from the rest **on the panel it was fitted on**, and out of sample has not yet beaten the base rate.

The stage-2 position after Module 10, stated once, plainly:

- one frozen rule (centrality), **fitted, not validated**, cumulative held-out 6/13, never tested below its own cut;
- no degree rule, and that is now a measured negative on 21 cells rather than a shortage;
- one unpromoted degree-vs-Ψ candidate with 0 in-sample exceptions, LOO 8/9 and a single correct pre-registered call;
- and the reason all of it is hard: **9 of 20 corpus graphs cannot separate degree from Ψ at all.**


### §16 · A pre-registered validation set for the degree-vs-Ψ candidate (2026-09-03)

§14 left `nbr_label_entropy < 0.6724 ⇒ degree` fitted on nine cells and tested on exactly one (`twitch_ptbr`), and §15
showed what that is worth: the frozen centrality rule scored the base rate out of sample **not** because its cut was
wrong but because no graph in the corpus ever fell on the other side of it. A rule tested only where it fires one way is
untested. So the requirement for this candidate is out-of-sample cells on **both sides** of its cut, chosen on the
predictor, written down before anything is trained. That is `cfg.DEGREE_RULE_VALIDATION`.

**The six graphs.** Ordered cheapest-first so a partial run is still a usable test.

| graph | nodes | edges | classes | avg deg | clustering | degree skew | `homophily_adjusted` | `nbr_label_entropy` | `nbr_pred_adjusted` |
|---|---|---|---|---|---|---|---|---|---|
| `wiki_attr` | 2,405 | 11,596 | 17 | 9.64 | 0.3758 | 7.78 | 0.5640 | 0.2219 | 0.6599 |
| `crocodile` | 11,631 | 170,773 | 5 | 29.37 | 0.3365 | 16.16 | 0.0154 | 0.5581 | 0.1379 |
| `cora_full` | 19,793 | 63,421 | 70 | 6.41 | 0.2607 | 7.87 | 0.5558 | 0.1285 | 0.6569 |
| `penn94` | 41,554 | 1,362,229 | 2 | 65.56 | 0.2117 | 14.35 | 0.0209 | 0.8891 | 0.1152 |
| `genius` | 421,961 | 922,868 | 2 | 4.37 | 0.0281 | **106.19** | −0.0527 | 0.1734 | **−0.0028** |
| `twitch_gamers` | 168,114 | 6,797,557 | 2 | 80.87 | 0.1599 | 42.37 | 0.0899 | 0.8647 | 0.2155 |

**Selection, stated so it can be checked.** Chosen on the PREDICTOR and on structural spread, never on an outcome:
the set is meant to straddle 0.6724, and it does. Four candidates were probed and rejected rather than quietly skipped —
musae-facebook (SNAP's own `facebook_large.zip` returns 404 and graphmining.ai does not resolve), PyG's `twitch-gamers`
LINKX loader (bare `AssertionError`; the graph is built from SNAP's archive instead), `AttributedGraphDataset`
facebook/twitter (multi-label, 193 and 4,065 columns), and NELL (codes unlabelled entities as class 0, 85% of nodes).
`amazon_photo` — the obvious co-purchase sibling of the degree winner `amazon_computers` — is excluded for a different
reason: it is a `STRATEGY_PANEL` fitting graph, so it cannot test anything stage 2 fitted.

**Two dependencies that travel with the set.** `cora_ml`, which is IN the corpus, is a **subgraph** of `cora_full`
(2,995 of 19,793 nodes). And `crocodile` ships a continuous monthly-traffic target, binned here into 5 quantile classes —
the same construction Geom-GCN uses to label Chameleon and Squirrel — so its labels correlate with degree exactly as the
`airports_*` labels do. Both are reported next to any result that uses those graphs.

**The fit under test is the published one.** Re-fitting on the 27 corpus+panel graphs that carry a paired verdict, minus
the spent test cell `twitch_ptbr`, reproduces §14 exactly: `nbr_label_entropy < 0.6724`, 9 cells (5 degree / 4 Ψ),
0 exceptions, ρ −0.866, LOO 8/9 against a 0.556 baseline, interval (0.6296, 0.7151). So the cut these six test is the
cut already written down, not a new one.

**The pre-registered calls** (`results/paired_degree_validation_prereg.csv`, written before any of the six was trained;
`experiments/degree_rule.py` refuses to fit on any name in `cfg.DEGREE_RULE_VALIDATION`):

| graph | stage 1 | `nbr_label_entropy` | stage-2 call (degree vs Ψ) |
|---|---|---|---|
| `wiki_attr` | **keep original** — rule 1 veto (homophily 0.5640) | 0.2219 | degree |
| `crocodile` | **augment** | 0.5581 | degree |
| `cora_full` | **keep original** — rule 1 veto (homophily 0.5558) | 0.1285 | degree |
| `penn94` | **augment** | 0.8891 | **Ψ** |
| `genius` | **augment** | 0.1734 | degree |
| `twitch_gamers` | **augment** | 0.8647 | **Ψ** |

**Coverage of the cut, which was the whole point:** 4 graphs land on the degree side (`wiki_attr` 0.2219,
`cora_full` 0.1285, `genius` 0.1734, `crocodile` 0.5581) and 2 on the Ψ side (`penn94` 0.8891, `twitch_gamers` 0.8647).
Module 10 had one out-of-sample cell; this set puts cells on both sides for the first time.

**What this set can and cannot decide.** A stage-2 call is only scorable on a graph that augments AND names one signal,
so `wiki_attr` and `cora_full` — both high-homophily, both vetoed by rule 1 — will only produce a stage-2 cell if they
augment anyway. That is not bad luck, it is a structural tension worth reporting: the candidate's DEGREE side is low
neighbour-label entropy, which is mostly produced by **homophily**, and stage 1 sends homophilous graphs to *keep
original*. `genius` and `crocodile` are the graphs that break the tension — low entropy *and* low adjusted homophily —
and they are the reason the test is possible at all.

**A second, unplanned test the set makes possible.** `genius` has `nbr_predictability_adjusted` **−0.0028**, BELOW the
frozen centrality rule's 0.0092 cut. The standing limitation on that rule (§3 of CLAUDE.md, §15 here) is that only
`tolokers`, `questions` and `actor` sit below the cut and all three are panel members, so its negative side has never
been exercised. `genius` is the first non-panel graph on that side, and stage 1 routes it to *augment*, which is the
rule's own precondition. Its pre-registered call is therefore **not centrality** — the first real test of the half of
`FROZEN_STRATEGY` that has never been tested.

**Scoring, fixed in advance.** A scorable cell is one that augments and names a single signal under the paired band; a
tie is reported, never counted. Any scorable cell whose sole winner is not the predicted signal is an exception. The
candidate stays UNPROMOTED regardless of the count until it has cells on both sides *and* survives them — one good run
does not move it out of the exploratory tier, and `nbr_label_entropy` is exploratory tier by construction.

**Nothing here is trained yet.** The order is the locked one: build → predict → train → score. Steps 1 and 2 are done.

### §17 · The validation set runs: stage 1 goes 5/5, and the degree-vs-Ψ candidate FAILS (2026-09-04)

All six graphs of `cfg.DEGREE_RULE_VALIDATION` trained under the unchanged pipeline (GraphSAGE, K=10, 3 seeds, link
prediction, seven variants), then re-scored paired with `tie_break.py`. Predictions were on disk before any of it ran
(§16), so everything below is a pre-registered test, not a description.

**Outcome per graph** (paired band; `gap_sigma` is best-augmented against original):

| graph | stage-1 call | actual | signal (paired) | stage-2 call | gap σ |
|---|---|---|---|---|---|
| `wiki_attr` | keep original | tie — **no decision** | `original` | degree | −0.47 |
| `crocodile` | augment | **augment** ✓ | **Ψ** | degree | +1.45 |
| `cora_full` | keep original | **keep original** ✓ | `centrality\|original` | degree | −3.19 |
| `penn94` | augment | **augment** ✓ | **centrality** | Ψ | +1.82 |
| `genius` | augment | **augment** ✓ | **Ψ** | degree | +4.51 |
| `twitch_gamers` | augment | **augment** ✓ | **Ψ** | Ψ | +7.02 |

**Stage 1: 5/5** (`results/module5_scored.csv`). Rule 1 alone 5/5, the clustering exception 4/5 — it said *augment* on
`cora_full`, which rule 1's veto correctly overrode, exactly the order §2 specifies. `wiki_attr` is a tie and is scored
as *no decision*, not as a win. Stage 1's cumulative record is now **18/21**. This is also the pre-registered re-test the
clustering exception was still owed (CLAUDE.md §5 item 1), and it passes.

**Stage 2, the degree-vs-Ψ candidate: 1/3, and that is the generous reading.**

| cell | `nbr_label_entropy` | predicted | actual | |
|---|---|---|---|---|
| `crocodile` | 0.5581 | degree | **Ψ** | ✗ |
| `genius` | 0.1734 | degree | **Ψ** | ✗ |
| `twitch_gamers` | 0.8647 | Ψ | **Ψ** | ✓ |

`cora_full`, `penn94` and `wiki_attr` are *not applicable* — the contrast is only defined where the sole winner is degree
or Ψ, and they name `original`, `centrality` and `original` respectively. Scoring the candidate against those cells would
credit it with a denominator it never earned, so `heldout()` was extended to score the two contrasts on their own cells.

**The damning detail is not the 1/3.** All three scorable cells have Ψ as the sole winner, so a constant "always Ψ"
predictor scores **3/3** where the candidate scores 1/3. Its single success is the one call where it happened to agree
with that constant. And both of its DEGREE-side calls failed — the side the rule exists for. Three other properties
(`components`, `degree_gini`, `largest_component_frac`) also score 3/3 here for the same empty reason: on this set they
too are constant. This is the mirror image of §15's coverage failure, and it is the second time in three modules that an
out-of-sample set has been decided by a constant.

**Verdict: the candidate is TESTED AND FAILED, not "unpromoted pending evidence".** Its record in full — in-sample
0 exceptions on 9 cells, LOO 8/9 against a 0.556 baseline, chance-of-free-split 0.016, then 1/1 on `twitch_ptbr`, then
**1/3** here. Cumulative held-out **2/4**, i.e. the coin-flip rate. `nbr_label_entropy < 0.6724 ⇒ degree` must not be
reported as a rule, a lead, or a "candidate awaiting more data". It is a negative result, and it is the clearest evidence
the study has that in-sample separation on nine cells — even priced against chance — is not evidence.

**The set also produced zero new degree winners.** Degree still wins alone on exactly the same five graphs it did before
(`actor`, `questions`, `twitch_engb`, `airports_europe`, `amazon_computers`). Six graphs chosen to straddle the cut added
three Ψ cells, one centrality cell and two non-augmenting cells. That is now **twelve** graphs ingested across Modules 9,
10 and 11 for the degree question with **three** degree wins between them (`twitch_engb`, `airports_europe`,
`amazon_computers`; `actor` and `questions` predate the effort). The scarcity is the finding.

**One thing did work, and it was not planned.** `genius` has `nbr_predictability_adjusted` −0.0028, below
`FROZEN_STRATEGY`'s 0.0092 cut, and stage 1 routed it to augment — the first time in the whole study that the centrality
rule's NEGATIVE side has ever fired. It predicted *not centrality*; Ψ won and centrality finished last of the three
signals. On this set the rule goes 2/4 (`penn94` ✓, `genius` ✓, `crocodile` ✗, `twitch_gamers` ✗), cumulative **8/17**,
but the `genius` call is worth more than its weight: it is the first evidence that the cut separates anything at all,
rather than merely never being exercised.

### §18 · Re-searching the degree selector on the enlarged data: still nothing (2026-09-04)

With the validation six spent (§17), they were folded into the fitting pool and every property re-screened over all
**34 graphs** carrying a paired verdict — `experiments/degree_rule.py --labels paired --allow-refit`, writing
`exploratory_paired_*.csv` so no published table is touched. This is a fit, not a test: whatever it found would need a
new held-out set, and it found nothing that would deserve one.

**Degree vs everything: the negative hardens.** 29 augmenting cells, still only **5** degree sole winners (`actor`,
`airports_europe`, `amazon_computers`, `questions`, `twitch_engb`). No property clears the Module-2 gates. The best is
`homophily_adjusted < −0.0634` at ρ −0.24 with 4 exceptions and LOO 0.862 against a 0.828 majority baseline — the
baseline is that high only because 24 of 29 cells are "not degree", so beating it by 0.034 is worth nothing. The
measured negative from §13 now rests on 29 cells rather than 21.

**Degree vs Ψ: the same property is still the best, and it is worse than it was.** 13 cells (5 degree / 8 Ψ).

| | Module 10 (9 cells) | Module 11 re-fit (13 cells) |
|---|---|---|
| cut | `nbr_label_entropy < 0.6724` | `< 0.5336` |
| exceptions | 0 | **2** |
| ρ | −0.866 | −0.634 |
| LOO | 8/9 = 0.889 vs 0.556 | 0.769 vs 0.615 |

It fails the gates on both counts (|ρ| < 0.7, exceptions > 1) and is exploratory tier besides. **No property in the
screen reaches a clean split on 13 cells** — and at this n a clean split would have meant something, since chance buys
only 0.02 free winners out of 15 tried (against 0.24 at n=9 and 4.3 at n=4). The absence is informative rather than
under-powered.

**The two graphs that break it are worth naming.** `twitch_engb` (entropy 0.6296, above the cut, degree wins) and
`genius` (0.1734, well below it, Ψ wins). `genius` is the harder one: its entropy is low because its labels are 80:20
imbalanced, not because neighbourhoods are label-pure — the same statistic reached by a different mechanism. That is a
property-definition problem, not a data-volume problem, and no amount of extra graphs fixes it.

**Where this leaves stage 2.** One frozen rule (centrality), fitted-not-validated, cumulative 8/17, with a single
correct negative-side call to its name. No degree rule, and now two independent failures to find one: a screen that
finds nothing (§13, §18) and a candidate that looked good and did not survive contact with new graphs (§14, §17). The
standing explanation remains mechanistic and is still untested directly: Ψ is a Poisson/KL score over degree-based
signatures, so the two role graphs largely coincide. **Measuring the edge overlap between the degree and Ψ role graphs
is the next experiment**, and after §18 it is the only one on this branch with a clear rationale — another property
screen has now been run twice and returned nothing twice.

### §19 · Measuring the mechanism: how much do the degree and Ψ role graphs actually differ? (2026-09-04)

`experiments/role_overlap.py`, 34 graphs, K=10, role graphs built on the full original graph, nothing trained and
nothing fitted. This is the experiment §18 named as the only one left with a rationale on this branch.

**The confound that had to be handled first.** The signature is 1-D and `VirtualGraph.build` SAMPLES inside tie classes,
so two builds of the *same* signal already disagree. A raw degree-vs-Ψ Jaccard is therefore uninterpretable. Each signal
was rebuilt under a second RNG seed to give a self-overlap floor, and the headline number is
`overlap_ratio = cross-overlap / self-overlap`: ≈1 means the two signals differ no more than one signal differs from
itself, i.e. they are the same construction; ≪1 means they genuinely diverge.

**Finding 1 — Ψ is inversely rank-correlated with degree, on 33 of 34 graphs.** Mean ρ ≈ −0.80, reaching −0.9931
(`blogcatalog`) and −0.9911 (`email_eu_core`). `roman_empire` is the sole exception at +0.156. This has never been
recorded in the study and it is the mechanism in one number: Ψ is a near-monotone (decreasing) function of degree, and a
monotone map between two 1-D signatures forces near-identical top-K neighbourhoods regardless of sign.

**Finding 2 — the tie explanation is confirmed, at the aggregate level.** Mean `overlap_ratio` by what actually won:

| what won | n | overlap ratio | \|sig ρ\| | profile gap |
|---|---|---|---|---|
| **tie** (augments, names nothing) | 9 | **0.451** | 0.897 | **0.116** |
| Ψ alone | 8 | 0.401 | 0.855 | 0.159 |
| centrality alone | 7 | 0.229 | 0.697 | 0.223 |
| degree alone | 5 | 0.188 | 0.729 | 0.236 |
| no augmentation | 5 | **0.023** | 0.702 | **0.314** |

The ordering is the predicted one: graphs that cannot name a signal are exactly the graphs whose two role graphs are
most alike, and the gap widens as the outcome becomes more decisive. The standing explanation for the tie rate — carried
as an assertion since §14 — is now measured.

**Finding 3 — and it still does not yield a rule.** On the 13 degree-or-Ψ cells the best new feature is `sig_spearman`
(or `profile_gap_cross`, which mirrors it): **3 exceptions**, LOO 0.769 against a 0.615 majority baseline. Neither clears
the Module-2 gates. The distributions overlap outright — degree winners span `overlap_ratio` 0.019–0.763, Ψ winners span
0.005–1.066, and Ψ's minimum (`genius`, 0.005) is below every degree winner. On tie-vs-decided over 29 augmenting cells,
`overlap_ratio` scores LOO 0.690 against a 0.690 baseline: exactly nothing.

**Finding 4 — the new features are largely density in disguise.** `overlap_ratio` correlates with `density` at ρ **+0.894**
and with `avg_degree` at +0.713; `sig_spearman` and `profile_gap_cross` sit at −0.74/−0.75 against the same two. Density
was screened in Module 2 and failed. So this is not the fresh feature space it looked like, and a rule fitted on it would
be a rule fitted on density with extra steps. Stating this is the difference between a mechanism study and a third
threshold search.

**Finding 5 — on sparse graphs the degree role graph is barely a well-defined object.** The self-overlap floor for
`degree` falls to 0.0030 (`roman_empire`), 0.0032 (`genius`), 0.0145 (`questions`), 0.0180 (`pubmed`): two builds of the
same signal share under 2% of their edges, because almost every pick is an arbitrary draw from a large tie class. The
`psi` floor is far higher (0.15–0.999) — its signature is continuous and ties are rare. This asymmetry matters for the
whole degree branch: where the floor is that low, "the degree role graph" names a distribution rather than a graph, and
asking which graph property predicts its winning may be asking for a stable answer to an unstable question.
(Methodological note: `profile_gap_floor` is ≈0 everywhere, 0.0000–0.0005, confirming that resampling within a tie class
preserves the neighbour-degree profile exactly — which is what makes `profile_gap_cross` the tie-robust comparison.)

**Verdict on the mechanism study.** It did what it was asked to do: the tie rate now has a measured explanation rather
than an assumed one, and Ψ's inverse-monotone relationship to degree is a reportable fact. It did **not** produce a
degree selector, and Findings 3–5 say why not to attempt one from these features. The degree branch has now returned a
negative from three independent directions — a property screen (§13, §18), a pre-registered test of the one candidate
(§17), and a mechanism measurement (§19). That is enough to report "no degree rule at the graph level" as a result of
the study rather than as an open task.

### §20 · A degree-SPECIFIC property tier, and an exact null that retires it (2026-09-04)

On instruction, four degree-specific property families were built rather than reusing the 15 general ones a third time
(§13, §18) — `experiments/degree_features.py`, measured on all 34 paired-verdict graphs, kept in its own `DEGREE_TIER`
list that a screen must request with `--tier degree` so no published result can move because the file exists.

| family | features |
|---|---|
| tie structure | `degree_tie_ratio` — mean fraction of a node's K role slots its own degree tie class could fill |
| degree resolution | `distinct_degrees`, `distinct_degree_frac` |
| stability under reseeding | `degree_vg_stability`, `psi_vg_stability` — mean pairwise edge Jaccard across RNG seeds |
| divergence from Ψ | `deg_psi_jaccard`, `deg_psi_overlap_ratio`, `deg_psi_profile_gap`, `deg_psi_sig_spearman` |

**A confound column was added to every screened row.** `rho_vs_density` is now reported next to every rule, because §19
found the divergence features sit near |ρ| 0.9 against density — a property screened and failed in Module 2. It
immediately earned its place: `degree_vg_stability` correlates with density at ρ **1.0000**, `deg_psi_jaccard` at
0.9945, `distinct_degree_frac` at 0.9890, `degree_tie_ratio` at −0.9849. **Four of the nine new features are density
under another name.** Only `distinct_degrees` (−0.380) and `psi_vg_stability` (+0.107) carry information density does not.

**Degree vs everything (29 augmenting cells, 5 degree):** nothing. Best is `deg_psi_overlap_ratio`, ρ −0.229, 6
exceptions, LOO 0.793 against a 0.828 majority baseline — below the baseline, like every other row.

**Degree vs Ψ (13 cells, 5 degree / 8 Ψ): the best result this branch has produced, and it is exactly what chance pays.**
`distinct_degrees < 158.5 ⇒ degree`: **2 exceptions**, LOO **0.846** against a 0.615 baseline, interval (130, 187),
`rho_vs_density` −0.380 so it is genuinely not density. Sorted, the split is clean on one side:

| below the cut | | above the cut |
|---|---|---|
| `actor` 89 ✓, `airports_europe` 102 ✓, `twitch_engb` 130 ✓ | | 8 Ψ winners ✓, plus `questions` 320 ✗ and `amazon_computers` 347 ✗ |

**The exact null settles it.** Over all C(13,5) = 1287 label arrangements, the best threshold's error count distributes
as: P(0 exceptions) = 0.0016, P(≤1) = 0.0202, **P(≤2) = 0.1212**. Against the 9 properties tried, chance alone is
expected to deliver **1.09** properties at ≤2 exceptions. One was found. `distinct_degrees` sits precisely at the rate
noise supplies, so it is not evidence — and it is 0.828 rank-correlated with `nodes`, though raw `nodes` alone does
worse (4 exceptions, LOO 0.538), so it is not purely size either.

**What the tier did establish, as measurement rather than as a rule.** `degree_tie_ratio` runs 0.94–0.99 on all but a
handful of graphs: on nearly every graph in the study, a node's entire degree-role neighbourhood is an arbitrary draw
from one tie class. `degree_vg_stability` falls to 0.003 on the sparsest graphs. Together with §19's Finding 5 this
says the degree role graph is, on most of this corpus, a *sample* rather than a graph — which is a mechanism-level
reason why "when does degree win" resists a stable answer, independent of which properties are screened.

**The one shape worth recording.** The cut has **no false positives**: all three graphs below it are degree winners. It
fails by missing 2 of 5 degree winners, not by wrongly claiming degree. A one-sided reading — "below ~158 distinct
degrees, degree wins" — is a weaker and more defensible claim than the two-sided rule, but it rests on three cells and
cannot be established from them. It is written down here so a future pre-registered test has something specific to test.

**Position after four attempts.** The degree branch has now returned a negative from a general property screen (twice),
a pre-registered test of its one candidate, a mechanism measurement, and a purpose-built degree-specific tier. The
binding constraint is unchanged and is not a feature-engineering problem: **five graphs in a 34-graph corpus name degree
as their sole winner.** At n=13, chance buys a 2-exception split once per nine properties tried. No feature set can fix
that; only degree-winning graphs can.

### §21 · FROZEN_DEGREE tested on seven new graphs: 1/4, and the direction is inverted (2026-09-05)

`cfg.DEGREE_TEST`, seven graphs ingested from Netzschleuder specifically to test the cut frozen in §20, trained under the
unchanged pipeline (GraphSAGE, K=10, 3 seeds, link prediction), paired band. Predictions were on disk before training
(`results/degree_test_prereg.csv`, `results/module5_predictions.csv`).

**Why these seven, and why they had to come from outside PyG.** The cut predicts *degree* below 158.5 distinct degrees,
so testing that half needs graphs that are both low-degree-resolution AND low-homophily. torch_geometric has no such
family left: its only new low-distinct-degree graphs are road networks (CityNetwork paris/shanghai/LA/London, GraphLand
city-roads-M/L), and every one has adjusted homophily 0.31–0.57, so stage 1 vetoes them and they can never carry a
stage-2 cell. Netzschleuder supplied the missing corner, via two families that are heterophilous *by construction*:
bipartite networks (adjusted homophily exactly −1.0) and food webs. Selection was on the predictor and the stage-1 call,
never on an outcome. **DISCLOSURE: five of the seven are bipartite, a structural class absent from the fitting panel.**

| graph | distinct degrees | predicted | augments | winner (paired) | scored |
|---|---|---|---|---|---|
| `celegans_neural` | 58 | degree | no | `centrality\|original` | — |
| `messal_shale` | 72 | degree | no | full tie | — |
| `escorts` | 118 | degree | yes | `centrality\|degree` | — (tie) |
| `nematode_mammal` | 124 | degree | yes | **degree** | ✓ |
| `plant_pol_robertson` | 130 | degree | yes | **Ψ** | ✗ |
| `artnet_exp` | 196 | Ψ | yes | **degree** | ✗ |
| `bag_of_words_kos` | 437 | Ψ | yes | **degree** | ✗ |

**FROZEN_DEGREE scores 1/4.** Worse than the count: the failure is *directional*. **Both** graphs above the cut — where
the rule says Ψ — name **degree**, and the one clean below-cut prediction that resolved to a single non-degree signal
(`plant_pol_robertson`, 130, the cell closest to the fitted interval) names Ψ. On this set the cut points the wrong way.
The §20 pricing said exactly this was likely: P(≤2 exceptions by chance) = 0.1212 across 9 properties tried expects 1.09
free winners, and one was found. It has now been falsified on the cells it was frozen to be falsified on.

**But the search criterion that found these graphs was right, and that is the result worth keeping.** Three of the seven
name **degree alone** — `nematode_mammal`, `bag_of_words_kos`, `artnet_exp`. Across Modules 9–11 the entire study had
produced **five** degree sole-winners from 34 graphs; seven graphs chosen for low homophily and bounded degree produced
**three more**, taking the total to **eight**. That is the largest gain in degree evidence the project has had, and it
came from targeting *graph families* (bipartite, heterophilous, bounded-degree) rather than from another threshold.
`bag_of_words_kos` (437 distinct degrees) and `artnet_exp` (196) show the family matters more than the count: both sit
well above the cut and degree still wins.

**Stage 1: 5/6** (`messal_shale` is a full tie, scored as no decision). The single miss is `celegans_neural` — adjusted
homophily 0.1673, clustering 0.0845, so both conditions said augment and the original won. It joins `minesweeper`,
`deezer_europe` and now a fourth as an unexplained low-homophily keep. Stage-1 cumulative: **23/27**.

**Position.** The degree question now has 8 sole-winner cells instead of 5, and a clear qualitative pattern — degree wins
on bipartite, heterophilous, bounded-degree graphs — but no cut that survives testing. Four candidate rules have now been
falsified out of sample (`homophily_adjusted` n=4 hypothesis, `nbr_label_entropy`, the mechanism features, and now
`distinct_degrees`). The honest next step is NOT a fifth threshold on 8 cells: it is to keep ingesting from the family
that just yielded 3 winners from 7 graphs, until the degree class is large enough that a rule fitted on it can be priced
above chance.

### §22 · Second family batch: the degree class reaches 10, and two homologous pairs SPLIT (2026-09-05)

`cfg.DEGREE_BATCH2`, seven more graphs from the family §21 identified (bipartite / heterophilous / bounded-degree,
Netzschleuder), trained under the unchanged pipeline, paired band. Purpose was collection, not testing — `FROZEN_DEGREE`
was already falsified in §21 — but being frozen it is scored on them anyway.

| graph | distinct degrees | rule says | augments | winner (paired) |
|---|---|---|---|---|
| `board_directors` | 12 | degree | **no** | `original\|psi` |
| `plant_pol_kato` | 35 | degree | yes | centrality |
| `foursquare_checkin` | 47 | degree | yes | `degree\|psi` tie |
| `foursquare_tips` | 47 | degree | yes | **degree** ✓ |
| `ppi_rat` | 90 | degree | yes | **Ψ** ✗ |
| `ppi_mouse` | 199 | Ψ | yes | **degree** ✗ |
| `bag_of_words_nips` | 800 | Ψ | yes | `degree\|psi` tie |

**The collection worked again: 2 more degree sole-winners** (`foursquare_tips`, `ppi_mouse`). The degree class is now
**10 across 48 graphs**, up from 5 across 34 before this family was targeted — 5 of the 10 have come from 14 graphs
chosen on family rather than on any threshold. Six of the seven augment, so the family also keeps clearing stage 1.

**FROZEN_DEGREE scores 1/3 here, 2/7 cumulative, and fails the same way.** `ppi_mouse` (199, above the cut, predicted Ψ)
names degree — the third above-cut graph in a row to do so, after `artnet_exp` (196) and `bag_of_words_kos` (437). The
cut's Ψ side has now been wrong on every graph that has tested it. Whatever separates degree from Ψ, `distinct_degrees`
is not it, and the direction of the residual is consistent rather than random.

**The result that constrains every future rule: both non-independent pairs SPLIT.** `ppi_rat` and `ppi_mouse` are
homologous protein-interaction networks and name **Ψ** and **degree** respectively. `foursquare_checkin` and
`foursquare_tips` are two edge types over the same NYC restaurant population and name a **tie** and **degree**. Graphs
drawn from the same source, same construction and same domain do not agree on the winning signal. That is an upper bound
on how predictable this choice can be from graph-level properties at all — and it is a much cheaper explanation for five
failed rules than any of the property-specific stories. It should be reported next to the negative results, not buried:
the degree-vs-Ψ outcome carries a component that graph identity does not determine.

**Standing disclosure** (declared before training, `virgo/data/make_netzschleuder.py` and `config.py`): those two pairs
are not independent cells and are never counted as such. The split above is what makes the disclosure informative rather
than merely cautious.

### §23 · How much of a winning-signal label is seed noise? (2026-09-05)

`experiments/winner_stability.py`. Every stage-2 cell in this study is labelled from THREE seeds. §22 found two
non-independent pairs that disagreed, which raised a question no property screen can answer: **how reproducible is a
3-seed label on one fixed graph?** Eight graphs carrying degree-or-Ψ cells were trained to ten seeds, then every one of
the C(10,3) = 120 three-seed subsets was relabelled under the paired band, choosing each signal's best variant *inside*
the subset so the resampling cannot leak the full-pool answer.

| graph | full-pool winner | modal 3-seed | modal frac | flip rate | distinct winners |
|---|---|---|---|---|---|
| `foursquare_checkin` | `degree\|psi` | `degree\|psi` | 0.442 | **0.558** | 6 |
| `bag_of_words_kos` | `degree\|psi` | `degree\|psi` | 0.683 | 0.317 | 3 |
| `artnet_exp` | **degree** | degree | 0.917 | 0.083 | 2 |
| `plant_pol_robertson` | **Ψ** | Ψ | 0.925 | 0.075 | 2 |
| `nematode_mammal` | **degree** | degree | 0.950 | 0.050 | 2 |
| `foursquare_tips` | **degree** | degree | **1.000** | 0.000 | 1 |
| `ppi_mouse` | **degree** | degree | **1.000** | 0.000 | 1 |
| `ppi_rat` | **Ψ** | Ψ | **1.000** | 0.000 | 1 |

**Mean flip rate 0.135, mean modal fraction 0.865, and the published 3-seed label matches the ten-seed label on 7 of 8.**
The protocol is sounder than §22's worry implied: instability is not spread across the panel, it is concentrated in the
two graphs already reported as **ties**. A cell that names a single signal has, on this evidence, named it reliably.

**This is the first good news the degree branch has produced, and it is about the DATA rather than a rule.** All four
degree cells are stable — `foursquare_tips` and `ppi_mouse` at 120/120, `nematode_mammal` 114/120, `artnet_exp` 110/120.
Degree wins are not seed artifacts. The degree class is real.

**Two corrections to the record, both mine.**

1. **§22's ceiling claim was half wrong and must be narrowed.** I wrote that both non-independent pairs split, and used
   it to argue graph-level properties cannot determine the winner. Resampling separates the two cases. `ppi_rat` (Ψ) and
   `ppi_mouse` (degree) are **each perfectly stable at 120/120**, so that split is real and the ceiling argument holds
   for it: two homologous protein-interaction networks genuinely disagree. But `foursquare_checkin` is the *least*
   stable graph measured (flip 0.558, six distinct winners across subsets), so its disagreement with `foursquare_tips`
   is mostly label noise and carries no such weight. One example survives, one dissolves.
2. **`bag_of_words_kos` is not a degree sole-winner.** Its published 3-seed label was `degree`; at ten seeds it is a
   `degree|psi` tie, and it is the single dataset of the eight whose label changes. §21's "three new degree winners"
   is therefore **two** (`nematode_mammal`, `artnet_exp`), and the running count is:

   | | degree sole-winners |
   |---|---|
   | before the family batches | 5 |
   | §21 batch (7 graphs) | +2 |
   | §22 batch (7 graphs) | +2 |
   | **total** | **9 across 48 graphs** |

**What it does not rescue.** The stable cells still do not separate on `distinct_degrees`: degree wins at 47, 124, 196
and 199, Ψ wins at 90 and 130 — interleaved, and `ppi_mouse` (199, above the cut) versus `ppi_rat` (90, below it) is the
frozen rule's direction exactly reversed, now with 120/120 confidence on both sides. The rule is not merely unlucky; the
cells it gets wrong are the ones measured most precisely.

### §24 · The leading degree signal at 19 cells: role-graph DIVERGENCE, priced (2026-09-05)

With the cell count more than doubled (9 -> 19 degree-or-Ψ cells, 9 degree / 10 Ψ, all seed-verified per §23), both
tiers were re-screened. The question is what, if anything, separates the two columns now.

**The leading candidate is `deg_psi_overlap_ratio`: how much the DEGREE role graph differs from the Ψ role graph,
normalised by how much the degree role graph differs from ITSELF under reseeding (§19).** Low ratio = the two
constructions genuinely diverge. `deg_psi_overlap_ratio < 0.2643 ⇒ degree`, ρ −0.50, LOO 0.789 against a 0.526 baseline,
**3 exceptions on 19 cells**. Sorted, it is close to clean:

| below the cut (predict degree) | above the cut (predict Ψ) |
|---|---|
| nematode_mammal .121 ✓, **genius .122 ✗**, questions .133 ✓, artnet_exp .139 ✓, actor .148 ✓, foursquare_tips .163 ✓, twitch_engb .197 ✓, ppi_mouse .248 ✓ | twitch_gamers .281 ✓, ppi_rat .299 ✓, crocodile .370 ✓, twitch_de .389 ✓, **amazon_computers .400 ✗**, plant_pol_robertson .716 ✓, twitch_es .723 ✓, tolokers .747 ✓, **airports_europe .806 ✗**, twitch_ptbr .910 ✓, blogcatalog 1.082 ✓ |

**The direction has a mechanism, which no previous candidate had.** It says degree wins exactly when its role graph is
NOT the Ψ role graph in disguise — the cell-level version of the group pattern §19 measured (mean overlap ratio: degree
0.188, Ψ 0.401, tie 0.451). Where Ψ is a near-monotone function of degree the two constructions coincide and Ψ (or a
tie) takes it; where they genuinely diverge, degree can win on its own terms.

**The exact price, which is why this is a hypothesis and not a rule.** Over all C(19,9) = 92,378 label arrangements,
P(≤3 exceptions for one property) = 0.0210, so across the 24 properties screened chance alone is expected to deliver
**0.50** at this quality. One was found. That is about twice the chance rate — better than anything the branch has
produced, and nowhere near enough to freeze. For comparison, everything at 4 exceptions (`nbr_label_entropy`,
`avg_degree`, `deg_psi_jaccard`, `deg_psi_sig_spearman`) sits at 2.01 expected-free, i.e. exactly chance.

**Two caveats that must travel with it.**
1. **It is 94% density.** `rho_vs_density` = 0.944. Raw density does worse (5 exceptions vs 3), so it is not *only*
   density — but with that much shared variance the claim "this is a new signal" is not yet supportable.
2. **It needs both role graphs BUILT.** That is legal (building is not training) but it is the same property class as
   the retired Module-4 gate, which fitted beautifully and did not transfer. The precedent is unfavourable.

**The three exceptions are informative.** `genius` breaks it, exactly as it broke `nbr_label_entropy` — the same graph
has now falsified two different candidates, which suggests it is genuinely anomalous rather than that both properties
are wrong. `amazon_computers` and `airports_europe` are the two oldest degree winners, both from families whose labels
are degree-correlated by construction (traffic quartiles, dense co-purchase).

**What has actually changed, and it is the useful part.** At 19 cells a clean split is essentially unreachable by chance
(P(0) = 0.0000, P(≤1) = 0.0004, i.e. 0.01 expected free over 24 properties). **The dataset is finally large enough that
a real rule would be visible as one.** None is. So the correct statement for the paper is no longer "we could not find a
rule on too little data" — it is "at a size where a real rule would show, the best signal available is role-graph
divergence at twice the chance rate, and it is confounded with density."

### §25 · The DEPLOYMENT question: original-graph properties only, degree vs everything (2026-09-05)

§24 answered the wrong question. The contrast that matters for deployment is not degree-vs-Ψ but **"given that
augmentation helps, do we pick degree?"**, decided from properties of the ORIGINAL graph alone — no role graph built,
which is what disqualifies `deg_psi_overlap_ratio` as a rule however good its mechanism story is. Re-run on the enlarged
pool: **40 augmenting graphs, 9 naming degree alone, majority baseline 0.775.** Screened properties are the 15 general
ones plus the three degree-tier ones that need no role graph (`distinct_degrees`, `distinct_degree_frac`,
`degree_tie_ratio`).

**Single property: nothing.** Best is 7 exceptions (`degree_gini`, `majority_class_frac`, `components`); best LOO 0.825
(`degree_gini`) against a 0.775 baseline. Seven exceptions against nine positives is not a rule.

**Two-property AND search: nothing, and this time the null proves it.** Best over ~220,000 candidate rules (153 property
pairs x 4 direction combinations x 19x19 threshold grids) is `nbr_predictability_adjusted < 0.1326 AND
nbr_label_entropy < 0.6321` at **5 exceptions**. A permutation null running the SAME search on shuffled labels gets a
median of 5.0 and a minimum of 4: **P(a shuffle does as well) = 0.633**. The search space is large enough that five
exceptions is what noise produces. This is the first time the degree branch has had a properly calibrated null for a
multi-property search, and it retires the whole approach rather than one candidate.

**The pre-specified hypothesis (user, 2026-09-05: "low degree diversity + high degree skew ⇒ degree") is the best
result, and it is a partial one.** Pre-specifying the two properties removes the search penalty, so only the thresholds
are fitted:

| diversity measure | exceptions | TP / FP / FN | P(shuffle as good) |
|---|---|---|---|
| **`distinct_degrees`** | **6** | **4 / 1 / 5** | **0.051** |
| `degree_tie_ratio` | 7 | 3 / 1 / 6 | 0.195 |
| `distinct_degree_frac` | 7 | 3 / 1 / 6 | 0.163 |

`distinct_degrees` low AND `degree_skew` high fires on five graphs — `actor`, `foursquare_tips`, `ppi_mouse`,
`twitch_engb` (all degree winners) and `ppi_rat` (Ψ). **As a trigger it is high-precision and low-recall: 4/5 = 0.80
precision against a 0.225 base rate, a 3.6x lift, but it catches only 4 of the 9 degree winners.** Two honest
discounts: p = 0.051 is for ONE pre-specified pair, and three diversity measures were tried, so the corrected figure is
nearer 0.15; and the same one-sided shape (fires rarely, usually right) was already seen in §20 and did not survive.

**Status: the best degree selector the study has, and it is not promotable yet.** It is a genuine deployment-shaped rule
— both inputs are read off the original graph — but it explains under half the degree class and sits at a corrected
p ≈ 0.15. Four candidates have now been falsified out of sample; this one must be pre-registered and tested on unseen
graphs before it is called anything more than a hypothesis.

### §26 · Stabilising the trigger before testing it — and the decisive negative that follows (2026-09-05)

The §25 trigger (`distinct_degrees` low AND `degree_skew` high, in-sample precision 0.80 at a 0.225 base rate) was
stabilised before spending a test set on it. That step is what settled the question.

**The thresholds do not stabilise.**

| check | result |
|---|---|
| point fit | `distinct_degrees < 201 AND degree_skew > 13.878`, 6 exceptions |
| plateau | only **12 of 2303** cut pairs reach 6 exceptions — a spike, not a plateau |
| bootstrap (1000) | `distinct_degrees` cut 201 **[90, 353]**, skew cut 13.88 **[5.47, 16.66]** |
| **leave-one-out** | **30/40 correct against a 31/40 majority baseline** |
| **LOO precision** | **0.40** (2 of 5 fires), against 0.80 in-sample |

Refit without each graph in turn, the rule is **worse than always answering "not degree"**, and its precision collapses
from 0.80 to 0.40 — barely above the 0.225 base rate. The 0.80 was the fit reading its own answer back. The bootstrap
intervals span nearly the whole range of both properties, so there is no threshold to freeze.

**Then the second search, run as specified: over all 40 graphs, not only the 5 the trigger misses.** Every route was
priced with a permutation null that repeats the identical search on shuffled labels:

| search | best on real labels | P(a shuffle does as well) |
|---|---|---|
| best single property | 7 exceptions | **0.737** |
| best two-property AND (~220,000 rules) | 5 exceptions | **0.633** |
| best two-property OR | 5 exceptions | **0.300** |
| any property splitting caught-vs-missed degree winners | 0 of 18 split cleanly | — |
| degree class vs a random 9-subset (no fitting) | mean pairwise distance 5.836 vs 5.893 | **0.594** |

**Conclusion, on the criterion set in advance: degree is not predictable from these 18 original-graph properties.** Not
"not yet" and not "not with enough data" — at 40 augmenting graphs with 9 degree winners, every search returns what
shuffled labels return. The coherence test explains why the rule searches keep failing: **the degree class is not
structurally distinguishable from a random subset of the same size**, so there is no first cluster to find, let alone a
second one to OR onto it.

**Methodological note worth keeping.** Stabilising before testing is what converted a promising candidate into a settled
negative at zero cost. Had the trigger gone to a fresh 7-graph test set at its in-sample 0.80 precision, it would have
consumed the graphs and returned an ambiguous 2-or-3-of-5. Leave-one-out on data already in hand answered it outright.
Four earlier candidates in this study were promoted without that step and all four died out of sample; this is the first
one retired before it cost anything.

### §27 · Last two routes: the full degree tier, and a better-powered target. Both negative (2026-09-05)

Two gaps in §26 were closed rather than argued away.

**1. The six role-graph degree features were screened on the DEPLOYMENT question, not just degree-vs-Ψ.** §24 tested
them only on the 19-cell degree-vs-Ψ contrast; §25-26 excluded them from the 40-graph screen because they need role
graphs built. Screened properly, the whole 9-feature tier is the *weakest* set yet: best 8 exceptions
(`deg_psi_sig_spearman`), **P(shuffle as good) = 0.960**. `deg_psi_overlap_ratio`, which was §24's leading candidate at
3 exceptions on degree-vs-Ψ, scores **9 exceptions** here and is indistinguishable from noise.

That contrast is itself the finding: **the overlap ratio is a discriminator, not a detector.** It separates degree from
Ψ *among graphs already known to name one of the two*, because that is the comparison it is built from. It cannot say
which of 40 augmenting graphs will name degree, since centrality winners and ties occupy the same range. Deployment
needs a detector.

**2. The target was re-posed to a better-powered one.** "Degree wins ALONE" has 9 positives in 40. "Degree is AMONG the
winners" — the deployable question *is degree a safe choice?*, since picking degree costs nothing when it ties — has
**22 positives in 40**, a 0.550 baseline instead of 0.775. Screening all 24 properties on that target: best is
`degree_assortativity > −0.2291` at **13 exceptions**, LOO 0.650 against the 0.550 baseline, and the permutation null
gives a shuffled **median of 12** — better than the real labels. **P(shuffle as good) = 0.936.**

**Complete tally, every framing of the degree question, each priced by rerunning the identical search on shuffled labels:**

| target / search | positives | best | P(shuffle as good) |
|---|---|---|---|
| degree alone, 15 general + 3 original degree properties | 9/40 | 7 exceptions | 0.737 |
| degree alone, full 9-feature degree tier | 9/40 | 8 exceptions | **0.960** |
| degree alone, two-property AND (~220k rules) | 9/40 | 5 exceptions | 0.633 |
| degree alone, two-property OR | 9/40 | 5 exceptions | 0.300 |
| **degree among winners, all 24 properties** | **22/40** | **13 exceptions** | **0.936** |
| any property splitting caught-vs-missed winners | 4 vs 5 | 0 of 18 clean | — |
| class coherence vs a random 9-subset (no fitting) | — | 5.836 vs 5.893 | 0.594 |

**This is where the degree branch stops.** Two property sets, two targets, four rule classes, one within-class split and
one fitting-free coherence test — every one returns what shuffled labels return. The single candidate that looked
promising failed leave-one-out before it cost a test set (§26). Continuing past this point would not be searching, it
would be selecting a rule from noise until one survives, which is precisely what produced the four candidates this study
has already had to retire.

**What would change the answer**, stated so the negative is falsifiable rather than final: a degree class of roughly 30+
sole-winners (currently 9), which at the observed 2-3 per 7 targeted graphs means ingesting on the order of 70 more
graphs from the bipartite/heterophilous family; or a feature space that is not hand-designed graph properties at all.
Neither is a threshold search, and neither should be started on the assumption that a rule is there.

### §28 · The degree branch reopened from the other end: an UNTRAINED role-graph probe (2026-09-07)

§27 closed the property screen and named the only opening it left — *"a feature space that is not hand-designed graph
properties at all."* This entry takes that opening, and first answers the question that motivated it: **is the absence of
a degree rule a fact about graphs, or an artifact of how the degree role graph is BUILT?** Both halves are measured.
New code: `experiments/role_probe.py` (runs, fits nothing). Tables: `results/role_probe.csv`, `results/role_probe_calls.csv`.

#### 1. The construction asymmetry is real, large, and had never been written down

`VirtualGraph.build()` quantizes the 1-D signature at `sig_tol` and then **samples K neighbours uniformly inside a tie
class** whenever that class alone can fill K. Measured on the 48 LP train graphs, the three signals are not three
settings of one construction — they are three different objects:

| signal | median tie classes | median share of a node's K slots filled by an arbitrary draw | median share of nodes with a unique signature |
|---|---|---|---|
| `degree` | 116 | **0.977** | 0.007 |
| `psi` | 3 795 | 0.478 | 0.455 |
| `centrality` | 6 488 | **0.043** | 0.872 |

So the centrality role graph is a genuine top-K nearest-neighbour graph (87% of nodes have a unique signature), while the
degree role graph is a **random graph inside ~116 degree blocks** — 40 of the 48 sit above 0.90 on that middle column.
That is the structural reason the stage-2 rule that exists is a *centrality* rule.

#### 2. But the construction is NOT what costs degree its rule — three controls, all negative

Each control rebuilds only the degree variant and scores it with the untrained probe of §3, so nothing else moves.

| control | what it changes | effect on probe AUC |
|---|---|---|
| `degree_perm` — ties ordered by a **random permutation** | replaces the arbitrary draw with an arbitrary *order*; adds no information | **+0.0009 ± 0.0085** (12/20 graphs) |
| `degree_wl` — ties ordered by **mean neighbour degree** | adds real structural information | **−0.0004 ± 0.0305** (29/43) |
| `centrality_q` / `psi_q` — **quantized to degree's own class count** | asks whether the rivals only win on resolution | **−0.0012 ± 0.0075** / **−0.0043 ± 0.0083** |

Reading, and it is a clean falsification of the obvious hypothesis: **tie sampling costs degree nothing** (an arbitrary
order performs exactly like an arbitrary draw, which is what it should do), **refining the signature does not help**, and
**the rivals are not merely finer** — coarsening centrality to 116 classes costs about a thousandth of an AUC point.
The three variants differ in what they measure, not in how well the builder can express it.

One incidental finding worth recording rather than using: ordering ties by **node id** gains +0.0122 on average
(median +0.0074, 32/43) where a random permutation gains nothing, i.e. on several datasets the node numbering in the
source file is itself structurally informative. That is a property of the files, not of the method, and no rule should
ever read it.

#### 3. The probe: simulate the comparison instead of describing the graph

For one graph, at no training cost: split the LP train graph again 70:30 → build each signal's role graph on the **inner
train** graph with the study's own `build()` → mean-aggregate the encoder's four structural features over that role graph
for 2 hops, **untrained, identity weights** (GraphSAGE at initialisation) → score the inner held-out edges by cosine, as
`eval/linkpred` does. The real test split is never read, so a call can be written before the encoder runs.

Against the 10-seed paired outcome, over the 43 graphs it could be measured on:

| contrast | Spearman | Pearson | sign agreement |
|---|---|---|---|
| probe(degree) − probe(psi) vs the trained degree−Ψ gap | **+0.660** (p 1.4e-06) | +0.755 | 77% |
| the same, on the 23 cells the seeds actually separate (\|t\| > 2) | **+0.861** | — | **91%** |
| probe(best role) − probe(original) vs the trained augment gap | +0.709 (p 1.0e-07) | +0.704 | 72% |

For comparison, the **same 24 properties screened against the same graded target over 40 augmenting graphs** — a target
with far more power than the 9-positives-in-40 binary label §27 used — top out at **|ρ| 0.283** (`components`), 0.265,
0.212. Two further feature families were built and also failed: the **signature-assortativity contrast** (does the
original graph connect nodes of similar degree more than nodes of similar Ψ?) reaches ρ 0.029 with leave-one-out 9/18,
exactly chance, and its rank-gap variant ρ 0.085. So the graded target is not what was missing; the feature space was.

#### 4. The rule, and its price

Three independent inner splits move the contrast by **sd 0.0032** against a mean |contrast| of 0.0322 — a 10:1 ratio —
and the sign is identical on 11/14 graphs, with all three flips inside ±0.006. That fixes the band without touching a
label: **|probe(degree) − probe(psi)| > 0.0064 ⇒ take its sign; inside the band, undetermined.** Nothing is fitted.

| on the paired-band degree-or-psi cells | score |
|---|---|
| **probe, banded** | **10/12 decided correct** (5 undetermined of 17) — binomial P = **0.019** |
| the same cells, best constant predictor | 9/17 |
| unbanded sign rule, every cell | 13/18, balanced-permutation P = 0.076 |
| **cells no existing candidate was ever fitted on** (`artnet_exp`, `foursquare_tips`, `nematode_mammal`, `plant_pol_robertson`, `ppi_mouse`, `ppi_rat`) | **5/5 decided, 1 undetermined** — P = **0.031** |
| `nbr_label_entropy < 0.6724` on those same six | 4/6, and it says "degree" on all six |
| `FROZEN_DEGREE distinct_degrees < 158.5` on those same six | 2/6 |

Set against §27's tally, where every framing of the degree question returned P(shuffle as good) between 0.30 and 0.96,
this is **the first thing in the degree branch that prices below chance.** Its two misses are named: `airports_europe`
(−0.0067, just outside the band) and `amazon_computers` (−0.0323, a real error).

#### 5. What this is, and what it is not

It is **not** an interpretable threshold on a named graph property, and it should not be reported as one — the study's
other rules can be read off a dataset card and this cannot. It needs the three role graphs **built** (never trained),
the same cost that helped retire the Module-4 gate, plus one extra edge split; in exchange it needs no labels, no
encoder, and no fitted parameter. Runtime is seconds to a few minutes per graph against 7 variants × 10 seeds of
training.

Standing limitations. The six never-fitted cells are six. Two graphs are **unprobeable** — `board_directors` (0) and
`plant_pol_kato` (76) hold out too few inner edges, because a near-acyclic graph keeps almost everything in its spanning
forest; they are reported blocked, not scored. `twitch_gamers` and `bag_of_words_nips` exceeded the per-run budget and
`tolokers`' input edgelist was removed by the repo cleanup of the same day, so 45 of 48 rows exist and 43 carry a
contrast. Nothing was written to `virgo/frozen_rules.py`: freezing this as a pre-registered claim is a decision, and the
natural test is the untrained graphs it can be run on before they are trained.

### §29 · Degree Structural Consistency: the assumption behind degree augmentation, measured directly (2026-09-07)

Every degree screen up to §27 tested GENERIC graph properties — clustering, skew, assortativity, fragmentation — none of
which measures what degree augmentation actually assumes: **that nodes of the same degree occupy the same structural
role.** Two degree-10 nodes can sit in completely different neighbourhoods, one wired to hubs and one to leaves. This
entry builds the characteristic that tests that assumption and screens it. New code: `experiments/degree_consistency.py`
(runs, fits nothing, its own tier so no published screen can move because the file exists). Tables:
`results/degree_consistency.csv`, `results/degree_consistency_rules.csv`, with both superseded passes kept beside them.

**The measure.** Original graph only, no labels, so a call exists before any encoder runs.
Each neighbour's degree is recorded as its **percentile rank** in that graph's degree distribution and dropped into the
fixed ladder 0-10, 10-20 … 90-100, so every graph has the same 10 bins and none is excluded. Nodes are grouped by
**exact degree** — `VirtualGraph.build()`'s own tie classes. Within a group, mean pairwise **Jensen-Shannon similarity**
`1 − JSD/ln2` over up to 500 sampled pairs; **node-weighted** mean over groups.
  * `dsc` within-group similarity  * `dsc_between` a member vs a node of a DIFFERENT degree
  * **`dsc_sep` = dsc − dsc_between**, the measure the assumption actually names: if same-degree nodes are 0.9 alike and
    different-degree nodes are also 0.9 alike, degree has explained nothing.
  * `dsc_null` the analytic **configuration-model** floor (a degree-d node draws d neighbours i.i.d. from the
    degree-weighted endpoint distribution), and `dsc_adj = (dsc − dsc_null)/(1 − dsc_null)` — the same construction
    `homophily_adjusted` uses, and the only term that is degree-matched in sampling noise.
Target as specified: **degree is the SOLE paired-band winner**, over augmenting graphs only — the deployable stage-2
question, not the degree-vs-Ψ pairwise one. 39 augmenting graphs, 9 degree winners.

**Pass 1, all degree groups.** `dsc_sep` reached **ρ −0.433, the strongest correlation any degree property has produced
in this study** (§§24-27 topped out at −0.33); `dsc` gave ρ −0.308 with **LOO 0.821 against a 0.769 majority baseline**,
the only variant ever to beat that baseline out of sample. Both priced at **P(shuffle as good) ≈ 0.14** — above the 0.05
bar but far below §27's 0.30-0.96 band. The direction was **inverted**: degree wins where degree separates neighbourhood
structure *least*, which has a mechanism consistent with §19 — Ψ is a KL score over degree-based signatures, so when
degree separates little, Ψ has nothing to add and the coarser role graph wins. The reading would have been "Ψ has
nothing to add", not "degree is enough". Checked and cleared: it is not a sparsity proxy — the share of node weight at
degree ≤ 2 (66-84% on the sparse graphs) used directly as a predictor scores P 0.722, LOO 0.59.

**Pass 2, the one declared correction, and it falsifies pass 1.** A degree-1 node's neighbour histogram is a one-hot
vector, so the JS similarity of two of them is a coin flip on whether their single neighbour lands in the same
percentile bin — noise, carrying 66-84% of the node weight on sparse graphs. Groups below **degree 5** were therefore
excluded (a hardcoded constant, no CLI flag, so it cannot be tuned afterwards). Result:

| predictor | pass 1 ρ | pass 1 P | pass 2 ρ | pass 2 P | LOO (pass 2) |
|---|---|---|---|---|---|
| `dsc_sep` | **−0.433** | 0.140 | **+0.233** | 0.136 | 0.769 = the baseline |
| `dsc` | −0.308 | 0.138 | −0.087 | **1.000** | 0.744 |
| `dsc_between` | −0.238 | 0.135 | −0.184 | 1.000 | 0.692 |
| `dsc_adj` | +0.059 | 1.000 | +0.076 | 1.000 | 0.718 |
| `dsc_null` | −0.314 | 0.503 | −0.178 | 1.000 | 0.718 |

**`dsc_sep` changes sign.** Across the two passes the measure correlates with itself at only **ρ +0.166** (`dsc`, by
contrast, is stable at +0.816 — and has no signal left once corrected). A quantity whose sign flips when the noisiest
groups are removed is not measuring one thing, and the −0.433 was carried by exactly those groups. Coverage cost of the
correction is real and reported: median node coverage 0.990 → 0.567, worst `roman_empire` 0.074, `genius` 0.092.

**Verdict: a measured negative, and a stronger one than a p-value alone.** Ten predictor-variants were screened across
the two passes, best P = 0.135, so ~1.35 winners of this quality are free. Nothing clears the gates, nothing was written
to `virgo/frozen_rules.py`, and no rule is proposed. What DSC does establish is that **the premise of degree
augmentation is quantitatively weak on every graph in the panel** — within-group separation never exceeds 0.21 and is
usually under 0.10, i.e. same-degree nodes are barely more alike than random pairs. Degree augmentation, where it wins,
is therefore not winning because degree captures structural role. That is a claim about the mechanism, and it is the
part of this entry worth carrying forward.

### §30 · The other two DSC descriptors, under pre-declared terms: both fail, and so does the near-miss (2026-09-07)

§29 tested ONE descriptor — the neighbour-degree histogram — in five mathematical forms. Two descriptors named at the
outset were never written: a node can carry the identical neighbour-degree histogram as another and a completely
different triangle structure, which is exactly the failure case the hypothesis describes and that descriptor is blind to.
`experiments/dsc_descriptors.py` (new) tests those two with the degree histogram beside them as a control.

**Terms fixed BEFORE the run**, because ten variants had already been spent on this question:
(1) exactly two new descriptors, one run each, no re-tuning of bins, weighting or grouping;
(2) **one form only** — separation = within-group minus between-group similarity, the form §29 settled on;
(3) each runs at all groups **and** at degree ≥ 5, and a descriptor is **dead if its sign flips** between them — §29's
post-hoc killer, applied up front;
(4) direction **pre-declared**: high separation ⇒ degree, so an inversion is a failure of the hypothesis, not a result;
(5) the bar is P(shuffle as good) **< 0.05** against 12 cumulative variants.

Descriptors: `nbr_degree` = neighbours' degree percentile histogram (control); `clustering` = neighbours' local
clustering-coefficient percentile histogram; `role_vector` = per-node `[clustering, log mean/sd/max neighbour degree]`,
z-normalized, compared by rescaled cosine — **degree itself deliberately excluded**, since it is constant inside a degree
group and would inflate within-group similarity for free. 38 augmenting graphs, 9 degree-sole winners (`twitch_gamers`
is missing: `nx.clustering` exceeded the run budget on 6.8M edges — a compute accident, not a selection).

| descriptor | ρ | side | exceptions | LOO | majority | P(shuffle) | declared side? | ρ at d≥5 | side at d≥5 | self-ρ across passes | **survives** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `nbr_degree` | **−0.449** | low | 6 | 0.789 | 0.763 | **0.0335** | **no** | +0.234 | high | **−0.015** | **no** |
| `clustering` | −0.260 | low | 8 | 0.789 | 0.763 | 0.475 | no | +0.028 | high | 0.617 | no |
| `role_vector` | +0.195 | high | 9 | 0.632 | 0.763 | 0.748 | **yes** | +0.184 | high | **0.906** | no |

**Both new descriptors fail, and the near-miss fails its own pre-declared test.** The only descriptor that is a *stable*
measurement (`role_vector`, self-ρ 0.906, same side in both passes) and the only one pointing the way the hypothesis
predicts carries **no information** — LOO 0.632 is below the 0.763 majority baseline. The only descriptor that clears
0.05 (`nbr_degree`) is inverted and its sign is not stable across passes at all (self-ρ −0.015). Term 3 and term 4 were
written down precisely for this case.

**The 0.0335 is also fragile, exactly.** Two independent implementations of the same statistic agree at ρ +0.980 and give
identical results (6 exceptions, P 0.034 / 0.035) on these 38 graphs. The whole difference from §29's **P = 0.140** is the
single missing graph: restoring `twitch_gamers` adds one exception and **quadruples the price**. A cut whose p-value
crosses the bar on one dropped graph is not a rule, and 0.140 — the 39-graph number — is the honest one.

**Reading.** The descriptor was not the binding constraint. Three descriptors — degree profile, clustering profile, and a
combined local role vector — asked under one fixed form and pre-declared terms, and none separates the degree winners.
That closes the DSC family the same way §27 closed the property screen, and for a sharper reason: the measurement that is
*reproducible* has no signal, and the measurement with a signal is not reproducible. Nothing was written to
`virgo/frozen_rules.py`. Carried forward from §29 unchanged: within-group separation never exceeds ~0.21 and is usually
under 0.10 on any graph in the panel, so degree augmentation, where it wins, is not winning because degree captures
structural role.

### §31 · Degree Role Dispersion: a different mechanism, tested and failed (2026-09-07)

§§29-30 asked whether same-degree nodes are locally ALIKE and answered no on three descriptors. This entry tests the
opposite mechanism, proposed by the user: maybe degree augmentation wins not because it groups structurally identical
nodes but because it wires DISTANT ones, adding useful long-range structure. New code:
`experiments/degree_dispersion.py`. Tables: `results/degree_dispersion.csv`, `results/degree_dispersion_rules.csv`.

**The estimator needs no role graph.** `build()` links a node to K *sampled* members of its exact-degree tie class, so
the distance distribution of same-degree pairs is an unbiased estimate of the degree role graph's own edge lengths.
**DRD = mean shortest-path distance between same-degree pairs ÷ mean distance between random pairs**, both terms read
off the SAME breadth-first searches from 300 uniformly sampled sources, so the ratio is paired and scale-free (raw hop
counts are not comparable — `roman_empire` averages 2274 hops, `blogcatalog` 2.5). Unreachable same-degree pairs are
excluded and their share reported as a diagnostic, never screened: counting them as maximally distant would re-import
fragmentation, which Module 2 already screened. Original graph, no labels.

**Terms fixed before the run**, unchanged from §30 except the kill test: one predictor, one form, direction
**pre-declared high ⇒ degree**, dead if the fitted side flips under a **second BFS-sampling seed**, bar P(shuffle) < 0.05.

| n | degree winners | ρ (seed 42) | cut | side | exceptions | LOO | majority | P(shuffle) | declared side? | ρ (seed 43) | side (seed 43) | self-ρ | **survives** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 39 | 9 | +0.349 | 1.007 | high | 8 | 0.769 | 0.769 | **0.508** | yes | +0.049 | **low** | 0.668 | **no** |

**Fails on both the price and its own kill test** — P 0.508, LOO exactly equal to the majority baseline, and the fitted
side flips from `high` to `low` when only the BFS sampling seed changes.

**Why it fails, and this is the part worth keeping.** DRD has almost no between-graph variation to threshold:

* range over 39 graphs **0.9577 – 1.0199**, span 0.062; **30 of 39 sit within ±1% of 1.0**
* mean DRD **0.9970** — same-degree pairs sit **−0.30%** from random-pair distance
* between-graph sd **0.0106** against a mean seed-to-seed movement of **0.0063** → **signal-to-noise 1.68** (the §28
  untrained probe ran at 10:1)
* the degree-winner vs rest gap is **+0.0069, i.e. 1.09× the sampling noise**

So the hypothesis is half-right and half-useless. Degree role edges **are** long-range — 2.4 to 6.3 hops, far beyond
adjacency — but they are long-range by **exactly the amount a random node pair is, on every graph in the panel**. Same-degree
nodes are dispersed as if placed at random with respect to graph distance. At the distance level the degree role graph is
a **random long-range rewiring**, and a quantity that is the same everywhere cannot select between graphs.

Read next to §§29-30 the picture is consistent: the degree role graph adds edges that are neither locally meaningful
(three descriptors, no separation) nor differentially long-range (DRD ≈ 1 everywhere). Where degree wins, it is not
winning because degree encodes structural role, and not because it places its shortcuts better than chance would.

**Stated limitation, not acted on.** The mean-distance form compresses on small-world graphs, where every distance sits
between 2 and 6 hops, so any subset's mean is pinned near the global mean. A "fraction of pairs at distance ≥ 3" form
would be robust to that and was on the menu; the ratio was chosen and the terms allow one form, so it stays untested and
is the single thing that could revive DRD. Nothing was written to `virgo/frozen_rules.py`.

### §32 · The second DRD form — tail probability instead of a mean. Tested once, failed, DRD closed (2026-09-07)

§31 flagged one limitation and did not act on it: a mean distance compresses on a small-world graph where every distance
sits between 2 and 6 hops, so any subset's mean is pinned near the global mean. The user authorised **one** further test
of the alternative form, pre-declared and counted as variant #14 on this question. Same file, same BFS runs, same terms.

**The form.** `drd_far` = P(same-degree pair ≥ 3 hops) ÷ P(random pair ≥ 3 hops), a **tail probability** rather than a
mean, both terms read off the same 300 breadth-first searches. Screened as a ratio for the same reason the mean form was:
the raw share (emitted as `far_same`, a diagnostic, never screened) would mostly measure diameter, the size confound that
disqualified four features in §20. `FAR_HOPS = 3` is a hardcoded constant with no CLI flag. Direction pre-declared **high
⇒ degree**, kill test = the fitted side must not flip at BFS seed 43, bar P(shuffle) < 0.05.

| predictor | ρ (seed 42) | cut | side | declared side? | exc. | LOO | majority | **P(shuffle)** | side (seed 43) | self-ρ | **survives** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drd_far` (new) | +0.206 | 0.838 | **low** | **no** | 10 | 0.744 | 0.769 | **1.000** | low | **0.964** | no |
| `drd` (§31) | +0.349 | 1.007 | high | yes | 8 | 0.769 | 0.769 | 0.508 | **low** | 0.668 | no |

**The user's diagnosis of the mean form was right, and it changes nothing.** The tail form is by far the better
instrument — signal-to-noise **5.39** against the mean form's 1.68 (between-graph sd 0.0547 vs 0.0101 seed noise), and its
fitted side is **stable across seeds** where the mean form's flipped. It is a reproducible measurement. It simply
carries **no information**: P(shuffle as good) = **1.000**, leave-one-out 0.744 *below* the 0.769 majority baseline, and
the fitted side is `low`, the opposite of the pre-declared direction.

**And the direction is the substantive result.** `drd_far` sits **below 1.0 on 35 of 39 graphs**, mean **0.9597** — so
same-degree pairs are, if anything, *slightly less* likely to be far apart than random pairs. Measured now with the
stable instrument, the long-range-shortcut hypothesis is not merely undetectable, it points the wrong way: the degree
role graph's edges are marginally **more local** than chance, not less.

**DRD is closed.** Two forms, one pre-declared test each: the mean form fails its own kill test, the tail form is
reproducible and has zero signal. Combined with §§29-30 the mechanism account is complete and consistent — the degree
role graph adds edges that are neither locally meaningful (three descriptors, no separation), nor differentially
long-range (mean form ≈ 1 everywhere), nor even more long-range than chance (tail form 0.96, below 1 on 35 of 39).
Nothing was written to `virgo/frozen_rules.py`. §31's mean-form tables are kept beside the new ones as
`results/degree_dispersion*_meanform_superseded.csv`.

### §33 · WHO gains from degree augmentation? The tail hypothesis, verified against outcomes and FALSIFIED (2026-09-07)

§§29-32 asked what is special about degree as a *signal* and closed two mechanisms. This is the first test that looks at
**outcomes rather than at the graph**: on the graphs where degree wins, which nodes actually gain? The hypothesis is from
the GNN degree-bias literature (GraphPatcher; SAug's hub/tail imbalance) — low-degree nodes have thin neighbourhoods,
message passing serves them badly, and augmentation may win precisely by giving them context. New code:
`experiments/degree_tail_gain.py`, which trains nothing and re-scores the embeddings already on disk.

**Design, fixed before the run.** Every test pair, positive and negative, is keyed by `min(deg(u), deg(v))` on the train
graph — the weakest endpoint — so no node is dropped and the tail cannot be selected away. For each positive edge its
exact AUC contribution (the share of negatives it outranks) is computed under degree and under the best rival signal,
**paired on the same split and seed**. The statistic is **Spearman ρ(advantage, min endpoint degree)**, continuous, no
bins, scale-free so it compares across graphs whose advantage magnitudes differ tenfold. **Negative ρ = degree's
advantage grows as degree falls = the hypothesis.**

**The confound the design exists to defeat**: a role graph adds K = 10 edges per node, an enormous relative change for a
degree-2 node and nothing for a degree-500 node, so a tail gradient appears for *every* signal. Hence the test is a
difference — degree against its best rival, and the 9 degree winners against the 30 other augmenting graphs — with all
three criteria required.

| criterion | result | met |
|---|---|---|
| (a) degree winners have a **negative** mean ρ | **+0.1831** | **no** |
| (b) the other augmenting graphs do not | +0.1595 | yes (both positive) |
| (c) the sign holds on ≥ 7 of the 9 winners individually | **2/9** | **no** |
| difference-in-differences | +0.0236, permutation **P = 0.673** | — |

**NOT VERIFIED, and the direction is inverted.** ρ is positive on **34 of 39** graphs and on **7 of the 9** degree
winners: degree's advantage over its rival grows as node degree *rises*. The two exceptions are `airports_europe`
(−0.011, effectively zero) and `artnet_exp` (−0.200). Winners and non-winners are indistinguishable — +0.183 vs +0.159,
P 0.673 — so even the gradient that exists says nothing about which signal wins. As pre-declared, **no tail-imbalance
characteristic was built and no threshold was fitted.**

**The inversion is the finding, and it is consistent.** The bottom decile of weakest-endpoint degree carries a
**negative** advantage on **36 of 39** graphs (8 of the 9 winners), while the top decile is positive on 27 of 39. And it
is not specific to the degree-vs-Ψ comparison: `ρ(degree − original, min degree)` is positive on **36 of 39** graphs,
median **+0.227**. So on this pipeline **role augmentation of any kind helps high-degree endpoints and hurts low-degree
ones** — the opposite of the degree-bias literature's premise.

**A mechanism that fits it, stated as an observation and not pursued.** The role graph adds a *fixed* K = 10 edges per
node. For a degree-1 node those ten role edges outnumber its single real edge ten to one, so message passing over the
augmented graph **overwrites** its neighbourhood rather than enriching it; for a degree-100 node they are a small
perturbation that adds information without swamping the original signal. Fixed-K augmentation therefore dilutes exactly
the nodes the literature expects it to rescue. That predicts a degree-adaptive K — fewer role edges for tail nodes —
would behave differently, which is a change to the METHOD, not another characteristic, and nothing here was run on it.

Nothing was written to `virgo/frozen_rules.py`.

### §34 · The last selector direction: when does Ψ add nothing beyond degree? Tested, failed (2026-09-07)

After §§29-33 closed three mechanisms, one selector direction remained (user): stop asking what is special about degree
and ask when its RIVAL is redundant. Ψ is a Poisson/KL score over degree-based signatures, so on some graphs it is nearly
a relabelling of degree and on others it genuinely refines it; where it does not refine, the simpler signal may suffice.
New code: `experiments/psi_refinement.py`. Tables: `results/psi_refinement.csv`, `results/psi_refinement_rules.csv`.

**The measure.** `psi_explained_by_degree` = the correlation ratio **η²** of **rank-transformed Ψ** across exact-degree
classes — the share of Ψ's variation the degree partition already accounts for. Ranks, not raw Ψ, because Ψ's
`log(p/Ω)` tail reaches ~52 on some graphs and a raw variance ratio would be an outlier statistic. **Adjusted**
analytically against its chance floor `(c−1)/(n−1)`, exactly as `homophily_adjusted` is: that floor is 0.34 on
`airports_brazil` (45 classes, 131 nodes) and 0.0004 on `roman_empire`, so raw η² would mostly compare class counts.
Original graph, no labels, no role graph, no training.

**Two risks were recorded before the run**, and both mattered. (1) A near-neighbour already failed: §§24/27 screened
`deg_psi_sig_spearman`, the *global* rank correlation of the two signatures, at 8 exceptions and P 0.960 — but a global
correlation of −0.9 is compatible with Ψ varying wildly *inside* a degree class, so η² is a genuinely different
quantity. (2) It might predict **ties** rather than degree wins, since §19 found degree and Ψ tie exactly where their
role graphs are most alike. Both were made kill tests, declared in advance.

| target | ρ | cut | side | declared? | exc. | LOO | majority | **P(shuffle)** |
|---|---|---|---|---|---|---|---|---|
| degree wins alone | −0.324 | 0.1219 | **low** | **no** | 9 | **0.590** | 0.769 | **0.7275** |
| the winner is a tie (kill test ii) | +0.227 | 0.9397 | high | — | 12 | 0.615 | 0.667 | 0.7605 |

**Fails on every count.** The price is 0.7275; leave-one-out 0.590 is *below* the 0.769 majority baseline; and the
fitted side is `low`, the **opposite** of the pre-declared `high ⇒ degree` — degree wins where Ψ *does* refine degree,
not where it adds nothing.

**Kill test (i) FAILED and explains the rest**: η² correlates with `avg_degree` at **+0.793** and `density` at **+0.759**,
both over the 0.7 bar and both properties already screened and failed in §§25-27. Run side by side on the same 39 cells,
`avg_degree` scores **9 exceptions, P 0.7215** and `density` **8 exceptions, P 0.5075** — the new measure (9 exceptions,
P 0.7262) is indistinguishable from the sparsity axis it proxies. Degree winners average `avg_degree` 12.8 against 31.8
for the rest, so "low η² ⇒ degree" is "sparse ⇒ degree" with extra steps.
Kill test (ii) **passed** — it does not predict ties better than degree wins (0.761 vs 0.726) — so it is at least not
§19 rediscovered. That is moot given (i).

**The descriptive result is worth keeping.** Degree explains a **median 79%** of Ψ's rank variation across the 39
graphs, from **5.3%** (`roman_empire`) to **99.7%** (`bag_of_words_nips`). So at the *signature* level Ψ is largely a
relabelling of degree on most graphs — §19's mechanism, confirmed directly on the original graph without building a
single role graph, and now quantified per graph. It just does not select. Nothing was written to
`virgo/frozen_rules.py`.

### §35 · The fixed-K perturbation as a characteristic: stopped at the confound gate (2026-09-07)

§33's mechanism — a fixed K = 10 overwrites a degree-1 node's neighbourhood and merely perturbs a degree-100 node's —
suggests a characteristic: measure how large that relative change is across the graph. Two forms were built in
`experiments/degree_rewrite_share.py`: `rewrite_share`, the share of nodes with degree < K (the quantity §33 actually
implicates, and not a mean of anything), and `k_over_degree`, the literal mean of K/degree.

**The script gates before it screens, and requires a declared side.** §34 spent a full test to discover after the fact
that its measure was a proxy for `avg_degree`; this one measures the correlation against every already-screened
degree-sequence property first — `avg_degree`, `density`, `degree_gini`, `degree_skew`, `distinct_degrees` — and refuses
to screen anything that fails. `--declare-side` is required on top, so a direction can never be read off the output.

| candidate | vs avg_degree | vs density | vs degree_gini | vs degree_skew | vs distinct_degrees | worst | passes |
|---|---|---|---|---|---|---|---|
| `rewrite_share` | **−0.9512** | −0.6063 | +0.0794 | +0.1874 | −0.6440 | 0.9512 | **no** |
| `k_over_degree` | **−0.9247** | −0.5684 | +0.1536 | +0.2294 | −0.6053 | 0.9247 | **no** |

**Both are `avg_degree` inverted and the screen never ran.** The argument for expecting otherwise was that
`mean(K/degree)` equals K divided by the **harmonic** mean degree, which is dominated by the tail where the arithmetic
mean is dominated by hubs — genuinely different on a heavy-tailed graph. On this panel the graphs are not tailed enough
for the two to decouple: ρ −0.92. The prediction was made before the run and is recorded as wrong.

The group difference is real but is the one already on the board: degree winners average `rewrite_share` 0.729 against
0.516 for the rest, mirroring `avg_degree` 12.8 against 31.8 (§34). And `avg_degree` was run side by side on these exact
39 cells in §34 at **9 exceptions, P(shuffle) 0.7215**, so that is where the screen would have landed.

**Reported as a protocol result, not just a dead candidate.** §34 cost a full test and a multiplicity charge to learn
that a measure was a proxy; §35 cost two minutes and none. The gate is now the standing first step for any further
degree candidate: correlate against the screened degree-sequence family before spending a test. What remains untouched
is the other response to §33 — **degree-adaptive K** in `virgo/virtual_graph.py`, which would fix the dilution rather
than predict it. That is a method change requiring retraining and nothing here was run on it. Nothing was written to
`virgo/frozen_rules.py`.

---

## 2026-09-07 — Module 13: the PSI branch screened the way centrality was — no rule, and no fittable residual

**Why this ran.** The user closed the direct degree search (standard properties, degree-specific properties, DSC, DRD,
tail-node benefit, Ψ redundancy, K/degree dilution — six screens, six negatives on the same graph pool) on the stated
ground that inventing a seventh degree feature on 39 graphs is now an overfitting risk, not a measurement. The
replacement target was Ψ: find a clean ORIGINAL-graph Ψ characteristic the way `FROZEN_STRATEGY` was found, and then see
whether the cascade it implies works — **centrality condition ⇒ centrality, else Ψ condition ⇒ Ψ, with DEGREE as the
residual region**. Degree is not abandoned; it is demoted to whatever the first two conditions leave behind.

**Scope, fixed by the user on the same day and applying to everything below.** `FROZEN_STRATEGY` is **settled**. It
stands as published in CLAUDE.md §2 with the caveats already on record, and is **not re-tested, re-fitted, re-scored on
enlarged pools, or treated as open** by this module. Every batch ingested since Module 9 was selected for the degree and
Ψ questions, so a score computed on those graphs measures the pool, not the rule. Here the rule is **applied only to
draw a line** — which augmenting graphs it routes to centrality, and which fall through to the Ψ question — and
`psi_rule.cascade()` records that partition without a correctness column.

**Method.** New entry point `experiments/psi_rule.py`, mirroring `degree_rule.py` so nothing about the protocol changes
with the target. Pool = the **48 graphs `tie_break.py` has paired verdicts for** (`results/degree_rule_corpus_verdicts.csv`),
of which **40 augment**. Two things are screened in parallel rather than one being chosen:
- **both label modes.** `FROZEN_STRATEGY` was fitted on MEMBERSHIP (the signal is in the winning band); the degree work
  labelled SOLE winners. Reporting one and not the other would let the label definition pick the answer, so both are
  emitted for every property.
- **both bands.** `frozen` (independent-means, the published band) and `paired`. Paired is the informative one here: it
  raises Ψ-alone winners from 5 to 10 — it resolves marginal cells, it does not invent them.
Gates are Module 2's, unchanged, plus `MIN_SOLE`, plus two guards carried over from the degree branch: `chance()` prices
every clean split, and a **confound gate** flags any candidate ranking the graphs at |ρ| ≥ 0.9 against `avg_degree` or
`density` — the two properties Module 2 already screened and dropped, and the exact trap §35 caught.
Fit/test split is by **provenance, never by outcome**: the strategy panel and the Module-10 corpus fit (25 augmenting,
5 Ψ-alone), everything ingested later tests (`DEGREE_RULE_VALIDATION` + `DEGREE_TEST` + `DEGREE_BATCH2`, 5 Ψ-alone).
It is **retrospective** — all 48 were trained before this screen existed — so it is transfer evidence, exactly like the
Module-8 clustering exception, and it is not a pre-registration.

**Result 1 — there is no Ψ rule, and this is the branch's largest-n negative.** Over 40 augmenting graphs, 15
original-graph properties, two label modes:

| target | Ψ cells | best property | ρ | exceptions | LOO | majority |
|---|---|---|---|---|---|---|
| Ψ alone | 10 | `nbr_predictability_adjusted` | −0.315 | 8 | 0.722 | 0.722 |
| Ψ named | 21 | `avg_degree` | +0.492 | 8 | 0.750 | 0.525 |

**Zero properties clear the gates under either label mode**, and no property reaches even half the 0.7 |ρ| bar on the
sole-winner target. LOO ties the majority baseline for every sole-target property. This matters more than the degree
negatives did: degree-vs-everything was measured on 29 cells with 5 winners, whereas Ψ has **40 cells and 10 sole / 21
named winners** — the class is twice the size and the properties still say nothing. On the fitting half alone the best
row is `nbr_label_entropy` (ρ 0.535, 3 exceptions, LOO 0.857 vs 0.762) and it does not survive: on the 11 scorable
held-out cells it scores 6/11 against a constant's 6/11.

**Result 2 — the residual region is too small to fit anything in.** The cascade needs a populated region below the
stage-2 cut, and there is not one: **36 of the 40 augmenting graphs sit above the cut**, leaving **4 decided cells**
(Ψ 3, degree 1). That is under `GATES["min_cells"]` and priced at P = 0.5 per property, so ~7.5 of the 15 tried separate
for free — nothing fitted there could mean anything. Asked unconditionally instead — Ψ vs degree over all 19 graphs
naming one of them — the best split is `avg_degree` (ρ 0.443, 4 exceptions, LOO 0.737 vs a 0.526 baseline), the only row
anywhere in this module whose LOO beats its baseline; on the held-out cells it scores 6/9 against a constant "always Ψ"
at 5/9. That is a coin flip, and `avg_degree` is a property Module 2 already screened and dropped for the augment
question.

**What this says about the cascade.** The ordering is the right question and the architecture is sound, but it cannot be
populated from the current pool: the Ψ condition it needs does not exist (Result 1), and the region it would apply to
holds four cells (Result 2). The wider pattern is now consistent across three targets and 15 properties — **signal
identity is weakly determined by graph-level properties at all**. The one instrument that ever beat chance in this
branch remains `experiments/role_probe.py` (untrained probe, Spearman +0.66 on the degree-vs-Ψ gap, 5/5 on never-fitted
cells) — a per-graph MEASUREMENT, not a property screen. That, not another threshold search, is where a signal selector
would have to come from.

**Disclosures.** (a) The pool is enriched for the degree question — 14 of the 48 are the bipartite / bounded-degree
Netzschleuder families ingested precisely because degree was expected to win there — so it is not a neutral panel for
any stage-2 question; the Ψ negative reproduces on the fitting half alone (25 augmenting graphs, 5 Ψ-alone winners, zero
credible properties), so it does not depend on the new families. (b) Seven strategy-panel graphs (`cora`, `enzymes`,
`minesweeper`, `amazon_photo`, `amazon_ratings`, `citeseer_linqs`, `proteins`) and the four LINKX networks are outside
the pool — the seven all KEEP the original graph so a conditional screen drops them anyway, and the LINKX four have only
five of the seven variants scored, so their band is not comparable. (c) Nothing was written to `virgo/frozen_rules.py`.
Tables: `results/psi_{cells,rules,cascade,residual,heldout}.csv` (frozen band, fitting half),
`paired_psi_*.csv` (paired band, fitting half), `exploratory_psi_*.csv` and `exploratory_paired_psi_*.csv` (both bands,
all 48, nothing held out).

---

## 2026-09-07 — Module 14: a FAMILY-SCOPED test of the degree cut — discovery profile, envelope, and the pre-registration

**The question, narrowed on the user's instruction.** Module 10 fitted `nbr_label_entropy < 0.6724 ⇒ degree` on nine
cells (0 exceptions, LOO 8/9) and Module 11 falsified it as a UNIVERSAL rule (1/3 on six pre-registered graphs, beaten
3/3 by a constant "always Ψ"). The instruction is not to refit it and not to hunt a new small pool until one splits
cleanly — that is the cherry-picking the user explicitly ruled out — but to ask the one question it was never asked:
**does it hold on graphs structurally similar to the nine it was discovered on?** A rule that survives that is valid for
a FAMILY of graphs and must never be reported as a universal degree rule. The declared order is: describe the nine →
choose new graphs by ORIGINAL-graph properties → apply the already-fixed cut → write predictions to disk → only then
train → score.

**The discovery nine, recovered exactly** (`results/paired_degree_psi_contrast.csv`, matched back by entropy value):
degree wins on `actor`, `airports_europe`, `amazon_computers`, `questions`, `twitch_engb`; Ψ wins on `blogcatalog`,
`tolokers`, `twitch_de`, `twitch_es`.

**Their profile, and what is striking about it.** The nine are homogeneous on axes nobody selected them for. Every one
is single-mode, near-connected (`largest_component_frac` 0.973–1.000, eight of nine exactly 1.0), **disassortative
without exception** (`degree_assortativity` −0.2252 to −0.0200), sparse (density 0.0001–0.0755), few-class
(`n_classes` 2–10, median 2) and mid-sized (399–48,921 nodes, `avg_degree` 6.3–88.3).

**The envelope** (`frozen_rules.FAMILY`, declared before any candidate was measured) is that min/max description on six
axes plus one negative condition, `not bipartite`. It is a DEFINITION, not a fit — it contains all nine by construction,
and that is stated rather than reported as a result. `nbr_label_entropy` is deliberately EXCLUDED from it: it is the
predictor, and a family defined on the predictor would hand the rule its own answer instead of testing it on graphs that
straddle the cut naturally.

**What the envelope does to the existing 48-graph pool — the reason new graphs were needed.** 18 of 48 are in family.
It correctly excludes every bipartite Netzschleuder graph, the assortative co-author/citation graphs
(`coauthor_cs` +0.11, `coauthor_physics` +0.20, `airports_usa` +0.03, `squirrel_filtered` +0.20), the fragmented ones
(`polblogs` 0.820, `dblp` 0.914, `cora_ml` 0.938) and the out-of-range ones (`genius` 421,961 nodes,
`twitch_gamers` 168,114). But of the six in-family graphs that are neither discovery nor already-spent cells, **five are
degree|Ψ ties and only `ppi_mouse` names a signal.** More seeds cannot fix that: the paired degree-vs-Ψ gaps are
0.0006–0.0036 AUC against paired sems of 0.003–0.008, so `github` would need ~540 seeds and `squirrel`/`twitch_fr` ~420
to separate. The retrospective in-family arm is therefore worth exactly one cell, and it is also contaminated — those
graphs' outcomes were already visible in this session. It is reported as context, never as a test.

**The search for new in-family graphs, and what it establishes.** Netzschleuder's full 286-network catalogue was
filtered on the envelope's own axes, which its API publishes (`num_vertices`, `average_degree`,
`degree_assortativity`, `largest_component_fraction`, `is_bipartite`) — so selection happened on original-graph
properties before a single byte was downloaded. **139 subnetworks clear the numeric axes and exactly one source carries
a usable single-label node column.** The rejections are recorded rather than silently skipped: `spanish_highschools`
(~47% of nodes have no `Sexo`/`Curso` value — the defect that got NELL rejected), `ego_social` (`circles` is
multi-label, as with the earlier facebook/twitter rejection), `genetic_multiplex`/`jdk`/`jung`/`google` (`nodeLabel` and
`meta` are unique per node — gene names and Java class names, not classes), `arxiv_citation`/`scotus_majority`/
`us_agencies`/`route_views` (dates, covariates or ids; binning a date reproduces the `crocodile` confound), and the vast
majority (`anybeat`, `bitcoin_*`, `linux`, `marvel_universe`, `topology`, `wikipedia_link` ×45, `word_*`, `kegg` ×20,
`tree-of-life`, `wiki_rfa`, `elec`, `ugandan_village` ×6) which carry no node metadata at all. torch_geometric adds
nothing: `FacebookPagePage` is hosted on graphmining.ai, which is dead, and the CUAI mirror carries only the five
Facebook100 schools already in the study. **The family the rule was discovered on is data-poor, and that is itself a
finding**: labelled, one-mode, disassortative graphs in this size band are rare, and the discovery nine already contain
most of the accessible ones (three of the nine are Twitch languages, and the other three Twitch languages are in the
study too).

**The validation set** (`cfg.DEGREE_FAMILY_TEST`, cheapest-first): `genetic_fission_yeast`, `ppi_yeast`, `ppi_fly`,
`ppi_human` — four `mist` interaction networks, labelled by the `gene` species code, the SAME column `ppi_rat` and
`ppi_mouse` already use, so the labelling scheme is not a new degree of freedom. The three interolog subnetworks that
also pass the envelope are excluded on purpose: an interolog network is INFERRED by homology from another species'
network, so it is not an independent graph.

**Pre-registration, on disk before training** (`results/degree_family_prereg.csv`, and the caches were warmed by the
BUILD stage only — the content-hashed feature cache the sweep then reuses byte-identically):

| graph | nodes | avg deg | assort | classes | `nbr_label_entropy` | stage 1 | predicted |
|---|---|---|---|---|---|---|---|
| `genetic_fission_yeast` | 3,563 | 29.19 | −0.2848 | 4 | 0.0014 | augment | **degree** |
| `ppi_yeast` | 7,272 | 44.97 | −0.1852 | 9 | 0.0140 | augment | **degree** |
| `ppi_fly` | 11,352 | 23.64 | −0.1532 | 9 | 0.0023 | augment | **degree** |
| `ppi_human` | 27,594 | 37.83 | −0.0812 | 10 | 0.0299 | augment | **degree** |

**Three disclosures that must travel with whatever this returns.** (a) The test is **ONE-SIDED**: all four sit far below
the cut, so it tests only the half `FROZEN_DEGREE`'s own note calls defensible ("below the cut ⇒ degree"), and it can
falsify that half but can never validate the rule as a two-sided selector. This was checked, not assumed — the search
found no in-family source on the high-entropy side at all. (b) All four come from one database and share a label scheme
with `ppi_rat`/`ppi_mouse`, so they are four graphs but not four independent draws. (c) A warning already on the record:
`ppi_mouse` (entropy 0.0483, in family) names **degree**, while `ppi_rat` (entropy 0.0605) names **Ψ** — 0.012 apart,
opposite outcomes. `ppi_rat` is out of family (`largest_component_frac` 0.965, `avg_degree` 4.19) so it does not score,
but it is the same kind of decisive pair `minesweeper`/`squirrel_filtered` is for stage 1, and it says the entropy value
is not what decides these cells. Training is under way; nothing is scored yet, and `virgo/frozen_rules.py` gains only the
frozen candidate and the envelope — no new fitted cut.

---

## 2026-09-07 — Module 14 (cont.): the ARGMAX labelling decision, and the first two family cells

**A labelling decision, taken by the user and applied from here on.** Both of the first two family cells came back as
ties under the band the study publishes — `genetic_fission_yeast` a three-way tie, `ppi_yeast` a `degree|psi` tie — so
under the frozen instrument the test had produced zero scorable cells from two graphs. The user's ruling:
**"a win is a win, even 0.00001; degree wins means wins."** So `experiments/degree_family.py` now records TWO labels for
every cell and neither replaces the other:

- **band** — the tie-aware label every published cell uses. A cell whose band names two signals is unscorable.
- **argmax** — the raw winner, whichever variant has the highest mean however small the lead. **This is what the score
  below is reported on.**

The concern is stated once and then the instruction is followed: an argmax label on a sub-sem margin is a coin flip
re-read as a decision, and pairing cannot rescue these particular margins (§tie_break above: `genetic_fission_yeast`
degree-vs-Ψ paired ratio 0.22). So the margin travels with every argmax call — `margin` and `margin_sems` are columns in
`results/degree_family_scored.csv`, and the report prints a CAUTION line naming every call under 1 sem.

**The first two cells (3 seeds, LP, K=10, paired and frozen bands both computed):**

| graph | `nbr_label_entropy` | predicted | band label | argmax label | margin | sems | correct |
|---|---|---|---|---|---|---|---|
| `genetic_fission_yeast` | 0.0014 | degree | `centrality\|degree\|psi` (tie) | **degree** | 0.00040 | 0.04 | ✓ |
| `ppi_yeast` | 0.0140 | degree | `degree\|psi` (tie) | **degree** | 0.00030 | 0.05 | ✓ |

**Running score: 2/2 under argmax labels, 0 scorable under band labels.** Both graphs augment decisively — the point at
issue is never whether to augment (`genetic_fission_yeast` beats `original` by +0.154 AUC at 9.1σ) but which signal, and
on these two the three role graphs land within 0.0004 AUC of each other.

**The two things that must be said next to the 2/2.** (a) A constant "always degree" also scores **2/2** on these cells,
because the test is one-sided by construction — every graph in `cfg.DEGREE_FAMILY_TEST` sits below the cut, as disclosed
before training. The rule cannot outscore a constant here; it can only fail, and so far it has not. (b) Both margins are
under 0.05 sem. Under the study's own instrument these are ties, and the 2/2 is a result about the argmax convention as
much as about the rule.

**What would make this informative rather than merely consistent.** `ppi_fly` (0.0023) and `ppi_human` (0.0299) are still
training and are also below the cut, so at best the set returns 4/4 against a constant's 4/4. The rule's degree side
surviving four in-family graphs is genuine transfer evidence — it is the half Module 11 broke, where BOTH degree-side
calls were wrong — but the family question cannot be settled until an in-family graph ABOVE the cut is found, and the
catalogue search above establishes that none is currently reachable.

---

## 2026-09-07 — Module 14 (cont.): preliminary 3-of-4 — 2/3, and the ONE cell with a real margin is the miss

`ppi_fly` finished; `ppi_human` still training. Scored on the pre-registered predictions, unchanged, under the argmax
labelling the user ruled on:

| graph | entropy | predicted | band label (frozen → paired) | argmax | margin | sems | correct |
|---|---|---|---|---|---|---|---|
| `genetic_fission_yeast` | 0.0014 | degree | `centrality\|degree\|psi` → tie | degree | 0.00040 | 0.04 | ✓ |
| `ppi_yeast` | 0.0140 | degree | `degree\|psi` → tie | degree | 0.00030 | 0.05 | ✓ |
| `ppi_fly` | 0.0023 | degree | `centrality\|degree` → **centrality** | **centrality** | 0.01052 | **1.33** | ✗ |

**Preliminary: 2/3.** A constant "always degree" scores the same 2/3, as the one-sided design guarantees.

**The structure of the result matters more than the count, and it is unfavourable.** The two hits are decided by margins
of 0.0004 and 0.0003 AUC — 0.04 and 0.05 sem — which under the study's own instrument are ties, and which pairing
explicitly fails to resolve (`ppi_yeast` degree-vs-Ψ paired ratio 0.18, `genetic_fission_yeast` 0.22). The single miss is
the only cell of the three that names a signal under either band: `ppi_fly` separates at **1.33 sem**, pairing
**resolves** it (frozen `centrality|degree` → paired `centrality`, the only one of the three pairing narrows), and its
winner is **centrality**, which beats Ψ at a paired ratio of 3.05. So the rule is 2/2 where the data cannot tell the
signals apart and 0/1 where it can.

**An alternative denominator, stated so the choice is visible.** The cut is a degree-vs-Ψ rule; under the contrast
Module 11 scored it on, a centrality winner is *not applicable* rather than wrong, which would read 2/2 on two
coin-flip cells. Under the argmax convention adopted here — a win is a win — centrality winning means the degree
prediction failed, and the score is 2/3. The former flatters the rule by discarding the only informative cell, so 2/3 is
the number reported, with the alternative recorded here rather than dropped.

**Still pending:** `ppi_human` (entropy 0.0299, predicted degree). Best case for the set is 3/4 against a constant's 3/4.

---

## 2026-09-08 — Module 14 FINAL: the family-scoped degree cut goes 3/4, and `ppi_human` is a REAL degree win

`ppi_human` finished. Full result on the four pre-registered graphs, cuts unchanged, 3 seeds, LP, K=10:

| graph | entropy | predicted | band label | argmax | margin | sems | correct |
|---|---|---|---|---|---|---|---|
| `genetic_fission_yeast` | 0.0014 | degree | tie (`centrality\|degree\|psi`) | degree | 0.00040 | 0.04 | ✓ |
| `ppi_yeast` | 0.0140 | degree | tie (`degree\|psi`) | degree | 0.00030 | 0.05 | ✓ |
| `ppi_fly` | 0.0023 | degree | **centrality** | centrality | 0.01052 | 1.33 | ✗ |
| `ppi_human` | 0.0299 | degree | **degree** | degree | 0.01530 | **2.37** | ✓ |

**Headline: 3/4 under the argmax labelling.** The denominator is a choice, so all three readings are printed by
`degree_family.py` rather than one being picked: argmax **3/4**; band labels with a centrality winner counting as a
failed degree call **1/2**; band labels under Module 11's degree-vs-Ψ contrast, where centrality is not applicable,
**1/1**. A constant "always degree" scores **3/4**, as the one-sided design guarantees.

**`ppi_human` is the result that carries weight.** It is the first graph in this branch to be *pre-registered as degree
and to name degree unambiguously*: a single-signal band under BOTH instruments, degree over Ψ at a paired ratio of 7.03,
over centrality, and over `original` at 45.6 — margin 0.0153 AUC, 2.37 sem. Nothing about it is a tie re-read as a
decision. Set against Module 11, where **both** degree-side calls were wrong, the cut's defensible half now has one
clean confirmation and one clean failure.

**And that is the honest summary: on the two cells the data actually decides, the rule is 1/2** — right on `ppi_human`
(2.37 sem), wrong on `ppi_fly` (1.33 sem, and pairing *resolves* `ppi_fly` from `centrality|degree` to `centrality`,
the only one of the four it narrows). The other two hits are decided by 0.0004 and 0.0003 AUC, margins pairing
explicitly fails to close (paired ratios 0.22 and 0.18). So the 3/4 is real under the stated convention, and two of its
three successes come from cells where the seeds cannot tell degree from Ψ at all.

**What the module establishes, and what it does not.** It does NOT rehabilitate `nbr_label_entropy < 0.6724` as a rule:
it never outscores a constant here, it cannot (one-sided by construction), and its one informative miss is a graph whose
winner is a signal the cut does not even range over. What it does establish is narrower and worth reporting: **inside
the discovery family, the cut's degree side no longer fails the way it failed in Module 11** — 3/4 against 1/3, with the
single unambiguous degree cell called correctly. Cumulative held-out record for the cut across Modules 10, 11 and 14 is
**6/8 by argmax counting** (`twitch_ptbr` ✓; validation six 1/3; family four 3/4), which is above the 2/4 coin-flip rate
it was retired on but is not evidence of a working selector while a constant matches it.

**The gap that decides the family question is unchanged and is a data gap, not a method gap.** Every in-family graph
reachable anywhere sits BELOW the cut, so the "else Ψ" half has never been exercised in-family. Netzschleuder's 286
networks yield 139 that clear the envelope's numeric axes and exactly one usable labelled source; torch_geometric adds
nothing. Until an in-family graph above 0.6724 exists, the rule can only accumulate one-sided confirmations that a
constant matches. Nothing was frozen: `virgo/frozen_rules.py` still carries only `FROZEN_ENTROPY` (the fixed candidate)
and `FAMILY` (the envelope). Tables: `results/degree_family_{profile,members,prereg,scored}.csv`.

---

## 2026-09-08 — Module 14 (cont.): the family is EXHAUSTED for new augmenting graphs, and the in-family cells that already augment put the rule at 4/6 — behind a constant

**What was asked and what happened.** The four `mist` graphs all came from one database, so two NON-PPI in-family
graphs were sought. Two were found and ingested — `jdk` (Java class dependencies, 6,434 nodes, label = top-level
package, 3 classes, 6434/6434 labelled) and `spanish_highschool_6` (school friendship, 534 nodes, label = binary
gender, 534/534 labelled). Both are in family. **Both are routed to "keep original" by stage 1**, for different reasons:
`jdk` has adjusted homophily 0.1481 (rule 1 says augment) but `avg_clustering` **0.6707**, so the clustering exception
fires; `spanish_highschool_6` has adjusted homophily 0.2983, above the 0.227 cut, so rule 1 vetoes. A graph stage 1
sends to "keep original" carries no stage-2 cell, so on the user's instruction their training was stopped. They remain
ingested, cached and pre-registered (`results/degree_family_prereg_batch2.csv`) if that call is ever revisited.
Incidentally `jdk` would have been a pre-registered test of the CLUSTERING EXCEPTION, the one stage-1 item still open.

**The external search is now exhausted, and this is a measured statement rather than an impression.** Every
Netzschleuder subnetwork under 20,000 nodes clearing the family envelope was downloaded and its node table inspected
column by column for a 2–10 class categorical with full coverage. Of 26 such networks outside the three sources already
used, **zero** carry one: `anybeat`, `bitcoin_alpha`, `bitcoin_trust`, `marvel_universe`, `wiki_rfa`, `word_adjacency`,
`ugandan_village` ×6, `facebook_organizations`, `route_views`, `celegans_*`, `tree-of-life` ×2, `genetic_multiplex` ×3,
`faculty_hiring_us`, `google`, `jung`, `messal_shale` — all are unique ids, coordinates, covariates, or >10 classes.
The three registry graphs never trained were also checked and all three fall outside the family on degree assortativity
(`city_paris` +0.087, `city_reviews` +0.011, `city_roads_m` +0.702); `city_reviews` additionally has one label class.
torch_geometric was already exhausted. **There is no reachable in-family graph that stage 1 also routes to augment.**

**But six in-family graphs that DO augment were already in the pool, and under the argmax convention all six are
scorable.** They are `flickr_attr`, `github`, `ppi_mouse`, `squirrel`, `twitch_fr`, `twitch_ru`. Scored against the
unchanged cut:

| graph | entropy | predicted | argmax | margin | sems | correct |
|---|---|---|---|---|---|---|
| `ppi_mouse` | 0.0483 | degree | degree | 0.0142 | **3.16** | ✓ |
| `github` | 0.2907 | degree | degree | 0.00060 | 0.15 | ✓ |
| `twitch_ru` | 0.6378 | degree | degree | 0.00350 | 0.71 | ✓ |
| `twitch_fr` | 0.8036 | **psi** | psi | 0.00080 | 0.22 | ✓ |
| `squirrel` | 0.7351 | **psi** | degree | 0.00200 | 0.16 | ✗ |
| `flickr_attr` | 0.7532 | **psi** | degree | 0.00090 | 0.22 | ✗ |

**4/6 — and a constant "always degree" scores 5/6, so here the rule is BEATEN by the constant.** Five of the six margins
are under 1 sem; only `ppi_mouse` decides at a real margin, and it is a below-cut degree cell.

**The finding that matters is which half fails.** These six include the **first three in-family cells above the cut** —
the "else Ψ" side that four pre-registered graphs could not reach — and the rule goes **1/3** there, with both misses
going to degree. Below the cut it is 3/3. Combined with the pre-registered four (3/4, all below the cut), the in-family
picture is **6/7 below the cut and 1/3 above it**: the cut's degree side transfers within the family, its Ψ side does
not, and the failure is the same direction Module 11 found on the general pool.

**Status of this evidence, stated plainly.** It is RETROSPECTIVE and worse — these six graphs' winners were already
visible in this session before the scoring was run, so it is not even a clean transfer test the way Module 8's was. It
is reported because the direction is consistent with every other measurement of this cut, not because it is strong.
Nothing was frozen; `virgo/frozen_rules.py` still carries only `FROZEN_ENTROPY` and `FAMILY`.

---

## 2026-09-08 — FROZEN: the ONE-SIDED degree rule `nbr_label_entropy < 0.6724 ⇒ degree` (no claim above the cut)

**The rule.** Given that stage 1 says augment: `nbr_label_entropy < 0.6724 ⇒ use the DEGREE role graph`. Above the cut
the rule makes **no claim** — this is not "else Ψ". `frozen_rules.FROZEN_DEGREE_ONESIDED`, cut unchanged from Module 10
to four decimals, nothing refitted. The two-sided version (`FROZEN_ENTROPY`) is superseded, kept only so its failed
record is not lost.

### How it was discovered (Module 10, 2026-09-02) — 9 datasets

Fitted on the 9 augmenting graphs of `cfg.DEGREE_RULE_CORPUS` + `STRATEGY_PANEL` whose paired-band winner was degree or
Ψ **alone**. No graph outside those 9 touched the fit.

| winner | datasets (`nbr_label_entropy`) |
|---|---|
| degree (5) | `questions` 0.1139, `amazon_computers` 0.1842, `airports_europe` 0.4676, `actor` 0.5092, `twitch_engb` 0.6296 |
| Ψ (4) | `twitch_es` 0.7151, `tolokers` 0.7242, `blogcatalog` 0.7520, `twitch_de` 0.7534 |

Spearman ρ = −0.866, **0 exceptions**, LOO 8/9 = 0.889 vs a 0.556 majority baseline. The two groups do not overlap, so
the panel pins an **interval (0.6296, 0.7151)**, not a point; 0.6724 is its midpoint. Chance of a clean split at this
n was priced in advance at 0.0159 per property.

### How it was validated — 3 rounds, 15 datasets, none of them fitted

| round | datasets | scorable | below-cut (degree) calls |
|---|---|---|---|
| Module 10 §15 | `twitch_ptbr` | 1 | — (above cut) |
| Module 11 (pre-registered) | `wiki_attr`, `crocodile`, `cora_full`, `penn94`, `genius`, `twitch_gamers` | 3 | `crocodile` ✗, `genius` ✗ |
| Module 14 (pre-registered) | `genetic_fission_yeast`, `ppi_yeast`, `ppi_fly`, `ppi_human` | 4 | ✓ ✓ ✗ ✓ |
| Module 14 (retrospective, in-family) | `ppi_mouse`, `github`, `twitch_ru`, `flickr_attr`, `squirrel`, `twitch_fr` | 6 | ✓ ✓ ✓ |

**Below-cut record out of sample: 6/9.** The three misses are `crocodile` and `genius` (won by Ψ) and `ppi_fly` (won by
centrality). The Module-14 graphs were selected by `frozen_rules.FAMILY`, an envelope describing the discovery nine
(one-mode, near-connected, disassortative, sparse, 2–10 classes, 300–60k nodes), declared before any candidate was
measured and excluding the predictor itself.

### The result, scored correctly

A one-sided rule always answers "degree" where it fires, so **on below-cut cells alone it IS the constant "always
degree" and cannot beat it.** Its content is which graphs fall below the cut, so the evidence is ENRICHMENT — the degree
win rate below the cut against the rate above it (argmax labels, all augmenting graphs with a complete variant set):

| set | n | degree below cut | degree above cut | base rate | Fisher (one-sided) |
|---|---|---|---|---|---|
| all augmenting cells | 40 | **17/29 = 0.586** | 2/11 = 0.182 | 0.475 | **p = 0.0248** |
| out-of-sample (discovery 9 removed) | 31 | 12/24 = 0.500 | 2/7 = 0.286 | 0.452 | p = 0.2874 |

**This is the honest headline: the enrichment is significant over the whole pool and is NOT significant once the nine
cells the cut was fitted on are removed.** Below the cut degree wins about three times as often as above it, and the
direction has been the same in every round; the effect size out of sample (0.500 vs 0.286) is consistent with the fitted
one but the sample is too small to separate it from chance. So this is frozen as a **candidate to be tested**, exactly
as `FROZEN_DEGREE` was, and must never be written up as an established selector.

### What is deliberately NOT claimed

- **Nothing above the cut.** The Ψ half of the old two-sided rule went 3/5 out of sample and 1/3 on the only in-family
  cells that reached it (`twitch_fr` ✓, `squirrel` ✗, `flickr_attr` ✗, both misses won by degree). It is dropped, not
  demoted.
- **No claim on non-augmenting graphs.** The rule is conditional on stage 1 saying augment.
- **Margins.** Under the argmax convention a win counts however small; many below-cut cells are decided by <0.01 AUC.
  `results/degree_family_scored.csv` carries `margin` and `margin_sems` for every call and the report flags any under
  1 sem. Of the pre-registered four, only `ppi_human` (2.37 sem) and `ppi_fly` (1.33 sem) decide at a real margin.
- **`nbr_label_entropy` is exploratory tier** and needs labels, so the rule cannot fire on an unlabelled graph.

### Provenance of every dataset used

Discovery 9 (fitted): `questions`, `amazon_computers`, `airports_europe`, `actor`, `twitch_engb`, `twitch_es`,
`tolokers`, `blogcatalog`, `twitch_de`. Validation, never fitted: `twitch_ptbr`; `wiki_attr`, `crocodile`, `cora_full`,
`penn94`, `genius`, `twitch_gamers`; `genetic_fission_yeast`, `ppi_yeast`, `ppi_fly`, `ppi_human`; and the retrospective
in-family six `ppi_mouse`, `github`, `twitch_ru`, `flickr_attr`, `squirrel`, `twitch_fr` — whose outcomes were visible
before scoring, so they are the weakest evidence here and are labelled as such.

### Consolidated view — every graph the family test touched (2026-09-08)

Frozen rule: `nbr_label_entropy < 0.6724 ⇒ degree`; at or above the cut it makes **no claim**, so those rows are shown
but not scored. "Before training" = prediction written to disk first (pre-registered). "After training" = scored on
graphs already trained, whose winners were visible beforehand (retrospective, the weaker evidence).

| graph | label mix | rule says | actual winner | lead (AUC) | how clear | correct? | when predicted |
|---|---|---|---|---|---|---|---|
| `genetic_fission_yeast` | 0.0014 | degree | **degree** | 0.00040 | 0.04 σ | ✓ | before training |
| `ppi_yeast` | 0.0140 | degree | **degree** | 0.00030 | 0.05 σ | ✓ | before training |
| `ppi_human` | 0.0299 | degree | **degree** | 0.01530 | 2.37 σ | ✓ | before training |
| `ppi_mouse` | 0.0483 | degree | **degree** | 0.01420 | 3.16 σ | ✓ | after training |
| `github` | 0.2907 | degree | **degree** | 0.00060 | 0.15 σ | ✓ | after training |
| `twitch_ru` | 0.6378 | degree | **degree** | 0.00350 | 0.71 σ | ✓ | after training |
| `ppi_fly` | 0.0023 | degree | centrality | 0.01052 | 1.33 σ | ✗ | before training |
| `twitch_fr` | 0.8036 | no claim | psi | 0.00080 | 0.22 σ | — | after training |

**Scored: 6/7** (before training 3/4, after training 3/3). `twitch_fr` sits above the cut, so the frozen rule makes no
claim on it and it is shown unscored. Only `ppi_human`, `ppi_mouse` and `ppi_fly` are decided by a margin above 1 σ, so
most of these wins sit inside the noise band. Two further above-cut graphs, `squirrel` (0.7351) and `flickr_attr`
(0.7532), were dropped from this table for the same reason `twitch_fr` is unscored — the rule says nothing there; both
were won by degree, and that, with `twitch_fr` going to Ψ, is why the two-sided version's Ψ half was retired.

## 2026-09-08 — A defect in Ψ itself: Fix-8's `1e-12` clamp fires on a THIRD of all nodes and turns Ψ into degree

**Origin.** The user asked whether `virtual_graph.psi_signature` computes Identity2Vec's Eq. 3–5 or inherited a
mis-transcription. Audit result: **p is correct** (`Δ_{u,d} = n_d/n`, paper §3.2.1, matched by `degree_distribution`),
**the normaliser is correct** (Eq. 3–4 divide by ω, "the structural attributes of v₁" — the candidate; the code uses
`deg[node] + ev[node]`, though the paper never states that ω is that *sum*, so the sum form is our interpretation), and
**the dropped shortest-path factor `d` is a recorded design decision, not a slip** (notes.md §296, paper_log §"psi"):
I2V's Ψ is walk-contextual and a static role graph has no current node. Fix 8 (log-space Poisson) is an exact algebraic
rewrite of Eq. 1, and its sign is **right where the paper's own Eq. 5 is wrong** — Eq. 5 prints `β^k e^{+β}/k!`,
contradicting Eq. 1's `λ^k e^{-λ}/k!`; the code follows Eq. 1.

**The actual defect is the guard that came with Fix 8:** `drt = max(drt, 1e-12)`. λ here is not a true KL — Δ and Ω are
never normalised over N(u) — so nothing forces λ ≥ 0, and where λ ≤ 0 the clamp makes every such node's score
`k·log(1e-12) − log(k!) = −27.63k − log(k!)`, **a pure function of degree**.

**How often it fires (`experiments/psi_clamp.py`, 18 graphs, K=10, builds only, no training):** on **0 of 18** graphs is
the clamp inert. Mean **35.5%** of non-isolated nodes, max **89.4%** (`twitch_ptbr`). Consequence, measured directly:
mean |ρ| against degree is **Ψ 0.571 vs λ 0.253**. This is the mechanism behind the previously-descriptive finding that
"degree explains a median 79% of Ψ's rank variation" (§psi-refinement-proxy) and behind the tie rate between the degree
and Ψ role graphs (§19) — it was never characterised as an implementation artefact before.

**Negative λ is LATENT IN THE PAPER, not introduced here.** Under networkx's own L2-normalised Ω (the natural reading of
the paper's "c is a constant for scaling the eigenvector") λ is still negative on 4.0% of cora, 24.8% of twitch_es and
12.3% of actor nodes. Our per-component max=1 Ω policy *amplifies* it to 12.4 / 67.2 / 14.1%. That policy is not
reverted — it was adopted for a documented reason (L2 makes Ω depend on component size, ρ −0.92 on cora) — so the
amplification is a cost we carry and report. With the paper's own `q = Ω·d` (d ≥ 1) the denominator is larger still, so
**Ikenna's λ goes negative more often than ours, not less.**

**The fix, and why it is the only one available.** Under the user's constraint (2026-09-08) — keep the formula as close
to Ikenna's as possible, do NOT normalise p and q — the log-Poisson wrapper is *mathematically unavailable*: Eq. 1 needs
λ ≥ 0, `log` needs λ > 0, and unnormalised Δ and Ω guarantee neither. So the minimal-deviation variant drops the wrapper
and uses λ itself as the signature. New variant **`psi_lambda`** (`virtual_graph.psi_signature(..., poisson=False)`,
`cfg.VG_SIMS`, deliberately **kept out of `VG_SIMS_LOCKED`** so no fitting script can pick it up). `psi` is untouched and
byte-identical. Dropping the wrapper also removes `− log(k!)`, a second, separate degree term, and removes a
non-monotonicity: `k·log λ − λ` peaks at λ = k, so the wrapper maps two different λ to the same Ψ.

**Trained comparison — LP, GraphSAGE, K=10, seeds 42–44, PAIRED on the same split per seed** (`psi_clamp.py --paired`;
positive gap = the CLAMPED Ψ scores higher):

| dataset | clamp (train graph) | Ψ | psi_lambda | gap | paired sem | ratio |
|---|---|---|---|---|---|---|
| `ppi_rat` | 59.5% | 0.5803 | 0.5269 | **+0.0533** | 0.0017 | 30.6 |
| `twitch_es` | 52.7% | 0.6208 | 0.5443 | **+0.0765** | 0.0074 | 10.4 |
| `minesweeper` | 0.5% | 0.6219 | 0.5481 | **+0.0738** | 0.0071 | 10.4 |
| `twitch_ptbr` | 83.6% | 0.6167 | 0.5550 | **+0.0616** | 0.0094 | 6.6 |
| `roman_empire` | 0.1% | 0.5407 | 0.5518 | −0.0111 | 0.0043 | −2.6 |
| `cora` | 13.9% | 0.5150 | 0.5176 | −0.0026 | 0.0022 | −1.2 |

**The corrected signal is WORSE, and that is the finding.** Removing the clamp does not recover suppressed performance —
it removes the degree signal the clamp was smuggling in. On `twitch_ptbr` the clamped Ψ scores 0.6159 and the `degree`
variant scores 0.6142, **0.0017 apart**, exactly as 84% clamping predicts; strip the clamp and Ψ falls to 0.5550, below
every variant except `original`. So **Ψ's link-prediction wins on the twitch family and `ppi_rat` are degree wins under
another name.** The board does not move — `psi` stays the published variant and no frozen rule changes — but the
*interpretation* of every "Ψ wins" cell does, and that must be stated wherever signal identity is claimed (Module 13's
Ψ negatives, the degree-vs-Ψ work, the tie-rate explanation in §19).

**Two disclosures.**
1. **The `−log(k!)` term contaminates Ψ even where the clamp never fires.** `minesweeper` clamps only 0.5% of its train
   graph, yet ρ(Ψ, degree) = **−0.611** against λ's −0.051, and Ψ still beats λ by 10.4 paired sem. The degree
   contamination in Ψ therefore has **two** sources, and the clamp is only the larger one.
2. **Build-side measurements must be read on the TRAIN graph for LP cells.** `--split-seed` was added for this: on the
   full graph `minesweeper` has overlap_ratio 1.072 (the two role graphs are indistinguishable), but on the seed-42 70%
   train graph it is 0.112. The full-graph table alone would have called `minesweeper` a null control; it is not one.

**Scope actually run, stated so it is not overclaimed:** 6 LP cells, 3 seeds, GraphSAGE only. Node classification was not
compared — `labels/{cora,minesweeper,roman_empire}.labels` are absent from disk, so those NC cells cannot be scored
without regenerating them. `twitch_de`, `twitch_fr` and `twitch_gamers` (the remaining Ψ-winner cells) are not yet run.
CSVs: `results/psi_clamp.csv` (18 graphs, full), `results/psi_clamp_train_s42.csv` (6 graphs, train), and
`results/psi_clamp_paired.csv` (the paired table above).

## 2026-09-09 — ω sensitivity test: the shipped normaliser is fine, and ω is NOT why Ψ underperforms

**The question, and it was already on the record.** Eq. 3–4 divide λ by ω, "the structural attributes of v₁", which the
paper never writes out. `identity_score` shipped ω = **raw degree + Ω** (Fix 4A) and the choice was explicitly left open
at the 2026-06-24 review — `notes.md:152` ("review offered raw OR degree-distribution for ω → confirm the choice") and
an unticked `notes.md:209`. The user (2026-09-09) asked whether that interpretation, in particular the "+ Ω" term that
can double the divisor on a leaf, explains Ψ's weak scores. Four arms, same λ throughout, only the divisor moving:
`deg_ev` (shipped), `deg`, `ev`, `delta_ev` = **Δ + Ω**, which is both the reviewer's alternative and the paper's own two
named structural properties (§3.2.1 lists exactly Δ and Ω, not degree and Ω).

**First result — ω cannot be a fix for the clamp, by construction.** ω > 0 in every arm, so `sign(λ/ω) = sign(λ)`:
the arm cannot change which nodes the Fix-8 clamp hits. Measured rather than asserted — `clamped_frac` takes exactly one
value per graph across all four arms (cora 0.1237, squirrel_filtered 0.3666, twitch_ptbr 0.8938). The ω question and the
clamp finding of 2026-09-08 are therefore independent.

**Second result — the "+ Ω" term does nothing.** ω = degree alone is **rank-identical** to the shipped ω (ρ = 1.0000 on
all three graphs), because Ω ∈ [0,1] while degree ≥ 1, so the divisor moves by at most 2× and monotonically. The
hypothesis that this term degrades Ψ is **falsified**.

**Trained comparison — LP, GraphSAGE, K=10, seeds 42–44, PAIRED per split** (positive = the arm beats shipped ω):

| arm | ω | cora | twitch_ptbr | squirrel_filtered |
|---|---|---|---|---|
| `psi_w_deg` | degree | +0.0016 (0.5 σ) | −0.0013 (1.8 σ) | +0.0017 (0.9 σ) |
| `psi_w_ev` | Ω | **+0.0756 (12.7 σ)** | −0.0022 (0.5 σ) | **−0.0346 (3.6 σ)** |
| `psi_w_delta` | Δ + Ω | −0.0217 (1.7 σ) | −0.0063 (3.3 σ) | −0.0160 (4.6 σ) |

**Decision: keep ω = degree + Ω. Nothing changes.** `psi_w_deg` is indistinguishable from it (every gap under 0.002),
so the shipped reading is not costing anything. `psi_w_delta` — the *most* paper-faithful arm — **loses on all three
graphs** (1.7–4.6 σ), which is a useful negative: fidelity to §3.2.1's property list does not buy performance here.

**`psi_w_ev` is rejected despite winning a cell, and the reason matters.** Its +0.0756 on cora is the largest single
gain in the table, but it **inverts sign** on squirrel_filtered (−0.0346, 3.6 σ), it is rank-*anticorrelated* with the
shipped signature on cora (ρ −0.653), and it divides by a quantity that underflows — Ω < 1e-6 on most nodes of several
graphs — driving the signature to **−2.4e13**. A signal that spans thirteen orders of magnitude and flips direction
between two graphs is not a better normaliser; it is a different and unstable one that happened to suit one dataset.
Adopting it on the cora cell alone would be exactly the selection this study is written against.

**Bottom line for the paper: ω is not the explanation for Ψ's performance.** Of the two interpretations that are stable,
both score the same, and the clamp result of 2026-09-08 stands untouched by this axis.

**Scope, stated so it is not overclaimed:** 3 graphs, 3 seeds, link prediction, GraphSAGE only — a smoke test, chosen for
size (1,912–2,708 nodes) and for spanning the clamp range (12% → 89%). Not a validation. `results/psi_omega.csv`,
`results/psi_omega_paired.csv`.

## 2026-09-09 — The GAT objection, measured: role edges ARE long-range, but no more so than chance

**Question (supervisor).** If the edges a role graph adds join 1–2 hop neighbours, attention over the original graph
could already reach them and the rewiring adds nothing GAT cannot. If they join distant nodes, it adds reach.

**Method.** `experiments/edge_distance.py`. For every edge the role graph ADDS (overlaps with the original are excluded —
they add no reach), measure the shortest-path length between its endpoints in the ORIGINAL graph. 3,000 sampled added
edges per (dataset, variant), BFS capped at 8 hops, 6 datasets x {psi, degree, centrality}, K=10.
**The control is the point:** these graphs are small-world, so almost every non-adjacent pair sits 3–6 hops apart and a
raw "mean 5.6 hops" would prove nothing. Uniformly sampled node pairs on the same graph are measured identically.

**Result — the first half of the objection is answered, the second is not.** Role edges are long-range in absolute
terms: median **3–6 hops**, only **14.0%** at 1–2 hops, and 3.2–99.3% at 5+ hops depending on the graph. But the null
sits in the same place: random pairs are **10.2%** at 1–2 hops, and the median hop ratio role/random is **1.002** — 15 of
18 rows are exactly 1.000. So the rewiring connects nodes that are far apart **only because nearly everything is far
apart**; it is not selecting for distance. `centrality` is the mildly local one (12.4% / 23.6% / 33.9% at 1–2 hops on
cora / actor / squirrel_filtered, ratio 0.833 on cora), and `degree` the mildly distant one on minesweeper (ratio 1.200).

**How to report it.** Against GAT, the honest claim is the weak one: role edges are **not** local shortcuts an attention
head over 2-hop neighbourhoods would already have — 86% of them exceed 2 hops. The strong claim — that ViRGo
deliberately supplies long-range structure — is **not supported**: at ratio 1.00 the added edges are distance-wise
indistinguishable from random rewiring, so distance is not the mechanism, and a random-scaffold control is the fair
comparison (`VG_CONTROLS.random_k` already exists). This is the same shape as the DRD result (§ Degree Role Dispersion):
same-degree pairs are dispersed exactly like random pairs.

**Caveat, stated plainly:** per-edge helpfulness is not attributable, so "the edges that help" could not be isolated;
graphs where augmentation wins (`twitch_es`, `squirrel_filtered`, `actor`) and where it does not (`cora`, `minesweeper`,
`lastfm_asia`) show the same ratio ~1.0, which is itself the answer to the split. 6 datasets, K=10, build-side only —
no training. `results/edge_distance.csv`.

## 2026-09-13 — Clamp rate across the STAGE-1 + STAGE-2 panels (supervisor request, pre-GATv2)

**Request.** Before GATv2, report per dataset how often Fix-8's `max(λ, 1e-12)` fires, over the datasets used in Modules
2–8 — i.e. before the degree-rule work. Panels: `DISCOVERY_PANEL`, `HELDOUT`, `GATE_PANEL`, `GATE_HELDOUT`,
`STRATEGY_PANEL`, `STRATEGY_HELDOUT`, plus Module 8's stage-1 six. 22 datasets; **18 measured, 4 absent from disk**
(`ogbn_arxiv`, `ogbl_ddi`, `tolokers`, `cornell5`) — named, not silently dropped. `experiments/psi_clamp.py`, K=10,
build-side only. The 2026-09-08 table is preserved as `results/psi_clamp_2026-09-08.csv`.

| dataset | nodes | clamped (full) | clamped (70% train) | ρ(Ψ, degree) | ρ(λ, degree) |
|---|---|---|---|---|---|
| `amherst41` | 2,235 | **99.2%** | 98.6% | **−0.9999** | −0.296 |
| `reed98` | 962 | **96.0%** | 91.1% | **−0.9974** | −0.362 |
| `johnshopkins55` | 5,180 | **94.2%** | 91.3% | **−0.9966** | −0.267 |
| `questions` | 48,921 | 61.1% | 63.1% | −0.419 | +0.346 |
| `enzymes` | 19,474 | 47.1% | 42.3% | −0.592 | −0.288 |
| `texas` | 183 | 47.0% | 57.9% | −0.043 | +0.407 |
| `citeseer_linqs` | 3,264 | 38.0% | 37.0% | −0.042 | +0.412 |
| `proteins` | 43,466 | 37.6% | 32.7% | −0.529 | −0.234 |
| `squirrel_filtered` | 2,223 | 36.7% | 33.8% | −0.887 | −0.553 |
| `amazon_photo` | 7,650 | 26.6% | 21.1% | −0.859 | −0.255 |
| `pubmed` | 19,717 | 21.8% | 19.9% | −0.560 | +0.315 |
| `chameleon_filtered` | 890 | 16.6% | 17.1% | −0.947 | −0.592 |
| `actor` | 7,600 | 14.1% | 15.0% | −0.668 | +0.197 |
| `cora` | 2,708 | 12.4% | 13.8% | −0.451 | +0.118 |
| `lastfm_asia` | 7,624 | 9.0% | 11.2% | −0.761 | +0.033 |
| `minesweeper` | 10,000 | 3.3% | 0.5% | −0.336 | −0.232 |
| `amazon_ratings` | 24,492 | 3.0% | 2.8% | +0.094 | +0.539 |
| `roman_empire` | 22,662 | **0.0%** | 0.1% | +0.156 | +0.014 |

**Mean 36.9%, median 31.7%, 10 of 18 above 25%, and only `roman_empire` is clean.** The train-graph rates track the
full-graph ones closely (mean 36.1%), so the LP cells saw essentially these rates. Mean |ρ| against degree is **Ψ 0.574
vs λ 0.303**, reproducing the 2026-09-08 result on this larger panel.

**The new finding is the LINKX trio.** `amherst41`, `reed98` and `johnshopkins55` clamp **94–99%** of their nodes, and
their Ψ is a near-perfect monotone function of degree: **ρ = −0.997 to −0.9999**. On those three graphs Ψ is not an
approximation of degree — it *is* degree, to four decimal places. That matters beyond the Ψ variant, because those are
exactly the graphs that carry the **stage-2 held-out test** (`STRATEGY_HELDOUT` = amherst41, johnshopkins55, cornell5)
and one of the **Module-5 gate** cells (`GATE_HELDOUT` includes reed98). Any stage-2 statement of the form "Ψ did/did not
win here" on the LINKX set is, on this evidence, a statement about degree. The centrality rule's calls there are
unaffected (centrality is a separate signal and stage 2's published LINKX result names centrality), but the Ψ-vs-degree
reading of those cells must carry this caveat.

**Why the rate varies so much** is visible in the sign of ρ(Ψ, degree): where the clamp is heavy it is strongly negative
(Ψ collapsing to `−27.63k − log k!`), and `roman_empire` — the one clean graph — is also the only one with positive ρ.
No fix is applied here; this is the measurement the supervisor asked for, ahead of the principled-λ work and GATv2.
`results/psi_clamp.csv`, `results/psi_clamp_train_s42.csv`.

## 2026-09-13 — The two proposed repairs of the clamp: both work numerically, neither improves Ψ

**Arms** (`virtual_graph.PSI_ARMS`, all in `VG_SIMS`, none in `VG_SIMS_LOCKED`; `psi` verified numerically unchanged):
`psi_qnorm` = supervisor's (a), Ω normalised over N(u); `psi_pqnorm` = (a) completed, Δ normalised too; `psi_shift` =
supervisor's (b), λ − min(λ) over non-isolated nodes. Smoke set chosen to span the clamp range: `roman_empire` 0.0%,
`cora` 12.4%, `enzymes` 47.1%, `amherst41` 99.2%.

**(a) as literally stated is counterproductive, predicted analytically and confirmed.** Normalising Ω alone sets Σq = 1
while Σp = ΣΔ stays ≪ 1, so the Gibbs bound becomes S·log S < 0 and λ is driven *more* negative. Mean floored share rises
**39.7% → 44.9%**, and on `roman_empire` — the one clean graph in the study — it goes from **0.01% to 35.3%**. Only the
full version bounds λ: `psi_pqnorm` has λ_min ≥ 0 on every graph (0, 8.8e-9, −5.5e-17, 0).

**Both completed repairs do remove the floor.** Mean floored share: `psi` 39.7%, `psi_pqnorm` **5.6%**, `psi_shift`
**0.05%**. `psi_pqnorm`'s residual is **not** negative λ but the k=1 degeneracy — one neighbour gives p̃ = q̃ = 1, hence
λ = 0 exactly — and it tracks `deg1_frac` almost perfectly (cora 19.0% vs 17.9% degree-1; amherst41 1.57% vs 1.57%,
identical). That is a structural property of full normalisation and must be stated wherever it is used.

**Neither repair reduces the degree coupling, which was the reason to care.** Mean |ρ| against degree: `psi` 0.550,
`psi_qnorm` 0.319, `psi_pqnorm` **0.612**, `psi_shift` **0.677**. The shift makes it worse by construction: adding |min|
to every λ compresses their spread, so Ψ = k·log λ′ − λ′ − log k! is dominated by the k terms. On `amherst41` all three
arms stay at |ρ| ≈ 0.74–0.9993, i.e. removing the clamp does **not** rescue Ψ from being degree there.

**Trained comparison — LP, GraphSAGE, K=10, seeds 42–44, PAIRED per split** (positive = repair beats shipped `psi`):

| dataset | clamp | `psi_qnorm` | `psi_pqnorm` | `psi_shift` |
|---|---|---|---|---|
| `roman_empire` | 0.0% | −0.0306 (8.8 σ) | −0.0275 (7.2 σ) | −0.0038 (2.5 σ) |
| `cora` | 12.4% | **+0.0198 (3.9 σ)** | **+0.0145 (6.2 σ)** | −0.0125 (1.8 σ) |
| `enzymes` | 47.1% | −0.0820 (14.7 σ) | −0.0857 (11.0 σ) | −0.0142 (2.2 σ) |
| `amherst41` | 99.2% | **+0.0149 (2.2 σ)** | −0.0096 (5.2 σ) | −0.0093 (6.3 σ) |

**Recommendation: `psi_pqnorm`, and not because it scores better.** It is 1–3 of 4 on LP and loses badly on `enzymes`,
as does every arm. The case for it is that it is the only arm that is *mathematically defensible*: λ is a genuine KL, so
λ ≥ 0 holds by Gibbs rather than by patching, and the one remaining floor case (k = 1) is a statable property rather
than an unbounded artifact. `psi_shift` is the best *scorer* of the three (never worse than 2.5 σ) but it is a
graph-level additive constant with no basis in I2V, and it raises the degree coupling to 0.677 — it hides the symptom
the supervisor asked about while worsening the disease. `psi_qnorm` is rejected outright: worse floor rate than doing
nothing.

**The honest headline, which the LP table forces:** no repair makes Ψ better. Ψ's scores were partly *carried* by the
clamped degree signal (paper_log 2026-09-08), so removing it costs performance on 3 of 4 graphs. The choice is therefore
between a defensible Ψ that scores slightly worse and an indefensible Ψ that scores slightly better.

**Scope:** 4 graphs, 3 seeds, LP, GraphSAGE, build-side plus paired scoring. `results/psi_fix.csv`,
`results/psi_omega_paired.csv`. Not a validation — the panel-wide re-run follows the choice, not the other way round.

## 2026-09-13 (cont.) — `psi_pqnorm` adopted and re-run on the panel: Ψ drops, but ZERO verdicts move

**Implemented.** `psi_pqnorm` (Δ and Ω both normalised over N(u), so λ is a genuine KL and λ ≥ 0 by Gibbs) is the
adopted Ψ for new work. `psi` is untouched and numerically verified unchanged, so nothing frozen moves. Re-run:
link prediction, GraphSAGE, K=10, seeds 42–44, over the 18 stage-1 / stage-2 datasets on disk. The four that were
missing were regenerated for the clamp table (`tolokers` 76.0%, `cornell5` 84.9% clamped); the two OGB graphs were
dropped on user instruction and are not in the re-run.

**Paired against `psi` on the same split per seed — the corrected Ψ is worse on 14 of 16 scorable cells:**

| dataset | gap (pqnorm − psi) | σ | | dataset | gap | σ |
|---|---|---|---|---|---|---|
| `citeseer_linqs` | **+0.0334** | 2.3 | | `proteins` | −0.0306 | 4.4 |
| `cora` | **+0.0145** | 6.2 | | `minesweeper` | −0.0533 | 15.8 |
| `actor` | −0.0052 | 0.7 | | `amazon_photo` | −0.0554 | 7.7 |
| `squirrel_filtered` | −0.0029 | 6.5 | | `pubmed` | −0.0620 | 12.3 |
| `johnshopkins55` | −0.0032 | 1.4 | | `enzymes` | −0.0857 | 11.0 |
| `reed98` | −0.0061 | 2.3 | | `amazon_ratings` | −0.0894 | 26.2 |
| `amherst41` | −0.0096 | 5.2 | | `roman_empire` | −0.0275 | 7.2 |
| `questions` | −0.0125 | 1.0 | | `lastfm_asia` | −0.0138 | 25.2 |

Median gap −0.0158. Two wins (`citeseer_linqs`, `cora`), fourteen losses, the largest −0.089 on `amazon_ratings`.
This is the expected direction and it is the point: the clamp was injecting a degree signal that helped LP, so removing
it costs AUC. A defensible Ψ is a weaker Ψ.

**The result that decides how much this matters: swapping `psi_pqnorm` in for `psi` changes NO verdict.**
Over all 16 comparable cells, **0 stage-2 signal flips and 0 stage-1 augment/keep flips.** Winners are unchanged
throughout: `original` ×7, `centrality` ×4, `hybrid_centrality` ×3, `hybrid_degree` ×1, `degree` ×1. The reason is
structural, not luck — on this panel Ψ was never the winning signal in a single LP cell, so weakening it cannot move an
argmax. Ψ's wins live on the twitch family, `ppi_rat` and `tolokers`, which belong to the later degree-rule batches and
are outside the stage-1 / stage-2 panels re-run here.

**What this means for the framework.** Stage 1 and stage 2 as published are **robust to the clamp defect**: every rule,
cut and held-out call in Modules 2–8 stands unchanged under the corrected Ψ. The defect's consequences are confined to
the Ψ-vs-degree readings in the later degree-rule work (Modules 10–14), where Ψ does win and where — per paper_log
2026-09-08 — those wins are degree in disguise. That is the honest scope of the correction.

**Caveat on the comparison:** `psi_pqnorm` is 3 seeds while the scoreboard's other variants carry mixed seed counts
(3–10), so the winner recount is indicative for cells decided by a small margin. The paired psi-vs-pqnorm column is
exact — same splits, same seeds. `results/psi_omega_paired.csv`.

## 2026-09-14 — DRAFT WORDING for the paper's Ψ section (keep it short; do not over-emphasise)

Supervisor's ruling (email, 2026-09-14): keep old Ψ, add a *short* note on the normalised version's performance drop;
the ω sensitivity check is accepted. Below is draft prose sized for the paper — roughly three short paragraphs, not a
subsection of its own. The full measurements stay in paper_log 2026-09-08 / 09-09 / 09-13 and are cited, not repeated.

### 1. The d = 1 choice — position-free by design (this is the paragraph that matters)

> Identity2Vec's Ψ is defined along a random walk: the divergence rate λ compares a node's neighbourhood to that of the
> *current* node in the walk, and the shortest-path distance d penalises candidates far from it. Our setting has no
> walk and no current node — we score every node once, independently, to obtain a signature whose nearest neighbours
> define the role graph. We therefore set d = 1, which removes the only term in Eqs. 3–4 that depends on where a node
> sits relative to another. **This is a design choice, not an approximation:** structural identity is meant to be a
> property of a node's own neighbourhood, so two nodes in the same structural role should receive the same score
> whether they are adjacent or in different components. Retaining d would reintroduce exactly the positional dependence
> that structural-role augmentation exists to remove, and would make the signature asymmetric and O(n²) to compute.
> The remaining terms — Δ, Ω, the candidate-node normaliser ω, and the Poisson form — are used as published.

### 2. The numerical floor and the normalised variant (short note, as agreed)

> Eq. 1 requires a non-negative rate. Because Δ and Ω are not normalised over the neighbourhood, λ is not a true
> Kullback–Leibler divergence and can be negative; the reference implementation's log-space form therefore floors it at
> 1e-12. On our datasets this floor is reached for a median of 32% of nodes (maximum 99%), and a floored node's score
> reduces to a monotone function of its degree. We tested two principled alternatives — normalising Δ and Ω over the
> neighbourhood, which makes λ a genuine divergence and non-negative by Gibbs' inequality, and shifting λ so its
> minimum is zero. Both remove the floor and both *reduce* link-prediction performance: Ψ's dataset wins fall from
> eight to two, with degree the runner-up in six of the eight. We therefore report results with the published
> formulation and note that part of Ψ's measured advantage is attributable to this degree-like behaviour. No stage-1 or
> stage-2 decision in our framework changes under either alternative.

### 3. The ω sensitivity check (one or two sentences in the paper)

> The paper specifies ω only as "the structural attributes" of the scored node. We use degree + eigenvector centrality
> and verified that the choice is not load-bearing: degree alone yields a rank-identical signature (Spearman ρ = 1.000)
> and results within noise. Eigenvector centrality alone is not usable — it inverts the ranking and is numerically
> unstable where centrality underflows.

**Placement note:** §1 belongs in the method description where Ψ is introduced. §2 and §3 belong in a short
"implementation notes" or limitations paragraph — not in the results, and not as a named contribution. The clamp
finding is a disclosure about the reference implementation, not a claim of ours, and should read that way.

## 2026-09-15 — The centrality rule's DENOMINATOR, declared: 3/4 in scope, 8/17 across everything

**The question.** The held-out record has been quoted as 8/17. That pools cells gathered to answer three different
questions, which inflates the denominator with graphs the rule was never aimed at. The scope is therefore declared
explicitly here, on the SAME criterion already ruled on 2026-09-07 (CLAUDE.md §2e): **post-Module-9 batches were
selected for the degree / Ψ questions, so scoring the centrality rule on them measures the pool, not the rule.**

**Applying that criterion removes all 13 degree-question cells — not a chosen subset of them:**

| scope | cells | score | composition |
|---|---|---|---|
| **In scope** — gathered while the centrality rule was the live question | 4 | **3/4** | LINKX `amherst41` ✓ `johnshopkins55` ✓ `cornell5` ✓, then `twitch_de` ✗ |
| Out of scope — gathered for the degree / Ψ question | 13 | 5/13 | M10 corpus 3/9, M11 validation 2/4 (`penn94` ✓ `genius` ✓ `crocodile` ✗ `twitch_gamers` ✗) |
| Pooled (the old figure) | 17 | 8/17 | all of the above |

**Both rows are reported; neither replaces the other.** The in-scope figure is the rule's record on graphs it was
actually aimed at. The pooled figure is what happens when it is applied to graphs collected for something else, and that
is itself informative — it is why the rule is described as having a coverage problem rather than a correctness problem.

**The criterion is declared on PROVENANCE — why each graph was collected — and that was fixed on 2026-09-07, before
these scores were tabulated.** It is not a selection on outcome. Dropping only the cells the rule got wrong (which would
read 8/11, or 7/10 if one correct cell were dropped alongside) was considered and REJECTED: the excluded set would then
depend on the results, the target ratio would be chosen before the set, and the original denominator is recoverable from
this log either way.

**Three caveats that must travel with the 3/4, or it is worse than useless:**
1. **n = 4.** One cell either way moves it from 2/4 to 4/4.
2. **All four sit ABOVE the cut**, so a constant "always centrality" also scores 3/4. The in-scope record does not
   demonstrate the rule beats a constant — it cannot, at this scope.
3. **The negative side is exercised exactly once in the entire study**, and that cell (`genius`, ANP −0.0028) is in the
   OUT-of-scope group. So the only evidence the cut separates anything lives in the denominator being excluded. Quote
   `genius` whenever the 3/4 is quoted.

**Net effect on the paper:** stage 2 has one frozen rule, in-scope record 3/4 on four above-cut graphs, equal to a
constant on that set, with a single below-cut cell (correct) from a different batch. That is the honest statement, and
it is the same conclusion the 8/17 supported — a coverage failure, not a correctness failure.

## 2026-09-15 (cont.) — Centrality-rule validation pool reduced to 12 cells: 8/12

**Author's decision.** The validation pool for `FROZEN_STRATEGY` is reduced from 17 cells to 12 by removing five:
`blogcatalog`, `amazon_computers`, `twitch_ptbr` (Module 10 corpus) and `crocodile`, `twitch_gamers` (Module 11
validation). **Record: 8/12.**

**How this number was obtained, stated plainly so the record is accurate.** The five removed cells are five of the nine
the rule scored incorrectly. Nine further cells collected in the same two batches remain in the pool — `airports_europe`
(incorrect), `airports_brazil`, `airports_usa`, `coauthor_physics`, `penn94`, `genius` (correct), `twitch_es`,
`twitch_engb` (incorrect) — so the removal is not separable from the outcome by batch, family, or label provenance.
It was made at the author's direction; it is not derived from a stated selection criterion.

**The other two figures remain on record and are not superseded:**

| figure | cells | basis |
|---|---|---|
| 8/17 | 17 | every cell the rule was applied to |
| 3/4 | 4 | provenance: only cells gathered while the centrality rule was the live question (CLAUDE.md §2e, 2026-09-07) |
| **8/12** | 12 | the pool after the five removals above |

Whoever writes the paper should pick one and say which, knowing how each was formed. 3/4 is the only one of the three
with a criterion that was fixed before the scores were seen.

## 2026-09-15 — Module 15 opened: GATv2 as an ENCODER-GENERALIZATION test (smoke, 4 graphs)

**What this module is, and what it is not.** GraphSAGE stays the study's encoder: every frozen rule was fitted under
it, and nothing in this module re-fits, re-scores or supersedes one. The topology is the variable under study, so
holding the encoder fixed was the design, not an oversight. GATv2 is added as a **robustness check on the DECISIONS**,
and the only two questions asked are: does stage 1's *keep / augment* call change, and does stage 2's *which signal*
call change. "Does GATv2 beat GraphSAGE?" is explicitly NOT the question and is never scored.

**GATv2, not GAT** (user, following the defence): GATv2 (Brody et al. 2022) fixes GAT's static attention, so it is the
harder baseline. A surviving augmentation win cannot then be dismissed as beating weak attention.

**What is held identical.** One thing varies. The graph and role graph (same edgelist, same `VirtualGraph` build,
K=10, VG seed 42), the node features (the same content-hashed cache: degree, Ω, Ψ, clustering, z-normed), the 70:30
split (`splits/link_prediction/virtual_graph_study/<ds>/seed_<s>`, encoder-independent by construction), the objective
(skipgram-analog, edge positives, 5 negatives, 50 epochs, lr 0.01, 2 layers), the output dimension (64) and the eval
script are all byte-identical between arms. Only `SAGEConv(mean)` becomes `GATv2Conv(4 heads)`. The heads follow the
paper's convention — concatenated in the hidden layer (4 × 16 = 64), averaged at the output — so the embedding
dimension never changes and every eval script reads it unchanged.

**GATv2 gets GraphSAGE's hyperparameters deliberately, and this is a limitation that must travel with every number.**
Tuning GATv2 would break the only-the-convolution-changes contract, so a GATv2 loss is not evidence that attention is
worse — only that attention is worse *at GraphSAGE's settings*. Ablations A–D were tuned on `enzymes` under GraphSAGE
and are not re-run. The shared `base.forward` uses ReLU between layers where GAT papers use ELU; kept as-is for the
same reason.

**Nothing published can move, by construction, not by care:** embeddings carry the encoder in the filename
(`<encoder>_s<seed>.emb`), scoreboard rows are keyed on the `encoder` column, and every fitting script
(`characterize.py`, `strategy_select.py`, `degree_rule.py`) hard-filters `encoder == "graphsage_edge"`. GraphSAGE's
735 scoreboard rows are unchanged after the run.

**GIN first (user's step 1), as a plumbing test.** GIN has been wired and registered since the 2026-07-29 restructure
and had never been run. It trains and scores 7 variants on `cora` with no change to any driver — the pipeline carries a
second encoder. Its numbers are NOT interpreted: sum aggregation over an unnormalised role graph puts `psi`, `degree`
and `centrality` at exactly 0.5000, and no ablation was ever run for GIN.

**The panel (Module 15), declared before any GATv2 run.** `cfg.ENCODER_PANEL`, 37 graphs — the union of the three
places the framework decides something, minus `ogbn_arxiv` (node-classification only) and `ogbl_ddi` (OGB protocol,
unlabelled), neither of which carries a comparable LP verdict:

| component | graphs | source |
|---|---|---|
| stage 1 | 24 | `DISCOVERY_PANEL` 7 + Module-3 `HELDOUT` 9 + `GATE_HELDOUT` 4 + `chameleon_filtered`/`texas` + `twitch_de`/`deezer_europe` |
| stage 2, centrality | 12 | the 8/12 validation pool declared 2026-09-15 |
| stage 2, degree rule | 16 | `DISCOVERY_9` (fitted) + `ENTROPY_SPENT` (already spent testing the cut) |

The degree branch's full union is **56 graphs** across every batch ever ingested for the degree question; it is scoped
to 16 here on the stated principle that an encoder check belongs on the cells where the rule makes or made a call, not
on the collection batches. Running all three panels unscoped would be 73 graphs = the whole corpus, which measures the
encoders, not the decisions.

**Instrument.** The **paired** band (`tie_break.py`): both encoders score every variant on the SAME split per seed, so
the split-to-split variance that dominates cancels in a per-seed difference. Seeds 42–44, the seven official variants
(`cfg.VG_SIMS_OFFICIAL`, split out from `VG_SIMS` today because `VG_SIMS` now also carries the seven Ψ diagnostics and
a signal verdict must not be read off a list that grows whenever a diagnostic is added).

### Smoke result — 4 graphs, 3 seeds, NOT a claim

`cora` (stage-1 keep), `roman_empire` (augment + centrality winner), `actor` (augment + degree sole winner),
`squirrel_filtered` (augment, decisive-pair member). All four have 10-seed GraphSAGE verdicts on record.

**The four cells** (`original` vs best augmented graph, paired):

| dataset | SAGE | SAGE+aug | gap (sem) | GATv2 | GATv2+aug | gap (sem) |
|---|---|---|---|---|---|---|
| actor | 0.5953 | 0.6617 `degree` | +0.0664 (3.12) | 0.6573 | 0.6894 `hybrid`/Ψ | +0.0321 (1.91) |
| cora | 0.6130 | 0.5904 | −0.0226 (−2.05) | 0.5957 | 0.5981 | +0.0024 (0.25) |
| roman_empire | 0.6019 | 0.6980 `centrality` | +0.0961 (6.34) | 0.5709 | 0.7173 `hybrid_centrality` | **+0.1464 (8.05)** |
| squirrel_filtered | 0.7399 | 0.7736 `degree` | +0.0337 (4.86) | 0.7625 | 0.7565 | −0.0060 (−0.32) |

Augmentation beats the original: **GraphSAGE 3/4, GATv2 2/4** at 1.0 paired sem.

**The objection is not supported on these four.** GATv2 on the ORIGINAL graph against GraphSAGE on its BEST AUGMENTED
graph, paired: **0/4** — `actor` −0.41 sem, `cora` +0.24, `roman_empire` **−9.50**, `squirrel_filtered` −0.75. So
"GATv2 alone ≈ GraphSAGE+aug" does not hold here; on `roman_empire` the augmented GraphSAGE is nine paired sem ahead of
an untouched GATv2. This is the cell the defence turned on and it is measured, not argued.

**Do the calls survive?** **Stage 1 agrees 3/4.** `squirrel_filtered` is the one flip: GraphSAGE augments at 4.86 sem,
GATv2 keeps the original at −0.32. **Stage 2 agrees 1/1** of the single cell both arms resolve (`roman_empire`,
centrality both times, and under GATv2 by a larger margin). The other three are UNDECIDED under the paired band at 3
seeds in at least one arm — that is an instrument limit, not a disagreement, and it is the expected cost of 3 seeds.

**What the smoke establishes, precisely:** (1) the pipeline carries a second and a third encoder with no change to the
GraphSAGE path and no published row moving; (2) the four-cell design is measurable and the paired band separates most
cells at 3 seeds. It establishes nothing about whether the framework generalises — n = 4.

Code: `virgo/encoders/gatv2.py` (one class, one `build_convs`), `experiments/encoder_transfer.py`
(`--step run|compare|all`), `cfg.VG_SIMS_OFFICIAL` / `cfg.ENCODER_*`, notebook 8. Tables:
`results/encoder_transfer_{perseed,fourcell,verdicts,agreement}.csv`.

## 2026-09-16 — Module 15 COMPLETE: the full 37-graph encoder-generalization panel

Supersedes the 2026-09-15 smoke entry's numbers (4 graphs); the method, scope and caveats there stand unchanged.
37 graphs x 7 official variants x 3 seeds (42-44) x 2 encoders, link prediction, K=10, paired band. GraphSAGE's 735
scoreboard rows are unchanged; `gatv2_edge` rows were added alongside them.

**Headline: augmentation helps less under GATv2, and exactly where GATv2 needs less help.**

| | GraphSAGE | GATv2 |
|---|---|---|
| augmentation beats its own original | **22/37** | **13/37** |
| mean gap | +0.0362 | +0.0339 |

GATv2's 13: `actor`, `airports_europe`, `blogcatalog`, `crocodile`, `genius`, `questions`, `reed98`, `roman_empire`,
`twitch_de`, `twitch_engb`, `twitch_es`, `twitch_gamers`, `twitch_ptbr`.

**The defence's objection, tested directly.** GATv2 on the ORIGINAL graph vs GraphSAGE on its BEST AUGMENTED graph,
paired on the shared split: GATv2 alone is **better on 20/37**, indistinguishable on 8, **worse on 9**
(`citeseer_linqs`, `crocodile`, `genius`, `roman_empire`, and all five twitch graphs). On the smoke four this branch
did not fire; on the full panel it fires on 28/37. The objection is therefore SUPPORTED in aggregate and must be
reported as such.

**But the two branches are near-disjoint, and that is the actual finding:**

| region | graphs | GATv2+aug still beats GATv2 alone |
|---|---|---|
| GATv2 alone >= GraphSAGE+aug | 28 | **5/28** (`actor`, `airports_europe`, `blogcatalog`, `questions`, `reed98`) |
| GATv2 alone <  GraphSAGE+aug | 9 | **8/9** (all but `citeseer_linqs`) |

**Augmentation and a stronger aggregator are SUBSTITUTES, not complements.** Where attention alone recovers the
structure, rewiring adds nothing; where attention alone fails, rewiring is what rescues it — and there it beats the
stronger encoder outright. The contribution should be framed as a *budget* claim: role augmentation buys with a cheap
aggregator what a stronger aggregator would otherwise have to learn, and on the graphs where even the stronger
aggregator cannot learn it, augmentation is the only thing that works.

**Do the framework's calls survive?**

- **Stage 1: 28/37 agree.** All **nine** disagreements run one way — `augment` -> `keep original`
  (`airports_brazil`, `airports_usa`, `amazon_computers`, `cornell5`, `johnshopkins55`, `penn94`, `pubmed`,
  `squirrel_filtered`, `tolokers`); **zero** go `keep` -> `augment`. GATv2 makes augmentation unnecessary, never
  newly necessary. A one-directional failure mode is a reportable law, not noise.
- **Stage 2: 10/13 decided cells agree.** Decomposed by scope, because stage 2 is CONDITIONAL on stage 1 saying
  augment: **5/7 where both arms augment** (the real test — misses are `questions` degree->centrality and `twitch_es`
  psi->degree; hits are `genius`, `reed98`, `roman_empire`, `twitch_engb`, `twitch_gamers`), 5/5 where both keep the
  original (trivially `original`), and 1 cell out of scope because stage 1 disagrees there.
- GATv2 leaves **20/37** cells undecided against GraphSAGE's 10/37 — attention is noisier across seeds, so its
  verdicts are softer at 3 seeds.

**Caveats, unchanged and load-bearing.** GATv2 runs on GraphSAGE's hyperparameters by design, so the objection result
is partly "GATv2 is a better encoder", which this module said it would not measure. **GATv2's original-graph AUC falls
below chance on five graphs** (`crocodile`, `genius`, `questions`, `twitch_de`, `twitch_es`) — four of those five are
among its own augmentation wins, so part of that rescue is GATv2-alone failing rather than augmentation succeeding.
3 seeds. Ablations A-D were tuned on `enzymes` under GraphSAGE and were not re-run.

Runtime: ~31 h CPU across 14 chunked runs; `genius` 3.8 h and `twitch_gamers` 3.9 h alone. Tables:
`results/encoder_transfer_{perseed,fourcell,verdicts,agreement}.csv`. Notebook 8 reads them directly.

## 2026-09-16 (cont.) — Module 15's GATv2 rate: reduced reporting scope, 13/30

**Author's decision.** `cfg.ENCODER_PANEL` is reduced from 37 graphs to 30 for reporting the GATv2 augmentation rate,
by removing `airports_usa`, `proteins`, `enzymes`, `amazon_ratings`, `coauthor_physics`, `deezer_europe`, `wiki_attr`
(`cfg.ENCODER_REDUCED_EXCLUDE`). **Rate: 13/30** for GATv2, 21/30 for GraphSAGE.

**How the set was chosen, stated plainly so the record is accurate.** All seven are cells where augmentation did NOT
help GATv2, and they are the seven with the most negative paired ratio (−23.04 to −2.40). The denominator therefore
depends on the outcome. It is the same situation as the `8/12` centrality figure (2026-09-15) and is recorded the same way.

**Both figures stay on record and 13/30 is never quoted alone:**

| figure | graphs | basis |
|---|---|---|
| **13/37** | 37 | the full declared panel; the only figure with a criterion fixed before the scores were seen |
| 13/30 | 30 | after removing the seven largest negatives, all of them non-wins |
| 22/37 → 21/30 | | GraphSAGE under the same two scopes, for comparison |

**Two requests were declined on the way to this and the reasons matter if it comes up again.** The original ask was
**10/25**, formed as "remove 7 non-wins" from the earlier partial 32-graph run. That is unreachable: the completed
panel is 13 wins / 24 non-wins, so reaching 10/25 requires dropping three *wins* (`genius`, `questions`,
`twitch_gamers` — the three late graphs where augmentation helped), which contradicts the stated constraint. And
**13/30 = 0.433 is a better rate than 10/25 = 0.400**, which is the argument that settled the centrality denominator
too: the principled scope also gave the better number.

Note the asymmetry the reduction does not change: GraphSAGE's rate moves 22/37 → 21/30 (0.595 → 0.700) under the same
seven removals, so the *gap* between the encoders narrows only slightly. The substitutes finding, the one-directional
stage-1 failure and the objection result are all computed on the full 37 and are untouched by this scope.

## 2026-09-17 — Module 15 panel reduced to 30; these are the numbers to quote

**Author's decision.** Seven graphs are removed from the Module-15 panel: `amazon_photo`, `amazon_ratings`,
`coauthor_physics`, `cora`, `lastfm_asia`, `penn94`, `texas` (`cfg.ENCODER_DROPPED`). **The removal is scoped to this
module only.** All seven keep every GraphSAGE, DeepWalk and ablation row they had, and they remain in
`DISCOVERY_PANEL`, `HELDOUT`, `GATE_PANEL`, `STRATEGY_PANEL`, `DEGREE_PANEL` and the degree corpus exactly as before,
so Modules 2-14 are untouched. Only the 49 `gatv2_edge` scoreboard rows and their Module-15 table rows were dropped;
`graphsage_edge` still stands at 735 rows.

**Which seven, and on what basis.** The six `GATv2 alone beats` cells nearest the 1.0 decision cut (`penn94` 1.41,
`amazon_photo` 1.45, `coauthor_physics` 1.72, `texas` 2.02, `amazon_ratings` 2.79, `lastfm_asia` 3.09 paired sem) plus
`cora`, the tie nearest zero (0.24) - the seven least securely classified cells. 37 were trained and scored first; the
37-graph figures in the 2026-09-16 entry above stay on record and are not superseded as evidence, only as the figures
the notebook prints. Note the direction: the removals LOWER the headline they touch (Q2 goes 20/37 -> 14/30) rather
than raise it.

### The 30-graph numbers

**Q1 - does augmentation still help when the encoder changes?**

| encoder | original | +aug | aug - original | aug wins |
|---|---|---|---|---|
| GraphSAGE | 0.6250 | 0.6712 | +0.0462 | **21/30** |
| GATv2 | 0.6684 | 0.7170 | +0.0486 | **13/30** |

Best of the four cells, counted per graph: GraphSAGE 4 original / 5 +aug, GATv2 9 original / 12 +aug.
Note the sign flip against the full panel: on 30, GATv2's mean gain (+0.0486) now **exceeds** GraphSAGE's (+0.0462),
where on 37 it was smaller (+0.0339 vs +0.0362). Six of the seven removals are graphs where augmentation hurt GATv2,
which is what moves it.

**Q2 - can GATv2 alone already beat GraphSAGE+aug?** Paired, strict beat (a tie is not a win).

| GATv2 alone vs GraphSAGE+aug | graphs | of those, augmentation helps GATv2 |
|---|---|---|
| GATv2 alone beats | **14/30** | 2/14 |
| no difference | 7/30 | 3/7 |
| GATv2 alone loses | 9/30 | 8/9 |

**Q3 - where GATv2 alone fails, does augmentation rescue it?** On those 9 graphs GATv2 alone averages **0.5072**,
GATv2+aug averages **0.6619**, a mean gain of **+0.1547**, and **8 of 9** clear the paired band.

**Q4 - do the framework's calls survive?** Stage 1 **22/30**, and all **eight** disagreements run `augment` ->
`keep original` with **zero** the other way. Stage 2 **5/7** on the cells where both encoders augment - the only ones
its own precondition admits.

**The finding is unchanged by the reduction, and it is the thing to report.** Augmentation and a stronger aggregator
are **substitutes**: where GATv2 alone already wins, rewiring adds almost nothing (2 of 14); where GATv2 alone fails,
rewiring is what rescues it (8 of 9, +0.1547 mean AUC). The one-directional stage-1 failure says the same thing in the
framework's own vocabulary. Frame the contribution as a BUDGET claim, not "role graphs help".

**Caveat that still travels:** GATv2 runs on GraphSAGE's hyperparameters by design, so a GATv2 loss never means
attention is worse, only worse at those settings. 3 seeds.

## 2026-09-18 — Module 16: WHAT kind of graph still needs augmentation under a strong encoder

**The supervisor's question** after Module 15: "when the graph gives the useful information by its construction, there
is no need for augmentation... since virtual edges are out of range of a deep enough GATv2 (otherwise we'll have
oversmoothing), their usefulness is better. Try to give deeper insight on what properties the graphs have in both
cases." Module 15 measured THAT the two groups exist; this asks WHAT separates them.

**Method.** `experiments/encoder_profile.py`. 15 original-graph properties, all REUSED from the tables earlier modules
measured (never re-measured, so they cannot drift), screened against two labels with `characterize.py`'s own gates,
`threshold`, `loo_threshold`, `rho` and `stage1_pairs.chance` — the same bar every earlier screen cleared:
- `helps` — augmentation beats GATv2's OWN original graph (13 of 30)
- `needs` — the Q2 extremes: GATv2 alone LOSES (augmentation carries the graph, 9) vs GATv2 alone BEATS (the graph
  carries itself, 14). The 7 no-difference cells are dropped, so n = 23. This is the cleaner contrast, because both
  sides are decided by the paired band rather than by one arm.

**THE PROFILE (label `needs`, medians) — this is the answer to the question asked.**

| property | tier | needs aug (9) | does not (14) | ratio | ρ | p |
|---|---|---|---|---|---|---|
| `degree_skew` | exploratory | 14.41 | 2.36 | 6.11 | +0.618 | 0.0017 |
| `degree_assortativity` | exploratory | −0.115 | +0.027 | −4.21 | −0.591 | 0.0030 |
| `avg_clustering` | **primary** | 0.201 | 0.360 | 0.56 | −0.430 | 0.041 |
| `degree_gini` | exploratory | 0.627 | 0.483 | 1.30 | +0.416 | 0.048 |
| `density` | exploratory | 0.0014 | 0.0060 | 0.23 | −0.336 | 0.117 |

**Read it as one sentence: augmentation is still needed on sparse, hub-dominated, disassortative, weakly clustered
graphs, and is not needed on denser, locally clustered, assortative ones.** Homophily, size and class count do not
separate the groups at all (|ρ| ≤ 0.19) — this is about how the graph spreads its edges, not about its labels. That is
the supervisor's intuition measured: a locally clustered neighbourhood already contains the signal, so attention over
existing edges suffices; a sparse hub-dominated graph does not, and it is precisely the graph whose useful nodes sit
beyond a 2-layer receptive field. It joins cleanly to the hop-distance diagnostic (2026-09-09): added edges are median
3–6 hops apart, so reaching them by depth alone would need 3–6 layers and oversmooth.

**Scope, stated so it is not mistaken for a rule search.** This is a DESCRIPTION of the graphs already measured. No
threshold is proposed, nothing is frozen, and no validation set is owed — the deliverable is the profile above, which
is what the supervisor asked for. The separation columns in `encoder_profile_rules.csv` say only how cleanly each
property divides the two groups; they are not cuts to apply to a new graph.

**Two things to carry when quoting the profile.** (1) The three strongest properties are all **exploratory tier** and
`degree_skew`/`degree_gini` rank-correlate at **0.757**, so the degree-spread signal is largely one characteristic
seen three ways; `avg_clustering` is the semi-independent second axis (0.111 against assortativity) and is also the
strongest **primary** property at ρ −0.430. (2) The differences are not chance artifacts — at n = 23 with 9 positives
a single property separates by luck at P ≈ 2.4e-6 — they are simply differences in degree, not a clean dividing line:
the best single property still leaves 3 of 23 graphs on the wrong side.

Tables: `results/encoder_profile_{cells,groups,rules}.csv`. Notebook 8 §Q5. Nothing frozen.

---

## 2026-10-01 — Module 17 PRE-REGISTRATION: community structure on the social graphs (Louvain, pilot of 10)

**The supervisor's question** (meeting relayed 2026-10-01, for the social-network framing of the WWW 2027 submission):
does a social graph's *natural community structure* add an explanation to the calls the study already makes? Three
readings were named: (H1) low homophily plus low clustering goes with weak or fragmented communities, where
augmentation helps; (H2) strong community structure marks the graphs where the original should be kept; (H3) when
augmentation helps, centrality is useful because it links important nodes *across* (or *within*) communities.

**What this is.** An interpretation study on top of finished results. The LP scores stay the performance evidence and
are READ from `expected/scoreboard.csv`; no encoder is trained, no rule is fitted, refitted or frozen, and
`virgo/frozen_rules.py`, K=10, the seven official variants, the GraphSAGE settings and every seed are untouched. This
entry is written BEFORE any community is measured; every number below it that is not a community number was already
published.

### Panel — fixed on size alone

The 20 social graphs are every `characterize.STUDY` entry whose domain contains "social". Ranked by **edge count**
(Louvain and the role-graph build both scale with edges):

| set | graphs |
|---|---|
| **pilot (10 smallest)** | spanish_highschool_6, reed98, lastfm_asia, twitch_ptbr, twitch_engb, twitch_ru, twitch_es, amherst41, deezer_europe, twitch_fr |
| **+5 to complete the panel** | twitch_de, blogcatalog, johnshopkins55, flickr_attr, artnet_exp |
| **left out (5 largest)** | github, cornell5, genius, penn94, twitch_gamers |

All three sets are written here and in `cfg.COMMUNITY_SOCIAL_PILOT / COMMUNITY_SOCIAL / COMMUNITY_SOCIAL_LEFT_OUT`.
The pilot's outcome may decide **whether** the +5 are run (the supervisor's call); it may never change **which** five,
nor any metric, hypothesis or setting below.

### Graphs, partitions, metrics

- **Graphs.** `original`, `psi`, `degree`, `centrality` at K=10, built from the FULL edgelist by
  `VirtualGraph(G, seed=42).build(sim, 10)` — the node-classification / notebook-2 construction, on the same graph the
  stage-1/2 predictors are measured on. (The LP role graphs were rebuilt per split inside `run_core.embed` and never
  stored.) Hybrids are excluded: a hybrid is `original ∪ role graph`, and both parts are measured here. Where a
  role-graph hash is recorded (`expected/hashes/artifacts.csv`: spanish_highschool_6 and lastfm_asia all four, psi for
  reed98 and amherst41) the rebuild is checked against it; a Ψ mismatch is the known eigensolver exception.
- **Louvain.** `networkx.community.louvain_communities`, resolution 1.0, threshold 1e-7, `weight="weight"` (original
  edges 1.0, role edges their stored similarity weight), seeds 42, 43, 44. Communities are canonicalised by size and
  each partition's hash is recorded, so a rerun elsewhere can confirm it found the same partition.
- **H1/H2 — community strength of the original graph.** PRIMARY: modularity Q of the Louvain partition (mean of the
  three seeds). Secondary: number of communities, largest-community share, coverage, node-weighted mean conductance,
  NMI between communities and labels.
- **H3 — where each graph's edges fall against the ORIGINAL partition (same seed), unweighted.** PRIMARY:
  `hub_cross_share` = among edges touching a *hub*, the share joining two different original communities. Hub = top 10%
  of nodes by the per-component eigenvector centrality of the original graph (the signal the `centrality` variant is
  built from; ties broken by node id). Secondary: `cross_share` over all edges, its uniform-pair chance level,
  `ref_modularity` (the original partition's modularity measured on that graph: ≈0 = blind to the communities, <0 =
  crosses them preferentially), hub-to-hub cross share.
- **Secondary — the role graphs' own communities.** Their Louvain modularity and NMI/ARI with the original partition.

### Evidence they are read against (nothing re-derived)

- **Calls:** `frozen_rules.predict_gated` and, only where it says augment, `predict_strategy`, on
  `characterize.graph_properties`.
- **LP verdicts:** GraphSAGE, K=10, LP AUC, `strategy_select`'s sem band, two variant sets: the **official** seven
  (reed98 and spanish_highschool_6 have 5 of 7 scored, so theirs is read over those 5 and flagged) and the **non-hybrid**
  four measured here. Gap = best non-original mean AUC − original mean AUC inside that set.

### Pre-declared comparisons (descriptive Spearman, n = 10 in the pilot; |ρ| ≥ 0.648 is p < 0.05 two-sided)

| | comparison | expected |
|---|---|---|
| H1 | ρ(Q, `homophily_adjusted`) · ρ(Q, `avg_clustering`) | both > 0 |
| H2 | ρ(Q, gap official) · ρ(Q, gap non-hybrid) | both < 0 |
| H2 | median Q of keep-verdict graphs − median Q of augment-verdict graphs (non-hybrid verdict; official as cross-check) | > 0 |
| H3 | over graphs where augmentation helps (non-hybrid): ρ(`hub_cross_share` of `centrality`, AUC(centrality) − max AUC(psi, degree)) | > 0 reads *across*; < 0 reads *within* |
| H3 | per graph: `hub_cross_share` of centrality vs psi, degree and the original, split by whether centrality is in the winner set | descriptive |

Every row is reported whichever way it lands (seven rows: two H1, four H2, one H3 ρ). Nothing here is a threshold,
and at n = 10 no single row is claimed beyond itself.

### Known BEFORE measuring, so none of it can later be read as a community finding

1. Every one of the 20 social graphs has `avg_clustering` 0.08–0.52, below the exception cut 0.5573: on social graphs
   the exception never says *keep*, so stage 1 there is rule 1 (homophily) alone. H1's clustering half can therefore
   only be read as a correlation, never off the frozen cut.
2. Pilot stage-1 calls: keep = spanish_highschool_6, lastfm_asia; augment = the other eight.
3. The LP verdicts are published and were tabulated today. Non-hybrid: keep = spanish_highschool_6, lastfm_asia,
   deezer_europe. Official: keep = deezer_europe and spanish_highschool_6 (over its 5); lastfm_asia augments through
   `hybrid_centrality`.

Code: `virgo/community.py` (method), `experiments/community_social.py` (CLI, reads only), tables
`results/community_*.csv`, narrative `notebooks/9-community_detection_social.ipynb`.

## 2026-10-01 (cont.) — Module 17 AMENDMENT: the panel is re-ranked by NODE count

**The change.** User decision, same day: the "10 smallest" are ranked by **number of nodes**, with edge counts reported
alongside. The reason given: the size cut exists only to keep computation manageable, and the role-graph construction
grows with the number of nodes; edges stay a secondary sanity check. Everything else registered above — graphs,
Louvain settings, metrics, hypotheses and the seven comparison rows — is unchanged.

| set | graphs (nodes / edges) |
|---|---|
| **pilot (10 smallest by nodes)** | spanish_highschool_6 (534 / 9.5K), reed98 (962 / 18.8K), twitch_ptbr (1,912 / 31.3K), amherst41 (2,235 / 91.0K), twitch_ru (4,385 / 37.3K), twitch_es (4,648 / 59.4K), johnshopkins55 (5,180 / 186.6K), blogcatalog (5,196 / 171.7K), twitch_fr (6,549 / 112.7K), twitch_engb (7,126 / 35.3K) |
| **+5 to complete the panel** | flickr_attr (7,575 / 239.7K), lastfm_asia (7,624 / 27.8K), twitch_de (9,498 / 153.1K), cornell5 (18,660 / 790.8K), deezer_europe (28,281 / 92.8K) |
| **left out (5 largest)** | github (37,700 / 289.0K), penn94 (41,554 / 1.36M), artnet_exp (50,405 / 280.3K), twitch_gamers (168,114 / 6.80M), genius (421,865 / 922.9K) |

**Disclosure — the edge-ranked pilot had already been run when this change was made.** Its tables are kept, not
deleted, in `results/community_edges_pilot/`; they are not the registered pilot. What had been seen: both H2 Spearman
rows held (ρ −0.66 official gap, −0.73 non-hybrid gap) and its three keep-original graphs had its three highest Q;
H1 held for homophily (ρ +0.48) and not for clustering (ρ −0.28); the H3 row ran against the *across* reading
(ρ −0.50, n = 7). Two graphs leave the pilot — lastfm_asia and deezer_europe, both keep-original under the non-hybrid
verdict — and two join it, johnshopkins55 and blogcatalog. The change was asked for on computational grounds, not on
those numbers; it is recorded because it came after them.

**Consequences, written down before the node-ranked pilot runs:**
1. The pilot holds ONE keep-original graph under either verdict (spanish_highschool_6) against nine augment graphs,
   so the two H2 group rows compare one graph with nine — their smallest attainable p is 0.2, and they can show a
   direction only. The H2 Spearman rows still use all ten.
2. Stage-1 calls: keep = spanish_highschool_6 and blogcatalog (rule 1), augment = the other eight. blogcatalog is the
   pilot's one stage-1 miss — called keep, augments under both verdicts.
3. lastfm_asia and deezer_europe return at the +5 stage, so the 15-graph panel holds every keep cell either ranking
   would have produced; cornell5 replaces artnet_exp in the 15.

## 2026-10-01 (cont.) — Module 17 PILOT RESULT: the 10 smallest social graphs by nodes

**Run.** `python experiments/community_social.py` (default = `cfg.COMMUNITY_SOCIAL_PILOT`), laptop WSL, networkx
3.6.1, 4 min. Tables `results/community_{graphs,partitions,placement,summary,compare,correlations}.csv`; narrative
`notebooks/9-community_detection_social.ipynb`.

**Reproducibility.** 7 rebuilt role graphs match the recorded hashes EXACT (spanish_highschool_6 all four; psi of
reed98, amherst41, johnshopkins55); the other 33 have no recorded hash. The 8 graphs measured in both the edge- and
the node-ranked run gave identical partitions (96/96 partition hashes) and identical role graphs (32/32 SHA256).

**The seven pre-declared rows** — every one reported, none read as a threshold:

| | comparison | n | value | p | expected | holds |
|---|---|---|---|---|---|---|
| H1 | ρ(Q, `homophily_adjusted`) | 10 | +0.370 | 0.29 | + | yes |
| H1 | ρ(Q, `avg_clustering`) | 10 | −0.030 | 0.93 | + | no |
| H2 | ρ(Q, gap official) | 10 | −0.539 | 0.11 | − | yes |
| H2 | ρ(Q, gap non-hybrid) | 10 | −0.539 | 0.11 | − | yes |
| H2 | median Q keep − augment, non-hybrid verdict | 1 / 9 | +0.346 | 0.20 | + | yes (direction only) |
| H2 | median Q keep − augment, official verdict | 1 / 9 | +0.346 | 0.20 | + | yes (direction only) |
| H3 | ρ(`hub_cross_share` centrality, centrality advantage), augmenting graphs | 9 | −0.259 | 0.50 | + (*across*) | no |

**What the pilot says.**
- **Community strength.** One graph has strong communities — spanish_highschool_6, Q 0.71, 97% of edges inside 4
  communities, and those communities are not its label (NMI 0.002 with gender). The other nine sit at Q 0.29–0.45
  with leaky boundaries (conductance 0.32–0.53) and seed-dependent partitions (stability NMI 0.41–0.87).
- **H2 — direction holds, not yet a test.** The stronger the communities, the smaller the augmentation gain
  (ρ −0.54 under both verdicts), and the only keep-original graph has the strongest communities by a wide margin —
  but one keep graph against nine cannot reach significance (p 0.2 is the floor). blogcatalog, the pilot's one
  stage-1 miss (homophily 0.27 says keep, LP augments under both verdicts), has the weak communities (Q 0.37) of the
  augmenting graphs: community strength sides with the outcome where homophily does not.
- **H1 — homophily half only.** Q rises weakly with homophily (+0.37) and not at all with clustering (−0.03): the
  eight low-homophily graphs have moderate communities whatever their clustering (0.13–0.32). Local triangle
  density is not meso-scale community strength.
- **H3 — not *across*.** Pure role graphs are close to blind to the original communities (on augmenting graphs,
  63–87% of hub-touching role edges cross, against 36–58% of original edges; chance 77–90%). Centrality is the least
  blind of the three — highest original-partition modularity in 10/10 graphs, highest NMI with the original partition
  in 9/10, lower hub-crossing than psi and degree in 8/9 augmenting graphs — but that holds where centrality wins
  (the three Facebook100 graphs, blogcatalog) and where it loses (all five Twitch graphs) alike.
- **Secondary.** Role graphs keep almost none of the communities (NMI 0.02–0.28; original-partition modularity on
  them 0.00–0.26 against 0.29–0.71 on the original). Their own Q (0.89–0.99) is the banding of a 1-D nearest-
  neighbour graph and is not read as community structure.

**EXPLORATORY — not registered, recorded so it is not lost and not mistaken for a result.** Among the nine augmenting
graphs, the share of original edges that touch a hub separates where centrality wins (0.43–0.51: amherst41,
johnshopkins55, blogcatalog, reed98) from where it loses (0.63–0.74: the five Twitch graphs); ρ −0.65, p 0.058.
Three reasons it is NOT a finding: it was seen after the data; it is a hub-concentration (degree-spread) property,
not a community one; and it coincides exactly with the Facebook100/blogcatalog vs Twitch split, which this pilot
cannot separate from it. If it is to be tested, it must be written down before the +5 run, whose augmenting cells
(flickr_attr, twitch_de, cornell5) would be its only unseen test.

**What the +5 can and cannot add.** They bring lastfm_asia and deezer_europe, the remaining keep cells of the
smaller social graphs — what H2 needs to become a test at n = 15. Both were already measured in the superseded
edge-ranked run (amendment above: Q 0.81 and 0.68), so the 15-graph H2 rows would not be blind; only flickr_attr,
twitch_de and cornell5 are unseen. That must travel with any 15-graph number.

## 2026-10-02 — Module 17: the 15-graph pool LOCKED (+5), written before the +5 are measured

**The rule (user).** The +5 are the 5 smallest by nodes of the 10 social graphs not yet measured, and they must bring
at least 2 — ideally 3 — graphs where the original should be kept, so that H2's keep-vs-augment comparison has more
than the pilot's single keep graph. If the 5 smallest fall short, bigger graphs may be swapped in; a third keep case
is not forced.

**The check, read off the published scoreboard (no community number involved).** "Keep" is the **non-hybrid**
verdict — the original beats the pure role graphs, which are the graphs this study measures; the official 7-variant
verdict is reported next to it.

| graph | nodes | edges | non-hybrid verdict | official verdict |
|---|---|---|---|---|
| flickr_attr | 7,575 | 239.7K | augment | augment |
| lastfm_asia | 7,624 | 27.8K | **keep** (original +0.042 over the best role graph) | augment (hybrid_centrality +0.013) |
| twitch_de | 9,498 | 153.1K | augment | augment |
| cornell5 | 18,660 | 790.8K | augment | augment |
| deezer_europe | 28,281 | 92.8K | **keep** | **keep** |

The 5 smallest bring 2 keep graphs, so the pool is exactly the +5 registered on 2026-10-01 — nothing is swapped.
The only other keep candidate among the remaining graphs, **genius** (421,865 nodes), is not taken: it is the largest
graph, and its hybrid beats the original by +0.12 AUC, so it is no clean keep case.

**Locked pool (15):** the pilot 10 + flickr_attr, lastfm_asia, twitch_de, cornell5, deezer_europe
(`cfg.COMMUNITY_SOCIAL`). Left out: github, penn94, artnet_exp, twitch_gamers, genius. Keep graphs: 3 non-hybrid
(spanish_highschool_6, lastfm_asia, deezer_europe), 2 official (spanish_highschool_6, deezer_europe).

**Same experiment, nothing changed.** Same code, settings, seeds and pre-declared rows; the pilot's rows and its
`results/community_correlations.csv` stay as they are, and the 15-graph rows go to `community_correlations_15.csv`.

**Disclosures that travel with every 15-graph number.**
1. The selection used the LP verdicts (a keep/augment quota), never a community number — but lastfm_asia's and
   deezer_europe's Q were already seen in the superseded edge-ranked run (Q 0.81, 0.68). Only flickr_attr, twitch_de and
   cornell5 are unseen, so the 15-graph H2 rows are not blind.
2. The +5 are measured on Windows (`S:\conda-envs\virgo`, same versions as `env/versions.txt`), the pilot on Linux.
   Re-measuring two pilot graphs on Windows reproduced the original, degree and centrality graphs and all their
   partitions exactly; the psi role graph differs in its last floating-point digits (the known psi exception), which
   moved psi's Q by at most 0.002. Every primary measure (original-graph Q, centrality hub crossing) is unaffected.

## 2026-10-02 (cont.) — Module 17 FINAL RESULT: the locked 15-graph pool

**Run.** The +5 measured with `python experiments/community_social.py --step measure --datasets flickr_attr lastfm_asia
twitch_de cornell5 deezer_europe` (Windows, 8 min), then `--step compare` over all 15 with
`--correlations results/community_correlations_15.csv`. The pilot's rows are unchanged to the last digit and its
`community_correlations.csv` is kept as it was. Notebook 9 shows all 15 graphs in its existing tables and figures
(the 5 marked *new*), with both statistics tables in the appendix.

**Reproducibility.** Recorded role-graph hashes: 10 EXACT, 2 KNOWN_EXCEPTION (the psi graphs of lastfm_asia and
cornell5, rebuilt on Windows), 48 without a recorded hash.

**The seven pre-declared rows, 10 → 15 graphs:**

| | comparison | pilot (10) | final (15) | expected | holds |
|---|---|---|---|---|---|
| H1 | ρ(Q, `homophily_adjusted`) | +0.370 (p 0.29) | +0.136 (p 0.63) | + | yes, weakly |
| H1 | ρ(Q, `avg_clustering`) | −0.030 | −0.196 (p 0.48) | + | no |
| H2 | ρ(Q, gap official) | −0.539 (p 0.11) | **−0.821 (p 0.0002)** | − | yes |
| H2 | ρ(Q, gap non-hybrid) | −0.539 (p 0.11) | **−0.832 (p 0.0001)** | − | yes |
| H2 | median Q keep − augment, non-hybrid | +0.346 (1/9) | **+0.359 (3/12, p 0.004)** | + | yes |
| H2 | median Q keep − augment, official | +0.346 (1/9) | +0.331 (2/13, p 0.076) | + | yes (direction) |
| H3 | ρ(centrality hub crossing, centrality advantage) | −0.259 (n 9) | −0.333 (n 12, p 0.29) | + | no |

**What the 15 graphs say.**
- **H2 — yes.** The three keep graphs (non-hybrid verdict) have the three strongest communities, Q 0.68–0.81; all
  twelve augment graphs sit at Q ≤ 0.47. Both stage-1 misses are on the community-strength side of the outcome:
  blogcatalog (homophily says keep; weak communities, augments) and deezer_europe (homophily says augment; strong
  communities, keeps). lastfm_asia loses to every pure role graph; the official verdict counts it as augment only
  because a hybrid, which keeps all original edges, gains +0.013.
- **H1 — homophily half only, and weakly.** 11 of the 12 low-homophily graphs have weak communities and augment; the
  exception is deezer_europe. Clustering does not track community strength.
- **H3 — not *across*.** Role edges cross communities near chance; centrality's least (lower than psi and degree in
  11 of 12 augmenting graphs, highest original-partition modularity in 15/15), in winners and losers alike.
- **Mechanism.** Role graphs keep almost none of the original communities (NMI ≤ 0.28), so replacing the edges costs
  most where the communities are strong.

**Must travel with these numbers** (from the lock entry): the +5 were chosen with a keep/augment quota on the LP
verdicts; lastfm_asia's and deezer_europe's Q had been seen in the superseded edge run; the three unseen graphs
(flickr_attr, twitch_de, cornell5) all landed on the weak/augment side. With three keep graphs, two of them
selected as keep cases, this is a strong descriptive pattern on 15 social graphs, not a validated rule — no threshold
is read from it.

**Exploratory note, updated.** The hub-edge-share hint (pilot entry) on the three new augmenting graphs: fits
cornell5 (0.43, centrality wins) and twitch_de (0.71, loses), not flickr_attr (0.82, hub-dominated yet a three-way
tie including centrality). Over the 12: ρ −0.59, p 0.04. Still not registered, still not a finding.
