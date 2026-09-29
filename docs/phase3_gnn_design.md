# Phase 3 — GraphSAGE encoder: architecture variants (v2, for discussion)

**Locked decisions (2026-07-04).** Encoder = **GraphSAGE trained unsupervised**, with a **Skipgram-analog objective** — the same role Skipgram plays in the I2V paper, so the GNN is a true drop-in for the walk+Skipgram back end. The study runs **on the virtual graphs** (starting with Ikenna's Poisson/KL Ψ graph); the original graph appears only as a small control row, not a focus. Output stays a 64-dim `.emb`, so all existing evaluation scripts run unchanged.

---

## 1. Core architecture (the locked GraphSAGE encoder)

```
original graph ──► cached structural signals ──► Ψ top-K virtual graph      (Phase 2, done)
                        (deg, Ω, Δ)                      │
                                                         ▼
                    structural features X ──► GraphSAGE (2 layers, mean) ──► z (64-d)
                                                         │
                             Skipgram-analog unsupervised loss:
                             positives = walk co-occurrence ON the virtual graph
                             negatives = degree^0.75 sampling (Q=5, matches I2V negative=5)
                                                         │
                                                         ▼
                                              .emb ──► eval_nodeclass / eval_linkpred (unchanged)
```

- **Input features `X`** (nodes have no attributes; structural-only keeps the method structural + inductive): `[degree, eigenvector centrality Ω, ψ score, clustering coefficient]`, z-normalized. All but clustering already cached in `identity2vec_cached.Graph`.
- **Loss** (GraphSAGE paper's unsupervised objective = Skipgram with the embedding-lookup replaced by the GNN):
  `L(u) = −log σ(z_u·z_v) − Σ_{i=1..Q} log σ(−z_u·z_{n_i})`, positive `v` co-occurs with `u` within window 10 on walks over the **virtual graph**; `Q=5` negatives ∝ deg^{3/4}.
- **Key comparability property:** generate the walk corpus with the exact I2V params already used by the Phase-2 DeepWalk bridge (num_walks=10, walk_length=40, window=10, seeds 42/43/44). Then Phase-2 bridge vs Phase-3 SAGE differ in **one thing only** — Skipgram's lookup table vs GraphSAGE message passing — the cleanest possible "replace Skipgram with a GNN" ablation.

---

## 2. Variant axes (the menu to discuss)

### A — Positive-pair source (what "context" means)
| | Variant | Tests |
|---|---|---|
| **A1 (default)** | walk co-occurrence on the virtual graph, I2V-matched corpus | faithful Skipgram replacement; isolates encoder effect vs Phase-2 bridge |
| A2 | 1-hop contrastive: positives = direct virtual neighbors, no walks | are walks still needed once similarity is *explicit* in the graph? (I2V needed walks to *find* role-similar nodes; the virtual graph already lists them) |

### B — Aggregation (how neighbors are combined)
| | Variant | Tests |
|---|---|---|
| **B1 (default)** | mean | standard SAGE |
| B2 | **Ψ-weighted mean** (weights = `1/(1+dist)` on virtual edges) | do similarity *strengths* carry signal beyond top-K membership? (Phase-2 bridge ignored them — open caveat) |
| B3 | max-pool | most-similar-neighbor dominance |
| B4 | sum | GIN-style expressivity inside the SAGE frame |

### C — Depth
1 / **2 (default)** / 3 layers. Virtual graphs are near-connected already at K=10 (Ψ: 2 components) → deeper nets risk over-smoothing fast; depth curve is a cheap, publishable ablation.

### D — Input features
**D1 (default)** structural-4 · D2 degree-only · D3 random/constant (does message passing over the *right graph* suffice even with uninformative inputs?)

### E — Message-passing graph (the Phase-2 variable, rerun under the GNN)
Ψ (start) / degree / centrality × K ∈ {5,10,20}; original-graph control row; **optional/stretch:** dual-branch (virtual + original fused) — proposed as follow-up, kept out of the headline study to keep "which virtual graph?" clean.

---

## 3. Proposed run order (Cora first, 3 seeds each)

1. **Stage 1 — lock the encoder** (Ψ, K=10): A1 vs A2 × B1 vs B2 → 4 configs. Pick winner.
2. **Stage 2 — the study:** winning config × {Ψ, degree, centrality} × K {5,10,20} → the "which virtual graph?" table, now under a GNN. Compare against the Phase-2 DeepWalk-bridge table (encoder effect) and I2V (headline).
3. **Stage 3 — ablations:** depth (C), features (D), original-graph control (E).
4. **Phase 4:** repeat winner matrix on the remaining 3 datasets + anomaly detection.

Fixed training details (proposal): hidden=64, out=64, full-batch (largest graph ~19.5k nodes — fits), Adam lr=0.01, epochs ≈ 20 with loss-plateau stop, dropout 0, seeds 42/43/44.

---

## 4. Implementation shape (once approved)

`encoder.py`, mirroring `train.py` per repo rules: one class `SageEncoder` (holds model + loss + walk-corpus reuse), exposes **`train(epochs)`** and writes `.emb`; `argparse` CLI `--sim {psi,degree,centrality} --k --positives {walk,1hop} --agg {mean,wmean,max,sum} --layers --seed`. Reads `output/<ds>/k<K>/virtual_<sim>.edgelist` (PyG `from_networkx`), writes `output/<ds>/k<K>/sage_<sim>_s<seed>.emb`. Env ready: torch 2.12.0 + torch-geometric 2.8.0.

---

## 5. Open questions for review

1. **A1 vs A2 as the headline loss** — we propose A1 (faithful Skipgram replacement, clean vs Phase-2); A2 kept as ablation. Agree?
2. **Dual-branch (virtual + original)** — include in this paper or defer to follow-up? We propose defer.
3. **Features** — structural-only OK? (Some datasets ship no node attributes, so structural-only also maximizes dataset coverage.)
