'''Characterization: MEASURE each dataset and the inputs GraphSAGE is fed (portion 1), then RELATE those properties to when augmentation helps (portion 2).'''
# Two families, kept separate on purpose:
#   Family 1 (graph characterization) = one row per ORIGINAL graph: size, density, degree spread, clustering,
#            assortativity, label homophily. Used to EXPLAIN results; never fed to the encoder.
#   Family 2 (node structural inputs) = the raw, pre-z-normalization distribution of the four per-node features
#            that ARE fed to GraphSAGE (degree, eigenvector centrality, Psi, clustering).
# Step 1 additionally freezes the locked GraphSAGE scores this characterization will later be joined against.
# Portion 2 then measures the original-vs-augmented gap INSIDE each dataset x task cell (so the metric is a constant
# that cancels), rank-correlates those gaps against the properties, and screens each primary property as an executable
# keep-vs-augment rule - descriptive only, n is one point per dataset, and a screened rule is a CANDIDATE awaiting
# validation on unseen datasets, never a proven one.
# Everything is read from artifacts that already exist - dataset edgelists, the encoder's feature caches, and
# results/scoreboard.csv - so nothing is recomputed and nothing can drift from what the study actually ran.

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import skew, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import graph_io
from virgo.encoders import feature_cache
from virgo.frozen_rules import DISCOVERY_PANEL, HELDOUT, STRATEGY_HELDOUT   # the locked fitting set + the two held-out sets: threshold-fitting may never see a dataset outside DISCOVERY_PANEL

# The eight study datasets. Domain / tasks / scope are DECLARED metadata (they cannot be measured from an edgelist);
# every other column in Family 1 is measured. scope records which graph the row describes: ogbl-ddi ships training edges only.
# roman_empire + tolokers added 2026-07-27: the first six are all homophilous, so the augment side of the rule went untested.
STUDY = {
    "cora":           {"domain": "citation",         "tasks": "NC+LP", "scope": "full"},
    "citeseer_linqs": {"domain": "citation",         "tasks": "NC+LP", "scope": "full"},
    "enzymes":        {"domain": "biological",       "tasks": "NC+LP", "scope": "full"},
    "proteins":       {"domain": "biological",       "tasks": "NC+LP", "scope": "full"},
    "ogbn_arxiv":     {"domain": "citation",         "tasks": "NC",    "scope": "full"},
    "ogbl_ddi":       {"domain": "drug interaction", "tasks": "LP",    "scope": "ogb_train_edges"},
    "roman_empire":   {"domain": "linguistic",       "tasks": "NC+LP", "scope": "full"},
    "tolokers":       {"domain": "crowdsourcing",    "tasks": "NC+LP", "scope": "full"},
    "questions":      {"domain": "q&a interaction",  "tasks": "NC+LP", "scope": "full"},
    # Module-3 held-out (2026-08-02), added to STUDY so the pipeline can measure them; NOT in DISCOVERY_PANEL (fixed literal).
    "pubmed":         {"domain": "citation",         "tasks": "NC+LP", "scope": "full"},
    "actor":          {"domain": "film",             "tasks": "NC+LP", "scope": "full"},
    "minesweeper":    {"domain": "grid",             "tasks": "NC+LP", "scope": "full"},
    "amazon_photo":   {"domain": "co-purchase",      "tasks": "NC+LP", "scope": "full"},
    "lastfm_asia":    {"domain": "music social",     "tasks": "NC+LP", "scope": "full"},   # added 2026-08-04
    "amazon_ratings": {"domain": "co-purchase",      "tasks": "NC+LP", "scope": "full"},   # added 2026-08-05; Platonov's heterophilous co-purchase graph, NOT amazon_photo
    "squirrel_filtered": {"domain": "wikipedia",     "tasks": "NC+LP", "scope": "full"},   # added 2026-08-05; de-duplicated Squirrel, avg degree ~42 = by far the densest in the study
    # LINKX Facebook100 college networks (2026-08-14), registered so the pipeline can measure them; held out, never fitted on.
    "reed98":         {"domain": "college social",   "tasks": "NC+LP", "scope": "full"},
    "amherst41":      {"domain": "college social",   "tasks": "NC+LP", "scope": "full"},
    "johnshopkins55": {"domain": "college social",   "tasks": "NC+LP", "scope": "full"},
    "cornell5":       {"domain": "college social",   "tasks": "NC+LP", "scope": "full"},
    # Rebuilt 2026-08-31 as the Module-8 test set: never in any fitting panel, scores kept in results/module7_withdrawn.csv.
    "chameleon_filtered": {"domain": "wikipedia",     "tasks": "NC+LP", "scope": "full"},
    "texas":          {"domain": "web pages",        "tasks": "NC+LP", "scope": "full"},
    # 2026-09-01 degree-evidence batch
    "wisconsin":      {"domain": "web pages",        "tasks": "NC+LP", "scope": "full"},
    "cornell_webkb":  {"domain": "web pages",        "tasks": "NC+LP", "scope": "full"},
    "chameleon":      {"domain": "wikipedia",        "tasks": "NC+LP", "scope": "full"},
    "twitch_de":      {"domain": "social",           "tasks": "NC+LP", "scope": "full"},
    "deezer_europe":  {"domain": "social",           "tasks": "NC+LP", "scope": "full"},
    # 2026-09-02 DEGREE_RULE_CORPUS (separate stage-2 pool)
    "airports_usa":    {"domain": "air transport",       "tasks": "NC+LP", "scope": "full"},
    "airports_europe": {"domain": "air transport",       "tasks": "NC+LP", "scope": "full"},
    "airports_brazil": {"domain": "air transport",       "tasks": "NC+LP", "scope": "full"},
    "email_eu_core":   {"domain": "email",               "tasks": "NC+LP", "scope": "full"},
    "wikics":          {"domain": "wikipedia",           "tasks": "NC+LP", "scope": "full"},
    "twitch_engb":     {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "github":          {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "amazon_computers": {"domain": "co-purchase",         "tasks": "NC+LP", "scope": "full"},
    "polblogs":        {"domain": "web pages",           "tasks": "NC+LP", "scope": "full"},
    "blogcatalog":     {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "coauthor_cs":     {"domain": "coauthorship",        "tasks": "NC+LP", "scope": "full"},
    "twitch_es":       {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "twitch_fr":       {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "dblp":            {"domain": "paper citation",      "tasks": "NC+LP", "scope": "full"},
    "coauthor_physics": {"domain": "coauthorship",        "tasks": "NC+LP", "scope": "full"},
    "twitch_ru":       {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "twitch_ptbr":     {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "cora_ml":         {"domain": "paper citation",      "tasks": "NC+LP", "scope": "full"},
    "squirrel":        {"domain": "wikipedia",           "tasks": "NC+LP", "scope": "full"},
    "flickr_attr":     {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    # 2026-09-03 DEGREE_RULE_VALIDATION (the corpus's held-out half; measured here, never fitted on)
    "wiki_attr":       {"domain": "wikipedia",           "tasks": "NC+LP", "scope": "full"},
    "crocodile":       {"domain": "wikipedia",           "tasks": "NC+LP", "scope": "full"},
    "cora_full":       {"domain": "paper citation",      "tasks": "NC+LP", "scope": "full"},
    "penn94":          {"domain": "college social",      "tasks": "NC+LP", "scope": "full"},
    "genius":          {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    "twitch_gamers":   {"domain": "social",              "tasks": "NC+LP", "scope": "full"},
    # 2026-09-04 FROZEN_DEGREE test candidates
    "artnet_exp":      {"domain": "art social",          "tasks": "NC+LP", "scope": "full"},
    "city_reviews":    {"domain": "business reviews",    "tasks": "NC+LP", "scope": "full"},
    "hm_categories":   {"domain": "co-purchase",         "tasks": "NC+LP", "scope": "full"},
    "city_roads_m":    {"domain": "road network",        "tasks": "NC+LP", "scope": "full"},
    "city_paris":      {"domain": "road network",        "tasks": "NC+LP", "scope": "full"},
    "city_shanghai":   {"domain": "road network",        "tasks": "NC+LP", "scope": "full"},
    "city_la":         {"domain": "road network",        "tasks": "NC+LP", "scope": "full"},
    # 2026-09-05 DEGREE_TEST (Netzschleuder)
    "celegans_neural":     {"domain": "neural",          "tasks": "NC+LP", "scope": "full"},
    "messal_shale":        {"domain": "food web",        "tasks": "NC+LP", "scope": "full"},
    "plant_pol_robertson": {"domain": "pollination",     "tasks": "NC+LP", "scope": "full"},
    "escorts":             {"domain": "sexual contact",  "tasks": "NC+LP", "scope": "full"},
    "nematode_mammal":     {"domain": "host-parasite",   "tasks": "NC+LP", "scope": "full"},
    "bag_of_words_kos":    {"domain": "document-word",   "tasks": "NC+LP", "scope": "full"},
    # 2026-09-05 DEGREE_BATCH2
    "plant_pol_kato":      {"domain": "pollination",     "tasks": "NC+LP", "scope": "full"},
    "board_directors":     {"domain": "corporate board", "tasks": "NC+LP", "scope": "full"},
    "foursquare_checkin":  {"domain": "check-in",        "tasks": "NC+LP", "scope": "full"},
    "foursquare_tips":     {"domain": "check-in",        "tasks": "NC+LP", "scope": "full"},
    "ppi_rat":             {"domain": "protein",         "tasks": "NC+LP", "scope": "full"},
    "ppi_mouse":           {"domain": "protein",         "tasks": "NC+LP", "scope": "full"},
    # 2026-09-07, cfg.DEGREE_FAMILY_TEST: the Module-14 family validation set (mist PPI / genetic interaction networks).
    "ppi_human":            {"domain": "protein",         "tasks": "NC+LP", "scope": "full"},
    "ppi_fly":              {"domain": "protein",         "tasks": "NC+LP", "scope": "full"},
    "ppi_yeast":            {"domain": "protein",         "tasks": "NC+LP", "scope": "full"},
    "genetic_fission_yeast": {"domain": "protein",        "tasks": "NC+LP", "scope": "full"},
    # 2026-09-08, cfg.DEGREE_FAMILY_TEST2: the non-PPI half of the family test.
    "jdk":                  {"domain": "software",        "tasks": "NC+LP", "scope": "full"},
    "spanish_highschool_6": {"domain": "social",          "tasks": "NC+LP", "scope": "full"},
    "bag_of_words_nips":   {"domain": "document-word",   "tasks": "NC+LP", "scope": "full"},
}

# THE ACTIVE PANEL is the FIXED literal DISCOVERY_PANEL (virgo/frozen_rules.py), NOT derived here: citeseer_linqs and
# proteins were dropped 2026-07-29, but the panel must never widen just because a dataset is added to STUDY. Both panels
# must name registered datasets, so the study can actually load them.
assert set(DISCOVERY_PANEL) <= set(STUDY), f"discovery panel names an unregistered dataset: {sorted(set(DISCOVERY_PANEL) - set(STUDY))}"
assert set(HELDOUT) <= set(STUDY), f"held-out set names an unregistered dataset: {sorted(set(HELDOUT) - set(STUDY))}"
assert set(STRATEGY_HELDOUT) <= set(STUDY), f"strategy held-out set names an unregistered dataset: {sorted(set(STRATEGY_HELDOUT) - set(STUDY))}"

FEATURES = ["degree", "eigenvector_centrality", "psi", "clustering"]   # the cached column order, set by GNNEncoder.features()

# One primary metric per task: the OGB pair is scored under its official protocol, so its metric name differs by design.
PRIMARY = {"node classification (weighted F1)": "weighted_f1", "link prediction (AUC)": "auc",
           "node classification (OGB official)": "test_acc", "link prediction (OGB official)": "test_hits@20"}

# A feature is flagged degenerate when it can barely distinguish nodes: almost no distinct values, or almost all zero.
# The flag is DESCRIPTIVE ONLY - portion 2 judges a feature by its ablation score (feature_scores), never by this boolean.
DEGENERATE = {"pct_unique": 1.0, "pct_zero": 95.0}

# --- portion 2 ---------------------------------------------------------------------------------------------------
# How each non-original variant changes the graph: hybrid ADDS role edges to the real ones, the rest REPLACE them.
KIND = {"hybrid": "additive", "hybrid_degree": "additive", "hybrid_centrality": "additive",
        "psi": "replacement", "degree": "replacement", "centrality": "replacement"}

# One variant fixed per task, so the gap can also be read without the max-over-variants selection bias.
# POST-HOC: each is simply the variant that already wins most cells in the max-gap table (6/8 apiece), not a principled choice.
FIXED_VARIANT = {"node classification": "hybrid", "link prediction": "centrality"}

# A cell is unusable when the metric cannot resolve ANY variant apart and no seed moves - e.g. questions NC, where a
# 97% majority class pins weighted F1 (spread 1e-4, std 0). Such a cell carries no signal and is excluded from portion 2 (D).
DEGENERATE_CELL = {"spread": 1e-3, "std": 1e-6}

# Graph properties tested as predictors of the gap, in two TIERS. Only PRIMARY predictors may gate a candidate rule and
# only they carry the multiple-comparison correction; exploratory ones are still measured and reported, never promoted.
# homophily_adjusted / nbr_predictability_adjusted are the label properties: raw values are not comparable across
# datasets, since their chance floor moves with class count and balance.
PREDICTORS_PRIMARY = ["homophily_adjusted", "nbr_predictability_adjusted", "components", "largest_component_frac",
                      "avg_degree", "avg_clustering", "n_classes"]
PREDICTORS_EXPLORATORY = ["edge_homophily", "nbr_label_entropy", "density", "degree_gini", "degree_skew",
                          "degree_assortativity", "majority_class_frac", "nodes"]
PREDICTORS = PREDICTORS_PRIMARY + PREDICTORS_EXPLORATORY

# Predictors that measure the SAME THING are collapsed into one family so the credible list cannot count one finding twice.
# `components` and `largest_component_frac` rank the panel's graphs at Spearman -1.000: identical information, no exception.
# The canonical member is the scale-free one - "< 39.5 components" does not transfer to a graph ten times larger, whereas
# "> 0.9588 of nodes in one component" does, and a rule is only worth stating if it can be applied to an unseen dataset.
FAMILY = {"components": "fragmentation", "largest_component_frac": "fragmentation"}
CANONICAL = {"fragmentation": "largest_component_frac"}

# Screening gates for portion 2 (D). Deliberately NOT significance-gated: at n=8 the two-tailed 0.05 critical Spearman
# value is 0.738, so |rho| >= 0.7 already sits at about p <= 0.07, and demanding p as well would reject usable patterns
# on this few datasets. Corrected p is reported alongside instead. Passing means "credible CANDIDATE", never "proven rule" -
# the proof is Module 3, a pre-registered prediction on unseen datasets.
GATES = {"min_abs_rho": 0.7, "max_exceptions": 1, "min_cells": 4, "loo_above_majority": True}

# Ablation-D encoder names -> arm, and single-feature arms -> the Family 2 feature they switch on.
ARMS = {"graphsage_edge": "all", "features_only": "none_mp",
        **{f"graphsage_edge_feat_{a}": a for a in ["degree", "centrality", "psi", "clustering", "deg_cent", "random", "const"]}}
ARM_FEATURE = {"degree": "degree", "centrality": "eigenvector_centrality", "psi": "psi", "clustering": "clustering"}


# Reads the raw feature matrix the pipeline cached for a graph - the exact numbers GraphSAGE was fed, before z-normalization.
def load_cache(ds, G):
    '''Cached [degree, centrality, psi, clustering] matrix, ordered to match G's nodes.'''
    path = feature_cache(str(cfg.dataset(ds)["edgelist"]))
    assert Path(path).exists(), f"{path} missing -> run the pipeline for {ds} first (the encoder writes this cache)"
    z = np.load(path)
    f = dict(zip(z["nodes"].tolist(), z["X"]))          # keyed by node id, so file order can never misalign the rows
    assert set(f) == set(G.nodes), f"{ds}: cached node set differs from the graph - the cache is stale"
    return np.array([f[n] for n in G.nodes])


# Node -> label map, restricted to nodes that are actually in the graph (label files may list dropped isolated nodes).
def labels_of(labels, G):
    '''Read a label file into {node: label} for nodes present in G; empty dict when the dataset is unlabelled.'''
    if labels is None or not Path(labels).exists():
        return {}
    lab = {}
    for line in open(labels):
        p = line.split()
        if len(p) >= 2 and int(p[0]) in G:
            lab[int(p[0])] = p[1]
    return lab


# Gini of the degree sequence: 0 = every node has the same degree, 1 = one node holds every edge.
def gini(x):
    '''Gini coefficient of a non-negative sequence.'''
    x = np.sort(np.asarray(x, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


# Do connected nodes share a label? Reported three ways plus the chance baseline that makes them comparable across datasets.
def homophily(G, lab):
    '''(edge homophily, node homophily, adjusted homophily, chance null): adjusted removes the class-count/imbalance baseline (Platonov et al. degree-weighted null) so scores compare across datasets.'''
    if not lab:
        return float("nan"), float("nan"), float("nan"), float("nan")
    same = tot = 0
    deg_c = {}                                           # per-class labelled-degree D_c, for the degree-weighted null
    for u, v in G.edges:
        if u in lab and v in lab:
            tot += 1
            same += lab[u] == lab[v]
            deg_c[lab[u]] = deg_c.get(lab[u], 0) + 1
            deg_c[lab[v]] = deg_c.get(lab[v], 0) + 1
    per_node = [np.mean([lab[w] == lab[n] for w in G[n] if w in lab])
                for n in G.nodes if n in lab and any(w in lab for w in G[n])]
    edge_h = same / tot if tot else float("nan")
    null = sum((d / (2 * tot)) ** 2 for d in deg_c.values()) if tot else float("nan")     # Σ_c (D_c/2m)², m = tot
    adj_h = (edge_h - null) / (1 - null) if tot and null < 1 else float("nan")
    return (edge_h, float(np.mean(per_node)) if per_node else float("nan"), adj_h, null)


# Beyond homophily: even when a node's neighbours carry OTHER labels, do they carry a CONSISTENT mix that reveals its class?
# Homophily only asks "same label?"; this asks "informative label mix?", which is the candidate explanation for heterophilous graphs.
def neighbour_predictability(G, lab):
    '''(adjusted accuracy, raw accuracy, normalized neighbour-label entropy): read a node's label off its neighbours' class histogram, leave-one-out.'''
    classes = sorted(set(lab.values())) if lab else []
    nodes = [n for n in G.nodes if n in lab and any(w in lab for w in G[n])] if lab else []
    if len(classes) < 2 or not nodes:
        return float("nan"), float("nan"), float("nan")
    ci = {c: i for i, c in enumerate(classes)}
    V = np.zeros((len(nodes), len(classes)))                     # per node: how many neighbours of each class
    for r, n in enumerate(nodes):
        for w in G[n]:
            if w in lab:
                V[r, ci[lab[w]]] += 1
    y = np.array([ci[lab[n]] for n in nodes])
    M = np.zeros((len(classes), len(classes)))
    np.add.at(M, y, V)                                           # compatibility matrix: class -> the class mix it connects to
    counts = np.bincount(y, minlength=len(classes))
    prior = np.log(counts / len(y) + 1e-12)
    logp = lambda A: np.log((A + 1) / (A + 1).sum(-1, keepdims=True))    # Laplace-smoothed log P(neighbour class | node class)
    score = V @ logp(M).T + prior                                # naive Bayes over the neighbour histogram
    loo = logp(M[y] - V)                                         # leave-one-out: this node's own contribution removed from its class row
    score[np.arange(len(y)), y] = (V * loo).sum(1) + prior[y]
    acc, major = float((score.argmax(1) == y).mean()), float(counts.max() / len(y))
    p = V / V.sum(1, keepdims=True)
    ent = float(np.mean(-(p * np.log(np.where(p > 0, p, 1.0))).sum(1)) / np.log(len(classes)))   # 0 log 0 = 0
    return (round((acc - major) / (1 - major), 4) if major < 1 else float("nan")), round(acc, 4), round(ent, 4)


# FAMILY 1: one row describing the original graph a dataset ships - the properties that will explain the results.
def graph_properties(ds):
    '''Measure one dataset's original graph: size, connectivity, degree spread, triangles, mixing, label agreement.'''
    d, meta = cfg.dataset(ds), STUDY[ds]
    G = graph_io.load_graph(d["edgelist"])
    p = graph_io.properties(G)                                    # shared definition: nodes/edges/components/isolates/...
    deg = np.array([k for _, k in G.degree], dtype=float)
    clustering = load_cache(ds, G)[:, FEATURES.index("clustering")]   # already computed by the pipeline; exact, and free
    lab = labels_of(d["labels"], G)
    edge_h, node_h, adj_h, null_h = homophily(G, lab)
    pred_adj, pred_raw, nbr_ent = neighbour_predictability(G, lab)
    counts = pd.Series(list(lab.values())).value_counts(normalize=True) if lab else pd.Series(dtype=float)
    return {
        "dataset": ds, "domain": meta["domain"], "tasks": meta["tasks"], "graph_scope": meta["scope"],
        "directed_source": bool(d.get("directed_source", False)),
        "nodes": p["nodes"], "edges": p["edges"],
        "density": round(2 * p["edges"] / (p["nodes"] * (p["nodes"] - 1)), 6),
        "avg_degree": round(float(deg.mean()), 4), "max_degree": int(deg.max()),
        "degree_std": round(float(deg.std()), 4), "degree_skew": round(float(skew(deg)), 4),
        "degree_gini": round(gini(deg), 4), "distinct_degrees": p["distinct_degrees"],
        "components": p["components"], "largest_component_frac": p["largest_component_frac"],
        "isolates": p["isolates"],
        "avg_clustering": round(float(clustering.mean()), 4),
        "degree_assortativity": round(float(nx.degree_assortativity_coefficient(G)), 4),
        "n_classes": int(len(counts)) if lab else 0,
        "majority_class_frac": round(float(counts.iloc[0]), 4) if lab else float("nan"),
        "edge_homophily": round(edge_h, 4) if edge_h == edge_h else float("nan"),
        "node_homophily": round(node_h, 4) if node_h == node_h else float("nan"),
        "homophily_null": round(null_h, 4) if null_h == null_h else float("nan"),
        "homophily_adjusted": round(adj_h, 4) if adj_h == adj_h else float("nan"),
        "nbr_predictability": pred_raw, "nbr_predictability_adjusted": pred_adj, "nbr_label_entropy": nbr_ent,
    }


# FAMILY 2 (step 3): how each structural input is distributed BEFORE z-normalization, which is where the differences live.
def feature_properties(ds):
    '''One row per (dataset, feature): mean/std/skew/zeros/unique/min/max of the raw values fed to GraphSAGE.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    X = load_cache(ds, G)
    rows = []
    for i, feat in enumerate(FEATURES):
        c = X[:, i]
        pct_zero = round(100 * float(np.mean(np.isclose(c, 0.0))), 2)
        pct_unique = round(100 * len(np.unique(c)) / len(c), 4)
        rows.append({
            "dataset": ds, "feature": feat, "graph_scope": STUDY[ds]["scope"], "nodes": len(c),
            "mean": round(float(c.mean()), 6), "std": round(float(c.std()), 6),
            "skew": round(float(skew(c)), 4) if c.std() > 0 else 0.0,
            "pct_zero": pct_zero, "pct_unique": pct_unique,
            "min": round(float(c.min()), 6), "max": round(float(c.max()), 6),
            "degenerate": bool(pct_unique < DEGENERATE["pct_unique"] or pct_zero > DEGENERATE["pct_zero"]),
        })
    return rows


# STEP 1: freeze the experimental side, so the characterization is joined against a fixed table, not a live one.
def frozen_inputs(datasets):
    '''Snapshot the locked GraphSAGE (K=10, 3-seed) primary score for every dataset x task x graph variant.'''
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    b = board[(board["encoder"] == "graphsage_edge") & (board["top_K_neighbors"] == 10)
              # VG_SIMS_LOCKED, not VG_SIMS: the frozen rules were fitted on those five variants, and the gap is a max
              # OVER variants - letting a newly promoted variant in would silently refit Module 2 on a wider set.
              & board["graph_variant"].isin(cfg.VG_SIMS_LOCKED) & board["dataset"].isin(datasets)]
    b = b[[PRIMARY.get(t) == m for t, m in zip(b["task"], b["metric"])]]   # one primary metric per task, never mixed
    return (b[["dataset", "task", "graph_variant", "metric", "mean", "std", "seeds"]]
            .sort_values(["dataset", "task", "graph_variant"]).reset_index(drop=True))


# Spearman on the pairs where both sides exist; rank-based, so it is unaffected by the differing metric scales.
def rho(x, y):
    '''(n, Spearman rho, p) over the non-missing pairs, or (n, nan, nan) when there is too little to rank.'''
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return int(ok.sum()), float("nan"), float("nan")
    r, p = spearmanr(x[ok], y[ok])
    # scipy's t-approximation diverges at |rho| = 1 (n=5 gives p=1e-24), but the smallest p ANY permutation of n points
    # can produce is 2/n!. Floor there so a perfect rank match on five datasets cannot be quoted as overwhelming evidence.
    p = max(float(p), 2 / float(math.factorial(int(ok.sum()))))
    return int(ok.sum()), round(float(r), 4), round(p, 4)


# Shared reading of one variant against the original: signed gap, that gap in units of pooled 3-seed noise, and the verdict.
def compare(orig, row):
    '''(gap, relative gap, gap in sigma, verdict) for one augmented variant against the original.'''
    gap = float(row["mean"] - orig["mean"])
    noise = float(np.hypot(orig["std"], row["std"]) / np.sqrt(2))     # pooled 3-seed std of the two sides
    return (round(gap, 4), round(gap / float(orig["mean"]), 4), round(gap / noise, 2) if noise else float("nan"),
            "augment" if gap > noise else ("tie" if gap > -noise else "keep original"))


# PORTION 2 (A): the quantity to be explained - inside one cell, does any augmented graph beat the original graph?
# Reported twice: over the BEST variant (the headline, but max-over-4 is biased upward) and over ONE variant fixed per
# task (FIXED_VARIANT), which carries no selection bias. A rule that only holds for the max-gap is visibly the weaker one.
def gaps(inputs):
    '''Per dataset x task: original vs best-augmented and vs the task's fixed variant, each with gap, sigma and verdict.'''
    rows = []
    for (ds, task), g in inputs.groupby(["dataset", "task"]):
        s = g.set_index("graph_variant")
        assert "original" in s.index, f"{ds} / {task}: no original row to compare against"
        orig, aug = s.loc["original"], s.drop(index="original")
        family = task.split(" (")[0]
        best = aug["mean"].idxmax()
        gap, rel, sigma, verdict = compare(orig, aug.loc[best])
        fixed = FIXED_VARIANT[family]
        f_gap, f_rel, f_sigma, f_verdict = compare(orig, s.loc[fixed])
        rows.append({
            "dataset": ds, "task": task, "task_family": family, "metric": orig["metric"],
            "original": float(orig["mean"]), "best_augmented": float(aug.loc[best, "mean"]),
            "best_variant": best, "best_variant_kind": KIND[best],
            "gap_abs": gap,                                            # native units - comparable only within this cell
            "gap_rel": rel,                                            # scale-free, so gaps can be RANKED across cells
            "gap_sigma": sigma,
            "original_rank": int(s["mean"].rank(ascending=False)["original"]),
            "verdict": verdict,
            "fixed_variant": fixed, "fixed_augmented": float(s.loc[fixed, "mean"]),
            "gap_fixed_abs": f_gap, "gap_fixed_rel": f_rel, "gap_fixed_sigma": f_sigma,
            "verdict_fixed": f_verdict, "verdict_agrees": bool(f_verdict == verdict),
            # Unusable = no variant separates from any other AND no seed moves: the metric has no resolution in this cell.
            "usable": not (float(s["mean"].max() - s["mean"].min()) < DEGENERATE_CELL["spread"]
                           and float(s["std"].max()) < DEGENERATE_CELL["std"]),
        })
    return pd.DataFrame(rows).sort_values(["task_family", "dataset"]).reset_index(drop=True)


# PORTION 2 (B): the ablation replaces the degenerate flag - a feature is worth what it scores against random features.
# Classifies one ablation arm against the random-feature control. The feature column EXPLAINS a verdict, it never decides one.
# Band is 2 sigma, deliberately stricter than the 1 sigma gaps() uses for keep/augment: this table is descriptive, and
# "clearly worse than random" should not be claimed on a margin that three seeds cannot resolve.
def status(lift, sigma, flat):
    '''Ablation arm vs the random-feature control: beats / worse than / tied with random, plus not measured and metric constant.'''
    if not np.isfinite(lift):                         # no random-feature control on disk -> the arm was never run
        return "not measured"
    if flat:                                          # every arm ties every other AND seeds never move -> metric has no resolution
        return "metric constant"
    if not np.isfinite(sigma):
        return "tied with random"
    return "beats random" if sigma > 2 else ("worse than random" if sigma < -2 else "tied with random")


def feature_scores(fam2):
    '''Ablation D (psi graph, K=10): each feature arm's score, its lift over the random-feature control, and that feature's raw spread.'''
    board = pd.read_csv(cfg.SCOREBOARD_CSV)
    b = board[(board["graph_variant"] == "psi") & (board["top_K_neighbors"] == 10) & board["encoder"].isin(ARMS)
              & board["dataset"].isin(fam2["dataset"].unique())].copy()   # the scoreboard holds every dataset ever run; report only the panel
    b = b[[PRIMARY.get(t) == m for t, m in zip(b["task"], b["metric"])]]
    b["arm"] = b["encoder"].map(ARMS)
    b["task_family"] = [t.split(" (")[0] for t in b["task"]]
    ctrl = b[b["arm"] == "random"].set_index(["dataset", "task"])["mean"]          # D4: message passing, zero structural signal
    ctrl_sd = b[b["arm"] == "random"].set_index(["dataset", "task"])["std"]
    b["lift_vs_random"] = [round(m - ctrl.get((d, t), float("nan")), 4) for d, t, m in zip(b["dataset"], b["task"], b["mean"])]
    # Same 1-sigma band as gaps(): a lift only counts when it clears the pooled 3-seed noise of the arm and its control.
    noise = [np.hypot(s, ctrl_sd.get((d, t), float("nan"))) / np.sqrt(2) for d, t, s in zip(b["dataset"], b["task"], b["std"])]
    b["lift_sigma"] = [round(l / n, 2) if n and np.isfinite(n) and n > 0 else float("nan") for l, n in zip(b["lift_vs_random"], noise)]
    # Degenerate cell: no arm separates from any other AND no seed moves -> a constant predictor, e.g. a 97/3 majority collapse.
    flat = ((b.groupby(["dataset", "task"])["mean"].transform(lambda s: s.max() - s.min()) < 1e-3)
            & (b.groupby(["dataset", "task"])["std"].transform("max") < 1e-6))
    b["vs_random_features"] = [status(l, s, f) for l, s, f in zip(b["lift_vs_random"], b["lift_sigma"], flat)]
    b["feature"] = b["arm"].map(ARM_FEATURE)                                        # set only for the single-feature arms
    raw = fam2.rename(columns={"std": "raw_std", "skew": "raw_skew", "pct_zero": "raw_pct_zero", "pct_unique": "raw_pct_unique"})
    return (b.merge(raw[["dataset", "feature", "raw_std", "raw_skew", "raw_pct_zero", "raw_pct_unique"]], on=["dataset", "feature"], how="left")
            [["dataset", "task", "task_family", "arm", "feature", "metric", "mean", "std", "lift_vs_random",
              "lift_sigma", "vs_random_features", "raw_std", "raw_skew", "raw_pct_zero", "raw_pct_unique"]]
            .sort_values(["task_family", "dataset", "arm"]).reset_index(drop=True))


# PORTION 2 (C): rank-correlate both scopes. n is tiny at the graph scope (one point per dataset), so this is descriptive, not a test.
# Correction is Bonferroni WITHIN the primary tier only (that is the whole point of tiering); exploratory rows carry no corrected p.
def correlate(fam1, gap, feat):
    '''Spearman rho for two scopes: graph property vs the augmentation gap, and a feature's raw spread vs its ablation lift.'''
    rows = []
    joined = gap[gap["usable"]].merge(fam1, on="dataset")
    for name, g in list(joined.groupby("task_family")) + [("pooled", joined)]:
        for p in PREDICTORS:
            tier = "primary" if p in PREDICTORS_PRIMARY else "exploratory"
            for target in ["gap_rel", "gap_fixed_rel"]:
                n, r, pv = rho(g[p], g[target])
                adj = round(min(1.0, pv * len(PREDICTORS_PRIMARY)), 4) if tier == "primary" and pv == pv else float("nan")
                rows.append({"scope": "graph property -> augmentation gap", "task_family": name, "predictor": p,
                             "tier": tier, "target": target, "n": n, "spearman_rho": r, "p_value": pv,
                             "p_bonferroni": adj, "survives_correction": bool(adj == adj and adj < 0.05)})
    single = feat[feat["feature"].notna()]                                          # only arms that switch on one feature
    for name, g in list(single.groupby("task_family")) + [("pooled", single)]:
        for p in ["raw_std", "raw_pct_zero", "raw_pct_unique", "raw_skew"]:
            n, r, pv = rho(g[p], g["lift_vs_random"])
            rows.append({"scope": "feature spread -> ablation lift", "task_family": name, "predictor": p,
                         "tier": "exploratory", "target": "lift_vs_random", "n": n, "spearman_rho": r,
                         "p_value": pv, "p_bonferroni": float("nan"), "survives_correction": False})
    return pd.DataFrame(rows)


# Each gap column is screened against the verdict that was derived from it, never against the other one.
TARGETS = {"gap_rel": "verdict", "gap_fixed_rel": "verdict_fixed"}


# Turns a predictor into an executable decision: one split point, and which side of it says "augment".
# The split is the MIDPOINT between the two datasets straddling the boundary - a max-margin choice, but the data only
# ever identify the INTERVAL between those two datasets, so the interval is returned alongside and is what gets reported.
def threshold(x, y):
    '''Best midpoint split of x against the binary target y: (split, the side that augments, error count, accuracy, and the value interval every equally-good split lies in).'''
    u = np.unique(x)
    if len(u) < 2:
        return float("nan"), "", int(min((y == 0).sum(), (y == 1).sum())), float("nan"), float("nan"), float("nan")
    cuts = (u[:-1] + u[1:]) / 2
    keys = [(i, s) for i in range(len(cuts)) for s in ("high", "low")]
    err = {k: int(((x > cuts[k[0]] if k[1] == "high" else x < cuts[k[0]]) != y).sum()) for k in keys}
    i, side = min(keys, key=lambda k: err[k])          # ties break on the first cut, "high" before "low"
    e, lo, hi = err[(i, side)], i, i
    while lo > 0 and err[(lo - 1, side)] == e:         # widen across neighbouring cuts that separate exactly as well
        lo -= 1
    while hi < len(cuts) - 1 and err[(hi + 1, side)] == e:
        hi += 1
    return (round(float(cuts[i]), 4), side, e, round(1 - e / len(y), 4), round(float(u[lo]), 4), round(float(u[hi + 1]), 4))


# The threshold above is fitted and scored on the SAME cells, so "0 exceptions" is guaranteed whenever the classes separate
# at all - it is not evidence. This refits the split with one dataset hidden and predicts that dataset, once per dataset.
def loo_threshold(x, y):
    '''Leave-one-out refit of the split: (accuracy, correct, folds scored, lowest and highest cutoff the folds produced).'''
    ok, cuts = 0, []
    for i in range(len(x)):
        m = np.ones(len(x), dtype=bool)
        m[i] = False
        if len(np.unique(x[m])) < 2 or len(np.unique(y[m])) < 2:      # a fold that cannot be fitted is skipped, not scored
            continue
        t, side, _, _, _, _ = threshold(x[m], y[m])
        cuts.append(t)
        ok += int(((x[i] > t) if side == "high" else (x[i] < t)) == y[i])
    if not cuts:
        return float("nan"), 0, 0, float("nan"), float("nan")
    return round(ok / len(cuts), 4), ok, len(cuts), round(min(cuts), 4), round(max(cuts), 4)


# loo_threshold only debiases the SPLIT. The predictor itself was chosen by looking at every cell, so that choice is still
# fitted on the test data. This hides a dataset and re-runs the WHOLE screen - predictor choice included - inside each fold,
# which is the only number that answers "what would this study have predicted for a dataset it had never seen".
def nested_loo(fam1, gap, allow_refit=False):
    '''One row per held-out dataset: the predictor and split the remaining datasets choose, and whether they call it right.'''
    _assert_panel(fam1["dataset"], allow_refit)
    joined = gap[gap["usable"]].merge(fam1, on="dataset")
    rows = []
    for family, g in joined.groupby("task_family"):
        for target, vcol in TARGETS.items():
            d = g[g[vcol] != "tie"].reset_index(drop=True)
            y = (d[vcol] == "augment").to_numpy()
            if len(d) < GATES["min_cells"] or not y.any() or y.all():
                continue
            for i in range(len(d)):
                tr, pick = d.drop(i), None
                for p in PREDICTORS_PRIMARY:
                    s = tr[tr[p].notna()]
                    yt = (s[vcol] == "augment").to_numpy()
                    if s[p].nunique() < 2 or not yt.any() or yt.all():
                        continue
                    t, side, err, _, _, _ = threshold(s[p].to_numpy(), yt)
                    r = rho(s[p], s[target])[1]
                    key = (err, -abs(r) if r == r else 0.0)                   # fewest errors, then the stronger correlation
                    if pick is None or key < pick[0]:
                        pick = (key, p, t, side)
                v = float(d.loc[i, pick[1]]) if pick else float("nan")
                # A missing value is scored as a miss, not skipped: a rule that cannot be evaluated on a dataset (ogbl_ddi
                # has no labels, so no homophily) has genuinely failed to predict it.
                pred = "n/a" if not pick or v != v else ("augment" if (v > pick[2] if pick[3] == "high" else v < pick[2]) else "keep original")
                rows.append({"task_family": family, "target": target, "held_out": d.loc[i, "dataset"],
                             "picked_predictor": pick[1] if pick else "", "threshold": pick[2] if pick else float("nan"),
                             "augment_side": pick[3] if pick else "", "value": round(v, 4) if v == v else float("nan"),
                             "predicted": pred, "actual": d.loc[i, vcol], "correct": bool(pred == d.loc[i, vcol])})
    return pd.DataFrame(rows)


# Guards the two functions that FIT thresholds. Fitting on a dataset outside the frozen panel would let an unseen Module-3
# graph move the very rule it is meant to test - the data leak that makes validation circular. Predict such graphs with
# experiments/predict_module3.py instead; pass allow_refit=True only to re-fit the panel itself on purpose.
def _assert_panel(datasets, allow_refit):
    '''Raise unless every dataset is inside the frozen fitting panel (or a deliberate re-fit was requested).'''
    extra = sorted(set(datasets) - set(DISCOVERY_PANEL))
    assert allow_refit or not extra, (f"threshold fitting must stay on the discovery panel {DISCOVERY_PANEL}; {extra} are outside it. "
                                       "Predict them with experiments/predict_module3.py, or pass --allow-refit to re-fit deliberately.")


# PORTION 2 (D): screen every PRIMARY predictor as a binary keep-vs-augment rule, per task family.
# Gates are |rho|, leave-one-dataset-out sign stability and threshold errors - see GATES for why significance is not one of them.
# A predictor whose task family has no augment cell at all cannot be screened: there is nothing to separate, and it fails on errors.
def candidate_rules(fam1, gap, allow_refit=False):
    '''One row per task family x predictor x gap definition: rho, LODO stability, best threshold, exceptions, credible flag.'''
    _assert_panel(fam1["dataset"], allow_refit)
    joined = gap[gap["usable"]].merge(fam1, on="dataset")
    rows = []
    for family, g in joined.groupby("task_family"):
        for target, vcol in TARGETS.items():
            for p in PREDICTORS_PRIMARY:
                n, r, pv = rho(g[p], g[target])
                lodo = [rho(g[p].drop(i), g[target].drop(i))[1] for i in g.index]
                lodo = [v for v in lodo if v == v]
                stable = bool(lodo) and r == r and all(np.sign(v) == np.sign(r) for v in lodo)
                d = g[(g[vcol] != "tie") & g[p].notna()]                    # ties carry no decision to reproduce
                y = (d[vcol] == "augment").to_numpy()
                fit = len(d) >= GATES["min_cells"] and y.any() and not y.all()
                t, side, err, acc, lo, hi = (threshold(d[p].to_numpy(), y) if fit
                                             else (float("nan"), "", -1, float("nan"), float("nan"), float("nan")))
                l_acc, l_ok, l_n, l_lo, l_hi = (loo_threshold(d[p].to_numpy(), y) if fit
                                                else (float("nan"), 0, 0, float("nan"), float("nan")))
                major = round(float(max(y.mean(), 1 - y.mean())), 4) if len(y) else float("nan")
                fam = FAMILY.get(p, p)
                rows.append({
                    "task_family": family, "target": target, "predictor": p,
                    "predictor_family": fam, "canonical": bool(CANONICAL.get(fam, p) == p),
                    "n_cells": n, "n_decided": len(d), "n_augment": int(y.sum()),
                    # How many DISTINCT predictor values each side of the split spans. 1 on the augment side means every
                    # augmenting graph carries the SAME value, so the "threshold" is a group label, not a graded trend.
                    "distinct_values": int(d[p].nunique()),
                    "distinct_augment": int(d[p][y].nunique()),
                    "spearman_rho": r, "p_value": pv,
                    "p_bonferroni": round(min(1.0, pv * len(PREDICTORS_PRIMARY)), 4) if pv == pv else float("nan"),
                    "lodo_min_abs_rho": round(min(abs(v) for v in lodo), 4) if lodo else float("nan"),
                    "lodo_sign_stable": stable,
                    "threshold": t, "augment_side": side, "n_exceptions": err, "accuracy": acc,
                    # The panel pins an INTERVAL, not a point: every cut between these two values fits it exactly as well.
                    "interval_lo": lo, "interval_hi": hi,
                    # Out-of-sample: split refitted with each dataset hidden. `accuracy` above is in-sample and cannot fail.
                    "loo_accuracy": l_acc, "loo_correct": l_ok, "loo_folds": l_n,
                    "loo_threshold_lo": l_lo, "loo_threshold_hi": l_hi,
                    "majority_baseline": major,
                    "loo_beats_majority": bool(l_acc == l_acc and l_acc > major),
                    "rule": f"{family}: augment when {p} is {side} ({'>' if side == 'high' else '<'} {t})" if side else "",
                    # n_exceptions is in-sample and cannot fail on separable data, so the out-of-sample check gates too:
                    # a split that predicts a hidden dataset no better than always guessing the majority verdict is not a rule.
                    "credible": bool(r == r and abs(r) >= GATES["min_abs_rho"] and stable
                                     and 0 <= err <= GATES["max_exceptions"] and n >= GATES["min_cells"]
                                     and (l_acc == l_acc and l_acc > major if GATES["loo_above_majority"] else True)),
                })
    return pd.DataFrame(rows).sort_values(["task_family", "target", "predictor"]).reset_index(drop=True)


# PORTION 2 (D): the reading of the three tables - the when-to-augment rule the study set out to produce.
def rule(gap, corr, feat, cand, nest):
    '''Print the when-to-augment conclusion: per-cell verdicts, the strongest predictor per task, best feature per cell.'''
    print("\nVERDICT PER CELL (1 sigma band on the 3-seed noise; gap_fixed drops the max-over-variants selection bias)")
    print(gap[["dataset", "task_family", "metric", "original", "best_augmented", "best_variant", "gap_abs", "gap_sigma",
               "verdict", "fixed_variant", "gap_fixed_abs", "gap_fixed_sigma", "verdict_fixed", "usable"]].to_string(index=False))
    print("\n  " + " | ".join(f"{v}: {n}" for v, n in gap["verdict"].value_counts().items()))
    if (~gap["usable"]).any():
        print("  excluded (metric has no resolution): " + ", ".join(f"{r.dataset}/{r.task_family}" for r in gap[~gap["usable"]].itertuples()))
    if (~gap["verdict_agrees"]).any():
        print("  best-variant and fixed-variant verdicts DISAGREE: " +
              ", ".join(f"{r.dataset}/{r.task_family} ({r.verdict} vs {r.verdict_fixed})" for r in gap[~gap["verdict_agrees"]].itertuples()))
    print("\nSTRONGEST PREDICTORS OF THE GAP (usable cells only; descriptive - n is one point per dataset)")
    g = corr[(corr["scope"].str.startswith("graph")) & (corr["target"] == "gap_rel")
             & (corr["tier"] == "primary") & corr["spearman_rho"].notna()]
    for name, sub in g.groupby("task_family"):
        top = sub.reindex(sub["spearman_rho"].abs().sort_values(ascending=False).index).head(3)
        print(f"  {name:<22} " + " | ".join(f"{r.predictor} rho={r.spearman_rho:+.2f} (n={r.n}, p_bonf={r.p_bonferroni:.3f})" for r in top.itertuples()))
    print(f"\nCANDIDATE RULES (gates: |rho|>={GATES['min_abs_rho']}, LODO sign-stable, <={GATES['max_exceptions']} exception,"
          " leave-one-out above the majority baseline; significance REPORTED, not gated -> a pass is a credible CANDIDATE, Module 3 is the test)")
    print(cand[["task_family", "target", "predictor", "predictor_family", "canonical", "n_cells", "n_augment",
                "distinct_values", "spearman_rho", "p_bonferroni", "lodo_min_abs_rho", "lodo_sign_stable",
                "threshold", "interval_lo", "interval_hi", "augment_side", "n_exceptions",
                "loo_accuracy", "majority_baseline", "credible"]].to_string(index=False))
    keep = cand[cand["credible"] & cand["canonical"]]                 # one row per FINDING: collinear predictors collapse to their canonical member
    print("\n  CREDIBLE FINDINGS (collinear predictors merged): " +
          ("; ".join(dict.fromkeys(keep["rule"])) if len(keep) else "none - no predictor clears the gates"))
    dropped = cand[cand["credible"] & ~cand["canonical"]]["predictor"].unique()
    if len(dropped):
        print("  merged away as the same finding: " + ", ".join(f"{p} (-> {CANONICAL[FAMILY[p]]})" for p in dropped))
    # Named explicitly: a rule that passes every in-sample gate and fails only out of sample is the case this check exists for.
    near = cand[~cand["credible"] & cand["spearman_rho"].notna() & (cand["spearman_rho"].abs() >= GATES["min_abs_rho"])
                & cand["lodo_sign_stable"] & cand["n_exceptions"].between(0, GATES["max_exceptions"])]
    for r in near.drop_duplicates("rule").itertuples():
        print(f"  REJECTED out of sample - {r.rule}: leave-one-out {r.loo_accuracy} vs {r.majority_baseline} for guessing the majority verdict")
    thin = keep[keep["distinct_augment"] <= 1]["predictor"].unique()
    if len(thin):
        print("  WEAK EVIDENCE - every augmenting graph shares one predictor value, so this is a two-group split, not a graded trend: " + ", ".join(thin))
    for name, sub in cand.groupby("task_family"):
        if sub["n_augment"].max() == 0:
            print(f"  {name}: no augment cell exists -> no rule is learnable here, only the boundary 'never augment' is reportable")
    print("\n  OUT-OF-SAMPLE CHECK - the threshold above is fitted and scored on the SAME cells, so its exception count is")
    print("  guaranteed once the classes separate at all. These numbers refit the split with each dataset hidden:")
    for r in keep.drop_duplicates("rule").itertuples():                # a rule that clears on both gap definitions is ONE rule
        print(f"    {r.rule}\n      any cut in ({r.interval_lo}, {r.interval_hi}) fits the panel equally well -> quote the interval, not the point"
              f"\n      leave-one-out {r.loo_correct}/{r.loo_folds} = {r.loo_accuracy} vs {r.majority_baseline} for always guessing the "
              f"majority verdict; fold cutoffs ranged {r.loo_threshold_lo} to {r.loo_threshold_hi}")
    if len(nest):
        print("\n  NESTED LEAVE-ONE-OUT - the PREDICTOR is re-chosen inside each fold too, so choosing it on all cells is scored as well.")
        print("  A dataset the rule cannot be evaluated on (no labels -> no homophily) counts as a miss, not a skip:")
        for (family, target), sub in nest.groupby(["task_family", "target"]):
            miss = [f"{r.held_out} (said {r.predicted}, was {r.actual})" for r in sub.itertuples() if not r.correct]
            print(f"    {family} / {target}: {int(sub['correct'].sum())}/{len(sub)} correct, predictor picked "
                  f"{' + '.join(sorted(set(sub['picked_predictor'])))}" + ("; missed " + ", ".join(miss) if miss else ""))
    print("\nBEST SINGLE FEATURE PER CELL (ablation D lift over random features)")
    s = feat[feat["feature"].notna()]
    best = s.loc[s.groupby(["dataset", "task_family"])["mean"].idxmax()]
    print(best[["dataset", "task_family", "arm", "mean", "lift_vs_random", "raw_pct_zero", "raw_pct_unique"]].to_string(index=False))
    f = corr[(corr["scope"].str.startswith("feature")) & (corr["task_family"] == "pooled")]
    print("\n  feature spread vs usefulness (pooled): " +
          " | ".join(f"{r.predictor} rho={r.spearman_rho:+.2f}" for r in f.itertuples() if r.spearman_rho == r.spearman_rho))


# Runs both portions and writes the CSVs the characterization is built from; measure always runs, relate reads its output.
def main(args):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    fam1 = out["dataset_characterization"] = pd.DataFrame([graph_properties(ds) for ds in args.datasets])
    fam2 = out["node_feature_characterization"] = pd.DataFrame([r for ds in args.datasets for r in feature_properties(ds)])
    step1 = out["characterization_inputs"] = frozen_inputs(args.datasets)
    if args.step in ("relate", "rules", "all"):
        gap = out["characterization_gaps"] = gaps(step1)
        feat = out["feature_usefulness"] = feature_scores(fam2)
        corr = out["characterization_correlations"] = correlate(fam1, gap, feat)
        cand = out["candidate_rules"] = candidate_rules(fam1, gap, args.allow_refit)
        nest = out["nested_loo"] = nested_loo(fam1, gap, args.allow_refit)
    # An exploratory refit (--allow-refit) may fit rules on datasets outside the discovery panel, so it must NEVER overwrite
    # the frozen Module-2 evidence (candidate_rules.csv, nested_loo.csv, and the tables they are read against). It writes an
    # exploratory_ copy of every output instead; the frozen files are updated only by a plain, panel-only run.
    prefix = "exploratory_" if args.allow_refit else ""
    for name, df in out.items():
        df.to_csv(cfg.RESULTS_DIR / f"{prefix}{name}.csv", index=False)
        print(f"{len(df):3d} rows -> results/{prefix}{name}.csv", flush=True)
    print("\nFAMILY 1 - graph properties (measured on the original graph)")
    print(fam1.drop(columns=["directed_source", "distinct_degrees", "isolates"]).to_string(index=False))
    print("\nFAMILY 2 - raw structural inputs, pre-z-normalization")
    print(fam2[["dataset", "feature", "mean", "std", "skew", "pct_zero", "pct_unique", "max", "degenerate"]].to_string(index=False))
    if args.step in ("relate", "rules", "all"):
        rule(gap, corr, feat, cand, nest)


# Defines command-line options (mirrors run_core.py / run_ogb.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Characterization: measure the graphs and the encoder's inputs, then relate them to when augmentation helps.")
    p.add_argument('--datasets', nargs='+', default=DISCOVERY_PANEL, choices=list(STUDY),
                   help='Datasets to measure (any registered dataset may be MEASURED; fitting is guarded separately). Default: the fixed seven-dataset discovery panel.')
    p.add_argument('--step', default='all', choices=['measure', 'relate', 'rules', 'all'],
                   help="'measure' = portion 1 only; 'relate' / 'rules' / 'all' also join the properties to the gaps and screen candidate rules. Default: all.")
    p.add_argument('--allow-refit', action='store_true',
                   help='Deliberately re-fit thresholds on a dataset set that is not the discovery panel. OFF by default so Module-3 datasets cannot silently move the rules. '
                        'Writes exploratory_*.csv so the frozen candidate_rules.csv / nested_loo.csv are never overwritten.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
