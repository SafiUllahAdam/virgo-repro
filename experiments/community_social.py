'''Module 17: community structure on the social graphs - Louvain on the original graph and on its pure role graphs, read against the calls and LP verdicts the study already has.'''
# An INTERPRETATION study, pre-registered in docs/paper_log.md (2026-10-01) before anything was measured. It READS the
# frozen rules and the published scoreboard; it fits nothing, freezes nothing and trains nothing. Method code lives in
# virgo/community.py - this file wires the social panel to it and writes results/community_*.csv.
#   --step measure   build original + psi/degree/centrality (K=10, VG seed 42, full graph) -> Louvain x 3 seeds -> measure
#   --step compare   join the measurements with the stage-1/2 calls and both LP verdicts -> the pre-declared comparisons
# Incremental per dataset: measuring a graph replaces its own rows and keeps every other graph's, so the five graphs that
# complete the panel can be added later without touching the pilot's numbers.

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import community
from virgo import config as cfg
from virgo import frozen_rules as fr
from virgo import graph_io
from virgo.virtual_graph import VirtualGraph
from experiments.characterize import STUDY, graph_properties, labels_of
from experiments.strategy_select import SIGNAL, winners

K = 10                                                           # the locked K; the community study never sweeps it
SOCIAL = cfg.COMMUNITY_SOCIAL + cfg.COMMUNITY_SOCIAL_LEFT_OUT    # in node order: pilot, +5, left out
assert sorted(SOCIAL) == sorted(d for d, m in STUDY.items() if "social" in m["domain"]), \
    "the community panel must be exactly the social-domain graphs of characterize.STUDY"
assert cfg.COMMUNITY_VARIANTS[0] == "original", "original must be measured first - every role graph is placed against its partition"
OUT = {name: cfg.RESULTS_DIR / f"community_{name}.csv"
       for name in ["graphs", "partitions", "placement", "summary", "compare", "correlations"]}
RECORDED = cfg.PROJECT_ROOT / "expected" / "hashes" / "artifacts.csv"
BOARD = cfg.PROJECT_ROOT / "expected" / "scoreboard.csv"        # the published LP results - the evidence, never retrained
ROLE = [v for v in cfg.COMMUNITY_VARIANTS if v != "original"]

# The pre-declared comparisons (paper_log 2026-10-01), all descriptive: (hypothesis, x, y, subset, expected sign).
TESTS = [("H1", "orig_modularity", "homophily_adjusted", "all", +1),
         ("H1", "orig_modularity", "avg_clustering", "all", +1),
         ("H2", "orig_modularity", "official_gap", "all", -1),
         ("H2", "orig_modularity", "nonhybrid_gap", "all", -1),
         ("H3", "hub_cross_centrality", "centrality_advantage", "non-hybrid verdict augments", +1)]
# H2 also as two groups: median modularity of the keep-verdict graphs minus that of the augment-verdict graphs.
GROUP_TESTS = [("H2", "orig_modularity", "nonhybrid_verdict", +1), ("H2", "orig_modularity", "official_verdict", +1)]


# Byte-for-byte what run_ogb.ensure_virtual writes for the same build, so a recorded role-graph hash can vouch for it.
def build_row(ds, sim, V, recorded, parts):
    '''One row per graph built: size, its hash against the recorded one, and how stable Louvain is on it across seeds.'''
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "virtual_graph.edgelist"
        nx.write_weighted_edgelist(V, f)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
    ref = recorded.get(f"output/notebook2_create_vir_graph/virtual_graphs/{ds}/k{K}/{sim}/virtual_graph.edgelist", "")
    # Same statuses as verify/compare.py: a psi mismatch is the eigensolver-fallback KNOWN_EXCEPTION, not a failure.
    status = ("NO_REFERENCE" if not ref else "EXACT" if ref == sha
              else "KNOWN_EXCEPTION" if sim.startswith("psi") else "MISMATCH")
    return {"dataset": ds, "variant": sim, "K": K, "nodes": V.number_of_nodes(), "edges": V.number_of_edges(),
            "isolates": sum(1 for _, d in V.degree if d == 0), "components": nx.number_connected_components(V),
            "sha256": sha, "hash_status": status, "seed_stability_nmi": round(community.stability(parts), 4)}


# Every graph of one dataset: build -> Louvain per seed -> describe the partition -> place the edges against the
# ORIGINAL graph's partition of the same seed. Hubs come from the original graph's eigenvector centrality - the exact
# values the centrality role graph is built from (the same VirtualGraph core, per-component, max-scaled).
def measure(ds, recorded):
    '''(graph rows, partition rows, placement rows) for one dataset.'''
    G = graph_io.load_graph(cfg.dataset(ds)["edgelist"])
    vg = VirtualGraph(G, seed=cfg.REPRO["seed"])
    hub_set = community.hubs(vg.core.eigenvector_centrality(), cfg.COMMUNITY_HUB_FRAC)
    lab = labels_of(cfg.dataset(ds)["labels"], G)
    graphs, parts, place, orig = [], [], [], {}
    for sim in cfg.COMMUNITY_VARIANTS:
        V = vg.build(sim, K)
        assert set(V.nodes) == set(G.nodes), f"{ds}/{sim}: the build changed the node set"
        seeded = []
        for seed in cfg.COMMUNITY_SEEDS:
            p = community.louvain(V, seed, **cfg.LOUVAIN)
            if sim == "original":
                orig[seed] = p
            nmi_o, ari_o = community.agreement(p, orig[seed])
            parts.append({"dataset": ds, "variant": sim, "seed": seed,
                          **community.describe(V, p, weight=cfg.LOUVAIN["weight"]),
                          "label_nmi": community.agreement(p, lab, nodes=sorted(lab))[0] if lab else float("nan"),
                          "nmi_vs_original": nmi_o, "ari_vs_original": ari_o, "partition_sha": community.partition_sha(p)})
            place.append({"dataset": ds, "variant": sim, "seed": seed, "hubs": len(hub_set),
                          **community.placement(V, orig[seed], hub_set)})
            seeded.append(p)
        graphs.append(build_row(ds, sim, V, recorded, seeded))
        q = [r["modularity"] for r in parts[-len(cfg.COMMUNITY_SEEDS):]]
        print(f"  {ds:22s} {sim:11s} {V.number_of_edges():>8d} edges | Q {np.mean(q):.4f} ± {np.std(q, ddof=1):.4f} | "
              f"hash {graphs[-1]['hash_status']}", flush=True)
    return graphs, parts, place


# Re-measuring a dataset replaces only its own rows; rows are kept in panel order so the tables read pilot first.
def merge(path, rows, datasets):
    '''Write `rows` into a results table, replacing any earlier rows of the same datasets.'''
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dataset"])
    out = pd.concat([old[~old["dataset"].isin(datasets)], pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("dataset", key=lambda s: s.map({d: i for i, d in enumerate(SOCIAL)}), kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def step_measure(datasets):
    '''Measure every dataset and write the graphs / partitions / placement tables.'''
    recorded = dict(pd.read_csv(RECORDED).query("kind == 'role_graph'")[["file", "sha256"]].values)
    graphs, parts, place = [], [], []
    for ds in datasets:
        print(f"measure {ds}", flush=True)
        g, p, q = measure(ds, recorded)
        graphs += g
        parts += p
        place += q
    for name, rows in [("graphs", graphs), ("partitions", parts), ("placement", place)]:
        merge(OUT[name], rows, datasets)
        print(f"wrote {OUT[name]}", flush=True)


# strategy_select.winners' sem-band rule, applied to whichever of `variants` were scored. Needed because winners()
# insists on all seven official variants, and two pilot graphs carry five; where all seven exist, compare() asserts
# this reproduces winners() exactly.
def verdict(board, ds, variants):
    '''LP verdict over `variants`: best variant, the signals its tie band names, whether augmentation beats the original, the gap.'''
    s = board[(board["dataset"] == ds) & (board["encoder"] == "graphsage_edge") & (board["top_K_neighbors"] == K)
              & (board["task"] == "link prediction (AUC)") & (board["metric"] == "auc")
              & board["graph_variant"].isin(variants)].set_index("graph_variant")
    assert s.index.is_unique and "original" in s.index, f"{ds}: LP rows missing or duplicated on the board"
    n = s["seeds"].str.count(r"\|").astype(float) + 1
    top = s["mean"].idxmax()
    width = np.sqrt(s.loc[top, "std"] ** 2 / n[top] + s["std"] ** 2 / n)
    keep = sorted(s.index[(s.loc[top, "mean"] - s["mean"]) <= width])
    return {"scored": len(s), "seeds": int(n["original"]), "best": top,
            "winners": "|".join(sorted({SIGNAL[v] for v in keep})),
            "verdict": "keep original" if "original" in keep else "augment",
            "gap": float(s.drop(index="original")["mean"].max() - s.loc["original", "mean"]), "auc": s["mean"]}


# Per dataset x variant: the seed means of every measurement, the seed spread of modularity, and the seed stability.
def summarize(datasets):
    '''The per-seed tables averaged over the Louvain seeds.'''
    part, place, graphs = (pd.read_csv(OUT[n]) for n in ["partitions", "placement", "graphs"])
    missing = sorted(set(datasets) - set(part["dataset"]))
    assert not missing, f"no community measurements for {missing} - run --step measure first"
    m = part.merge(place, on=["dataset", "variant", "seed"])
    m = m[m["dataset"].isin(datasets)]
    cols = ["communities", "modularity", "largest_share", "effective_communities", "singleton_share", "coverage",
            "conductance", "label_nmi", "nmi_vs_original", "ari_vs_original", "cross_share", "chance_cross",
            "ref_modularity", "hub_edge_share", "hub_cross_share", "hubhub_cross_share"]
    g = m.groupby(["dataset", "variant"], sort=False)
    out = g[cols].mean().join(g["modularity"].std().rename("modularity_sd")).reset_index()
    out = out.merge(graphs[["dataset", "variant", "edges", "hash_status", "seed_stability_nmi"]], on=["dataset", "variant"])
    return out.round(4)


def step_compare(datasets, board_path):
    '''Join the measurements with the frozen calls and both LP verdicts, then run the pre-declared comparisons.'''
    summary = summarize(datasets)
    merge(OUT["summary"], summary.to_dict("records"), datasets)
    board = pd.read_csv(board_path)
    rows = []
    for ds in datasets:
        p = graph_properties(ds)
        _, _, call, why = fr.predict_gated(p)
        off, nh = verdict(board, ds, cfg.VG_SIMS_OFFICIAL), verdict(board, ds, cfg.COMMUNITY_VARIANTS)
        if off["scored"] == len(cfg.VG_SIMS_OFFICIAL):                       # the published verdict, reproduced exactly
            w = winners([ds], "sem", board=board).iloc[0]
            assert (w["best_variant"], w["winner_signals"], bool(w["beats_original"])) == \
                   (off["best"], off["winners"], off["verdict"] == "augment"), f"{ds}: verdict() drifted from winners()"
        S = summary[summary["dataset"] == ds].set_index("variant")
        auc = nh["auc"]
        row = {"dataset": ds, "nodes": p["nodes"], "edges": p["edges"],
               "homophily_adjusted": p["homophily_adjusted"], "avg_clustering": p["avg_clustering"],
               "nbr_predictability_adjusted": p["nbr_predictability_adjusted"],
               "stage1_call": call, "stage1_by": why,
               "stage2_call": fr.predict_strategy(p) if call == "augment" else "not consulted",   # stage 2 is conditional
               "official_scored": off["scored"], "official_seeds": off["seeds"], "official_best": off["best"],
               "official_winners": off["winners"], "official_verdict": off["verdict"], "official_gap": round(off["gap"], 4),
               "nonhybrid_best": nh["best"], "nonhybrid_winners": nh["winners"], "nonhybrid_verdict": nh["verdict"],
               "nonhybrid_gap": round(nh["gap"], 4),
               "centrality_advantage": round(float(auc["centrality"] - max(auc["psi"], auc["degree"])), 4)}
        for col in ["modularity", "modularity_sd", "communities", "largest_share", "coverage", "conductance",
                    "label_nmi", "seed_stability_nmi"]:
            row[f"orig_{col}"] = S.loc["original", col]
        for v in cfg.COMMUNITY_VARIANTS:
            row[f"hub_cross_{v}"] = S.loc[v, "hub_cross_share"]
            row[f"cross_{v}"] = S.loc[v, "cross_share"]
            row[f"ref_modularity_{v}"] = S.loc[v, "ref_modularity"]
        for v in ROLE:
            row[f"own_modularity_{v}"] = S.loc[v, "modularity"]
            row[f"nmi_vs_original_{v}"] = S.loc[v, "nmi_vs_original"]
        rows.append(row)
    cmp = merge(OUT["compare"], rows, datasets)
    cmp = cmp[cmp["dataset"].isin(datasets)]
    corr = correlations(cmp)
    corr.to_csv(OUT["correlations"], index=False)                    # always recomputed over exactly `datasets`
    print(f"wrote {OUT['summary']}\nwrote {OUT['compare']}\nwrote {OUT['correlations']}", flush=True)
    return cmp, corr


# Descriptive only: no threshold is read off any row, and every row is reported whichever way it lands.
def correlations(cmp):
    '''The pre-declared Spearman and two-group comparisons over the measured panel.'''
    rows = []
    for hyp, x, y, subset, sign in TESTS:
        d = cmp if subset == "all" else cmp[cmp["nonhybrid_verdict"] == "augment"]
        d = d[[x, y]].dropna()
        rho, pv = spearmanr(d[x], d[y]) if len(d) >= 3 else (float("nan"), float("nan"))
        rows.append({"hypothesis": hyp, "statistic": "spearman", "x": x, "y": y, "subset": subset, "n": str(len(d)),
                     "value": round(float(rho), 4), "p": round(float(pv), 4), "expected": "+" if sign > 0 else "-",
                     "direction_holds": bool(np.sign(rho) == sign) if rho == rho else None})
    for hyp, x, by, sign in GROUP_TESTS:
        keep, aug = cmp.loc[cmp[by] == "keep original", x], cmp.loc[cmp[by] == "augment", x]
        diff = float(keep.median() - aug.median()) if len(keep) and len(aug) else float("nan")
        pv = float(mannwhitneyu(keep, aug).pvalue) if len(keep) and len(aug) else float("nan")
        rows.append({"hypothesis": hyp, "statistic": "median keep - median augment", "x": x, "y": by, "subset": "all",
                     "n": f"{len(keep)} keep / {len(aug)} augment", "value": round(diff, 4), "p": round(pv, 4),
                     "expected": "+" if sign > 0 else "-",
                     "direction_holds": bool(np.sign(diff) == sign) if diff == diff else None})
    return pd.DataFrame(rows)


def main(args):
    if args.step in ("measure", "all"):
        step_measure(args.datasets)
    if args.step in ("compare", "all"):
        cmp, corr = step_compare(args.datasets, args.board)
        print(corr.to_string(index=False))


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 17: Louvain community structure on the social graphs (reads only).")
    p.add_argument('--step', default='all', choices=['measure', 'compare', 'all'],
                   help='measure = build + Louvain + measurements; compare = join with the calls and LP verdicts. Default all.')
    p.add_argument('--datasets', nargs='+', default=cfg.COMMUNITY_SOCIAL_PILOT, choices=SOCIAL,
                   help='Social graphs to run. Default: the 10-graph pilot (cfg.COMMUNITY_SOCIAL_PILOT); '
                        'cfg.COMMUNITY_SOCIAL is the full 15.')
    p.add_argument('--board', type=Path, default=BOARD,
                   help='Scoreboard the LP verdicts are read from. Default: the published expected/scoreboard.csv.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
