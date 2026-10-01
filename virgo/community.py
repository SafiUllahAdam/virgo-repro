'''Community structure of a graph and of its role-graph variants: seeded Louvain partitions and the measurements read on them.'''
# Module 17 (paper_log 2026-10-01). Method code only - partition, describe, compare two partitions, place a graph's edges
# against a reference partition. It reads no file and knows no dataset, so experiments/community_social.py stays a thin
# CLI. Louvain = networkx's modularity optimiser: no new dependency, weighted graphs supported, seeded and deterministic.

import hashlib
import math
from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# One seeded Louvain run. Ids are canonical (largest community = 0, ties broken by the smallest node id), so one
# partition always gets the same ids and the same hash, whatever order networkx returns the sets in.
def louvain(G, seed, resolution=1.0, threshold=1e-7, weight="weight"):
    '''node -> community id for one seeded Louvain run.'''
    comms = nx.community.louvain_communities(G, weight=weight, resolution=resolution, threshold=threshold, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    return {n: i for i, c in enumerate(comms) for n in c}


# networkx's quality functions take a list of node sets, not a node -> id map.
def groups(part):
    '''node -> id map back to the list of node sets, in id order.'''
    out = {}
    for n, c in part.items():
        out.setdefault(c, set()).add(n)
    return [out[c] for c in sorted(out)]


# Recorded next to every partition, so a rerun on another machine can confirm it found the SAME partition.
def partition_sha(part):
    '''Short content hash of a canonical partition.'''
    return hashlib.sha256("\n".join(f"{n} {c}" for n, c in sorted(part.items())).encode()).hexdigest()[:16]


# How strong, how many, how even, how separated. Conductance is averaged with each community weighted by its node
# count, so a graph's many tiny communities cannot outvote its few large ones; an edgeless community has no boundary
# to measure and is left out of that average.
def describe(G, part, weight="weight"):
    '''Modularity, community count and sizes, coverage and mean conductance of one partition of G.'''
    comms = groups(part)
    assert sum(len(c) for c in comms) == G.number_of_nodes(), "the partition does not cover the graph"
    sizes = np.array([len(c) for c in comms], dtype=float)
    share = sizes / sizes.sum()
    vol, cut = np.zeros(len(comms)), np.zeros(len(comms))
    inside = total = 0.0
    for u, v, w in G.edges(data=weight, default=1.0):
        cu, cv = part[u], part[v]
        vol[cu] += w
        vol[cv] += w
        total += w
        if cu == cv:
            inside += w
        else:
            cut[cu] += w
            cut[cv] += w
    denom = np.minimum(vol, 2 * total - vol)               # conductance = boundary / the smaller side's volume
    live = denom > 0
    return {"communities": len(comms),
            "modularity": nx.community.modularity(G, comms, weight=weight),
            "largest_share": float(share.max()),
            "effective_communities": float(math.exp(-(share * np.log(share)).sum())),   # exp(entropy of the size shares)
            "singleton_share": float(sizes[sizes == 1].sum() / sizes.sum()),
            "coverage": inside / total if total else float("nan"),
            "conductance": float(np.average(cut[live] / denom[live], weights=sizes[live])) if live.any() else float("nan")}


# Used three ways: a variant's partition against the original's, two seeds against each other, communities against labels.
def agreement(a, b, nodes=None):
    '''(NMI, ARI) between two node -> id maps over `nodes` (default: every node of a).'''
    nodes = sorted(a) if nodes is None else nodes
    x, y = [a[n] for n in nodes], [b[n] for n in nodes]
    return float(normalized_mutual_info_score(x, y)), float(adjusted_rand_score(x, y))


# How much of a partition is the graph and how much is the seed.
def stability(parts):
    '''Mean pairwise NMI between seeded partitions of one graph.'''
    vals = [agreement(a, b)[0] for a, b in combinations(parts, 2)]
    return float(np.mean(vals)) if vals else float("nan")


# "Important node" = top fraction by a centrality score; ties at the cut are broken by node id, so the set is fixed.
def hubs(score, frac=0.10):
    '''The top `frac` of nodes by `score` (node -> value).'''
    order = sorted(score, key=lambda n: (-score[n], n))
    return set(order[:math.ceil(frac * len(order))])


# Where a graph's edges fall against a REFERENCE partition - the original graph's communities. Unweighted on purpose:
# the question is which node pairs a graph links, and the locked encoder aggregates its neighbours unweighted.
# ref_modularity is the reference partition's modularity measured on this graph: about 0 = the edges ignore the
# original communities, below 0 = they cross them more than a degree-preserving random graph would.
def placement(V, ref, hub_set):
    '''Share of V's edges - all of them, those touching a hub, those joining two hubs - that cross the reference communities.'''
    assert set(V.nodes) == set(ref), "the graph and the reference partition cover different nodes"
    sizes = np.bincount(list(ref.values())).astype(float)
    n = sizes.sum()
    e = cross = he = hcross = hh = hhcross = 0
    for u, v in V.edges():
        c = ref[u] != ref[v]
        e += 1
        cross += c
        hu, hv = u in hub_set, v in hub_set
        if hu or hv:
            he += 1
            hcross += c
            if hu and hv:
                hh += 1
                hhcross += c
    ratio = lambda a, b: a / b if b else float("nan")
    return {"edges": e, "cross_share": ratio(cross, e),
            "chance_cross": float(1 - (sizes * (sizes - 1)).sum() / (n * (n - 1))),   # a uniformly random node pair
            "ref_modularity": nx.community.modularity(V, groups(ref), weight=None),
            "hub_edges": he, "hub_edge_share": ratio(he, e), "hub_cross_share": ratio(hcross, he),
            "hubhub_edges": hh, "hubhub_cross_share": ratio(hhcross, hh)}
