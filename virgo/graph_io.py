'''One entry point for loading a dataset graph, plus the property report every downstream stage silently assumes.'''
# Splits, virtual graphs and encoder features all read through load_graph(), so they can never disagree about the graph.
# check() exists because every bug found on 2026-07-17/18 was invisible in code and obvious in one measurement:
# a 125-node LP split, a degree-10 graph with a 6724-degree hub, negatives that were 99.9% cross-component, Omega at 1e-115.

from pathlib import Path

import networkx as nx

# THE graph policy: how every stage treats any dataset. One place to read, one place to change, recorded per run.
# Re-exported as benchmark_config.GRAPH_POLICY. Pass policy={...} to any consumer to override a single key.
GRAPH_POLICY = {
    "self_loops":   "drop",           # drop | keep - dropped everywhere, so splits and virtual graphs see ONE graph
    "directed":     "symmetrize",     # every graph is loaded UNDIRECTED: a global recorded deviation. Whether the underlying
                                      # relation is directed (politics = retweets) is dataset METADATA - an edgelist storing
                                      # each edge once is indistinguishable from a directed one, so it is declared, not detected.
    "centrality":   "per_component",  # per_component | global - global Omega underflows to 1e-115 on a disjoint union
    "sig_tol":      1e-9,             # signatures closer than tol x spread are TIED -> neighbors sampled, never ordered
    "lp_negatives": "component",      # component | uniform - uniform negatives are ~99.9% cross-component on disjoint unions
}

# Phase-1 reproduction contract: I2V's own behaviour. Deliverable #1 (byte-identical embeddings) is only valid under this.
I2V_BASELINE_POLICY = {**GRAPH_POLICY, "centrality": "global", "lp_negatives": "uniform"}


# Merges caller overrides onto the default policy.
def policy_of(policy=None):
    '''Return the effective policy: defaults, with any caller override applied.'''
    unknown = set(policy or {}) - set(GRAPH_POLICY)
    assert not unknown, f"unknown policy keys {sorted(unknown)}; valid: {sorted(GRAPH_POLICY)}"
    return {**GRAPH_POLICY, **(policy or {})}


# Detects the delimiter from the first real line (networkrepository ships comma-delimited edgelists, e.g. proteins).
def delimiter(path):
    '''Return the delimiter an edgelist uses, or None for whitespace.'''
    first = next(l for l in open(path) if l.strip() and not l.startswith('%'))
    return ',' if ',' in first else None


# Reads any dataset edgelist the same way everywhere, under the graph policy.
def load_graph(path, policy=None):
    '''Read an edgelist into an undirected graph under GRAPH_POLICY, so every stage sees ONE graph definition.'''
    p = policy_of(policy)
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph(), delimiter=delimiter(path))
    if p["self_loops"] == "drop":
        G.remove_edges_from(nx.selfloop_edges(G))   # not a valid link-prediction pair, and it inflates degree in every signature
    sidecar = Path(path).with_suffix(".nodes")      # OGB: a node with no (training) edge vanishes from an edgelist -> restore the FULL node set
    if sidecar.exists():
        G.add_nodes_from(int(l) for l in open(sidecar) if l.strip())
    return G


# Measures the facts downstream stages depend on; pass the source path to also detect a directed source.
def properties(G, path=None):
    '''Return the graph facts that quietly change results: components, degree ties, isolates, directed source.'''
    deg = [d for _, d in G.degree]
    comps = [len(c) for c in nx.connected_components(G)]
    p = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
         "components": len(comps), "largest_component": max(comps) if comps else 0,
         "isolates": sum(1 for d in deg if d == 0), "self_loops": nx.number_of_selfloops(G),
         "distinct_degrees": len(set(deg)), "max_degree": max(deg) if deg else 0}
    p["largest_component_frac"] = round(p["largest_component"] / p["nodes"], 4) if p["nodes"] else 0.0
    if path is not None:                        # how the FILE stores edges - NOT whether the relation is directed (undetectable)
        D = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph(), delimiter=delimiter(path))
        p["single_direction_edges"] = sum(1 for u, v in D.edges if not D.has_edge(v, u))
    return p


# Fails loudly on an unusable graph and prints the properties that silently changed results in earlier runs.
def check(G, name="graph", path=None, policy=None, strict=False, directed_source=False):
    '''Assert the graph is usable, then report every property a downstream stage assumes. strict=True raises on the flags.
       directed_source is DECLARED by the caller (DATASETS registry): an edgelist that stores each edge once is
       indistinguishable from a directed one, so guessing it from the file only produces false positives.'''
    pol, p = policy_of(policy), properties(G, path)
    assert p["nodes"] > 0 and p["edges"] > 0, f"{name}: empty graph ({p['nodes']} nodes, {p['edges']} edges)"
    if pol["self_loops"] == "drop":
        assert p["self_loops"] == 0, f"{name}: {p['self_loops']} self-loops survived load_graph()"

    flags = []
    if p["components"] > 1:                     # drove the largest-component split bug AND the Omega underflow
        flags.append(f"DISCONNECTED: {p['components']} components, largest holds {100 * p['largest_component_frac']:.1f}% of nodes"
                     f" -> splits cover every component, negatives='{pol['lp_negatives']}', centrality='{pol['centrality']}'")
    if directed_source:                         # caller-declared (registry metadata), never guessed from the file
        flags.append(f"DIRECTED RELATION loaded UNDIRECTED under directed='{pol['directed']}' -> recorded deviation; "
                     f"the relation is not symmetric and link prediction on it is an easier, different task")
    if p["distinct_degrees"] < 0.01 * p["nodes"]:   # coarse signature -> "top-K nearest" is ill-posed, ties are sampled
        flags.append(f"COARSE DEGREE SIGNATURE: {p['distinct_degrees']} distinct degrees over {p['nodes']} nodes"
                     f" -> degree top-K is sampled inside tie classes (sig_tol={pol['sig_tol']:g}), never ordered")
    if p["isolates"]:                           # degree 0 and Omega 0 -> they all land in ONE tied signature class
        flags.append(f"{p['isolates']} isolated nodes: they DO get K virtual edges (sampled inside that tie class), but carry"
                     f" no link-prediction pairs; on graph_variant='original' they get no edges and so no embedding at all")

    assert not (strict and flags), f"{name}: strict=True and " + " | ".join(flags)
    print(f"{name}: {p['nodes']} nodes, {p['edges']} edges, max degree {p['max_degree']}")
    for f in flags:
        print(f"  {f}")
    return p
