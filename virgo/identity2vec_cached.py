"""Identity2Vec walker with cached structural signals - much faster than the baseline.

Degree, eigenvector centrality, neighbor lists and shortest-path lengths are invariant for a static
graph, but the baseline recomputes them inside the walk loop. Caching them changes no computed value and removes the dominant cost. This is the I2V 'cache fix'."""

import networkx as nx
from virgo import identity2vec


class Graph(identity2vec.Graph):
    """Drop-in Graph that caches the structural signals the baseline recomputes per step."""

    def __init__(self, nx_Graph, e, per_component=False):
        super().__init__(nx_Graph, e)
        self.per_component = per_component   # False = baseline-identical global Ω (Phase-1 repro); True = per-component Ω (see eigenvector_centrality)
        self._deg = None
        self._deg_dist = None
        self._ev = None
        self._neigh = None
        self._spl = {}

    # Degree per node, computed once.
    def degree_node(self):
        if self._deg is None:
            self._deg = {n: d for n, d in self.G.degree}
        return self._deg
    
    # Cached degree-distribution Δ per node.
    # Δ_u = fraction of graph nodes with the same degree as node u.
    def degree_distribution(self):
        if self._deg_dist is None:
            deg_dict = self.degree_node()
            total_nodes = len(self.G.nodes)

            degree_counts = {}
            for deg in deg_dict.values():
                degree_counts[deg] = degree_counts.get(deg, 0) + 1

            self._deg_dist = {
                node: degree_counts[deg] / total_nodes
                for node, deg in deg_dict.items()
            }

        return self._deg_dist

    # Eigenvector centrality, computed once (deterministic power iteration).
    # per_component: Ω is a GLOBAL eigenvector, so on a disjoint union it is supported only on the component with the
    # largest spectral radius and every other component underflows toward 0 (enzymes: 81.6% of nodes below 1e-12, min 1e-115).
    # Computing it inside each component keeps Ω defined everywhere. Each component is then rescaled to max=1, so Ω reads
    # "how central am I relative to my component's most central node" - comparable ACROSS components, which is what a role
    # graph compares. NetworkX's own L2=1 normalization would instead make Ω depend on component SIZE (corr -0.92 on cora:
    # a leaf in a 2-node fragment would outrank a real hub).
    def eigenvector_centrality(self):
        if self._ev is None:
            if not self.per_component:
                self._ev = nx.eigenvector_centrality(self.G, max_iter=1000)
                return self._ev
            self._ev = {}
            for c in nx.connected_components(self.G):
                sub = self.G.subgraph(c)
                if sub.number_of_edges() == 0:
                    self._ev.update({n: 0.0 for n in c})          # isolated node: Ω undefined, no neighbors to borrow centrality from
                    continue
                try:
                    ev = nx.eigenvector_centrality(sub, max_iter=1000)
                except nx.PowerIterationFailedConvergence:
                    ev = nx.eigenvector_centrality_numpy(sub)     # exact eigensolver fallback (bipartite/regular components)
                ev = {n: max(0.0, v) for n, v in ev.items()}      # the eigensolver can return ~-1e-18; Ω is non-negative, and psi's q>0 test would drop those nodes
                top = max(ev.values())
                self._ev.update({n: v / top for n, v in ev.items()} if top > 0 else ev)   # rescale to max=1 per component
        return self._ev

    # Neighbor lists, looked up once; returns fresh copies because callers mutate them (skip_visited).
    def node_neighbors(self):
        if self._neigh is None:
            self._neigh = {n: list(self.G.neighbors(n)) for n in self.G.nodes}
        return {n: v[:] for n, v in self._neigh.items()}

    # Shortest-path length via one cached BFS per source (callers use only the length).
    def s_path(self, source, destination):
        if source not in self._spl:
            self._spl[source] = nx.single_source_shortest_path_length(self.G, source)
        dist = self._spl[source]
        if destination in dist:
            return None, dist[destination]
        return [], 0
