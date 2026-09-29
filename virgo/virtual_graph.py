'''Build a top-K structural-similarity virtual graph: same nodes, edges rewired to structural role-neighbors.'''
# Phase 2. Reuses I2V's cached structural signals; the virtual edgelist is a drop-in for the walk/GNN encoders.
# A virtual graph links nodes by structural role (hub<->hub, bridge<->bridge), not by original edges.

import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.special import gammaln

from virgo import graph_io
from virgo import identity2vec_cached


# Ψ arms: sim -> (Poisson wrapper?, reading of Eq. 3-4's ω, how p/q are normalized over N(u), shift λ so min=0?).
# `psi` is the published one; every other arm is a diagnostic and none of them is in cfg.VG_SIMS_LOCKED.
#   ω        OPEN item from 2026-06-24 (notes.md:152, :209) - the paper says only "the structural attributes of v1".
#   norm     the supervisor's fix (a). 'q' normalizes Ω over N(u); 'pq' normalizes Δ too, which is what actually makes
#            λ a true KL and so λ >= 0 by Gibbs. 'q' ALONE DOES NOT: with Σq=1 but Σp=S, the bound is S·log(S) < 0
#            whenever S < 1, and Δ = n_d/n is small, so Ω-only normalization drives λ negative rather than fixing it.
#   shift    the supervisor's fix (b). λ -> λ - min(λ) over the graph's non-isolated nodes: order-preserving, and the
#            1e-12 floor then bites ONE node (the minimum) instead of a third of them.
PSI_ARMS = {'psi': (True, 'deg_ev', 'none', False), 'psi_lambda': (False, 'deg_ev', 'none', False),
            'psi_w_deg': (True, 'deg', 'none', False), 'psi_w_ev': (True, 'ev', 'none', False),
            'psi_w_delta': (True, 'delta_ev', 'none', False),
            'psi_qnorm': (True, 'deg_ev', 'q', False), 'psi_pqnorm': (True, 'deg_ev', 'pq', False),
            'psi_shift': (True, 'deg_ev', 'none', True)}


class VirtualGraph():
    '''Connects each node to its top-K structurally most-similar nodes under a chosen similarity.'''

    def __init__(self, nx_Graph, e=2.7182, seed=42, policy=None):
        self.seed = seed                                     # ties are sampled, not ordered -> the build needs a seed
        self.policy = graph_io.policy_of(policy)             # centrality mode + tie tolerance come from THE graph policy
        self.G = nx_Graph
        self.core = identity2vec_cached.Graph(nx_Graph, e, per_component=self.policy["centrality"] == "per_component")

    # Reference-free I2V KL rate λ: KL(Δ_neigh || Ω_neigh), dropping the walk shortest-path term.
    def psi_lambda_of(self, node, neigh_map, deg_dist, ev, deg, omega='deg_ev', norm='none'):
        '''I2V's divergence rate λ for one node: signed, unclamped, no Poisson wrapper (Eq. 3-4 with d dropped).'''
        neigh = neigh_map[node]
        if len(neigh) == 0:
            return 0.0
        ps = [deg_dist[w] for w in neigh]                     # I2V p=Δ (degree-dist)
        qs = [ev[w] for w in neigh]                           # I2V q=Ω (eigenvector centrality)
        if norm in ('q', 'pq'):                               # supervisor's fix (a): Ω as a distribution over N(u)
            t = sum(qs)
            qs = [q / t for q in qs] if t > 0 else qs
        if norm == 'pq':                                      # ...and Δ too, which is what makes λ a true KL -> λ >= 0
            t = sum(ps)
            ps = [p / t for p in ps] if t > 0 else ps
        rt = sum(p * np.log(p / q) for p, q in zip(ps, qs) if p > 0 and q > 0)
        # Eq. 3-4 divide by ω, "the structural attributes of v1", which the paper never writes out; these are the readings.
        # ω > 0 in every arm, so the ω ARM CANNOT CHANGE WHICH NODES CLAMP - sign(λ/ω) = sign(λ). It changes magnitude only.
        w_node = {'deg_ev': deg[node] + ev[node],             # Fix 4A as shipped: raw degree + Ω
                  'deg': deg[node],                           # degree alone
                  'ev': ev[node],                             # Ω alone
                  'delta_ev': deg_dist[node] + ev[node]}[omega]   # Δ + Ω: the paper's OWN two named structural properties (3.2.1)
        return rt / w_node                                    # I2V Fix 4A normalizer; node has neighbors so deg>=1

    # I2V Fix 8: the Poisson score in log space. The 1e-12 floor is the clamp under audit - see paper_log 2026-09-08.
    def poisson_score(self, lam, k):
        '''Ψ = k·log(λ) − λ − log(k!), evaluated in log space; λ is floored at 1e-12 because log needs λ > 0.'''
        if k == 0:
            return 0.0
        return k * np.log(max(lam, 1e-12)) - max(lam, 1e-12) - gammaln(k + 1)

    # Per-node 1-D structural signature for a similarity variant (add a variant = add one branch here).
    def signatures(self, sim):
        '''node -> structural signature vector; distance in this space defines "structurally similar".'''
        nodes = list(self.G.nodes)
        if sim in PSI_ARMS:                                   # every Ψ arm: the wrapper, the ω reading and the λ repair vary
            poisson, omega, norm, shift = PSI_ARMS[sim]
            neigh_map, deg_dist = self.core.node_neighbors(), self.core.degree_distribution()
            ev, deg = self.core.eigenvector_centrality(), self.core.degree_node()
            vals = {n: self.psi_lambda_of(n, neigh_map, deg_dist, ev, deg, omega, norm) for n in nodes}
            if shift:                                         # fix (b): slide λ up so its minimum is 0, keeping every node's ORDER
                live = [vals[n] for n in nodes if len(neigh_map[n])]   # isolated nodes are defined as 0.0, not measured -> excluded
                lo = min(live) if live else 0.0
                vals = {n: (vals[n] - lo if len(neigh_map[n]) else 0.0) for n in nodes}
            if poisson:
                vals = {n: self.poisson_score(vals[n], len(neigh_map[n])) for n in nodes}
        elif sim == 'degree':
            vals = self.core.degree_node()
        elif sim == 'centrality':
            vals = self.core.eigenvector_centrality()
        else:
            raise ValueError(f"Unknown sim '{sim}'. Use: {', '.join(PSI_ARMS)}, degree, centrality, original, hybrid, hybrid_degree, hybrid_centrality.")
        X = np.array([[float(vals[n])] for n in nodes])
        return nodes, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)   # guard: no NaN/inf signatures

    # Build the virtual graph: same nodes, no self-loops, exactly top-K per node (undirected union), finite weights.
    def build(self, sim, k):
        '''Top-K nearest structural neighbors per node -> weighted undirected virtual graph.'''
        if sim == 'original':                                 # control: unchanged original edges, weight 1.0 -> GNN on the real graph (K unused)
            V = nx.Graph()
            V.add_nodes_from(self.G.nodes)
            V.add_edges_from(self.G.edges, weight=1.0)
            return V
        if sim in ('hybrid', 'hybrid_degree', 'hybrid_centrality'):   # original ∪ role top-K: physical neighbors (1.0) + role neighbors (sim weight)
            base = {'hybrid': 'psi', 'hybrid_degree': 'degree', 'hybrid_centrality': 'centrality'}[sim]   # hybrid = original ∪ psi (historical name)
            V = self.build(base, k)
            V.add_edges_from(self.G.edges, weight=1.0)        # overlap edge takes 1.0: original-edge semantics win
            return V
        # Density-matched controls: same "union of K per node" construction as the role graphs, only the neighbor SOURCE differs,
        # so edge counts are comparable by construction. original_k = K real neighbors (semantics without volume);
        # random_k = K arbitrary nodes (the null: does any sparse scaffold do as well as a role scaffold?).
        if sim in ('original_k', 'random_k'):
            rng = np.random.default_rng(self.seed)
            nodes = list(self.G.nodes)
            V = nx.Graph()
            V.add_nodes_from(nodes)
            for node in nodes:
                pool = list(self.G[node]) if sim == 'original_k' else \
                    [nodes[j] for j in rng.choice(len(nodes), min(k + 1, len(nodes)), replace=False)]
                pool = [x for x in pool if x != node]
                picked = pool if len(pool) <= k else [pool[j] for j in rng.choice(len(pool), k, replace=False)]
                for j in picked:
                    V.add_edge(node, j, weight=1.0)           # no similarity meaning here, so weight is constant
            assert V.number_of_nodes() == self.G.number_of_nodes(), "node set changed"
            return V
        nodes, X = self.signatures(sim)
        k = min(k, len(nodes) - 1)
        n = len(nodes)
        vals = X[:, 0]
        span = float(vals.max() - vals.min())
        tol = self.policy["sig_tol"]
        keys = np.round(vals / (span * tol)) if span > 0 else np.zeros(n)       # equal key = tied signature = interchangeable role
        order = np.argsort(keys, kind='stable')               # tie classes contiguous, ordered by signature value
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)                             # where each node sits in that order
        starts = np.flatnonzero(np.r_[True, keys[order][1:] != keys[order][:-1]])
        ends = np.r_[starts[1:], n]
        cls = np.searchsorted(starts, pos, side='right') - 1  # node -> its tie class
        rng = np.random.default_rng(self.seed)

        V = nx.Graph()
        V.add_nodes_from(nodes)                               # keep ALL original nodes, even isolated ones -> none missing
        for i, node in enumerate(nodes):
            lo, hi = int(starts[cls[i]]), int(ends[cls[i]])
            picked = []
            if hi - lo - 1 >= k:                              # class alone can fill K: members are tied, so "nearest" is undefined -> sample
                seen = set()
                while len(picked) < k:
                    j = int(order[rng.integers(lo, hi)])
                    if j != i and j not in seen:
                        seen.add(j)
                        picked.append(j)
            else:                                             # class too small: take it all, then widen to the nearest signature values
                picked = [int(j) for j in order[lo:hi] if j != i]
                left, right = lo - 1, hi
                while len(picked) < k:
                    dl = abs(vals[order[left]] - vals[i]) if left >= 0 else np.inf
                    dr = abs(vals[order[right]] - vals[i]) if right < n else np.inf
                    c = int(cls[order[left if dl <= dr else right]])   # widen a WHOLE tie class at a time, never one index at a time:
                    a, b = int(starts[c]), int(ends[c])               # its members are equidistant, so pick among them by sampling
                    cand = [int(j) for j in order[a:b] if j != i]
                    need = k - len(picked)
                    picked += cand if len(cand) <= need else [cand[j] for j in rng.choice(len(cand), need, replace=False)]
                    left, right = (a - 1, right) if dl <= dr else (left, b)
            for j in picked:                                  # exactly K per node, no self-loops
                V.add_edge(node, nodes[j], weight=1.0 / (1.0 + abs(float(vals[i] - vals[j]))))   # similarity in (0,1], always finite
        assert V.number_of_nodes() == self.G.number_of_nodes(), "node set changed"
        assert np.all(np.isfinite([w for _, _, w in V.edges(data='weight')])), "non-finite edge weight"
        return V


# Reads the input edgelist file into a networkx graph (shared loader: one graph definition for every stage).
def build_graph():
    '''Read input network.'''
    return graph_io.load_graph(args.input)


# Runs the whole pipeline: build graph -> build virtual graph -> save weighted edgelist.
def main(args):
    np.random.seed(args.seed)                                # build is deterministic given the seed (tied signatures are sampled)
    if args.output is None:
        ds = Path(args.input).stem
        args.output = f"output/notebook2_create_vir_graph/virtual_graphs/{ds}/k{args.k}/{args.sim}/virtual_graph.edgelist"
    G = build_graph()
    V = VirtualGraph(G, args.e, args.seed).build(args.sim, args.k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)   # auto-create per-dataset subfolder
    nx.write_weighted_edgelist(V, args.output)               # "u v weight" -> readable by NetworkX & PyG
    isolated = sum(1 for _, d in V.degree if d == 0)
    print(f"virtual graph | sim={args.sim} k={args.k} | nodes={V.number_of_nodes()} edges={V.number_of_edges()} "
          f"avg_deg={2 * V.number_of_edges() / V.number_of_nodes():.2f} isolated={isolated} -> {args.output}")


# Defines command-line options (mirrors experiments/train.py): input graph, output edgelist, similarity variant, K, seed.
def parse_args():
    '''Parses arguments.'''
    parser = argparse.ArgumentParser(description="Build a top-K structural-similarity virtual graph.")
    parser.add_argument('--input', nargs='?', default='input/cora.edgelist', help='Input graph path')
    parser.add_argument('--output', nargs='?', default=None,
                        help='Output virtual edgelist (default: output/notebook2_create_vir_graph/virtual_graphs/<ds>/k<K>/<sim>/virtual_graph.edgelist)')
    parser.add_argument('--sim', default='psi',
                        choices=list(PSI_ARMS) + ['degree', 'centrality', 'original', 'hybrid', 'hybrid_degree', 'hybrid_centrality', 'original_k', 'random_k'],
                        help='Structural similarity: psi=I2V KL/Poisson, psi_lambda=I2V KL rate only (unclamped), degree-only, centrality-only, '
                             'original=exact copy of input graph (control, K unused), hybrid[_degree|_centrality]=original + psi/degree/centrality top-K union, '
                             'original_k/random_k=density-matched controls (K real / K arbitrary neighbors per node). Default psi.')
    parser.add_argument('--k', type=int, default=10, help='Top-K structural neighbors per node. Default 10.')
    parser.add_argument('--e', type=float, default=2.7182, help='Euler constant (I2V).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default 42.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
