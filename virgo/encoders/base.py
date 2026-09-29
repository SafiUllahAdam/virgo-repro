'''Unsupervised GNN over a virtual graph: same Skipgram-analog objective as I2V, lookup table replaced by message passing.'''
# Phase 3. Features come from the ORIGINAL graph's cached structural signals; messages pass over the VIRTUAL graph.
# Everything here is architecture-independent - a new encoder subclasses GNNEncoder and defines build_convs() only.
# Positives = direct virtual edges by default (ablation-A winner); "walk" kept as the Phase-2-bridge-comparable option.
# Ablation D: the `feats` knob selects which structural features enter the GNN ("random" = the no-structure control).
# Ablation D6: layers=0 -> no convolutions, embeddings ARE the z-normed features (features without message passing; nothing to train).

import hashlib
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from virgo.virtual_graph import VirtualGraph

# Walk-corpus settings mirror I2V_PARAMS (virgo/config.py) - the Phase-2 bridge corpus.
WALKS = {"num_walks": 10, "walk_length": 40, "window": 10}


class GNNEncoder():
    '''Virtual-graph GNN trained with Skipgram-style positives/negatives; subclasses supply the conv stack.'''

    name = "gnn"                                           # scoreboard/filename id; every subclass overrides it

    def __init__(self, G, V, seed, hidden=64, dimensions=64, layers=2, agg="mean", feats="all", cache=None):
        # layers=0 (D6) -> no convs: forward() returns the features unchanged, V is unused, train() is invalid.
        self.G, self.V, self.seed, self.feats, self.cache = G, V, seed, feats, cache
        torch.manual_seed(seed); np.random.seed(seed)         # CPU-only + seeded -> bit-reproducible
        self.nodes = list(G.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        e = [(self.index[u], self.index[v]) for u, v in V.edges]
        self.edge_index = torch.tensor([[a for a, b in e] + [b for a, b in e],
                                        [b for a, b in e] + [a for a, b in e]], dtype=torch.long)   # undirected -> both directions
        if agg == "weighted":                                  # ablation B "weighted": Ψ edge weights, normalized per target -> weighted mean
            w = torch.tensor([float(d.get("weight", 1.0)) for _, _, d in V.edges(data=True)] * 2, dtype=torch.float)
            denom = torch.zeros(len(self.nodes)).scatter_add_(0, self.edge_index[1], w)
            self.edge_weight = w / denom[self.edge_index[1]].clamp(min=1e-12)
        else:
            self.edge_weight = None
        self.X = self.features()
        dims = [self.X.shape[1]] + [hidden] * (layers - 1) + [dimensions]
        self.convs = self.build_convs(dims, agg)

    def build_convs(self, dims, agg):
        '''Return a ModuleList of len(dims)-1 convolutions mapping dims[i] -> dims[i+1]; the ONE thing an encoder defines.'''
        raise NotImplementedError(f"{type(self).__name__} must define build_convs(dims, agg)")

    def features(self):
        '''Structural node features from the ORIGINAL graph, z-normalized; ablation D `feats` selects the subset; disk-cached via `cache`.'''
        if self.feats == "random":                             # D4 control: node identity only, zero structural signal
            X = np.random.default_rng(self.seed).normal(size=(len(self.nodes), 4))
        elif self.feats == "const":                            # D5 floor: identical rows -> z-norm gives all-zeros
            X = np.ones((len(self.nodes), 1))
        elif self.cache and Path(self.cache).exists():         # sweep-wide reuse: 5 variants x 3 seeds share ONE computation per graph
            z = np.load(self.cache)
            f = dict(zip(z["nodes"].tolist(), z["X"]))
            X = np.array([f[n] for n in self.nodes])
        else:
            vg = VirtualGraph(self.G)
            psi_nodes, psi = vg.signatures('psi')              # reference-free I2V KL->Poisson score per node
            psi = dict(zip(psi_nodes, psi[:, 0]))              # keyed by node, like deg/ev/clus: signatures() owns its own node order
            deg, ev = vg.core.degree_node(), vg.core.eigenvector_centrality()
            clus = nx.clustering(self.G)
            X = np.array([[deg[n], ev[n], float(psi[n]), clus[n]] for n in self.nodes])
            if self.cache:                                     # cache ALL FOUR columns; any feats subset selects from them below
                Path(self.cache).parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.cache, nodes=np.array(self.nodes), X=X)
        if self.feats not in ("random", "const"):
            X = X[:, {"degree": [0], "centrality": [1], "psi": [2], "clustering": [3],
                      "deg_cent": [0, 1], "all": [0, 1, 2, 3]}[self.feats]]   # cached column order: deg, Ω, ψ, clustering
        return torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-9), dtype=torch.float)

    def corpus(self, max_pairs=2_000_000, positives="walk"):
        '''Skipgram positives - ablation A: "walk" = A1 window co-occurrence on virtual-graph walks (= Phase-2 bridge corpus); "edge" = A2 direct virtual edges, no walks.'''
        if positives == "edge":
            pairs = [(self.index[u], self.index[v]) for u, v in self.V.edges]
            pairs += [(b, a) for a, b in pairs]            # both directions: each endpoint also learns as the center node
        elif positives == "walk":
            from node2vec import Node2Vec
            U = nx.Graph(); U.add_nodes_from(self.V.nodes); U.add_edges_from(self.V.edges)   # unweighted copy = the bridge's walks
            walks = Node2Vec(U, dimensions=1, walk_length=WALKS["walk_length"], num_walks=WALKS["num_walks"],
                             p=1, q=1, workers=1, seed=self.seed, quiet=True).walks
            w = WALKS["window"]
            pairs = [(self.index[int(a)], self.index[int(b)]) for walk in walks
                     for i, a in enumerate(walk) for b in walk[i + 1:i + 1 + w] if a != b]
        else:
            raise ValueError(f"Unknown positives '{positives}'. Use: walk (A1), edge (A2).")
        assert len(pairs) > 0, "no positive pairs - is the virtual graph edgeless?"
        pairs = torch.tensor(pairs, dtype=torch.long)
        if len(pairs) > max_pairs:                             # deterministic cap: runtime/memory guard on large graphs
            keep = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(self.seed))[:max_pairs]
            pairs = pairs[keep]
        return pairs

    def forward(self):
        '''Full-graph message passing over the virtual edges -> one embedding per node.'''
        z = self.X
        for i, conv in enumerate(self.convs):
            z = conv(z, self.edge_index) if self.edge_weight is None else conv(z, self.edge_index, self.edge_weight)
            if i < len(self.convs) - 1:
                z = F.relu(z)
        return z

    def train(self, epochs, lr=0.01, negatives=5, pairs_per_epoch=100_000, max_pairs=2_000_000, positives="walk"):
        '''Skipgram-analog objective: pull positive pairs together, push deg^0.75 negatives (true non-neighbors) apart.'''
        assert len(self.convs) > 0, "layers=0 (D6) has no parameters to train - save() the features directly instead"
        pairs = self.corpus(max_pairs, positives)
        deg = torch.tensor([d for _, d in self.V.degree(self.nodes)], dtype=torch.float) ** 0.75
        neg_dist = deg / deg.sum() if deg.sum() > 0 else torch.full((len(self.nodes),), 1.0 / len(self.nodes))
        # A word2vec negative may land on a real neighbor: harmless when the graph is sparse, but on a dense graph the
        # deg^0.75 bias hits hubs, which are the likely neighbors (ogbl-ddi: 32% of draws). Reject those and redraw.
        N = len(self.nodes)
        key = lambda a, b: np.minimum(a, b).astype(np.int64) * N + np.maximum(a, b)   # undirected pair -> one int
        known = np.sort(key(self.edge_index[0].numpy(), self.edge_index[1].numpy()))
        opt = torch.optim.Adam((p for c in self.convs for p in c.parameters()), lr=lr)
        g = torch.Generator().manual_seed(self.seed)
        losses = []
        for epoch in range(epochs):
            z = self.forward()
            idx = torch.randint(len(pairs), (min(pairs_per_epoch, len(pairs)),), generator=g)
            u, v = pairs[idx, 0], pairs[idx, 1]
            n = torch.multinomial(neg_dist, len(u) * negatives, replacement=True, generator=g)
            uu = u.repeat_interleave(negatives).numpy()
            for _ in range(10):                                              # collisions shrink geometrically; sparse graphs exit at once
                k = key(uu, n.numpy())
                bad = torch.tensor((known[np.clip(np.searchsorted(known, k), 0, len(known) - 1)] == k) | (uu == n.numpy()))
                if not bad.any():
                    break
                n[bad] = torch.multinomial(neg_dist, int(bad.sum()), replacement=True, generator=g)
            n = n.view(len(u), negatives)
            pos = F.logsigmoid((z[u] * z[v]).sum(-1))                        # pull positives together
            neg = F.logsigmoid(-(z[u].unsqueeze(1) * z[n]).sum(-1)).sum(-1)  # push Q negatives apart
            loss = -(pos + neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch:3d} | loss {loss:.4f}")
        return losses

    def save(self, out):
        '''Write embeddings in word2vec text format so the existing eval scripts read them unchanged.'''
        with torch.no_grad():
            z = self.forward().numpy()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"{len(self.nodes)} {z.shape[1]}\n")
            for node, row in zip(self.nodes, z):
                f.write(f"{node} " + " ".join(f"{x:.6f}" for x in row) + "\n")
        return out


# Content-hashed cache path: same graph bytes -> same cache file, so a changed/regenerated graph can never reuse stale features.
def feature_cache(input_path):
    '''Cache path for a graph's structural features, keyed by the edgelist (+ .nodes sidecar) content hash.'''
    from virgo.config import OUTPUT_DIR                  # lazy: keeps the encoders importable without the dataset registry
    p = Path(input_path)
    s = p.with_suffix(".nodes")
    h = hashlib.md5(p.read_bytes() + (s.read_bytes() if s.exists() else b"")).hexdigest()[:12]
    return str(OUTPUT_DIR / "feature_cache" / f"{p.stem}_{h}.npz")   # anchored to the repo root, not the cwd
