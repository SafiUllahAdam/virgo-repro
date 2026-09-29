'''Split graph edges 70/30 for link prediction, keep the train graph connected, sample equal non-edges. No leakage.'''

import argparse
import itertools
import os
import random
import networkx as nx

from virgo import graph_io


# Defines command-line options: which graph, output seed-folder, test fraction, seed.
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare leakage-free link-prediction splits.")
    parser.add_argument('--input', default='input/cora.edgelist', help='Input graph edgelist')
    parser.add_argument('--outdir', default=None,
                        help='Seed folder for the 4 split files (default: splits/link_prediction/original_graph/<dataset>/seed_<seed>)')
    parser.add_argument('--test-frac', type=float, default=0.3, help='Fraction of edges held out for test. Default 0.3.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    parser.add_argument('--negatives', default=None, choices=['component', 'uniform'],
                        help="Negative pairs: component = both endpoints in one component (default, as hard as the positives); "
                             "uniform = any node pair (Phase-1 / I2V protocol, inflates AUC on many-component graphs).")
    return parser.parse_args()


# Reads an edgelist into an undirected graph (WHOLE graph: disjoint-union datasets would lose ~99% of it to a largest-component filter).
def build_graph(path):
    '''Read input network via the shared loader. Every component is kept - the split must cover the dataset, not one component.'''
    return graph_io.load_graph(path)   # self-loops dropped there, so splits and virtual graphs see the SAME graph


# Holds out test_frac of edges as positives while forcing a spanning forest to stay in train (no component gets disconnected).
def split_edges(G, test_frac, seed):
    '''Spanning-forest edges stay in train; test positives are sampled from the rest.'''
    rng = random.Random(seed)
    tree = {frozenset(e) for e in nx.minimum_spanning_tree(G).edges()}   # disconnected input -> spanning FOREST, one tree per component
    removable = [frozenset(e) for e in G.edges() if frozenset(e) not in tree]
    rng.shuffle(removable)
    n_test = min(int(round(test_frac * G.number_of_edges())), len(removable))
    test_pos = removable[:n_test]
    train_pos = list(tree) + removable[n_test:]
    # sorted, not set-iteration order: the written row order decides node insertion order downstream, so it must not
    # depend on the interpreter's set hashing. Which edge is train vs test is already fixed by the seed above.
    return sorted(tuple(sorted(e)) for e in train_pos), sorted(tuple(sorted(e)) for e in test_pos)


# Samples k node pairs that are not real edges and not already used (the fake/negative pairs).
def sample_non_edges(G, k, seed, exclude, negatives=None):
    '''Draw k unique non-edges disjoint from the graph and from exclude.
       negatives='component' keeps both endpoints in ONE component (positives are within-component too, so the
       pair stays as hard as a real edge); 'uniform' draws from all node pairs (the Phase-1 / I2V protocol).
       None -> GRAPH_POLICY["lp_negatives"].'''
    negatives = negatives or graph_io.GRAPH_POLICY["lp_negatives"]
    rng = random.Random(seed + 1)                       # +1: split_edges already consumed this seed, keep the streams independent
    neg, seen = [], set()                               # list, not set: prepare() slices this into train/test, so draw order must be
                                                        # deterministic. Kept in RNG draw order - sorting here would put every low-id
                                                        # pair in train and every high-id pair in test.
    if negatives == 'uniform':
        nodes = list(G.nodes())
        while len(neg) < k:
            u, v = rng.choice(nodes), rng.choice(nodes)
            e = frozenset((u, v))
            if u != v and not G.has_edge(u, v) and e not in exclude and e not in seen:
                seen.add(e); neg.append(tuple(sorted(e)))
        return neg

    comps = [sorted(c) for c in nx.connected_components(G)]
    free = [len(c) * (len(c) - 1) // 2 - G.subgraph(c).number_of_edges() for c in comps]   # non-edges available inside each component
    pools = [c for c, f in zip(comps, free) if f > 0]   # a complete (or 1-node) component offers no non-edge
    # Components are drawn in proportion to their free non-edge capacity, then a pair is drawn inside the chosen one and
    # re-drawn if it hits an edge. That is NOT exactly uniform over all within-component non-edges: rejection makes a
    # denser component slightly less likely to yield an accepted pair, and the weights are not decremented as pairs are
    # consumed. The residual tilt toward sparser components is a few percent and does not affect which component is reachable.
    weights = list(itertools.accumulate(f for f in free if f > 0))                          # cumulative weights for rng.choices
    assert weights and weights[-1] >= k, \
        f"only {weights[-1] if weights else 0} within-component non-edges available, need {k} -> use negatives='uniform'"
    while len(neg) < k:
        pool = rng.choices(pools, cum_weights=weights)[0]
        u, v = rng.choice(pool), rng.choice(pool)
        e = frozenset((u, v))
        if u != v and not G.has_edge(u, v) and e not in exclude and e not in seen:
            seen.add(e); neg.append(tuple(sorted(e)))
    return neg


# Writes node-pair lines "u v" to a file, leaving it untouched when the content is unchanged.
def write_pairs(path, pairs):
    '''Save one "u v" pair per line. An unchanged file is not rewritten, so its mtime means "this split changed".'''
    text = "".join(f"{u} {v}\n" for u, v in pairs)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return                      # keep the mtime: reuse checks compare an embedding's mtime against this file
    with open(path, 'w') as f:
        f.write(text)


# Builds and writes the 4 split files into one seed folder; returns the counts. Importable by the runner.
def prepare(input_path, outdir, test_frac=0.3, seed=42, negatives=None):
    '''Split edges -> sample negatives -> write train.edgelist + train_neg/test_pos/test_neg.txt into outdir. Returns counts.'''
    os.makedirs(outdir, exist_ok=True)
    G = build_graph(input_path)
    train_pos, test_pos = split_edges(G, test_frac, seed)
    pos = {frozenset(e) for e in train_pos} | {frozenset(e) for e in test_pos}
    neg = sample_non_edges(G, len(train_pos) + len(test_pos), seed, pos, negatives)
    train_neg, test_neg = neg[:len(train_pos)], neg[len(train_pos):]

    p = str(outdir)
    write_pairs(os.path.join(p, "train.edgelist"), train_pos)   # retrain embeddings on THIS graph only
    write_pairs(os.path.join(p, "train_neg.txt"), train_neg)
    write_pairs(os.path.join(p, "test_pos.txt"), test_pos)
    write_pairs(os.path.join(p, "test_neg.txt"), test_neg)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "train_pos": len(train_pos), "train_neg": len(train_neg),
            "test_pos": len(test_pos), "test_neg": len(test_neg)}


# Runs the split from the command line and prints a summary.
def main(args):
    if args.outdir is None:
        ds = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = f"splits/link_prediction/original_graph/{ds}/seed_{args.seed}"
    c = prepare(args.input, args.outdir, args.test_frac, args.seed, args.negatives)
    print(f"{args.input}: nodes={c['nodes']} edges={c['edges']} seed={args.seed} negatives={args.negatives}")
    print(f"  train_pos={c['train_pos']} train_neg={c['train_neg']} test_pos={c['test_pos']} test_neg={c['test_neg']}")
    print(f"  wrote splits to {args.outdir}/  -> next: retrain the embedding on {args.outdir}/train.edgelist")


if __name__ == "__main__":
    main(parse_args())
