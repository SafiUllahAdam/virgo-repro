'''Train one GNN encoder over a saved virtual graph -> .emb (single-run CLI; the sweeps live in run_core.py / run_ogb.py).'''
# Architecture-agnostic: --arch picks any encoder registered in virgo.encoders, so a new encoder is runnable
# from the terminal the moment it is added to ENCODERS - this file never changes.

import argparse
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import graph_io
from virgo.encoders import ENCODERS, feature_cache

ARCHS = {c.name: c for c in ENCODERS.values()}                    # study id -> class, collapsed to architecture name -> class


# Reads an edgelist file into a networkx graph (shared loader: one graph definition for every stage).
def build_graph(path):
    '''Read input network.'''
    return graph_io.load_graph(path)


# Runs the whole pipeline: load original + virtual graph -> train the chosen encoder -> save .emb.
def main(args):
    ds = Path(args.input).stem
    cls = ARCHS[args.arch]
    tag = ("features_only" if args.layers == 0 else                          # D6: no message passing -> not an encoder row
           f"{cls.name}_{args.positives}" + ("" if args.agg == "mean" else f"_{args.agg}")
           + ("" if args.features == "all" else f"_feat_{args.features}"))   # each ablation writes its own files
    virtual = args.virtual or f"output/notebook2_create_vir_graph/virtual_graphs/{ds}/k{args.k}/{args.sim}/virtual_graph.edgelist"
    output = args.output or f"output/notebook3_gnn_encoder/node_classification/{ds}/k{args.k}/{args.sim}/{tag}_s{args.seed}.emb"
    G = build_graph(args.input)
    V = nx.read_weighted_edgelist(virtual, nodetype=int)
    V.add_nodes_from(G.nodes)                                  # edgelists omit isolated nodes -> restore the full node set
    enc = cls(G, V, args.seed, hidden=args.hidden, dimensions=args.dimensions, layers=args.layers,
              agg=args.agg, feats=args.features, cache=feature_cache(args.input))
    if args.layers:                                            # D6 (layers=0): the features are the embedding, nothing to train
        enc.train(args.epochs, lr=args.lr, negatives=args.negatives, positives=args.positives)
    print(f"virgo-{cls.name} | sim={args.sim} k={args.k} positives={args.positives} agg={args.agg} feats={args.features} "
          f"seed={args.seed} -> {enc.save(output)}")


# Defines command-line options (mirrors virtual_graph.py); defaults mirror virgo/config.GNN_PARAMS.
def parse_args():
    '''Parses arguments.'''
    parser = argparse.ArgumentParser(description="Train an unsupervised GNN encoder over a saved virtual graph.")
    parser.add_argument('--input', nargs='?', default='input/cora.edgelist', help='Original graph (features + node set)')
    parser.add_argument('--virtual', nargs='?', default=None,
                        help='Virtual edgelist (default: output/notebook2_create_vir_graph/virtual_graphs/<ds>/k<K>/<sim>/virtual_graph.edgelist)')
    parser.add_argument('--output', nargs='?', default=None,
                        help='Output .emb (default: output/notebook3_gnn_encoder/node_classification/<ds>/k<K>/<sim>/<encoder>_s<seed>.emb; '
                             'trains on the full graph -> for link-prediction embeddings pass a train graph via --input and set --output)')
    parser.add_argument('--arch', default='graphsage', choices=list(ARCHS),
                        help='Encoder architecture (any registered in virgo.encoders). Default graphsage = the locked encoder.')
    parser.add_argument('--sim', default='psi',
                        choices=['psi', 'degree', 'centrality', 'original', 'hybrid', 'hybrid_degree', 'hybrid_centrality'],
                        help='Virtual-graph variant (original=copy of input graph, hybrid[_degree|_centrality]=original + '
                             'psi/degree/centrality top-K union). Default psi.')
    parser.add_argument('--k', type=int, default=10, help='Top-K of the virtual graph. Default 10.')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs. Default 50.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate. Default 0.01.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden width. Default 64.')
    parser.add_argument('--dimensions', type=int, default=64, help='Embedding size (matches I2V). Default 64.')
    parser.add_argument('--layers', type=int, default=2,
                        help='Message-passing layers (ablation C depth 1-3; 0=D6 features-only, no message passing). Default 2.')
    parser.add_argument('--negatives', type=int, default=5, help='Negatives per positive (matches Word2Vec negative=5). Default 5.')
    parser.add_argument('--positives', default='edge', choices=['walk', 'edge'],
                        help='Ablation A: edge=A2 direct virtual edges (winner, default), walk=A1 walk co-occurrence (bridge-comparable).')
    parser.add_argument('--agg', default='mean', choices=['mean', 'weighted', 'sum', 'max'],
                        help='Ablation B neighbor aggregation: mean | weighted (Ψ-weighted mean) | sum | max. GraphSAGE only. Default mean.')
    parser.add_argument('--features', default='all',
                        choices=['all', 'degree', 'deg_cent', 'psi', 'centrality', 'clustering', 'random', 'const'],
                        help='Ablation D input features: all=D0 [deg,Ω,ψ,clustering] | degree=D1 | deg_cent=D2 | psi=D3 | '
                             'centrality=D7 | clustering=D8 | random=D4 control (no structural signal) | const=D5 floor. Default all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default 42.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
