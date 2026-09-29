'''
Codes created by: Muhammad Safi Ullah ADAM
Email: muhammadsafi2299@gmail.com
'''

import argparse
import sys
from pathlib import Path

from gensim.models import Word2Vec
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import graph_io
from virgo import identity2vec
from virgo import identity2vec_cached

'''This is the run file. It is the file you execute from terminal.

Its job is: load graph → call Identity2Vec walks → train Skipgram → save embeddings. '''

# Defines all command-line options (which graph to read, where to save, walk settings) and their default values.
def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run identity2vec.")

    parser.add_argument('--input', nargs='?', default='input/cora.edgelist',
                        help='Input graph path')        
    
    parser.add_argument('--output', nargs='?', default='output/cora.emb',
                        help='Output embedding path')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')
    
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--epochs', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers. Default is 1.')

    parser.add_argument('--min-count', type=int, default=0,
                        help='Minimum count of Training words. Default is 0.')
    
    parser.add_argument('--sg', type=int, default=1,
                        help='Training Algorithm. CBOW=0,SkipGram=1. Default is 1.')
    
    parser.add_argument('--e', type=int, default=2.7182,
                        help='Euler Constant')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility. Default 42.')

    parser.add_argument('--cached', action='store_true',
                        help='Use cached structural signals (identical output, much faster).')
    
    return parser.parse_args()

# Reads the input edgelist file into a networkx graph and gives every edge a weight of 1.
def build_graph():
    '''Read input network'''
    
    G = graph_io.load_graph(args.input)   # shared loader: same delimiter + self-loop policy as splits and virtual graphs
    for e in G.edges:
        G.edges[e]['weight'] = 1 
        
    return G
# NetworkX is a Python library for working with graphs. We need it because raw text like 10, 25 is not useful directly. NetworkX converts it into a graph object, to ask Eigenvector centrality, shortest path, neighbours etc


# Feeds the random walks into Word2Vec/SkipGram to learn one vector (embedding) per node, then saves them to the output file.
def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    identitywalks = [list(map(str, walk)) for walk in walks]
    print("Training Node Corpus...")
    model = Word2Vec(identitywalks, vector_size=args.dimensions, window=args.window_size, 
                     min_count=args.min_count, sg=args.sg, workers=args.workers, epochs=args.epochs,  
                    sample=1e-3, alpha=0.025, min_alpha=0.01, negative=5, seed=args.seed)
    print("Saving Embeddings...")
    model.wv.save_word2vec_format(args.output)
    
    return model

        
# Runs the whole pipeline in order: build graph -> generate identity walks -> train and save embeddings.
def main(args):
    np.random.seed(args.seed)                     # seed global np.random so identity2vec's walks are reproducible
    nx_Graph = build_graph()
    Graph = identity2vec_cached.Graph if args.cached else identity2vec.Graph
    G = Graph(nx_Graph, args.e)
    walks = G.identity2vec_walk(args.num_walks, args.walk_length)
    learn_embeddings(walks) 

if __name__ == "__main__":
    args = parse_args()
    main(args)
    


