'''Predict whether node pairs are real edges or fake edges using embeddings; report link-prediction AUC.'''

import argparse
import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Binary operators that turn two node vectors into one edge feature (Hadamard is the node2vec default).
OPERATORS = {
    'hadamard': lambda a, b: a * b,
    'average': lambda a, b: (a + b) / 2.0,
    'l1': lambda a, b: np.abs(a - b),
    'l2': lambda a, b: (a - b) ** 2,
}


# Defines command-line options: embedding file, split seed-folder, edge operator, seed.
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate link prediction (AUC) from embeddings.")
    parser.add_argument('--emb', required=True, help='Embedding trained on the TRAIN graph (no leakage)')
    parser.add_argument('--splits', default='splits/link_prediction/original_graph/cora/seed_42',
                        help='Seed folder holding train.edgelist + train_neg/test_pos/test_neg.txt')
    parser.add_argument('--op', default='hadamard', choices=list(OPERATORS), help='Edge feature operator (logreg mode only). Default hadamard.')
    parser.add_argument('--score', default='cosine', choices=['cosine', 'dot', 'logreg'],
                        help="Scoring: cosine/dot = paper-faithful unsupervised similarity; logreg = supervised classifier. Default cosine.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return parser.parse_args()


# Reads "u v" pair lines into a list of (str, str) node-id pairs.
def load_pairs(path):
    '''Load node-id pairs as strings (gensim keys are strings).'''
    with open(path) as f:
        return [tuple(line.split()) for line in f if line.strip()]


# Builds an edge-feature matrix from pairs, skipping pairs whose nodes are missing from the embedding.
def edge_features(kv, pairs, op):
    '''Apply the operator to each present pair; return features and how many were skipped.'''
    fn = OPERATORS[op]
    feats = [fn(kv[u], kv[v]) for u, v in pairs if u in kv and v in kv]
    return np.array(feats), len(pairs) - len(feats)


# Scores held-out edges vs non-edges -> test AUC. Importable by the runner.
def evaluate(emb, split_dir, op='hadamard', seed=42, score='cosine'):
    '''split_dir = seed folder with train.edgelist + train_neg/test_pos/test_neg.txt.
       score='cosine'|'dot' = paper-faithful UNSUPERVISED embedding-similarity ranking (no classifier);
       score='logreg' = supervised edge classifier (--op feature -> logistic regression).'''
    kv = KeyedVectors.load_word2vec_format(str(emb))
    p = str(split_dir)
    test_pos = load_pairs(os.path.join(p, "test_pos.txt"))
    test_neg = load_pairs(os.path.join(p, "test_neg.txt"))

    if score in ('cosine', 'dot'):                       # paper-style: rank test edges by embedding similarity, no training
        def sim(u, v):
            a, b = kv[u], kv[v]
            d = float(np.dot(a, b))
            return d / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) if score == 'cosine' else d
        y, s = [], []
        for pairs, label in ((test_pos, 1), (test_neg, 0)):
            for u, v in pairs:
                if u in kv and v in kv:
                    s.append(sim(u, v)); y.append(label)
        return roc_auc_score(y, s)

    # supervised alternative: edge feature (--op, default Hadamard) -> logistic regression trained on the 70% split
    sets = {k: load_pairs(os.path.join(p, f"{k}.txt")) if k != 'train_pos' else load_pairs(os.path.join(p, "train.edgelist"))
            for k in ['train_pos', 'train_neg', 'test_pos', 'test_neg']}
    feats = {k: edge_features(kv, pairs, op)[0] for k, pairs in sets.items()}
    X_train = np.vstack([feats['train_pos'], feats['train_neg']])
    y_train = np.r_[np.ones(len(feats['train_pos'])), np.zeros(len(feats['train_neg']))]
    X_test = np.vstack([feats['test_pos'], feats['test_neg']])
    y_test = np.r_[np.ones(len(feats['test_pos'])), np.zeros(len(feats['test_neg']))]
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train, y_train)
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


# Runs the evaluation from the command line and prints the AUC.
def main(args):
    auc = evaluate(args.emb, args.splits, args.op, args.seed, args.score)
    print(f"{args.splits} link prediction | score={args.score} seed={args.seed} | AUC={auc:.4f}")


if __name__ == "__main__":
    main(parse_args())
