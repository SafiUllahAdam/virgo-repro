'''Train logistic regression on node embeddings and report weighted F1 for node classification.'''

import argparse
import warnings
import numpy as np
from gensim.models import KeyedVectors
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


# Defines command-line options: embedding file, labels file, train fraction, seed.
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate node classification (weighted F1) from embeddings.")
    parser.add_argument('--emb', required=True, help='Embedding file (word2vec text format)')
    parser.add_argument('--labels', required=True, help='Labels file: "node_id label" per line')
    parser.add_argument('--train-frac', type=float, default=0.7, help='Fraction of labelled nodes for training. Default 0.7.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return parser.parse_args()


# Reads a labels file into a dict mapping node id (str) -> class label (str).
def load_labels(path):
    '''Load "node_id label" lines into a dict.'''
    labels = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                node, label = line.split()[:2]
                labels[node] = label
    return labels


# Aligns embeddings with labels, splits, trains logreg, returns (weighted F1, #nodes, #classes). Importable by runner.
def evaluate(emb, labels_path, train_frac=0.7, seed=42):
    '''Load embedding + labels -> stratified split -> logistic regression -> micro/macro/weighted F1.'''
    kv = KeyedVectors.load_word2vec_format(str(emb))
    labels = load_labels(labels_path)
    ids = [i for i in labels if i in kv]                 # keep only nodes that have both vector and label
    X = np.array([kv[i] for i in ids])
    y = np.array([labels[i] for i in ids])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_frac, stratify=y, random_state=seed)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))   # paper protocol: one-vs-rest LBFGS, 300 iters (L2 is sklearn's default penalty)
    with warnings.catch_warnings(record=True) as caught:   # the notebooks call filterwarnings("ignore"), which would otherwise
        warnings.simplefilter("always", ConvergenceWarning)   # hide non-convergence and make a too-low F1 look like a real result
        pred = clf.fit(X_train, y_train).predict(X_test)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print("  warning: logreg stopped at max_iter=300 without converging -> this F1 is a lower bound, not the protocol number")
    f1s = {avg: f1_score(y_test, pred, average=avg) for avg in ("micro", "macro", "weighted")}
    if len(ids) < len(labels):
        print(f"  warning: {len(labels) - len(ids)} labelled nodes had no embedding (skipped)")
    return f1s, len(ids), len(set(y))


# Runs the evaluation from the command line and prints the weighted F1.
def main(args):
    f1s, n, n_classes = evaluate(args.emb, args.labels, args.train_frac, args.seed)
    print(f"node classification | nodes={n} classes={n_classes} train_frac={args.train_frac} seed={args.seed} | "
          f"micro={f1s['micro']:.4f} macro={f1s['macro']:.4f} weighted={f1s['weighted']:.4f}")


if __name__ == "__main__":
    main(parse_args())
