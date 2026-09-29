'''Evaluate OGB-sourced datasets under the OFFICIAL OGB protocol: ogbn-arxiv Accuracy (official split), ogbl-ddi Hits@20.'''
# Separate from eval_nodeclass/eval_linkpred: OGB datasets keep OGB's FIXED split + OGB Evaluator, not the core random 70/30.
# No-test-peeking is ENFORCED by the split argument: one call scores ONE split. Use "valid" while choosing
# variants/encoders; call with "test" exactly once, after all choices are locked. Never both in one call.
# Same decoder + same linear probe for every variant -> fair internal comparison; only the split/metric are OGB's.

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

# Link-decoder settings. hidden/lr mirror OGB's own ddi reference model (examples/linkproppred/ddi/gnn.py);
# epochs + pairs_per_epoch are runtime caps (our embeddings are FROZEN, so only the small MLP trains).
DECODER = {"hidden": 256, "epochs": 50, "lr": 0.005, "pairs_per_epoch": 100_000, "batch": 32_768}


# Reads a labels file into {node_id(int) -> class(int)}.
def _load_labels(path):
    '''Load "node_id label" lines into an int->int dict.'''
    y = {}
    for line in open(path):
        if line.strip():
            n, lab = line.split()[:2]
            y[int(n)] = int(lab)
    return y


# ogbn-arxiv node classification: probe fits on official TRAIN ids, scores ONE chosen split (valid=selection, test=final).
def evaluate_nodeclass(emb, labels_path, split_npz, seed=42, split="valid"):
    '''Fit the logreg probe on official train nodes -> Accuracy (OGB Evaluator) + weighted/macro F1 on ONE split.'''
    from ogb.nodeproppred import Evaluator
    assert split in ("valid", "test"), f"split must be 'valid' (selection) or 'test' (final, read once), got '{split}'"
    kv = KeyedVectors.load_word2vec_format(str(emb))
    y = _load_labels(labels_path)
    s = np.load(split_npz)
    X = lambda ids: np.array([kv[str(int(i))] for i in ids])                       # same probe protocol as the core datasets
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))
    clf.fit(X(s["train"]), np.array([y[int(i)] for i in s["train"]]))
    pred = clf.predict(X(s[split]))
    true = np.array([y[int(i)] for i in s[split]])
    acc = Evaluator(name="ogbn-arxiv").eval({"y_true": true.reshape(-1, 1), "y_pred": pred.reshape(-1, 1)})["acc"]
    return {"split": split, "acc": acc,
            "weighted_f1": f1_score(true, pred, average="weighted"), "macro_f1": f1_score(true, pred, average="macro")}


# OGB's own ddi reference model does not score pairs by cosine: it trains a small MLP on the elementwise product of the
# two embeddings. Cosine cannot learn WHICH directions mean "linked", and it saturates at exactly 1.0 for structural twins
# (identical embeddings), where Hits@K's strict `>` can never be beaten. The decoder only ever sees TRAINING edges.
class LinkDecoder():
    '''OGB-style link predictor over FROZEN embeddings: hadamard(z_u, z_v) -> MLP -> score.'''

    def __init__(self, Z, index, seed, hidden=256):
        torch.manual_seed(seed)                                                    # CPU-only + seeded -> reproducible
        self.Z = torch.tensor(np.asarray(Z), dtype=torch.float)
        self.index = index                                                         # node id -> embedding row
        self.net = torch.nn.Sequential(torch.nn.Linear(self.Z.shape[1], hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 1))

    def rows(self, pairs):
        '''Map an (E,2) array of node ids to embedding row indices.'''
        return torch.tensor([[self.index[int(u)], self.index[int(v)]] for u, v in pairs], dtype=torch.long)

    def train(self, edges, epochs=50, lr=0.005, pairs_per_epoch=100_000, batch=32_768, seed=42):
        '''Fit on TRAINING edges (positives) vs uniform non-edges (negatives), BCE; no valid/test pair is ever seen.'''
        E = self.rows(edges).numpy()
        n = self.Z.shape[0]
        key = lambda a, b: np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)  # undirected pair -> one int
        known = np.sort(key(E[:, 0], E[:, 1]))                                     # sorted keys: membership without a python set
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(epochs):
            pos = torch.tensor(E[rng.integers(len(E), size=pairs_per_epoch)], dtype=torch.long)
            a, b = rng.integers(n, size=pairs_per_epoch), rng.integers(n, size=pairs_per_epoch)
            i = np.searchsorted(known, key(a, b))                                  # fresh negatives every epoch, as OGB does
            ok = (a != b) & ~(known[np.clip(i, 0, len(known) - 1)] == key(a, b))   # drop self-loops and real edges
            neg = torch.tensor(np.c_[a[ok], b[ok]], dtype=torch.long)
            for cp, cn in zip(pos.split(batch), neg.split(batch)):
                sp = self.net(self.Z[cp[:, 0]] * self.Z[cp[:, 1]]).squeeze(-1)
                sn = self.net(self.Z[cn[:, 0]] * self.Z[cn[:, 1]]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(sp, torch.ones_like(sp)) + \
                       F.binary_cross_entropy_with_logits(sn, torch.zeros_like(sn))
                opt.zero_grad(); loss.backward(); opt.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    decoder epoch {epoch:3d} | loss {loss.item():.4f}", flush=True)
        return self

    def score(self, pairs, batch=32_768):
        '''Rank score per pair (logits: Hits@K only needs the ordering).'''
        p = self.rows(pairs)
        with torch.no_grad():
            return torch.cat([self.net(self.Z[c[:, 0]] * self.Z[c[:, 1]]).squeeze(-1) for c in p.split(batch)]).numpy()


# ogbl-ddi link prediction: score ONE split's official pos/neg pairs -> Hits@20 (OGB Evaluator, K=20 fixed).
def evaluate_linkpred(emb, pairs_npz, split="valid", scorer="mlp", train_edgelist=None, seed=42, params=None):
    '''Score the chosen split's OGB pos+neg pairs -> Hits@20; scorer "mlp" = decoder trained on TRAINING edges only, "cosine" = fixed.'''
    from ogb.linkproppred import Evaluator
    assert split in ("valid", "test"), f"split must be 'valid' (selection) or 'test' (final, read once), got '{split}'"
    kv = KeyedVectors.load_word2vec_format(str(emb))
    P = np.load(pairs_npz)
    if scorer == "cosine":                                                         # the pre-decoder protocol, kept reproducible
        def f(pairs):
            a = np.array([kv[str(int(u))] for u, v in pairs]); b = np.array([kv[str(int(v))] for u, v in pairs])
            return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    else:
        assert train_edgelist, "the mlp scorer needs the TRAINING edgelist - its only source of positive pairs"
        p = params or DECODER
        d = LinkDecoder(kv.vectors, {int(k): i for k, i in kv.key_to_index.items()}, seed, hidden=p["hidden"])
        d.train(np.loadtxt(train_edgelist, dtype=int), epochs=p["epochs"], lr=p["lr"],
                pairs_per_epoch=p["pairs_per_epoch"], batch=p["batch"], seed=seed)
        f = d.score
    res = Evaluator(name="ogbl-ddi").eval({"y_pred_pos": f(P[f"{split}_pos"]), "y_pred_neg": f(P[f"{split}_neg"])})
    return {"split": split, "hits@20": res["hits@20"]}                             # K=20 is OGB's fixed cutoff for ddi


# Defines command-line options: OGB task + embedding + label/split/pairs paths + which split to score.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Evaluate OGB-sourced datasets under the official OGB metric (Accuracy / Hits@20).")
    p.add_argument('--task', required=True, choices=['nodeclass', 'linkpred'], help='ogbn-arxiv=nodeclass, ogbl-ddi=linkpred')
    p.add_argument('--emb', required=True, help='Embedding file (word2vec text format)')
    p.add_argument('--labels', help='ogbn-arxiv labels file (nodeclass)')
    p.add_argument('--split-npz', help='ogbn-arxiv split npz (nodeclass)')
    p.add_argument('--pairs', help='ogbl-ddi pos/neg pairs npz (linkpred)')
    p.add_argument('--scorer', default='mlp', choices=['mlp', 'cosine'],
                   help='linkpred pair scorer: mlp = OGB-style decoder trained on training edges (default), cosine = fixed.')
    p.add_argument('--train-edgelist', help='linkpred: the TRAINING edgelist the mlp decoder fits on (required for --scorer mlp)')
    p.add_argument('--split', default='valid', choices=['valid', 'test'],
                   help='valid = model selection (default); test = the FINAL number, run once after all choices are locked.')
    p.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return p.parse_args()


# Runs the OGB evaluation from the command line and prints the official metric for the chosen split.
def main(args):
    if args.task == 'nodeclass':
        r = evaluate_nodeclass(args.emb, args.labels, args.split_npz, args.seed, args.split)
        print(f"ogbn-arxiv node classification | split={r['split']} | acc={r['acc']:.4f} "
              f"(secondary: weighted_f1={r['weighted_f1']:.4f} macro_f1={r['macro_f1']:.4f})")
    else:
        r = evaluate_linkpred(args.emb, args.pairs, args.split, args.scorer, args.train_edgelist, args.seed)
        print(f"ogbl-ddi link prediction | split={r['split']} | scorer={args.scorer} | hits@20={r['hits@20']:.4f}")


if __name__ == "__main__":
    main(parse_args())
