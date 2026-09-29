"""Embedding-model wrappers: each .train() writes ONE Word2Vec-format .emb so the existing eval
scripts (virgo/eval/nodeclass.py / linkpred.py) read every model identically. Add a model = add a class."""

import subprocess
import sys
from pathlib import Path

from virgo import graph_io

_ROOT = Path(__file__).resolve().parents[2]     # virgo/encoders/walk.py -> repo root


class EmbeddingModel:
    """Base: a model turns an edgelist into a Word2Vec .emb file. Subclasses implement train()."""
    name = "base"

    def train(self, edgelist, out_emb, seed, params):
        """Learn an embedding for the graph at `edgelist` and save it to `out_emb` (word2vec text format)."""
        raise NotImplementedError


class Identity2VecModel(EmbeddingModel):
    """Identity2Vec via the existing train.py (cached fast path + fixed seed) - the frozen baseline, unchanged."""
    name = "identity2vec"

    def train(self, edgelist, out_emb, seed, params):
        cmd = [sys.executable, str(_ROOT / "experiments" / "train.py"),
               "--input", str(edgelist), "--output", str(out_emb),
               "--dimensions", str(params["dimensions"]), "--walk-length", str(params["walk_length"]),
               "--num-walks", str(params["num_walks"]), "--window-size", str(params["window_size"]),
               "--epochs", str(params["epochs"]), "--sg", str(params["sg"]),
               "--seed", str(seed), "--cached"]
        subprocess.run(cmd, check=True, cwd=str(_ROOT))
        return Path(out_emb)


class _RandomWalkModel(EmbeddingModel):
    """Shared node2vec engine; DeepWalk is just node2vec with p=q=1 (uniform walks). workers=1 for determinism."""
    p = 1.0
    q = 1.0

    def train(self, edgelist, out_emb, seed, params):
        from node2vec import Node2Vec
        G = graph_io.load_graph(str(edgelist))     # shared loader: one graph definition for every model
        n2v = Node2Vec(G, dimensions=params["dimensions"], walk_length=params["walk_length"],
                       num_walks=params["num_walks"], p=self.p, q=self.q, workers=1, seed=seed, quiet=False)
        model = n2v.fit(window=params["window_size"], min_count=0, sg=params["sg"],
                        epochs=params["epochs"], workers=1, seed=seed,
                        alpha=0.025, min_alpha=0.01, sample=1e-3, negative=5, hs=0)   # Skipgram config matched to I2V (fair benchmark)
        model.wv.save_word2vec_format(str(out_emb))
        return Path(out_emb)


class DeepWalkModel(_RandomWalkModel):
    """DeepWalk: uniform random walks (p=q=1) + Skipgram."""
    name = "deepwalk"
    p = 1.0
    q = 1.0


class Node2VecModel(_RandomWalkModel):
    """node2vec: biased walks (p=1, q=0.5) so it stays a distinct method from DeepWalk (p=q=1); paper does not fix p/q."""
    name = "node2vec"
    p = 1.0
    q = 0.5


class Struc2VecModel(EmbeddingModel):
    """struc2vec via the vendored official CLI (third_party/struc2vec/src/main.py). Note: no seed -> not bit-reproducible."""
    name = "struc2vec"

    def train(self, edgelist, out_emb, seed, params):
        import shutil
        src = _ROOT / "third_party" / "struc2vec" / "src"
        pdir = src.parent / "pickles"                                # struc2vec caches distances/walks under FIXED names here
        shutil.rmtree(pdir, ignore_errors=True); pdir.mkdir(parents=True, exist_ok=True)   # clear so each graph is fresh (no cross-dataset reuse)
        (src / "random_walks.txt").unlink(missing_ok=True)
        n_nodes = graph_io.load_graph(str(edgelist)).number_of_nodes()
        opt = "True" if n_nodes > 10000 else "False"   # exact struc2vec on small graphs (cora/citeseer/webkb); optimizations only for large (enzymes ~19.5k -> memory)
        cmd = [sys.executable, str(src / "main.py"),
               "--input", str(Path(edgelist).resolve()), "--output", str(Path(out_emb).resolve()),
               "--dimensions", str(params["dimensions"]), "--walk-length", str(params["walk_length"]),
               "--num-walks", str(params["num_walks"]), "--window-size", str(params["window_size"]),
               "--iter", str(params["epochs"]), "--workers", "1",
               "--OPT1", opt, "--OPT2", opt, "--OPT3", opt]
        subprocess.run(cmd, check=True, cwd=str(src))                 # cwd=src so its imports + side-files stay contained
        return Path(out_emb)


# Registry: name -> model instance. Loop files and the runner select models by name from here.
MODELS = {m.name: m for m in (Identity2VecModel(), DeepWalkModel(), Node2VecModel(), Struc2VecModel())}


# Returns the model instance for a name, or raises listing the valid names.
def get_model(name):
    """Look up an embedding model by name."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name]
