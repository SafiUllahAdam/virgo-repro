"""Shared helpers: seeding, embedding loading, auto-incrementing run IDs."""

import glob
import random
import re
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors


# Fixes the Python and NumPy random seeds so every run is reproducible.
def set_seed(seed=42):
    """Seed the RNGs (project standard seed is 42)."""
    random.seed(seed)
    np.random.seed(seed)


# Loads a word2vec-format .emb file as a gensim KeyedVectors (keys are node-id strings).
def load_embeddings(path):
    """Read an embedding file."""
    return KeyedVectors.load_word2vec_format(str(path))


# Finds the next NNN run id by looking at existing numbered CSVs in the results dir.
def next_run_id(results_dir):
    """Auto-increment the 3-digit run id from existing result filenames."""
    ids = [int(m.group(1)) for f in glob.glob(str(Path(results_dir) / "*.csv"))
           if (m := re.match(r"^(\d{3})\.", Path(f).name))]
    return max(ids, default=0) + 1
