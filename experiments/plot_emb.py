'''draws the node embeddings as a 2D picture, colored by degree, to check if hubs and leaves separate.'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on the path -> `import virgo` works from any cwd

from virgo import graph_io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# Read the three inputs from the command line: embedding file, edgelist file, and output image path.
emb_file, edge_file, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

# Load the trained node embeddings and the original graph.
kv = KeyedVectors.load_word2vec_format(emb_file)
G = graph_io.load_graph(edge_file)   # shared loader: the degree colouring must match the graph every other stage saw

# Stack every node's vector into a matrix and subtract the mean so the data is centered at zero.
X = np.array([kv[k] for k in kv.index_to_key])
X = X - X.mean(0)

# Squash the high-dimensional vectors down to 2D (PCA via SVD) so we can plot them on a flat picture.
P = X @ np.linalg.svd(X, full_matrices=False)[2][:2].T

# Get each node's degree (how many neighbors it has).
deg = np.array([G.degree(int(k)) for k in kv.index_to_key])

# Split nodes into three groups by degree: leaves (<2), hubs (>10), and everything in between.
low = np.where(deg < 2)[0]
high = np.where(deg > 10)[0]
mid = np.where((deg >= 2) & (deg <= 10))[0]

# Draw each group in its own color, then save the picture to file.
plt.figure(figsize=(6.5, 6))
plt.scatter(P[mid, 0], P[mid, 1], s=10, c="lightblue", alpha=0.5, label=f"degree 2-10 ({len(mid)} nodes)")
plt.scatter(P[low, 0], P[low, 1], s=14, c="tab:red", alpha=0.8, label=f"degree < 2 ({len(low)} nodes)")
plt.scatter(P[high, 0], P[high, 1], s=14, c="tab:green", alpha=0.8, label=f"degree > 10 ({len(high)} nodes)")
plt.legend()
plt.title(f"{emb_file} - hubs (green) vs leaves (red)")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved", out_png)