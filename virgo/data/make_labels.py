"""Build labels/<dataset>.labels from the original LINQS data, matching the edgelist's node numbering.

The author's edgelists number nodes by order of appearance in the LINQS '<name>.content' file
(verified: cora edges reproduced at 100% overlap). Planetoid uses a different numbering, so it is
NOT used here. Downloads are plain urllib (no fsspec)."""

import io
import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path

LINQS = {
    "cora":     "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    "citeseer": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
}

# WebKB: the I2V author's own repo (paper ref [16] = networkrepository). Wisconsin subset = same graph as input/webkb.edgelist.
WEBKB_ZIP = "https://raw.githubusercontent.com/ikenna-oluigbo/webkb-dataset/master/webkb.zip"


# Reads the local edgelist into its set of undirected integer edges.
def read_edgelist(path):
    edges = set()
    for line in open(path):
        if line.strip():
            a, b = map(int, line.replace(",", " ").split()[:2])   # some networkrepository edgelists are comma-delimited (proteins)
            edges.add(tuple(sorted((a, b))))
    return edges


# Downloads the LINQS .tgz and returns the .content and .cites lines for a dataset.
def fetch_linqs(name, retries=3, timeout=60):
    """Get original <name>.content and <name>.cites text from LINQS via urllib."""
    for attempt in range(1, retries + 1):
        try:
            raw = urllib.request.urlopen(LINQS[name], timeout=timeout).read()
            break
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {LINQS[name]}: {e}")
    tar = tarfile.open(fileobj=io.BytesIO(raw))
    pick = lambda end: tar.extractfile(next(m for m in tar.getmembers() if m.name.endswith(end))).read().decode()
    return pick(f"{name}.content").splitlines(), pick(f"{name}.cites").splitlines()


# Builds labels/<name>.labels in the edgelist's numbering, after verifying the edges match.
def make_labels(name):
    """Map paper-id -> 1-based node id by content order; verify against the edgelist; write labels."""
    name = name.lower()
    content, cites = fetch_linqs(name)
    paper_ids = [ln.split("\t")[0] for ln in content]
    paper_labels = [ln.split("\t")[-1] for ln in content]
    idmap = {pid: i + 1 for i, pid in enumerate(paper_ids)}     # content order = the author's node numbering

    cite_edges = {tuple(sorted((idmap[a], idmap[b])))
                  for a, b in (ln.split("\t")[:2] for ln in cites) if a in idmap and b in idmap}
    local_edges = read_edgelist(f"input/{name}.edgelist")
    overlap = len(cite_edges & local_edges) / len(local_edges)
    print(f"{name}: content_nodes={len(paper_ids)} labels={len(set(paper_labels))} edge_overlap={overlap:.4f}")

    if overlap < 0.90:
        print(f"STOP: {name} content-order does not match input/{name}.edgelist (overlap {overlap:.4f})")
        return
    Path("labels").mkdir(exist_ok=True)
    out = f"labels/{name}.labels"
    with open(out, "w") as f:
        for node_id, label in enumerate(paper_labels, start=1):
            f.write(f"{node_id} {label}\n")
    print("WROTE", out, f"({len(paper_labels)} nodes)")


# Builds input/webkb_wisc.edgelist + labels/webkb_wisc.labels from the author's WebKB zip (Wisconsin subset).
def make_webkb():
    """Download the author's WebKB; write the Wisconsin edgelist + node-class labels (content line order = node id)."""
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(WEBKB_ZIP, timeout=60).read()))
    content = [ln for ln in z.read("webkb/wisconsin.content").decode().splitlines() if ln.strip()]
    edges = [ln.strip() for ln in z.read("webkb/wisconsin_edges.edgelist").decode().splitlines() if ln.strip()]
    labels = [ln.split()[-1] for ln in content]                # last field of each .content line = class

    Path("input").mkdir(exist_ok=True)
    with open("input/webkb_wisc.edgelist", "w") as f:
        f.write("\n".join(edges) + "\n")
    Path("labels").mkdir(exist_ok=True)
    with open("labels/webkb_wisc.labels", "w") as f:
        for node_id, label in enumerate(labels, start=1):
            f.write(f"{node_id} {label}\n")

    node_ids = {int(x) for ln in edges for x in ln.split()[:2]}
    assert node_ids <= set(range(1, len(labels) + 1)), "edge node id outside label range"
    print(f"webkb(wisconsin): nodes={len(labels)} edges={len(edges)} classes={sorted(set(labels))}")
    print("WROTE input/webkb_wisc.edgelist + labels/webkb_wisc.labels")


# Builds an aligned LINQS Citeseer: input/citeseer_linqs.edgelist + labels/citeseer_linqs.labels from one source (ids match).
def make_citeseer_linqs():
    """Build the LINQS Citeseer graph AND labels together so node ids and labels align by construction."""
    content, cites = fetch_linqs("citeseer")
    paper_ids = [ln.split("\t")[0] for ln in content]
    paper_labels = [ln.split("\t")[-1] for ln in content]
    idmap = {pid: i + 1 for i, pid in enumerate(paper_ids)}     # content order = node id = the basis of the labels
    edges = sorted({tuple(sorted((idmap[a], idmap[b])))
                    for a, b in (ln.split("\t")[:2] for ln in cites)
                    if a in idmap and b in idmap and a != b})    # drop danglers + self-loops

    Path("input").mkdir(exist_ok=True)
    with open("input/citeseer_linqs.edgelist", "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    Path("labels").mkdir(exist_ok=True)
    with open("labels/citeseer_linqs.labels", "w") as f:
        for node_id, label in enumerate(paper_labels, start=1):
            f.write(f"{node_id} {label}\n")
    print(f"citeseer_linqs: nodes={len(paper_ids)} edges={len(edges)} classes={len(set(paper_labels))}")
    print("WROTE input/citeseer_linqs.edgelist + labels/citeseer_linqs.labels")


# Downloads a networkrepository .zip and returns the text of its edges + node_labels members.
def _fetch_nr(url, edges_member, labels_member, timeout=60):
    """Get (edges_text, node_labels_text) from a networkrepository zip via urllib."""
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=timeout).read()))
    return z.read(edges_member).decode(), z.read(labels_member).decode()


# Parses "u,v" or "u v" edge text into a set of undirected int edges (skips % comment lines).
def _parse_edges(text):
    edges = set()
    for ln in text.splitlines():
        ln = ln.strip().replace(",", " ")
        if ln and not ln.startswith("%"):
            a, b = map(int, ln.split()[:2])
            edges.add(tuple(sorted((a, b))))
    return edges


# Writes labels/<name>.labels if networkrepository ids match input/<name>.edgelist; else an aligned <name>_nr pair. Returns the version used.
def _make_nr_labels(name, url, edges_member, labels_member):
    """Verify source ids align with input/<name>.edgelist; write aligned labels, else build a self-aligned <name>_nr graph+labels."""
    edges_txt, labels_txt = _fetch_nr(url, edges_member, labels_member)
    labels = [ln.strip() for ln in labels_txt.splitlines() if ln.strip() and not ln.startswith("%")]
    src_edges = _parse_edges(edges_txt)
    local = read_edgelist(f"input/{name}.edgelist")
    overlap = len(src_edges & local) / len(local)
    print(f"{name}: source_nodes={len(labels)} classes={len(set(labels))} edge_overlap={overlap:.4f}")
    Path("labels").mkdir(exist_ok=True)
    if overlap >= 0.90:                                          # ids already align -> label by line number = node id
        with open(f"labels/{name}.labels", "w") as f:
            for node_id, lab in enumerate(labels, start=1):
                f.write(f"{node_id} {lab}\n")
        print("WROTE", f"labels/{name}.labels", f"({len(labels)} nodes)")
        return name
    Path("input").mkdir(exist_ok=True)                          # misaligned -> rebuild graph+labels from one source (ids align by construction)
    with open(f"input/{name}_nr.edgelist", "w") as f:
        for u, v in sorted(src_edges):
            f.write(f"{u} {v}\n")
    with open(f"labels/{name}_nr.labels", "w") as f:
        for node_id, lab in enumerate(labels, start=1):
            f.write(f"{node_id} {lab}\n")
    print(f"MISALIGNED (overlap {overlap:.4f}) -> WROTE input/{name}_nr.edgelist + labels/{name}_nr.labels (use '{name}_nr')")
    return f"{name}_nr"


# Builds Enzymes node labels (TU ENZYMES.node_labels, one per line = node id) aligned to input/enzymes.edgelist.
def make_enzymes():
    """Build/verify Enzymes node-class labels from networkrepository (paper ref [16])."""
    return _make_nr_labels("enzymes",
        "https://nrvis.com/download/data/labeled/ENZYMES.zip", "ENZYMES.edges", "ENZYMES.node_labels")


# Builds Proteins labels + a whitespace copy of the comma-delimited author edgelist, both from the same networkrepository source.
def make_proteins():
    """input/proteins.edgelist is comma-delimited (every reader here expects whitespace) -> rebuild graph+labels from its own source; ids align by construction."""
    edges_txt, labels_txt = _fetch_nr(
        "https://nrvis.com/download/data/labeled/PROTEINS.zip", "proteins.edges", "proteins.node_labels")
    edges = sorted(_parse_edges(edges_txt))
    labels = [ln.split(",")[-1].strip() for ln in labels_txt.splitlines()
              if ln.strip() and not ln.startswith("%")]           # PROTEINS ships "node_id,label"; ENZYMES ships a bare label per line
    overlap = len(set(edges) & read_edgelist("input/proteins.edgelist")) / len(edges)
    print(f"proteins: source_nodes={len(labels)} classes={len(set(labels))} edge_overlap={overlap:.4f}")
    assert overlap == 1.0, f"source != input/proteins.edgelist (overlap {overlap:.4f}); refusing to write unverified labels"
    Path("input").mkdir(exist_ok=True)
    with open("input/proteins_nr.edgelist", "w") as f:            # author file untouched: the copy only swaps commas for spaces
        for u, v in edges:
            f.write(f"{u} {v}\n")
    Path("labels").mkdir(exist_ok=True)
    with open("labels/proteins_nr.labels", "w") as f:
        for node_id, lab in enumerate(labels, start=1):
            f.write(f"{node_id} {lab}\n")
    print(f"WROTE input/proteins_nr.edgelist ({len(edges)} edges) + labels/proteins_nr.labels ({len(labels)} nodes)")
    return "proteins_nr"


# Politics node-class labels are unavailable for input/politics.edgelist (rt-pol): no verifiable same-dataset source. STOPs (LP-only).
def make_politics():
    """rt-pol ships no labels; the original Conover retweet source is offline/unmappable -> NO fabrication. NC unavailable, LP still works."""
    raise RuntimeError(
        "Politics NODE classification is unavailable. input/politics.edgelist is networkrepository 'rt-pol' "
        "(https://nrvis.com/download/data/rt/rt-pol.zip), which ships the graph only (no node_labels). The original "
        "left/right labels (Conover et al. 2011, ICWSM) are hosted at cnets.indiana.edu, which is offline, and rt-pol's "
        "node renumbering cannot be mapped back, so labels CANNOT be verified against this graph. Per policy we do not "
        "fabricate labels. LINK PREDICTION needs no labels and runs fine on politics. To enable NC, drop in a manually "
        "verified labels/politics.labels (node id = our edgelist numbering).")


# Maps a requested dataset to the version actually used; auto-aligns Citeseer (author graph has no matching labels).
def resolve_dataset(name):
    """Return the dataset version to use. Author 'citeseer' = paper's own graph (link prediction only, no aligned labels)."""
    name = name.lower()
    if name == "citeseer":
        print("Using author 'citeseer' graph (paper's own file) -> link prediction only (no aligned labels).")
    if name == "citeseer_linqs" and not os.path.exists("input/citeseer_linqs.edgelist"):
        make_citeseer_linqs()                                    # build the aligned pair before training
    return name


# Single entry the notebook/CLI call: reuse existing labels, else build them, else stop with a clear message.
def ensure_labels(dataset):
    """Create labels for a dataset if missing; the notebook no longer decides the method."""
    dataset = dataset.lower()
    path = f"labels/{dataset}.labels"

    if os.path.exists(path):
        print("Label file already exists:", path)
        return path

    if dataset in ("enzymes", "politics", "proteins"):           # networkrepository: verify-or-fall-back to <name>_nr
        return f"labels/{_ensure_nr(dataset)}.labels"
    if dataset in ("enzymes_nr", "politics_nr", "proteins_nr"):
        return f"labels/{_ensure_nr(dataset[:-3])}.labels"

    if dataset in {"cora", "citeseer"}:
        print("Creating Planetoid labels:", dataset)
        make_labels(dataset)
    elif dataset == "webkb_wisc":
        print("Creating WebKB-Wisc labels and clean graph")
        make_webkb()
    elif dataset == "citeseer_linqs":
        print("Creating aligned LINQS Citeseer (graph + labels)")
        make_citeseer_linqs()
    else:
        raise ValueError(f"No automatic label builder for dataset: {dataset}")

    if not os.path.exists(path):                       # build attempted but produced nothing -> stop clearly
        raise RuntimeError(
            f"Could not create {path}. For cora/citeseer this means the LINQS download failed "
            f"(no internet/proxy) or the edgelist did not match (overlap < 0.90). "
            f"Fix: restore internet and re-run, or place a verified {path} manually.")
    return path


# Dataset name -> (base, version, safe-name). 'safe' is the aligned graph+labels version to actually use.
_VERSIONS = {
    "cora":           ("cora", "orig", "cora"),
    "citeseer":       ("citeseer", "orig", "citeseer"),          # author's own graph (paper's file) -> link-pred only, no labels
    "citeseer_linqs": ("citeseer", "linqs", "citeseer_linqs"),
    "webkb_wisc":     ("webkb", "wisc", "webkb_wisc"),
    "enzymes":        ("enzymes", "orig", "enzymes"),            # networkrepository ids expected to align
    "enzymes_nr":     ("enzymes", "nr", "enzymes_nr"),           # self-aligned fallback if they don't
    "politics":       ("politics", "orig", "politics"),          # rt-pol; no verifiable labels -> link-pred only
    "proteins":       ("proteins", "nr", "proteins_nr"),         # author edgelist is comma-delimited -> the rebuilt whitespace pair is the only safe version
    "proteins_nr":    ("proteins", "nr", "proteins_nr"),
}


# Ensures networkrepository labels exist; returns the version actually aligned ('<name>' or '<name>_nr').
def _ensure_nr(name):
    """Build/verify enzymes|politics|proteins labels; return the safe version name."""
    if os.path.exists(f"labels/{name}.labels"):
        return name
    if os.path.exists(f"labels/{name}_nr.labels"):
        return f"{name}_nr"
    return {"enzymes": make_enzymes, "politics": make_politics, "proteins": make_proteins}[name]()   # builds + returns the aligned version


# One call to resolve + build a dataset: returns base/version/safe + edge/label paths. Author files stay untouched.
def prepare_dataset(name):
    """Resolve a dataset to its safe (aligned) version, build files if missing, return an info dict."""
    name = name.lower()
    if name not in _VERSIONS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(_VERSIONS)}")
    base, version, safe = _VERSIONS[name]
    if name == "citeseer":                                      # author graph = paper's own file, no aligned labels -> link-pred only
        print("Using author 'citeseer' graph (paper's own file) -> link prediction only (no aligned labels).")
    elif name in ("enzymes", "politics", "proteins"):           # may resolve to the self-aligned <name>_nr version
        safe = _ensure_nr(name)
        base, version, _ = _VERSIONS[safe]
    else:
        ensure_labels(safe)                                     # builds the aligned graph+labels if missing
    info = {"base": base, "version": version, "safe": safe,
            "edge_path": f"input/{safe}.edgelist", "label_path": f"labels/{safe}.labels"}
    print(f"dataset ready -> base={base} version={version} safe={safe}")
    print(f"  edges  = {info['edge_path']}")
    print(f"  labels = {info['label_path']}")
    return info


if __name__ == "__main__":
    make_labels("Cora")                                         # citeseer is link-pred only (author graph, no aligned labels)
    make_webkb()
