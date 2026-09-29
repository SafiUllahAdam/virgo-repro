'''Convert torch_geometric benchmark graphs to project edgelist/label/.nodes files: pubmed (Planetoid), actor (film) - structural-only.'''
# Added 2026-08-02 as Module-3 held-out datasets (see frozen_rules.HELDOUT). Same policy as make_hetero/make_ogb:
# node features (PubMed 500-dim TF-IDF, Actor 932-dim) are IGNORED by design, and any official train/val/test masks are
# ignored - these run the core protocol (stratified 70% node classification, 70:30 link prediction).
# lastfm_asia (2026-08-04) is the one exception to "download via the PyG class": torch_geometric.datasets.LastFMAsia
# fetches graphmining.ai, which no longer serves the file (TLS handshake failure / 404), so the raw graph is read from
# SNAP's primary archive of the same dataset (Rozemberczki & Sarkar 2020) instead. Node ids, labels and edges are the
# source's own; SNAP ships features as a liked-artist JSON rather than PyG's 128-dim matrix, which we would drop anyway.
# reed98/amherst41/johnshopkins55 (2026-08-14) are the LINKX non-homophilous Facebook100 college networks. Their label is
# gender, which is MISSING for some users - PyG codes that as y = -1, and those nodes are left out of the .labels file
# rather than written as a third class, exactly as the LINKX benchmark evaluates them. The graph keeps every node.

import argparse
from pathlib import Path

from virgo.data.make_ogb import _write_edges, _write_nodes   # shared writers; the import also installs the torch.load allowlist

RAW = "output/pyg_raw"                         # PyG download cache: derived, never hand-edited, kept out of input/

PYG = ["pubmed", "actor", "amazon_photo", "lastfm_asia", "reed98", "amherst41", "johnshopkins55", "cornell5", "texas",
       "wisconsin", "cornell_webkb", "chameleon", "twitch_de", "deezer_europe",
       "airports_usa", "airports_europe", "airports_brazil", "email_eu_core", "wikics", "twitch_engb", "github",
       "amazon_computers", "polblogs", "blogcatalog", "coauthor_cs", "twitch_es", "twitch_fr", "dblp",
       "coauthor_physics", "twitch_ru", "twitch_ptbr", "cora_ml", "squirrel", "flickr_attr",
       "wiki_attr", "crocodile", "cora_full", "penn94", "genius", "twitch_gamers",
       "artnet_exp", "city_reviews", "hm_categories", "city_roads_m",
       "city_paris", "city_shanghai", "city_la"]

# 2026-09-04, candidates for the FROZEN_DEGREE test. The cut is on distinct_degrees, so a test set must span it; these
# were probed for that first and for nothing else. CityNetwork road networks sit at 8-9 distinct degrees (far below the
# 158.5 cut) and GraphLand's artnet-exp at 196 (just above it), which is the near-boundary case worth the most per cell.
# DISCLOSURE: the CityNetwork label is a 10-quantile binning of node ECCENTRICITY and the GraphLand regression targets
# are binned into quintiles here - structural or traffic-derived targets in both cases, so they correlate with position
# in the graph exactly as the airports_* and crocodile labels do.
# REJECTED after probing, recorded so they are not retried: graphland twitch-views IS twitch_gamers (same 168114/6797557
# graph, regression target); web-topics and web-fraud are the same 2.89M-node graph, too large; tolokers-2 is a variant
# of the panel's tolokers.
CITY = {"city_paris": "paris", "city_shanghai": "shanghai", "city_la": "la"}
# name -> (GraphLand name, quantile bins; 0 = the target is already categorical)
GRAPHLAND = {"artnet_exp": ("artnet-exp", 0), "hm_categories": ("hm-categories", 0),
             "city_reviews": ("city-reviews", 5), "city_roads_m": ("city-roads-M", 5)}
LINKX = ["reed98", "amherst41", "johnshopkins55", "cornell5", "penn94", "genius"]

# 2026-09-01, the DEGREE-evidence batch (user): the stage-2 psi/degree branch is blocked because only two graphs in the
# corpus name degree as their sole winning signal. These six were chosen to add low-homophily graphs where a degree role
# graph is plausible; every one is screened and REPORTED, whether or not it produces a degree win.
# Two of the six were already in the corpus: WebKB Texas is `texas` (built 2026-08-31) and Platonov's de-duplicated
# Chameleon is `chameleon_filtered`. `chameleon` here is the ORIGINAL geom-gcn WikipediaNetwork copy, which has the
# duplicate-node problem that filtering removes - it is screened as a separate graph and that caveat travels with it.
# `cornell_webkb` is NOT `cornell5`: 183 web pages against a 18.6K-node Facebook100 network that shares the name.
DEGREE_BATCH = ["twitch_de", "deezer_europe", "wisconsin", "cornell_webkb", "chameleon", "texas"]

# 2026-09-02, the DEGREE_RULE_CORPUS (user): a SEPARATE stage-2 pool, kept out of the study panel, whose only purpose is
# to learn when a DEGREE role graph should be selected. Twenty graphs in three waves, chosen for structural SPREAD - not
# because degree is expected to win on any of them - across degree skew, clustering, assortativity, density, size and
# neighbour predictability. Protocol pre-registered in docs/paper_log.md 12 before the first one was trained.
# DISCLOSURE: the three `airports_*` graphs are the classic structural-identity benchmarks, and their labels are
# air-traffic quartiles, which correlate with degree. They are included because they are the standard role benchmark,
# and every result is reported with and without them so that correlation cannot silently carry a rule.
DEGREE_RULE_CORPUS = [
    "airports_usa", "airports_europe", "email_eu_core", "wikics", "twitch_engb", "github", "amazon_computers",   # wave 1
    "airports_brazil", "polblogs", "blogcatalog", "coauthor_cs", "twitch_es", "twitch_fr", "dblp",                # wave 2
    "coauthor_physics", "twitch_ru", "twitch_ptbr", "cora_ml", "squirrel", "flickr_attr",                         # wave 3
]

# 2026-09-03, the DEGREE_RULE_VALIDATION set (user): six graphs held out to TEST what the corpus fitted, never to fit on.
# The corpus produced one unpromoted candidate - nbr_label_entropy < 0.6724 => degree - on nine cells with a single
# pre-registered test cell, so what it needs is out-of-sample cells on BOTH sides of that cut, not another screen.
# Selected on the PREDICTOR and on structural spread, never on an outcome: three are expected low-entropy (degree side)
# and three high-entropy (psi side), and the predictions are written to disk before any of them is trained.
# Chosen from what is actually reachable: musae-facebook is gone (SNAP's own facebook_large.zip 404s and graphmining.ai
# is unreachable), PyG's twitch-gamers loader asserts, AttributedGraphDataset facebook/twitter are multi-label, and NELL
# codes unlabelled entities as class 0 - all four were probed and rejected rather than silently skipped.
# DISCLOSURE, two dependencies that travel with this set: `cora_ml` (in the corpus) is a SUBGRAPH of `cora_full`, and
# `crocodile` is labelled by binning its raw traffic target into quantiles, so its labels correlate with degree exactly
# as the airports graphs' do. Both are reported with every result fitted or tested on them.
DEGREE_RULE_VALIDATION = ["wiki_attr", "crocodile", "cora_full", "penn94", "genius", "twitch_gamers"]   # cheapest first

# archive -> (SNAP zip, edge file, target file, node-id column, label column, quantile bins). All of these are musae
# graphs whose PyG classes fetch graphmining.ai, which is unreachable; SNAP is the authors' primary archive of the same
# data. bins > 0 means the target is CONTINUOUS and is binned into that many quantile classes.
SNAP = {"twitch_de":     ("twitch", "twitch/DE/musae_DE_edges.csv", "twitch/DE/musae_DE_target.csv", "new_id", "mature", 0),
        "twitch_engb":   ("twitch", "twitch/ENGB/musae_ENGB_edges.csv", "twitch/ENGB/musae_ENGB_target.csv", "new_id", "mature", 0),
        "twitch_es":     ("twitch", "twitch/ES/musae_ES_edges.csv", "twitch/ES/musae_ES_target.csv", "new_id", "mature", 0),
        "twitch_fr":     ("twitch", "twitch/FR/musae_FR_edges.csv", "twitch/FR/musae_FR_target.csv", "new_id", "mature", 0),
        "twitch_ptbr":   ("twitch", "twitch/PTBR/musae_PTBR_edges.csv", "twitch/PTBR/musae_PTBR_target.csv", "new_id", "mature", 0),
        "twitch_ru":     ("twitch", "twitch/RU/musae_RU_edges.csv", "twitch/RU/musae_RU_target.csv", "new_id", "mature", 0),
        "github":        ("git_web_ml", "git_web_ml/musae_git_edges.csv", "git_web_ml/musae_git_target.csv", "id", "ml_target", 0),
        "deezer_europe": ("deezer_europe", "deezer_europe/deezer_europe_edges.csv", "deezer_europe/deezer_europe_target.csv", "id", "target", 0),
        "crocodile":     ("wikipedia", "wikipedia/crocodile/musae_crocodile_edges.csv", "wikipedia/crocodile/musae_crocodile_target.csv", "id", "target", 5),
        "twitch_gamers": ("twitch_gamers", "large_twitch_edges.csv", "large_twitch_features.csv", "numeric_id", "mature", 0)}


# Returns the single PyG graph for a supported benchmark; node features are dropped downstream (structural-only).
def _load(name):
    '''Load one PyG benchmark graph. PubMed = Planetoid citation (3 classes); Actor = Geom-GCN film (5); Amazon Photo = co-purchase (8); LastFM Asia = music social (18); the LINKX graphs = Facebook100 college networks (2, gender) plus genius (2).'''
    from torch_geometric.datasets import (Actor, Airports, Amazon, AttributedGraphDataset, CitationFull, Coauthor,
                                          EmailEUCore, LINKXDataset, Planetoid, PolBlogs, WebKB, WikiCS, WikipediaNetwork)
    if name == "pubmed":
        return Planetoid(root=RAW, name="PubMed")[0]
    if name == "actor":
        return Actor(root=f"{RAW}/actor")[0]
    if name == "amazon_photo":
        return Amazon(root=RAW, name="Photo")[0]
    if name in LINKX:
        return LINKXDataset(root=RAW, name=name)[0]
    if name == "texas":
        return WebKB(root=RAW, name="Texas")[0]           # 2026-08-31: rebuilt to TEST the stage-1 second condition; 183 nodes, DIRECTED web links
    if name in ("wisconsin", "cornell_webkb"):
        return WebKB(root=RAW, name={"wisconsin": "Wisconsin", "cornell_webkb": "Cornell"}[name])[0]
    if name == "chameleon":
        return WikipediaNetwork(root=RAW, name="chameleon", geom_gcn_preprocess=True)[0]   # the ORIGINAL copy; duplicate nodes are a known caveat
    if name.startswith("airports_"):
        return Airports(root=f"{RAW}/airports", name=name.split("_")[1].capitalize())[0]   # labels = traffic quartiles (see the DISCLOSURE above)
    if name == "email_eu_core":
        return EmailEUCore(root=f"{RAW}/email_eu")[0]
    if name == "wikics":
        return WikiCS(root=f"{RAW}/wikics")[0]
    if name == "polblogs":
        return PolBlogs(root=f"{RAW}/polblogs")[0]
    if name in ("coauthor_cs", "coauthor_physics"):
        return Coauthor(root=RAW, name={"coauthor_cs": "CS", "coauthor_physics": "Physics"}[name])[0]
    if name == "amazon_computers":
        return Amazon(root=RAW, name="Computers")[0]
    if name in ("blogcatalog", "flickr_attr", "wiki_attr"):
        return AttributedGraphDataset(root=RAW, name={"blogcatalog": "BlogCatalog", "flickr_attr": "Flickr", "wiki_attr": "Wiki"}[name])[0]
    if name in ("dblp", "cora_ml", "cora_full"):
        return CitationFull(root=RAW, name={"dblp": "DBLP", "cora_ml": "Cora_ML", "cora_full": "Cora"}[name])[0]   # cora_ml is a SUBGRAPH of cora_full (see the DISCLOSURE above)
    if name == "squirrel":
        return WikipediaNetwork(root=RAW, name="squirrel", geom_gcn_preprocess=True)[0]   # ORIGINAL copy; squirrel_filtered is the de-duplicated one
    if name in CITY:
        from torch_geometric.datasets import CityNetwork
        return CityNetwork(root=f"{RAW}/city", name=CITY[name])[0]      # label = 10 quantiles of eccentricity (see the DISCLOSURE above)
    if name in GRAPHLAND:
        import numpy as np
        import pandas as pd
        import torch
        from torch_geometric.datasets import GraphLandDataset
        gl, bins = GRAPHLAND[name]
        g = GraphLandDataset(root=f"{RAW}/graphland", name=gl, split="RH")[0]   # split unused: the core protocol makes its own
        y = g.y.reshape(-1).numpy()
        g.y = torch.from_numpy((pd.qcut(y, bins, labels=False, duplicates="drop") if bins else y).astype(np.int64))
        return g
    if name in SNAP:
        return _snap(name)                                # graphmining.ai is dead for all of these, exactly as for lastfm_asia
    if name == "lastfm_asia":
        return _lastfm_asia()                             # SNAP archive, not the PyG class: its host is dead (see header)
    raise ValueError(f"Unknown dataset '{name}'. Use: {PYG}.")


# Reads SNAP's musae archives for the two graphs whose PyG host (graphmining.ai) no longer serves the file.
def _snap(name):
    '''One musae graph from SNAP's primary archive: the six Twitch language networks and Twitch Gamers (label = mature content), GitHub (web vs ML developer), Deezer Europe (gender) or Crocodile (traffic quantiles).'''
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Data, download_url, extract_zip
    archive, e_file, t_file, id_col, y_col, bins = SNAP[name]
    raw = Path(RAW) / name / "raw"
    if not (raw / e_file).exists():
        extract_zip(download_url(f"https://snap.stanford.edu/data/{archive}.zip", str(raw)), str(raw))
    e = pd.read_csv(raw / e_file).to_numpy(dtype=np.int64)                       # one row per undirected pair
    t = pd.read_csv(raw / t_file)
    # Twitch keys its target file by the ORIGINAL streamer id and carries the graph id in new_id; the others are 0..n-1.
    # musae_FR ships 6551 rows for 6549 distinct new_id values, so the id column is NOT always a permutation: duplicates
    # are dropped (first row kept) and the array is filled with -1, which make_pyg writes as UNLABELLED rather than as a
    # class. Sizing by len(t) and filling an np.empty would have written uninitialised memory as labels.
    t = t.drop_duplicates(subset=[id_col], keep="first")
    # Crocodile ships raw monthly traffic, not classes; geom-gcn builds Chameleon's and Squirrel's labels by binning the
    # same target, so the same construction is used here rather than a new one.
    idx, y = t[id_col].to_numpy(), t[y_col].to_numpy()
    y = (pd.qcut(y, bins, labels=False, duplicates="drop") if bins else y).astype(np.int64)
    n = int(max(idx.max(), e.max())) + 1
    lab = np.full(n, -1, dtype=np.int64)
    lab[idx] = y                                                                 # reorder onto the edge file's node ids
    return Data(edge_index=torch.from_numpy(e.T), y=torch.from_numpy(lab), num_nodes=n)


# Reads SNAP's lastfm_asia archive (downloaded once into RAW) as a PyG graph; no data.x, the liked-artist JSON is not loaded.
def _lastfm_asia():
    '''LastFM Asia from SNAP: mutual-follower graph of 7,624 Asian users, 18 country classes. Structural-only, so features are skipped.'''
    import numpy as np
    import torch
    from torch_geometric.data import Data, download_url, extract_zip
    raw = Path(RAW) / "lastfm_asia" / "raw"
    inner = raw / "lasftm_asia"                           # the archive's own folder name, misspelled at the source
    if not (inner / "lastfm_asia_edges.csv").exists():
        extract_zip(download_url("https://snap.stanford.edu/data/lastfm_asia.zip", str(raw)), str(raw))
    e = np.loadtxt(inner / "lastfm_asia_edges.csv", delimiter=",", skiprows=1, dtype=np.int64)     # one row per undirected pair
    y = np.loadtxt(inner / "lastfm_asia_target.csv", delimiter=",", skiprows=1, dtype=np.int64)
    assert (y[:, 0] == np.arange(len(y))).all(), "lastfm_asia target ids are not 0..n-1 in order - labels would misalign"
    return Data(edge_index=torch.from_numpy(e.T), y=torch.from_numpy(y[:, 1]), num_nodes=len(y))


# Builds input/<name>.edgelist + .nodes + labels/<name>.labels from the PyG copy; drops data.x (structural-only).
def make_pyg(name):
    '''One PyG benchmark -> edgelist + .nodes + labels. node_features and official masks ignored by design.'''
    g = _load(name)
    n = g.num_nodes
    ne = _write_edges(f"input/{name}.edgelist", g.edge_index.numpy())   # dedup+sort symmetrizes; self-loops dropped
    _write_nodes(f"input/{name}.edgelist", n)
    y = g.y.numpy().reshape(-1)
    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{name}.labels", "w") as f:
        for i in range(n):
            if y[i] >= 0:                                 # Facebook100 codes a missing gender as -1: unlabelled, not a class
                f.write(f"{i} {int(y[i])}\n")
    feats = f"data.x ({g.x.shape[1]}-dim) IGNORED" if g.x is not None else "node features NOT LOADED"
    print(f"{name}: nodes={n} edges={ne} classes={len(set(y[y >= 0]))} labelled={int((y >= 0).sum())}/{n} avg_degree={2 * ne / n:.2f} | "
          f"{feats} (structural-only) | official masks IGNORED (core protocol)")
    print(f"WROTE input/{name}.edgelist + .nodes + labels/{name}.labels")


# Defines command-line options: which PyG dataset to build (default all).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert torch_geometric benchmark graphs to project edgelist/label/.nodes files (structural-only).")
    p.add_argument('--dataset', default='all', choices=['all'] + PYG, help='Which dataset to build. Default all.')
    return p.parse_args()


# Builds the requested dataset(s) from the command line.
def main(args):
    for name in PYG:
        if args.dataset in ('all', name):
            make_pyg(name)


if __name__ == "__main__":
    main(parse_args())
