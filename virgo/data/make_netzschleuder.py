'''Convert Netzschleuder networks to project edgelist/label/.nodes files: the small heterophilous graphs PyG does not ship.'''
# Added 2026-09-05 to test FROZEN_DEGREE. Its cut is on distinct_degrees, so a test set must span it - and the side that
# predicts DEGREE needs graphs that are BOTH low-degree-resolution and low-homophily. Nothing new in torch_geometric is
# both: the only new low-distinct-degree family there is road networks (CityNetwork / GraphLand city-roads), and every
# one of them has adjusted homophily 0.31-0.57, so stage 1 vetoes them and they can never carry a stage-2 cell.
# Netzschleuder supplies the missing corner. Two structural families are heterophilous BY CONSTRUCTION:
#   bipartite     every edge joins the two sides, so edge homophily is 0 and adjusted homophily is exactly -1.0
#   food webs     predator-prey links join different trophic groups
# Both also have bounded degree, which is what puts them below the cut. Same policy as make_pyg: node covariates are
# NOT loaded, only the single categorical column named below becomes the label (structural-only).
# DISCLOSURE: five of the seven are bipartite, a structural class absent from the 34-graph fitting panel. That is what
# made them findable at all, and it is a confound that must travel with any result read off this set.

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo.data.make_ogb import _write_edges, _write_nodes   # shared writers; one edgelist convention for every source

# name -> (netzschleuder network, subnetwork, the ONE categorical column used as the label)
NETS = {
    "plant_pol_robertson": ("plant_pol_robertson", "plant_pol_robertson", "is_pollinator"),   # bipartite, 130 distinct degrees - the closest below-cut cell to the fitted interval
    "messal_shale":        ("messal_shale", "messal_shale", "group"),                         # Eocene food web, NOT bipartite
    "celegans_neural":     ("celegans_2019", "hermaphrodite_chemical", "node_type"),          # C. elegans chemical synapses, NOT bipartite
    "escorts":             ("escorts", "escorts", "male"),                                    # bipartite escort-client network
    "nematode_mammal":     ("nematode_mammal", "nematode_mammal", "is_host"),                 # bipartite host-parasite
    "bag_of_words_kos":    ("bag_of_words", "kos", "is_word"),                                # bipartite document-word, 437 distinct degrees = ABOVE the cut
    # 2026-09-05 second batch: the family that produced 3 degree winners from 7 graphs, mined further (paper_log 21).
    # Purpose is to GROW the degree class, not to test FROZEN_DEGREE - that rule is already falsified (paper_log 21).
    # DISCLOSURE, pseudo-replication to state with any result: foursquare_checkin/foursquare_tips are two edge types over
    # the same NYC restaurant population, and ppi_mouse/ppi_rat are homologous protein-interaction networks. Neither pair
    # is two independent graphs, so they are reported as such and never counted as two unrelated cells.
    "plant_pol_kato":      ("plant_pol_kato", "plant_pol_kato", "pollinator"),                # bipartite pollination, 35 distinct degrees
    "board_directors":     ("board_directors", "net2m_2011-05-01", "gender"),                 # bipartite director-board, ONE 2011 snapshot of 112 (the rest are the same network over time)
    "foursquare_checkin":  ("foursquare", "NYC_restaurant_checkin", "is_user"),               # bipartite user-restaurant check-ins
    "foursquare_tips":     ("foursquare", "NYC_restaurant_tips", "is_user"),                  # bipartite user-restaurant tips, same population as the check-in graph
    "ppi_rat":             ("mist", "ppi_rat", "gene"),                                       # rat protein interactions, near-zero clustering
    "ppi_mouse":           ("mist", "ppi_mouse", "gene"),                                     # mouse protein interactions, 199 distinct degrees = ABOVE the cut
    "bag_of_words_nips":   ("bag_of_words", "nips", "is_word"),                               # bipartite NIPS document-word, 800 distinct degrees = ABOVE the cut
    # 2026-09-07, Module 14 (cfg.DEGREE_FAMILY_TEST): graphs inside frozen_rules.FAMILY, the envelope of the nine cells
    # `nbr_label_entropy < 0.6724 => degree` was discovered on. The catalogue was filtered on the envelope's own axes -
    # num_vertices, average_degree, degree_assortativity, largest_component_fraction, is_bipartite - which the API
    # publishes, so selection happened on ORIGINAL-graph properties before anything was downloaded. 139 subnetworks
    # clear those axes and only `mist` carries a usable label. The four rejections, recorded rather than silently
    # skipped: spanish_highschools (~47% of nodes have no Sexo/Curso value - the defect that got NELL rejected),
    # ego_social (`circles` is MULTI-label), genetic_multiplex / jdk / jung / google (`nodeLabel` and `meta` are unique
    # per node - gene names and Java class names, not classes), arxiv_citation / scotus_majority / us_agencies / 
    # route_views (only dates, covariates or ids; binning a date reproduces the crocodile confound).
    # The three mist INTEROLOG subnetworks that also pass are excluded on purpose: an interolog network is INFERRED by
    # homology from another species' network, so it is not an independent graph.
    "ppi_human":            ("mist", "ppi_human", "gene"),                                    # human PPI, 27,594 nodes
    "ppi_fly":              ("mist", "ppi_fly", "gene"),                                      # fly PPI, 11,352 nodes
    "ppi_yeast":            ("mist", "ppi_yeast", "gene"),                                    # budding-yeast PPI, 7,272 nodes
    "genetic_fission_yeast": ("mist", "genetic_fission_yeast", "gene"),                       # fission-yeast GENETIC interactions - different edge semantics to the PPI three
    # 2026-09-08, the NON-PPI half of the family test (cfg.DEGREE_FAMILY_TEST2). The four mist graphs all came from one
    # database with one label scheme, so a result on them cannot separate "the rule works in this family" from "the rule
    # works on protein networks". These two are in the same STRUCTURAL family and in different domains, and they are
    # independent of each other and of mist. They are the ONLY two the catalogue offers: rescanning every subnetwork
    # under 20k nodes that clears the envelope, everything else carries just `name`/`_pos`/coordinates, or is bipartite,
    # or was rejected before. `jung` and `google` were rejected here for new reasons - `jung` is a near-DUPLICATE of jdk
    # (identical java.* subtree: java.util 639, java.awt 549, java.lang 220, java.security 212 in both), and `google`'s
    # `meta` is a URL whose only categorical reading is the scheme, 15690 http vs 73 https.
    "jdk":                  ("jdk", "jdk", "meta"),                                           # Java class-dependency network; label = top-level PACKAGE (javax/java/org)
    "spanish_highschool_6": ("spanish_highschools", "6", "Sexo"),                             # school friendship network; label = binary gender, 534/534 labelled
}

# A few sources ship the categorical as part of a longer string. The transform is applied to that ONE column and is a
# prefix, never a new measurement or a binning of a number - `jdk`'s `meta` is a fully-qualified class name whose
# top-level package IS the categorical (javax 3304 / java 2379 / org 751, 6434 of 6434 nodes labelled).
LABEL_FN = {"jdk": lambda v: str(v).split(".")[0]}


# Netzschleuder ships one zip per network holding edges.csv / nodes.csv, with the header line prefixed by "# ".
def _read(net, sub):
    '''The edge and node tables for one Netzschleuder network, with the header's comment prefix stripped.'''
    import pandas as pd
    url = f"https://networks.skewed.de/net/{net}/files/{sub}.csv.zip"
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=180).read()))
    out = []
    for suffix in ("edges.csv", "nodes.csv"):
        f = [n for n in z.namelist() if n.endswith(suffix)][0]
        t = pd.read_csv(z.open(f), skipinitialspace=True, on_bad_lines="skip")
        t.columns = [c.lstrip("# ").strip() for c in t.columns]
        out.append(t)
    return out


# Builds input/<name>.edgelist + .nodes + labels/<name>.labels; only the named column is read, every covariate ignored.
def make_netzschleuder(name):
    '''One Netzschleuder network -> edgelist + .nodes + labels. Node covariates ignored by design (structural-only).'''
    import numpy as np
    net, sub, col = NETS[name]
    e, nd = _read(net, sub)
    ids = nd.iloc[:, 0].to_numpy()
    index = {v: i for i, v in enumerate(ids)}                       # source ids -> 0..n-1, the project's edgelist convention
    n = len(ids)
    pairs = np.array([[index[a], index[b]] for a, b in zip(e.iloc[:, 0], e.iloc[:, 1])
                      if a in index and b in index], dtype=np.int64)
    ne = _write_edges(f"input/{name}.edgelist", pairs.T)             # dedup+sort symmetrizes; self-loops dropped
    _write_nodes(f"input/{name}.edgelist", n)
    vals = nd[col].map(LABEL_FN[name]) if name in LABEL_FN else nd[col]
    codes = {k: i for i, k in enumerate(sorted({v for v in vals if v == v}, key=str))}
    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{name}.labels", "w") as f:
        for i, v in enumerate(vals):
            if v == v:                                              # a missing covariate is UNLABELLED, never its own class
                f.write(f"{i} {codes[v]}\n")
    print(f"{name}: nodes={n} edges={ne} classes={len(codes)} labelled={int(sum(v == v for v in vals))}/{n} "
          f"avg_degree={2 * ne / n:.2f} | label column '{col}', every other covariate IGNORED (structural-only)")
    print(f"WROTE input/{name}.edgelist + .nodes + labels/{name}.labels")


# Defines command-line options (mirrors make_pyg.py / make_hetero.py).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert Netzschleuder networks to project edgelist/label/.nodes files (structural-only).")
    p.add_argument('--dataset', default='all', choices=['all'] + list(NETS), help='Which dataset to build. Default all.')
    return p.parse_args()


# Builds the requested dataset(s) from the command line.
def main(args):
    for name in NETS:
        if args.dataset in ('all', name):
            make_netzschleuder(name)


if __name__ == "__main__":
    main(parse_args())
