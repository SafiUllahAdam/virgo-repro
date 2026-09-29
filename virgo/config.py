"""Single source of truth: paths, dataset registry, I2V hyperparameters, reproduction defaults."""

from pathlib import Path

# Project root = one level above virgo/. Everything else is derived from it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS_DIR = PROJECT_ROOT / "splits"
LABELS_DIR = PROJECT_ROOT / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"

# Notebook-first output zones (2026-07-08 layout): every path reads notebook -> content -> dataset -> K -> variant.
NB1_DIR = OUTPUT_DIR / "notebook1_reproduce_i2v"        # Phase-1 embeddings: <dataset>/<task>/<model>_s<seed>.emb
NB2_DIR = OUTPUT_DIR / "notebook2_create_vir_graph"     # virtual_graphs/ + bridge embeddings per task folder
NB3_DIR = OUTPUT_DIR / "notebook3_gnn_encoder"          # GraphSAGE embeddings per task folder
LP_SPLITS_ORIG = SPLITS_DIR / "link_prediction" / "original_graph"       # Phase-1 splits: <dataset>/seed_<s>/
LP_SPLITS_VG = SPLITS_DIR / "link_prediction" / "virtual_graph_study"    # shared Phase-2/3 splits: <dataset>/seed_<s>/
SCOREBOARD_CSV = RESULTS_DIR / "scoreboard.csv"         # THE master table (one row per dataset x encoder x graph x K x task)
GRAPH_HEALTH_CSV = RESULTS_DIR / "graph_health.csv"     # one row per virtual graph built
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"               # per-run comparison CSVs
FIGURES_DIR = RESULTS_DIR / "figures"                   # characterization figures (notebook 5)

# Dataset registry: name -> edgelist + label file (labels may not exist yet -> None or a path to be filled).
DATASETS = {
    "cora":     {"edgelist": INPUT_DIR / "cora.edgelist",     "labels": LABELS_DIR / "cora.labels"},
    "citeseer": {"edgelist": INPUT_DIR / "citeseer.edgelist", "labels": None},  # author's own graph (paper's file) -> link-pred only, no aligned labels
    "citeseer_linqs": {"edgelist": INPUT_DIR / "citeseer_linqs.edgelist", "labels": LABELS_DIR / "citeseer_linqs.labels"},  # aligned (graph+labels from LINQS)
    "politics": {"edgelist": INPUT_DIR / "politics.edgelist", "labels": None, "directed_source": True},  # rt-pol ships no labels -> link-pred only; retweet = DIRECTED relation, loaded undirected (recorded deviation)
    "enzymes":  {"edgelist": INPUT_DIR / "enzymes.edgelist",  "labels": LABELS_DIR / "enzymes.labels"},  # labels built+verified by make_labels.make_enzymes
    "enzymes_nr": {"edgelist": INPUT_DIR / "enzymes_nr.edgelist", "labels": LABELS_DIR / "enzymes_nr.labels"},  # aligned fallback if enzymes ids mismatch
    "proteins": {"edgelist": INPUT_DIR / "proteins_nr.edgelist", "labels": LABELS_DIR / "proteins_nr.labels"},  # author input/proteins.edgelist is comma-delimited -> make_labels.make_proteins rebuilds a whitespace copy + labels from the same source (edge overlap 1.0)
    "proteins_nr": {"edgelist": INPUT_DIR / "proteins_nr.edgelist", "labels": LABELS_DIR / "proteins_nr.labels"},  # explicit alias: same rebuilt pair
    # OGB-sourced, structural-only, scored under the OFFICIAL OGB protocol (fixed split + OGB Evaluator), NOT the core random 70/30. Build via make_ogb.py.
    "ogbn_arxiv": {"edgelist": INPUT_DIR / "ogbn_arxiv.edgelist", "labels": LABELS_DIR / "ogbn_arxiv.labels",
                   "directed_source": True, "eval": "ogb", "split": SPLITS_DIR / "ogb" / "ogbn_arxiv_idx.npz"},   # full transductive graph, official time split -> Accuracy; data.x (128-dim text) ignored
    "ogbl_ddi":   {"edgelist": INPUT_DIR / "ogbl_ddi_train.edgelist", "labels": None,
                   "eval": "ogb", "pairs": SPLITS_DIR / "ogb" / "ogbl_ddi_pairs.npz"},   # graph = TRAINING edges only (no leakage); official pos/neg -> Hits@20; featureless
    # Heterophilous benchmarks (Platonov et al. 2023), structural-only, scored under the CORE protocol (stratified 70% NC, 70:30 LP)
    # -> no "eval" key, so they route exactly like the core four. Added 2026-07-27: the panel had no low-homophily graph, so the
    # augment side of the rule was never tested. Their official train/val/test masks are ignored. Build via make_hetero.py.
    "roman_empire": {"edgelist": INPUT_DIR / "roman_empire.edgelist", "labels": LABELS_DIR / "roman_empire.labels",
                     "directed_source": True},   # word order + dependency arcs = a DIRECTED relation, symmetrized upstream by PyG; 300-dim fastText features ignored
    "tolokers":     {"edgelist": INPUT_DIR / "tolokers.edgelist", "labels": LABELS_DIR / "tolokers.labels"},   # worked-the-same-task = symmetric by construction; 10-dim worker profile ignored
    "questions":    {"edgelist": INPUT_DIR / "questions.edgelist", "labels": LABELS_DIR / "questions.labels",
                     "directed_source": True},   # answered-your-question = a DIRECTED relation, symmetrized upstream by PyG; 301-dim fastText features ignored
    # Module-3 HELD-OUT (2026-08-02): genuinely unseen datasets to validate the two frozen rules. Structural-only, core
    # protocol (no "eval" key -> route like the core four). Build via make_pyg.py (pubmed, actor) / make_hetero.py (minesweeper).
    "pubmed":      {"edgelist": INPUT_DIR / "pubmed.edgelist",      "labels": LABELS_DIR / "pubmed.labels"},        # Planetoid citation, 3 classes; 500-dim TF-IDF ignored
    "actor":       {"edgelist": INPUT_DIR / "actor.edgelist",       "labels": LABELS_DIR / "actor.labels"},         # Geom-GCN film co-occurrence, low homophily, 5 classes; 932-dim features ignored
    "minesweeper": {"edgelist": INPUT_DIR / "minesweeper.edgelist", "labels": LABELS_DIR / "minesweeper.labels"},   # Platonov grid, adjusted homophily ~0.01, 2 classes; 7-dim features ignored
    "amazon_photo": {"edgelist": INPUT_DIR / "amazon_photo.edgelist", "labels": LABELS_DIR / "amazon_photo.labels"},  # co-purchase ("bought together"), 8 classes; 745-dim bag-of-words ignored
    "lastfm_asia": {"edgelist": INPUT_DIR / "lastfm_asia.edgelist", "labels": LABELS_DIR / "lastfm_asia.labels"},   # 2026-08-04: music-platform friendships, 18 classes (country); friendship = symmetric; 128-dim features ignored
    "amazon_ratings": {"edgelist": INPUT_DIR / "amazon_ratings.edgelist", "labels": LABELS_DIR / "amazon_ratings.labels"},   # 2026-08-05: Platonov co-purchase ("bought together"), 5 rating classes, heterophilous; NOT amazon_photo; 300-dim fastText ignored
    "squirrel_filtered": {"edgelist": INPUT_DIR / "squirrel_filtered.edgelist", "labels": LABELS_DIR / "squirrel_filtered.labels"},   # 2026-08-05: Platonov's DE-DUPLICATED Squirrel (the original WikipediaNetwork copy leaks train->test), 5 classes, very dense; 2089-dim features ignored
    # LINKX non-homophilous Facebook100 college networks (Lim et al. 2021), 2026-08-14. Structural-only, core protocol.
    # Label = gender, missing for some users -> those nodes are UNLABELLED (not a third class), as LINKX itself evaluates them.
    "reed98":         {"edgelist": INPUT_DIR / "reed98.edgelist",         "labels": LABELS_DIR / "reed98.labels"},          # ~962 nodes, the smallest of the three
    "amherst41":      {"edgelist": INPUT_DIR / "amherst41.edgelist",      "labels": LABELS_DIR / "amherst41.labels"},       # ~2.2K nodes
    "johnshopkins55": {"edgelist": INPUT_DIR / "johnshopkins55.edgelist", "labels": LABELS_DIR / "johnshopkins55.labels"},  # ~5.2K nodes
    "cornell5":       {"edgelist": INPUT_DIR / "cornell5.edgelist",       "labels": LABELS_DIR / "cornell5.labels"},       # ~18.6K nodes; the stand-in after reed98's stage-1 call came back "keep original"
    # Rebuilt 2026-08-31 to TEST the stage-1 second condition (Module 8): withdrawn from the strategy batch on 2026-08-14,
    # so their 10-seed link-prediction scores survive in results/module7_withdrawn.csv and no retraining is needed - only
    # the graphs, for the properties the candidate rule reads. Structural-only, core protocol.
    "chameleon_filtered": {"edgelist": INPUT_DIR / "chameleon_filtered.edgelist", "labels": LABELS_DIR / "chameleon_filtered.labels"},   # Platonov's de-duplicated Chameleon, 5 classes; 2325-dim features ignored
    "texas":              {"edgelist": INPUT_DIR / "texas.edgelist",              "labels": LABELS_DIR / "texas.labels",
                           "directed_source": True},   # WebKB university web pages, 5 classes; hyperlink = DIRECTED, loaded undirected (recorded deviation); 1703-dim features ignored
    # 2026-09-01, the DEGREE-evidence batch: added to break the stage-2 deadlock (only 2 graphs in the corpus name degree
    # as their sole winning signal). Screened and reported whichever way each one lands - see docs/paper_log.md.
    "wisconsin":          {"edgelist": INPUT_DIR / "wisconsin.edgelist",          "labels": LABELS_DIR / "wisconsin.labels",
                           "directed_source": True},   # WebKB, 251 nodes / 5 classes; hyperlink = DIRECTED, loaded undirected; 1703-dim features ignored
    "cornell_webkb":      {"edgelist": INPUT_DIR / "cornell_webkb.edgelist",      "labels": LABELS_DIR / "cornell_webkb.labels",
                           "directed_source": True},   # WebKB Cornell, 183 nodes / 5 classes - NOT cornell5, which is the 18.6K-node Facebook100 network
    "chameleon":          {"edgelist": INPUT_DIR / "chameleon.edgelist",          "labels": LABELS_DIR / "chameleon.labels",
                           "directed_source": True},   # ORIGINAL geom-gcn WikipediaNetwork copy, 2277 nodes / 5 classes; has the duplicate-node issue chameleon_filtered removes - report that caveat
    "twitch_de":          {"edgelist": INPUT_DIR / "twitch_de.edgelist",          "labels": LABELS_DIR / "twitch_de.labels"},   # Twitch DE streamers, 9498 nodes, binary mature-content label; SNAP archive (graphmining.ai dead)
    "deezer_europe":      {"edgelist": INPUT_DIR / "deezer_europe.edgelist",      "labels": LABELS_DIR / "deezer_europe.labels"},  # Deezer Europe users, 28281 nodes, binary gender label; SNAP archive
    # 2026-09-02, DEGREE_RULE_CORPUS: a SEPARATE stage-2 pool for the degree rule, never part of the study panel.
    # Chosen for structural spread, not for expected outcome; protocol pre-registered in docs/paper_log.md 12.
    "airports_usa":      {"edgelist": INPUT_DIR / "airports_usa.edgelist", "labels": LABELS_DIR / "airports_usa.labels"},  # USA air-traffic network, 1190 airports, 4 activity quartiles - a classic structural-identity benchmark (labels correlate with degree; see the DISCLOSURE in make_pyg)
    "airports_europe":   {"edgelist": INPUT_DIR / "airports_europe.edgelist", "labels": LABELS_DIR / "airports_europe.labels"},  # European air-traffic network, 399 airports, 4 activity quartiles
    "airports_brazil":   {"edgelist": INPUT_DIR / "airports_brazil.edgelist", "labels": LABELS_DIR / "airports_brazil.labels"},  # Brazilian air-traffic network, 131 airports, 4 activity quartiles
    "email_eu_core":     {"edgelist": INPUT_DIR / "email_eu_core.edgelist", "labels": LABELS_DIR / "email_eu_core.labels", "directed_source": True},  # EU research-institution email, 1005 members / 42 departments; sending an email = DIRECTED, loaded undirected
    "wikics":            {"edgelist": INPUT_DIR / "wikics.edgelist", "labels": LABELS_DIR / "wikics.labels", "directed_source": True},  # Wikipedia computer-science articles, 10 classes; hyperlink = DIRECTED, loaded undirected; 300-dim GloVe features ignored
    "twitch_engb":       {"edgelist": INPUT_DIR / "twitch_engb.edgelist", "labels": LABELS_DIR / "twitch_engb.labels"},  # Twitch ENGB streamers, binary mature-content label; SNAP archive
    "github":            {"edgelist": INPUT_DIR / "github.edgelist", "labels": LABELS_DIR / "github.labels"},  # GitHub developers, 37.7K nodes, binary web-vs-ML label; SNAP archive
    "amazon_computers":  {"edgelist": INPUT_DIR / "amazon_computers.edgelist", "labels": LABELS_DIR / "amazon_computers.labels"},  # Amazon Computers co-purchase, 10 classes; 767-dim bag-of-words ignored
    "polblogs":          {"edgelist": INPUT_DIR / "polblogs.edgelist", "labels": LABELS_DIR / "polblogs.labels", "directed_source": True},  # US political blogs, 2 classes; hyperlink = DIRECTED, loaded undirected; no node features
    "blogcatalog":       {"edgelist": INPUT_DIR / "blogcatalog.edgelist", "labels": LABELS_DIR / "blogcatalog.labels"},  # BlogCatalog bloggers, 6 classes; 8189-dim attributes ignored
    "coauthor_cs":       {"edgelist": INPUT_DIR / "coauthor_cs.edgelist", "labels": LABELS_DIR / "coauthor_cs.labels"},  # Microsoft Academic co-authorship (CS), 15 fields; 6805-dim keyword features ignored
    "twitch_es":         {"edgelist": INPUT_DIR / "twitch_es.edgelist", "labels": LABELS_DIR / "twitch_es.labels"},  # Twitch ES streamers, binary mature-content label; SNAP archive
    "twitch_fr":         {"edgelist": INPUT_DIR / "twitch_fr.edgelist", "labels": LABELS_DIR / "twitch_fr.labels"},  # Twitch FR streamers, binary mature-content label; SNAP archive
    "dblp":              {"edgelist": INPUT_DIR / "dblp.edgelist", "labels": LABELS_DIR / "dblp.labels", "directed_source": True},  # DBLP citation network, 4 venues; citation = DIRECTED, loaded undirected; 1639-dim features ignored
    "coauthor_physics":  {"edgelist": INPUT_DIR / "coauthor_physics.edgelist", "labels": LABELS_DIR / "coauthor_physics.labels"},  # Microsoft Academic co-authorship (Physics), 5 fields; 8415-dim keyword features ignored
    "twitch_ru":         {"edgelist": INPUT_DIR / "twitch_ru.edgelist", "labels": LABELS_DIR / "twitch_ru.labels"},  # Twitch RU streamers, binary mature-content label; SNAP archive
    "twitch_ptbr":       {"edgelist": INPUT_DIR / "twitch_ptbr.edgelist", "labels": LABELS_DIR / "twitch_ptbr.labels"},  # Twitch PTBR streamers, binary mature-content label; SNAP archive
    "cora_ml":           {"edgelist": INPUT_DIR / "cora_ml.edgelist", "labels": LABELS_DIR / "cora_ml.labels", "directed_source": True},  # Cora-ML citation network, 7 classes; citation = DIRECTED, loaded undirected
    "squirrel":          {"edgelist": INPUT_DIR / "squirrel.edgelist", "labels": LABELS_DIR / "squirrel.labels", "directed_source": True},  # ORIGINAL geom-gcn Squirrel, 5 classes - squirrel_filtered is the de-duplicated version; the duplicate-node caveat applies here
    "flickr_attr":       {"edgelist": INPUT_DIR / "flickr_attr.edgelist", "labels": LABELS_DIR / "flickr_attr.labels"},  # Flickr user network (attributed release), 9 groups; 12047-dim attributes ignored
    # 2026-09-03, DEGREE_RULE_VALIDATION: held out to TEST the degree-vs-psi candidate, never fitted on (see make_pyg).
    "wiki_attr":         {"edgelist": INPUT_DIR / "wiki_attr.edgelist", "labels": LABELS_DIR / "wiki_attr.labels", "directed_source": True},  # Wikipedia article hyperlink graph, 17 classes; hyperlink = DIRECTED, loaded undirected; 4973-dim features ignored
    "crocodile":         {"edgelist": INPUT_DIR / "crocodile.edgelist", "labels": LABELS_DIR / "crocodile.labels"},  # musae Wikipedia crocodile pages, labels = 5 quantile bins of monthly traffic (DISCLOSURE: traffic correlates with degree, as for airports_*)
    "cora_full":         {"edgelist": INPUT_DIR / "cora_full.edgelist", "labels": LABELS_DIR / "cora_full.labels", "directed_source": True},  # full Cora citation network, 70 classes (DISCLOSURE: cora_ml is a SUBGRAPH of this graph); citation = DIRECTED, loaded undirected
    "penn94":            {"edgelist": INPUT_DIR / "penn94.edgelist", "labels": LABELS_DIR / "penn94.labels"},  # LINKX Facebook100 Penn, binary gender; missing gender is UNLABELLED, not a third class, as for the other FB100 graphs
    "genius":            {"edgelist": INPUT_DIR / "genius.edgelist", "labels": LABELS_DIR / "genius.labels"},  # LINKX genius.com users, binary marked-account label, ~80:20 imbalance; 12-dim features ignored
    "twitch_gamers":     {"edgelist": INPUT_DIR / "twitch_gamers.edgelist", "labels": LABELS_DIR / "twitch_gamers.labels"},  # Twitch Gamers mutual-follower network, binary mature-content label; SNAP archive (PyG's LINKX loader asserts)
    # 2026-09-04, candidates for the FROZEN_DEGREE test (see make_pyg). Registered so they can be MEASURED; which of
    # them become the test set is decided on distinct_degrees and the stage-1 call, never on an outcome.
    "artnet_exp":        {"edgelist": INPUT_DIR / "artnet_exp.edgelist", "labels": LABELS_DIR / "artnet_exp.labels"},  # GraphLand artnet, binary; distinct_degrees 196 = just ABOVE the 158.5 cut
    "city_reviews":      {"edgelist": INPUT_DIR / "city_reviews.edgelist", "labels": LABELS_DIR / "city_reviews.labels"},  # GraphLand city-reviews, target binned into 5 quantiles (DISCLOSURE: traffic-derived)
    "hm_categories":     {"edgelist": INPUT_DIR / "hm_categories.edgelist", "labels": LABELS_DIR / "hm_categories.labels"},  # GraphLand H&M co-purchase, 21 categories; very dense (10.7M edges)
    "city_roads_m":      {"edgelist": INPUT_DIR / "city_roads_m.edgelist", "labels": LABELS_DIR / "city_roads_m.labels"},  # GraphLand road network, target binned into 5 quantiles; distinct_degrees 8
    "city_paris":        {"edgelist": INPUT_DIR / "city_paris.edgelist", "labels": LABELS_DIR / "city_paris.labels"},  # CityNetwork Paris junctions, 10 eccentricity quantiles; distinct_degrees 8
    "city_shanghai":     {"edgelist": INPUT_DIR / "city_shanghai.edgelist", "labels": LABELS_DIR / "city_shanghai.labels"},  # CityNetwork Shanghai junctions, 10 eccentricity quantiles
    "city_la":           {"edgelist": INPUT_DIR / "city_la.edgelist", "labels": LABELS_DIR / "city_la.labels"},  # CityNetwork Los Angeles junctions, 10 eccentricity quantiles
    # 2026-09-05, DEGREE_TEST: Netzschleuder graphs ingested to TEST frozen_rules.FROZEN_DEGREE (see make_netzschleuder).
    "celegans_neural":     {"edgelist": INPUT_DIR / "celegans_neural.edgelist", "labels": LABELS_DIR / "celegans_neural.labels", "directed_source": True},  # C. elegans chemical synapses, 7 neuron types; synapse = DIRECTED, loaded undirected
    "messal_shale":        {"edgelist": INPUT_DIR / "messal_shale.edgelist", "labels": LABELS_DIR / "messal_shale.labels", "directed_source": True},  # Eocene Messel Shale food web, 152 taxon groups; predation = DIRECTED, loaded undirected
    "plant_pol_robertson": {"edgelist": INPUT_DIR / "plant_pol_robertson.edgelist", "labels": LABELS_DIR / "plant_pol_robertson.labels"},  # bipartite plant-pollinator network; 130 distinct degrees = the closest below-cut cell to the fitted interval
    "escorts":             {"edgelist": INPUT_DIR / "escorts.edgelist", "labels": LABELS_DIR / "escorts.labels"},  # bipartite escort-client network, binary side label
    "nematode_mammal":     {"edgelist": INPUT_DIR / "nematode_mammal.edgelist", "labels": LABELS_DIR / "nematode_mammal.labels"},  # bipartite nematode-host network, binary side label
    "bag_of_words_kos":    {"edgelist": INPUT_DIR / "bag_of_words_kos.edgelist", "labels": LABELS_DIR / "bag_of_words_kos.labels"},  # bipartite KOS document-word network; 437 distinct degrees = ABOVE the cut
    # 2026-09-05 DEGREE_BATCH2: more of the family that produced 3 degree winners from 7 (see make_netzschleuder).
    "plant_pol_kato":      {"edgelist": INPUT_DIR / "plant_pol_kato.edgelist", "labels": LABELS_DIR / "plant_pol_kato.labels"},  # bipartite plant-pollinator (Kato)
    "board_directors":     {"edgelist": INPUT_DIR / "board_directors.edgelist", "labels": LABELS_DIR / "board_directors.labels"},  # bipartite director-board, one 2011 snapshot
    "foursquare_checkin":  {"edgelist": INPUT_DIR / "foursquare_checkin.edgelist", "labels": LABELS_DIR / "foursquare_checkin.labels"},  # bipartite user-restaurant check-ins
    "foursquare_tips":     {"edgelist": INPUT_DIR / "foursquare_tips.edgelist", "labels": LABELS_DIR / "foursquare_tips.labels"},  # bipartite user-restaurant tips; SAME population as foursquare_checkin
    "ppi_rat":             {"edgelist": INPUT_DIR / "ppi_rat.edgelist", "labels": LABELS_DIR / "ppi_rat.labels"},  # rat protein interactions
    "ppi_mouse":           {"edgelist": INPUT_DIR / "ppi_mouse.edgelist", "labels": LABELS_DIR / "ppi_mouse.labels"},  # mouse protein interactions; HOMOLOGOUS to ppi_rat
    "bag_of_words_nips":   {"edgelist": INPUT_DIR / "bag_of_words_nips.edgelist", "labels": LABELS_DIR / "bag_of_words_nips.labels"},  # bipartite NIPS document-word
    # 2026-09-07, DEGREE_FAMILY_TEST: the only labelled source in the whole Netzschleuder catalogue that lands inside
    # frozen_rules.FAMILY - see make_netzschleuder for the search that establishes it. Label = the `gene` species code,
    # the SAME column ppi_rat/ppi_mouse already use, so the labelling scheme is not a new degree of freedom.
    "ppi_human":            {"edgelist": INPUT_DIR / "ppi_human.edgelist", "labels": LABELS_DIR / "ppi_human.labels"},  # human protein interactions
    "ppi_fly":              {"edgelist": INPUT_DIR / "ppi_fly.edgelist", "labels": LABELS_DIR / "ppi_fly.labels"},  # D. melanogaster protein interactions
    "ppi_yeast":            {"edgelist": INPUT_DIR / "ppi_yeast.edgelist", "labels": LABELS_DIR / "ppi_yeast.labels"},  # S. cerevisiae protein interactions
    "genetic_fission_yeast": {"edgelist": INPUT_DIR / "genetic_fission_yeast.edgelist", "labels": LABELS_DIR / "genetic_fission_yeast.labels"},  # S. pombe GENETIC interactions - different edge semantics to the PPI three
    # 2026-09-08, DEGREE_FAMILY_TEST2: the NON-PPI half - same structural family, different domains, independent sources.
    "jdk":                  {"edgelist": INPUT_DIR / "jdk.edgelist", "labels": LABELS_DIR / "jdk.labels"},  # Java class dependencies; label = top-level package
    "spanish_highschool_6": {"edgelist": INPUT_DIR / "spanish_highschool_6.edgelist", "labels": LABELS_DIR / "spanish_highschool_6.labels"},  # school friendship network; label = binary gender
}
# webkb / webkb_wisc removed 2026-07-02: input edgelists deleted deliberately (recoverable from git history if ever needed).

# Cross-model benchmark scope: which datasets and methods the comparison loop sweeps (experiments/benchmark_baselines.py).
# (politics dropped: no verifiable labels; webkb_wisc dropped 2026-07-02 with its deleted input files.)
DEGREE_RULE_CORPUS = ["airports_usa", "airports_europe", "airports_brazil", "email_eu_core", "wikics", "twitch_engb", "github", "amazon_computers", "polblogs", "blogcatalog", "coauthor_cs", "twitch_es", "twitch_fr", "dblp", "coauthor_physics", "twitch_ru", "twitch_ptbr", "cora_ml", "squirrel", "flickr_attr"]  # the stage-2 degree pool, in wave order
# The VALIDATION half of the same pool (2026-09-03, user): the corpus above FITS, these six only ever TEST. Kept as a
# separate list, not appended to the corpus, because degree_rule.py refuses to fit on anything named here - a rule that
# has seen its own test set cannot be validated by it. Ordered cheapest-first so a partial run is still a usable test.
DEGREE_RULE_VALIDATION = ["wiki_attr", "crocodile", "cora_full", "penn94", "genius", "twitch_gamers"]
# The test set for frozen_rules.FROZEN_DEGREE (2026-09-05), cheapest-first. Chosen to span the distinct_degrees cut with
# graphs stage 1 routes to AUGMENT - selection on the PREDICTOR and on the stage-1 call, never on an outcome. Nothing
# here is in DEGREE_PANEL, so none of it could have moved the cut.
DEGREE_TEST = ["celegans_neural", "messal_shale", "plant_pol_robertson", "escorts", "nematode_mammal",
               "bag_of_words_kos", "artnet_exp"]
# 2026-09-05: more of the bipartite / heterophilous / bounded-degree family, ingested to GROW the degree class from 8
# (paper_log 21). FROZEN_DEGREE is already falsified, so these are collection, not its test - though being frozen, it is
# scored on them anyway. Two pairs are NOT independent: foursquare_* share a population, ppi_* are homologous.
DEGREE_BATCH2 = ["plant_pol_kato", "board_directors", "foursquare_checkin", "foursquare_tips", "ppi_rat",
                 "ppi_mouse", "bag_of_words_nips"]
# 2026-09-07, DEGREE_FAMILY_TEST: the validation set for the FAMILY-SCOPED question (Module 14) - does the already-fixed
# cut `nbr_label_entropy < 0.6724 => degree` hold on graphs STRUCTURALLY SIMILAR to the nine it was discovered on?
# Selected by frozen_rules.FAMILY, an envelope drawn from those nine and from nothing else, applied to Netzschleuder's
# full 286-network catalogue: 139 subnetworks clear the numeric axes and exactly ONE source survives the requirement of
# a usable single-label node column (see make_netzschleuder for the four rejections). Interolog subnetworks are excluded
# deliberately - they are INFERRED by homology from other species' networks, so they are not independent graphs.
# DISCLOSURE, and it must travel with any result: all four come from one database and share a label scheme with
# ppi_rat/ppi_mouse, which are already in the study; and their entropy is expected LOW, so this tests the cut's
# degree side only. That is the half FROZEN_DEGREE's own note calls defensible, and a one-sided test is reported as one.
DEGREE_FAMILY_TEST = ["genetic_fission_yeast", "ppi_yeast", "ppi_fly", "ppi_human"]   # cheapest-first
# 2026-09-08, the NON-PPI extension of the same test (user): the four above are one database with one label scheme, so
# they cannot separate "holds in this family" from "holds on protein networks". These two are in-family by the same
# envelope, in unrelated domains, and independent of each other. They are all the catalogue has - see make_netzschleuder.
# DISCLOSURE on the label choice, declared BEFORE measuring: spanish_highschool_6 ships Sexo (binary gender), Curso
# (4 school years) and Grupo (5 classrooms), all 534/534 labelled. `Sexo` is chosen because it is the same KIND of label
# as the discovery family's binary graphs (Twitch mature-content, Facebook100 gender) and because a gender label in a
# friendship network should sit ABOVE the entropy cut - the side the rule has never been tested on in-family. That is
# selection on the PREDICTOR, which this study's protocol allows; `Curso` is the homophilous alternative and would
# almost certainly have landed below the cut, i.e. on the side already covered four times.
DEGREE_FAMILY_TEST2 = ["spanish_highschool_6", "jdk"]   # cheapest-first
assert not (set(DEGREE_RULE_CORPUS) & set(DEGREE_RULE_VALIDATION)), "a dataset is in BOTH the degree corpus and its validation set - it must be in exactly one"

# --- Module 15: the ENCODER-GENERALIZATION panels (2026-09-15) ----------------------------------------------------
# GATv2 is an EXTENSION, never a replacement: GraphSAGE stays the frozen encoder every rule was fitted under, and the
# only question asked here is whether the stage-1 and stage-2 CALLS survive a change of aggregator. So the panel is the
# graphs where the framework actually decides something, not the whole corpus - re-running 73 datasets would measure
# the encoders, which is not the question.
ENCODER_STAGE1 = ["cora", "enzymes", "ogbn_arxiv", "ogbl_ddi", "roman_empire", "tolokers", "questions",          # discovery
                  "citeseer_linqs", "proteins", "pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia",  # Module 3
                  "amazon_ratings", "squirrel_filtered",
                  "reed98", "amherst41", "johnshopkins55", "cornell5", "chameleon_filtered", "texas",             # Module 8 test
                  "twitch_de", "deezer_europe"]                                                                   # post-freeze
# The centrality rule's 12-cell validation pool (paper_log 2026-09-15). The rule itself stays SETTLED - it is applied
# here, never re-scored as right or wrong; what is scored is whether GATv2 names the same winning signal GraphSAGE did.
ENCODER_STAGE2_CENTRALITY = ["amherst41", "johnshopkins55", "cornell5", "twitch_de",
                             "airports_europe", "airports_brazil", "airports_usa", "coauthor_physics",
                             "penn94", "genius", "twitch_es", "twitch_engb"]
# The one-sided degree cut's whole evidence base: frozen_rules.DISCOVERY_9 (fitted) + ENTROPY_SPENT (already tested on).
# Every other degree batch is collection, not decision, so it is out of scope for an encoder check.
ENCODER_STAGE2_DEGREE = ["actor", "airports_europe", "amazon_computers", "questions", "twitch_engb",
                         "blogcatalog", "tolokers", "twitch_de", "twitch_es",
                         "twitch_ptbr", "wiki_attr", "crocodile", "cora_full", "penn94", "genius", "twitch_gamers"]
# THE Module-15 panel: the graphs where the framework decides something and both encoders carry a comparable
# link-prediction verdict. ogbn_arxiv is node-classification only and ogbl_ddi is scored under the OGB protocol,
# so neither can be compared across encoders; the three lists above record where each graph came from.
ENCODER_PANEL = ["enzymes", "roman_empire", "tolokers", "questions", "citeseer_linqs", "proteins", "pubmed",
                 "actor", "minesweeper", "squirrel_filtered", "reed98", "amherst41", "johnshopkins55", "cornell5",
                 "chameleon_filtered", "twitch_de", "deezer_europe", "airports_europe", "airports_brazil",
                 "airports_usa", "genius", "twitch_es", "twitch_engb", "amazon_computers", "blogcatalog",
                 "twitch_ptbr", "wiki_attr", "crocodile", "cora_full", "twitch_gamers"]
# One graph per decision the framework makes, for a cheap smoke run before the full panel.
ENCODER_SMOKE = ["roman_empire", "actor", "squirrel_filtered"]

BENCH_DATASETS = ["cora", "citeseer", "enzymes"]  # citeseer = author graph, link-pred only (no aligned labels)
BENCH_MODELS = ["identity2vec", "deepwalk", "node2vec", "struc2vec"]

# Identity2Vec embedding hyperparameters (mirror experiments/train.py defaults; walk_length=40 = repo default, paper's 80 is a recorded deviation, see notes.md).
I2V_PARAMS = {
    "dimensions": 64, "walk_length": 40, "num_walks": 10,
    "window_size": 10, "epochs": 1, "sg": 1, "e": 2.7182,
}

# Virtual-graph study (Phase 2): variants + K sweep. SAME K across variants + SAME seeds = fair comparison.
# The variants the LOCKED analyses were fitted on (Module 2 rules, Module 4 gate). Frozen: promoting a new variant must
# never silently move a frozen rule, so characterize.py / gate_rules.py read THIS list, not VG_SIMS.
VG_SIMS_LOCKED = ["psi", "degree", "centrality", "original", "hybrid"]
# THE official candidate set. psi = I2V Ψ; degree/centrality = simpler baselines; original = unchanged-graph control (K unused);
# hybrid = original ∪ psi top-K. hybrid_degree / hybrid_centrality promoted 2026-08-12 out of the notebook-2/3 experiment:
# same union, role side = degree / eigenvector centrality -> the study now varies WHICH structural augmentation is added.
# psi_lambda added 2026-09-08: same I2V λ (Eq. 3-4, d dropped, Fix-4A normalizer) used DIRECTLY as the signature.
# psi wraps λ in Fix-8's log-Poisson, which needs λ>0 and so clamps at 1e-12; λ is negative on 12-67% of nodes
# (Δ and Ω are not normalized over N(u)), and every clamped node collapses to k*log(1e-12) - log(k!) = a pure degree
# function. Dropping the wrapper removes the clamp AND the -log(k!) degree term. Kept OUT of VG_SIMS_LOCKED.
# psi_w_deg / psi_w_ev / psi_w_delta added 2026-09-09: the same Ψ with a different reading of Eq. 3-4's ω
# ("the structural attributes of v1", never written out in the paper). Open since 2026-06-24, notes.md:152/:209.
# psi_qnorm / psi_pqnorm / psi_shift added 2026-09-13: the two principled repairs for Fix-8's 1e-12 clamp
# (supervisor, pre-GATv2). qnorm = Ω normalized over N(u); pqnorm = Δ normalized too, which is what actually
# guarantees λ >= 0; shift = λ - min(λ), order-preserving. All DIAGNOSTIC, none in VG_SIMS_LOCKED.
VG_SIMS = VG_SIMS_LOCKED + ["hybrid_degree", "hybrid_centrality", "psi_lambda", "psi_w_deg", "psi_w_ev",
                            "psi_w_delta", "psi_qnorm", "psi_pqnorm", "psi_shift"]
# The SEVEN official variants, split out 2026-09-15 because VG_SIMS now also carries the psi diagnostics: a signal
# verdict must be read off the seven the study publishes, never off a list that grows whenever a diagnostic is added.
VG_SIMS_OFFICIAL = VG_SIMS_LOCKED + ["hybrid_degree", "hybrid_centrality"]
# Density-matched controls, kept OUT of VG_SIMS so the Phase-2/3 study is unchanged; opt in with VG_SIMS + VG_CONTROLS.
# The original graph's density floats per dataset (ddi avg degree 500 vs role graphs 13; cora 3.9 vs 12) and even flips
# direction, so "role vs original" confounds edge meaning with edge count. These two hold the count fixed.
VG_CONTROLS = ["original_k", "random_k"]    # original_k = K real neighbors per node; random_k = K arbitrary nodes (null scaffold)
VG_K = [5, 10, 20]                          # top-K sweep (sparsity vs over-smoothing tradeoff)
VG_SEEDS = [42, 43, 44]                     # deterministic build; extra seeds cover the downstream walk/GNN encoder

# GraphSAGE encoder (Phase 3): unsupervised GraphSAGE over the virtual graph, Skipgram-analog loss.
# Walk corpus for the positives reuses I2V_PARAMS (num_walks/walk_length/window) -> only the encoder changes vs the Phase-2 bridge.
GNN_PARAMS = {
    "hidden": 64, "dimensions": 64, "layers": 2,
    "agg": "mean",                                        # ablation B: "mean" | "weighted" (Ψ-weighted mean) | "sum" | "max"
    "lr": 0.01, "epochs": 50, "negatives": 5,             # Q=5 matches the bridge's Word2Vec negative=5
    "pairs_per_epoch": 100_000, "max_pairs": 2_000_000,   # deterministic corpus caps (runtime/memory on large graphs)
    "positives": "edge",                                  # ablation A DECIDED 2026-07-07: "edge" (A2) won LP, tied NC; "walk" = A1 bridge-comparable
    "features": "all",                                    # ablation D baseline (D0): all four structural input features
}

# Ablation D (input features): does the GNN win come from the structural features or from message passing?
# "random" (D4) is THE control - message passing with zero structural signal; compare it against the deepwalk bridge.
D_FEATURES = {
    "all":      ("D0 all",       "degree + centrality + psi + clustering"),
    "degree":   ("D1 degree",    "degree only"),
    "deg_cent": ("D2 deg+cent",  "degree + eigenvector centrality"),
    "psi":      ("D3 psi",       "psi only (confounded: the psi graph was built from it)"),
    "random":   ("D4 random",    "seeded random features - control: message passing alone"),
    "const":    ("D5 constant",  "identical rows -> z-norm zeros - floor, expect AUC ~ 0.50"),
    "none_mp":  ("D6 features only", "raw features, no message passing (layers=0) - control: features alone"),
    "centrality": ("D7 centrality", "eigenvector centrality only (per-component normalized, see GRAPH_POLICY)"),
    "clustering": ("D8 clustering", "local clustering coefficient only"),
}

# THE graph policy (defined in graph_io.py, the module that owns graph semantics) re-exported so config lives at one import.
# How every stage treats ANY dataset: self-loops, directed sources, centrality mode, signature ties, LP negatives.
from virgo.graph_io import GRAPH_POLICY, I2V_BASELINE_POLICY

# Reproduction defaults - fixed for every run so results are repeatable.
REPRO = {
    "seed": 42,
    "linkpred_test_frac": 0.30,    # 70:30 edge split
    "nodeclass_train_frac": 0.70,  # stratified split (paper sweeps 30-70%)
    "linkpred_op": "hadamard",     # edge operator, only used when linkpred_score='logreg' (node2vec protocol)
    "linkpred_score": "cosine",    # main result: unsupervised cosine similarity ranking (I2V-paper-faithful, basis of the Phase-1 repro); 'logreg' = supervised Hadamard->logistic-regression alternative, optional robustness check only
}


# Returns the registry entry for a dataset, or raises listing the valid names.
def dataset(name):
    """Look up a dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
