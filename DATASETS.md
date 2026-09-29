# Datasets

Every graph is used **structurally only** — edges, plus node labels where they exist. Published node features are
ignored by design, so a gain cannot be credited to the attributes instead of to the rewiring under test. The registry
lives in `virgo/config.py`; the builders in `virgo/data/`.

## Role in the study

| group | datasets | role |
|---|---|---|
| discovery panel | cora, enzymes, ogbn_arxiv, ogbl_ddi, roman_empire, tolokers, questions | stage-1 rules fitted here |
| held-out (Module 3) | citeseer_linqs, proteins, pubmed, actor, minesweeper, amazon_photo, lastfm_asia, amazon_ratings, squirrel_filtered | pre-registered test of the stage-1 rules |
| held-out (Modules 5, 7) | reed98, amherst41, johnshopkins55, cornell5 | the two-stage framework run end to end |
| transfer test (Module 8) | the four above + chameleon_filtered, texas | the clustering exception scored on six graphs it was not fitted on — **retrospective**: all six were trained before the screen existed. `chameleon_filtered` supplies its only out-of-sample negative case (low homophily, high clustering, keeps the original) |
| degree batch — fitting | twitch_de, chameleon, cornell_webkb | added 2026-09-01 to break the stage-2 deadlock (only two graphs in the corpus name degree as their sole winning signal). Split fixed on graph properties **before** any of them was trained |
| degree batch — validation | deezer_europe, wisconsin, texas | held out while the degree rule is fitted; the largest graph is deliberately on this side |
| earlier batch | twitch_pt | measured before stage 1 existed; no pre-registered call |
| **`DEGREE_RULE_CORPUS`** (separate pool, never in the study panel) | wave 1: airports_usa, airports_europe, email_eu_core, wikics, twitch_engb, github, amazon_computers · wave 2: airports_brazil, polblogs, blogcatalog, coauthor_cs, twitch_es, twitch_fr, dblp · wave 3: coauthor_physics, twitch_ru, twitch_ptbr, cora_ml, squirrel, flickr_attr | 20 graphs ingested 2026-09-02 for the stage-2 **degree** question only, chosen for structural spread rather than expected outcome; protocol pre-registered in `docs/paper_log.md` §12. Wave 3 was trained after the degree-vs-Ψ candidate was written down, so it is that candidate's held-out test |

| **`DEGREE_RULE_VALIDATION`** (the corpus's held-out half) | wiki_attr, crocodile, cora_full, penn94, genius, twitch_gamers | 6 graphs ingested 2026-09-03 to **test** what the corpus fitted, never to fit on. `degree_rule.py` refuses to fit on any of them, and every fitted cut's prediction is written to `results/paired_degree_validation_prereg.csv` **before** they are trained. Chosen on the predictor, not the outcome: three sit either side of the degree-vs-Ψ candidate's `nbr_label_entropy` cut, which the corpus tested on exactly one cell. Two disclosures travel with the set — `cora_ml` (in the corpus) is a **subgraph** of `cora_full`, and `crocodile`'s labels are quantile bins of its raw traffic target, so they correlate with degree exactly as the `airports_*` labels do |

Two registered graphs carry no scoreboard row: `citeseer` (the Identity2Vec author's graph, no aligned labels) and
`politics` (ships no labels). Both are link-prediction only.

The Facebook100 label is **gender, missing for ~10% of users** (LINKX codes it −1). Those nodes are left out of the
`.labels` file rather than written as a third class; the graph keeps every node.

## Sources

| dataset(s) | domain | reference |
|---|---|---|
| `cora`, `citeseer`, `citeseer_linqs`, `pubmed` | paper citation | Sen et al., *Collective Classification in Network Data*, AI Magazine 29(3), 2008. Cora also McCallum et al., *Information Retrieval* 3(2):127–163, 2000; PubMed also Namata et al., MLG workshop, 2012 |
| `enzymes`, `proteins` | protein structure | Borgwardt et al., *Bioinformatics* 21(suppl_1):i47–i56, 2005. PROTEINS also Dobson & Doig, *J. Mol. Biol.* 330(4):771–783, 2003 |
| `politics` (rt-pol) | Twitter retweet | Rossi & Ahmed, *The Network Data Repository*, AAAI, 2015 |
| `ogbn_arxiv` | paper citation (MAG) | Hu et al., *Open Graph Benchmark*, NeurIPS, 2020; graph from Wang et al., *Microsoft Academic Graph*, QSS 1(1), 2020 |
| `ogbl_ddi` | drug–drug interaction | Hu et al., *Open Graph Benchmark*, NeurIPS, 2020; interactions from Wishart et al., *DrugBank 5.0*, NAR 46(D1), 2018 |
| `roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`, `questions`, `squirrel_filtered` | Wikipedia text · co-purchase · synthetic grid · crowdsourcing · Q&A · Wikipedia web | Platonov et al., *A Critical Look at the Evaluation of GNNs under Heterophily*, ICLR, 2023. `squirrel_filtered` is their de-duplicated rebuild of the graph of Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 |
| `actor` | film co-occurrence | Pei et al., *Geom-GCN*, ICLR, 2020; induced from Tang et al., KDD, 2009 |
| `amazon_photo` | co-purchase | Shchur et al., *Pitfalls of GNN Evaluation*, NeurIPS R2L workshop, 2018; from McAuley et al., SIGIR, 2015 |
| `lastfm_asia` | music social network | Rozemberczki & Sarkar, *Characteristic Functions on Graphs*, CIKM, 2020 |
| `reed98`, `amherst41`, `johnshopkins55`, `cornell5`, `penn94` | Facebook100 college social | Lim et al., *Large Scale Learning on Non-Homophilous Graphs*, NeurIPS, 2021; networks from Traud, Mucha & Porter, *Physica A* 391(16), 2012 |
| `genius` | genius.com user network | Lim et al., *Large Scale Learning on Non-Homophilous Graphs*, NeurIPS, 2021; data from Lim & Benson, *Expertise and Dynamics within Crowdsourced Musical Knowledge Curation*, WWW, 2021 |
| `twitch_gamers` | Twitch mutual-follower network | Rozemberczki & Sarkar, *Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings*, 2021. Built from SNAP's archive: PyG's LINKX loader for this graph asserts rather than downloading |
| `crocodile` | Wikipedia web | Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 — the third musae-Wikipedia graph alongside `chameleon` and `squirrel`. Ships a continuous monthly-traffic target; labelled here by binning it into 5 quantiles, the same construction Geom-GCN uses for the other two |
| `cora_full` | paper citation | Bojchevski & Günnemann, *Deep Gaussian Embedding of Graphs*, ICLR, 2018; graph from McCallum et al., *Information Retrieval* 3(2), 2000. **`cora_ml` is a subgraph of this graph** |
| `wiki_attr` | Wikipedia web | Yang et al., *Network Representation Learning with Rich Text Information*, IJCAI, 2015, via the attributed-network collection of Huang, Li & Hu, WSDM, 2017 |
| `texas`, `wisconsin`, `cornell_webkb`, `chameleon` | university web pages · Wikipedia web | Pei et al., *Geom-GCN*, ICLR, 2020. WebKB pages from the CMU World Wide Knowledge Base project, 1998; `chameleon` from Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 |
| `twitch_engb`, `twitch_es`, `twitch_fr`, `twitch_ptbr`, `twitch_ru`, `github` | streaming social · developer social | Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 (musae) — read from SNAP, PyG's host is dead |
| `airports_usa`, `airports_europe`, `airports_brazil` | air transport | Ribeiro, Saverese & Figueiredo, *struc2vec*, KDD, 2017 — labels are traffic quartiles, which correlate with degree (disclosed in paper_log §12) |
| `wikics` | Wikipedia articles | Mernyei & Cangea, *Wiki-CS*, ICML GRL workshop, 2020 |
| `coauthor_cs`, `coauthor_physics`, `amazon_computers` | coauthorship · co-purchase | Shchur et al., *Pitfalls of GNN Evaluation*, NeurIPS R2L workshop, 2018 |
| `email_eu_core` | email | Leskovec, Kleinberg & Faloutsos, TKDD 1(1), 2007; Yin et al., KDD, 2017 |
| `polblogs` | political blogs | Adamic & Glance, *LinkKDD*, 2005 |
| `blogcatalog`, `flickr_attr` | social | Huang, Li & Hu, *Label Informed Attributed Network Embedding*, WSDM, 2017 |
| `dblp`, `cora_ml` | paper citation | Bojchevski & Günnemann, *Deep Gaussian Embedding of Graphs*, ICLR, 2018 |
| `squirrel` | Wikipedia | Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 — the ORIGINAL geom-gcn copy; `squirrel_filtered` is the de-duplicated one |
| `twitch_de`, `deezer_europe` | streaming social · music social | Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 (Twitch); Rozemberczki & Sarkar, *Characteristic Functions on Graphs*, CIKM, 2020 (Deezer) |

**How they were obtained.** `cora`, `citeseer`, `enzymes`, `proteins` and `politics` come from the Identity2Vec
author's `input.zip` (Network Repository files); their labels are rebuilt from LINQS (`cora`, `citeseer_linqs`) or
Network Repository (`enzymes`, `proteins`). `pubmed`, `actor`, `amazon_photo`, the four LINKX graphs and the four
Geom-GCN web/Wikipedia graphs are built through `torch_geometric`; `lastfm_asia`, `twitch_de` and `deezer_europe` from
SNAP directly (PyG's `graphmining.ai` host returns 503 for all three); the six Platonov graphs from the authors' `.npz`
release; the two OGB graphs through `ogb`.

Two name collisions worth stating once: **`cornell_webkb` (183 web pages) is not `cornell5`** (18,660-node Facebook100
network), and **`chameleon` is the original Geom-GCN copy** whose duplicate nodes Platonov's `chameleon_filtered`
removes — both are registered, and any result on `chameleon` carries that caveat.
