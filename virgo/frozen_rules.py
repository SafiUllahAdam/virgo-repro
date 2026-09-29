'''THE frozen rule artifacts: the Module-2 fitting panel and its two rules (locked 2026-07-29), and the second stage-1
condition fitted on top of them - the Module-4 retention gate (locked 2026-08-05, SUPERSEDED) and the Module-8 clustering
exception that replaced it (locked 2026-09-01).

Nothing here is recomputed. The point and interval of each rule are the exact numbers candidate_rules() produced on the
panel (results/candidate_rules.csv, link-prediction credible rows) and must NEVER be re-derived from data that includes an
unseen dataset - that would make Module 3 circular. Prediction uses the POINT midpoint (deterministic); the interval is
carried only for honest reporting, since every cut inside it fits the panel equally well.'''

from collections import namedtuple

# The datasets the rules were FITTED on. A FIXED LITERAL, never derived from STUDY: adding a dataset to the study must not
# silently join the discovery panel and overwrite the seven-dataset Module-2 result. Anything outside this set is unseen -
# it may be PREDICTED, never fitted.
DISCOVERY_PANEL = ["cora", "enzymes", "ogbn_arxiv", "ogbl_ddi", "roman_empire", "tolokers", "questions"]

# The Module-3 held-out datasets, stored SEPARATELY so they can never leak into fitting. Append more as they are ingested;
# predict_module3.py defaults to this list. A dataset must live in exactly one of the two panels, never both.
HELDOUT = ["citeseer_linqs", "proteins", "pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia", "amazon_ratings", "squirrel_filtered"]

assert not (set(DISCOVERY_PANEL) & set(HELDOUT)), "a dataset is in BOTH the discovery panel and the held-out set - it must be in exactly one"

# predictor = the graph property; op = the side that says "augment"; point = the split used for prediction;
# interval = every cut that fits the panel equally well (report this, not the 4-decimal point); needs_labels = property is n/a on an unlabelled graph.
Rule = namedtuple("Rule", "name predictor op point interval needs_labels")
FROZEN_RULES = [
    Rule("rule1", "homophily_adjusted",     "<", 0.227,  (0.0926, 0.3613), True),
    Rule("rule2", "largest_component_frac", ">", 0.9588, (0.9177, 1.0),    False),
]


# The lead rule, frozen with the rules themselves: Module 2 designated rule 1 the primary (rho -1.00 on the bias-free gap,
# graded across 5 distinct values, vs rule 2's two-group split). It is what breaks a disagreement - a single call, not two.
LEAD = FROZEN_RULES[0]


def predict_one(rule, value):
    '''augment / keep original for one rule's point, or n/a when the property is missing (e.g. homophily on an unlabelled graph).'''
    if value is None or value != value:                    # NaN -> the property needs labels this graph does not have
        return "n/a"
    hit = value < rule.point if rule.op == "<" else value > rule.point
    return "augment" if hit else "keep original"


def predict(props):
    '''Per-rule verdicts for one property dict, plus the combined call: the firing rules' shared verdict, or that they disagree.'''
    calls = {r.name: predict_one(r, props.get(r.predictor)) for r in FROZEN_RULES}
    live = [v for v in calls.values() if v != "n/a"]
    combined = "no rule fires" if not live else ("rules disagree" if len(set(live)) > 1 else live[0])
    return calls, combined


# --- Module 4: the retention GATE, locked 2026-08-05, SUPERSEDED 2026-09-01 by FROZEN_EXCEPTION below ------------------
# Kept because Modules 4 and 5 are published against it and gate_rules.py asserts the screen still reproduces its cut. It
# is NO LONGER part of the stage-1 call: predict_gated() consults the clustering exception instead. Why it was dropped -
# it needs the role graph BUILT before the decision can be made, and on the LINKX test its only differentiating call was
# its only error (reed98 augments at 2.55 sigma; the gate said keep).
# Module 3 showed rule 1 fails ASYMMETRICALLY: above the boundary it called "keep original" and was right 3/3; below it
# called "augment" and was right only 2/4. So rule 1 is a VETO, and a second variable is needed on one side only - inside
# the low-homophily zone. This gate is that variable: a property of the VIRTUAL graph, so it needs no labels, and the
# rewiring must already be BUILT (never trained) to measure it.
# Numbers are the exact low-homophily-zone / gap_rel row of results/gate_candidates.csv - never re-derive them from data
# that includes the graph being predicted.
Gate = namedtuple("Gate", "name predictor op point interval needs_labels applies_when fitted_on")
FROZEN_GATE = Gate("gate", "original_retention", "<", 0.0119, (0.0062, 0.0176), False,
                   "rule1 == augment", "GATE_PANEL")

# The Module-4 FITTING panel: the Module-2 discovery panel (minus ogbn_arxiv, which is node-classification only) PLUS the
# nine Module-3 datasets, whose verdicts are already published and so are no longer unseen. Declared, because the gate is
# FITTED here: a dataset in this list can never test the gate, only a genuinely unseen third set can.
GATE_PANEL = ["cora", "enzymes", "ogbl_ddi", "roman_empire", "tolokers", "questions",
              "pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia", "amazon_ratings", "squirrel_filtered"]

# Module-5 held-out: the third, genuinely unseen set the GATE is tested on. Append datasets as they are ingested.
# The LINKX Facebook100 trio (2026-08-14) is measured here FIRST, as stage 1: only the graphs this combined call sends to
# "augment" go on to test the strategy rule, since that rule is conditional on augmentation being indicated.
GATE_HELDOUT = ["reed98", "amherst41", "johnshopkins55", "cornell5"]

assert not (set(GATE_PANEL) & set(GATE_HELDOUT)), "a dataset is in BOTH the gate panel and the gate held-out set - it must be in exactly one"


# --- Module 6: the STRATEGY rule, locked 2026-08-13 -------------------------------------------------------------------
# Modules 2-4 answer WHETHER to augment. This answers WHICH structural signal, and only partly: it separates centrality
# from the rest, and says nothing about psi vs degree below the cut. It is CONDITIONAL - it may only be consulted once
# augmentation is already indicated, because over all 14 panel datasets the same split takes 4 exceptions.
# Status, which must travel with it: FITTED, not validated. Nothing was held out - every labelled dataset is in the panel.
#   in-sample  0 exceptions over the 7 augmenting datasets, rho 0.866, graded over 0.0123 -> 0.8498 (not a two-group split)
#   out-of-sample  leave-one-out 5/7 against a 4/7 majority baseline - ONE call better than guessing
#   band-dependent  it exists under the sem tie band only; under the Module-2 sigma band the ties re-inflate and it takes
#                   2 exceptions. The band choice is a methodological decision, so it is part of the rule, not context.
#   co-predictor  degree_assortativity (exploratory tier) produces the identical split at rho 0.866; the two correlate at
#                 0.71, so which one is THE predictor is not determined by this data.
#   NOT homophily  rho 0.32 against homophily_adjusted. roman_empire is the proof: lowest adjusted homophily in the panel
#                 (-0.0468) yet centrality wins. This is the same property Module 2 screened and DROPPED for the augment
#                 question - a different question, with a different answer.
Strategy = namedtuple("Strategy", "name signal predictor op point interval needs_labels applies_when fitted_on")
FROZEN_STRATEGY = Strategy("strategy1", "centrality", "nbr_predictability_adjusted", ">", 0.0092, (0.006, 0.0123), True,
                           "augmentation is already indicated", "STRATEGY_PANEL")

# The Module-6 FITTING panel: every labelled dataset with a link-prediction score, at 10 seeds. ogbn_arxiv is
# node-classification only and ogbl_ddi is unlabelled, so neither can carry a signal label.
STRATEGY_PANEL = ["cora", "enzymes", "roman_empire", "tolokers", "questions", "squirrel_filtered", "amazon_ratings",
                  "amazon_photo", "lastfm_asia", "pubmed", "actor", "minesweeper", "citeseer_linqs", "proteins"]

# Module-7 held-out: the genuinely new datasets the strategy rule is tested on. Append as they are ingested.
# 2026-08-14, the LINKX Facebook100 networks that PASSED stage 1 - the combined augment call above sent them to "augment",
# which is the rule's own precondition, so they can actually test it. reed98 was measured with them and is deliberately
# ABSENT: stage 1 called it "keep original", so a strategy test on it would be scored on a graph the earlier rules say
# should not be augmented at all. Selecting the test set on the STAGE-1 CALL (never on an outcome) is what makes this
# conditional; the cost is that stage 2 is only ever tested on graphs stage 1 sends here.
STRATEGY_HELDOUT = ["amherst41", "johnshopkins55", "cornell5"]

assert not (set(STRATEGY_PANEL) & set(STRATEGY_HELDOUT)), "a dataset is in BOTH the strategy panel and its held-out set - it must be in exactly one"


def predict_strategy(props):
    '''Which signal to add, once augmentation is indicated: "centrality", or "psi or degree" which the rule cannot split.'''
    v = props.get(FROZEN_STRATEGY.predictor)
    if v is None or v != v:                                # needs labels; an unlabelled graph gets no strategy call
        return "n/a"
    return FROZEN_STRATEGY.signal if v > FROZEN_STRATEGY.point else "psi or degree (undetermined)"


# --- Module 8: the CLUSTERING EXCEPTION, locked 2026-09-01 ------------------------------------------------------------
# THE second stage-1 condition, replacing the retention gate in the same slot: inside the low-homophily zone only. The
# difference that motivated it is that this one is measured on the ORIGINAL graph - no role graph is built, no labels are
# needed, nothing about the rewiring is assumed.
# Framing, which is not cosmetic: adjusted homophily still MAKES the decision. High homophily keeps the original, low
# homophily normally augments, and this is the EXCEPTION that catches the low-homophily graphs augmentation does not
# help - the ones whose neighbourhoods are already dense, so role edges add nothing they did not already have.
#   fitted     the GATE_PANEL zone (results/stage1_pair_rules.csv): rho -0.82 on gap_rel, one in-sample exception
#              (minesweeper), 8/10 decided panel cells against rule 1 alone at 7/10.
#   tested     6 unseen graphs (GATE_HELDOUT + chameleon_filtered + texas), 4 of them decided: 4/4, against rule 1 alone
#              3/4 and the retention gate 3/4. chameleon_filtered IS the negative side - adjusted homophily 0.0295 (low),
#              clustering 0.5769, and it keeps the original at -1.04 sigma.
#   caution    the cut is a CANDIDATE, not a universal constant. The negative side rests on two graphs (amazon_ratings
#              in-sample, chameleon_filtered out of sample), and the test was retrospective - all six were trained and
#              scored before this screen existed - so it is transfer evidence, not a pre-registration.
#   interval   panel-fitted (0.5329, 0.5816); chameleon_filtered at 0.5769 narrows the upper end, so the honest interval
#              after the test is (0.5329, 0.5769). The POINT did not move.
FROZEN_EXCEPTION = Gate("exception", "avg_clustering", "<", 0.5573, (0.5329, 0.5769), False,
                        "rule1 == augment", "GATE_PANEL, tested on 6 unseen graphs")


def predict_gated(props):
    '''Stage 1: rule 1 decides, the clustering exception overrides it inside the low-homophily zone, rule 2 covers unlabelled graphs.'''
    calls, _ = predict(props)
    exc = predict_one(FROZEN_EXCEPTION, props.get(FROZEN_EXCEPTION.predictor))
    if calls["rule1"] == "keep original":                  # the veto side: 3/3 in Module 3, no second variable needed
        return calls, exc, "keep original", "rule 1 veto"
    if calls["rule1"] == "augment":                        # the ambiguous side: the only place the exception speaks
        return calls, exc, (exc if exc != "n/a" else "augment"), ("clustering exception" if exc != "n/a" else "rule 1 (clustering not measurable)")
    # No labels -> rule 1 cannot fire at all. The exception's cut was fitted INSIDE the zone, so it does not transfer
    # here; fall back to rule 2, exactly as Module 3 did for ogbl_ddi, and carry the clustering call as information only.
    return calls, exc, calls["rule2"], "rule 2 (no labels)"


# --- Module 12: the DEGREE candidate, frozen 2026-09-04 ---------------------------------------------------------------
# THE first degree-branch cut ever written down. Frozen on the user's instruction so it can be TESTED; freezing here
# means "this exact claim is now pre-registered", NOT "this rule is validated". Read the status block before quoting it.
# It comes from the degree-SPECIFIC tier (experiments/degree_features.py), not from the 15 general properties that
# returned nothing three times (paper_log 13, 18, and the degree-vs-everything half of 20).
#   fitted     13 degree-or-psi cells over 34 paired-verdict graphs, 5 degree / 8 psi (results/degree_features.csv)
#              2 exceptions, LOO 0.846 against a 0.615 majority baseline, interval (130, 187)
#   NOT density  rho -0.380 against density, the confound that disqualified four of the nine features in its own tier
#              (degree_vg_stability sits at rho 1.0000 against density). It IS 0.828 rank-correlated with `nodes`, but
#              raw `nodes` alone splits worse - 4 exceptions, LOO 0.538 - so it is not merely size.
#   THE PRICE, and it is the reason this is a candidate and not a rule: over all C(13,5) = 1287 label arrangements the
#              best threshold's error count gives P(<=2 exceptions) = 0.1212, so across the 9 properties tried chance
#              alone is expected to deliver 1.09 at this quality. Exactly one was found. This cut sits AT the rate noise
#              supplies. It is frozen to be falsified, not because the evidence is good.
#   one-sided  the cut has NO false positives - all three graphs below it (actor 89, airports_europe 102, twitch_engb
#              130) name degree alone. It fails by MISSING degree winners: questions (320) and amazon_computers (347)
#              are degree winners above the cut. So "below the cut => degree" is the defensible half; "above the cut =>
#              psi" is the half with both exceptions, and a test should score the two sides separately.
#   needs the role graph  distinct_degrees is read off the ORIGINAL graph, so unlike the retired Module-4 gate this one
#              does NOT need the role graph built. That is deliberate: the Module-4 gate was retired partly for that.
Degree = namedtuple("Degree", "name signal predictor op point interval needs_labels applies_when fitted_on")
FROZEN_DEGREE = Degree("degree1", "degree", "distinct_degrees", "<", 158.5, (130, 187), False,
                       "augmentation is indicated AND the winner is degree or psi", "DEGREE_TIER over 34 paired-verdict graphs")

# The 34 graphs the cut was fitted on. Anything here can never test it.
DEGREE_PANEL = ["actor", "airports_brazil", "airports_europe", "airports_usa", "amazon_computers", "blogcatalog",
                "coauthor_cs", "coauthor_physics", "cora_full", "cora_ml", "crocodile", "dblp", "email_eu_core",
                "flickr_attr", "genius", "github", "lastfm_asia", "penn94", "polblogs", "pubmed", "questions",
                "roman_empire", "squirrel", "squirrel_filtered", "tolokers", "twitch_de", "twitch_engb", "twitch_es",
                "twitch_fr", "twitch_gamers", "twitch_ptbr", "twitch_ru", "wiki_attr", "wikics"]


def predict_degree(props):
    '''Which of degree or psi, given that augmentation is indicated: "degree", "psi", or "n/a" when the input is missing.'''
    v = props.get(FROZEN_DEGREE.predictor)
    if v is None or v != v:
        return "n/a"
    return FROZEN_DEGREE.signal if v < FROZEN_DEGREE.point else "psi"


# --- Module 14: the FAMILY-SCOPED degree candidate, frozen 2026-09-07 --------------------------------------------------
# Module 10 fitted `nbr_label_entropy < 0.6724 => degree` on nine cells and Module 11 falsified it as a UNIVERSAL rule
# (1/3 on six pre-registered graphs, beaten 3/3 by a constant "always psi"). The user's 2026-09-07 instruction is not to
# refit it - the cut stays EXACTLY where it was - but to ask the narrower question it was never asked:
#     does it hold on graphs STRUCTURALLY SIMILAR to the nine it was discovered on?
# A rule that survives that is a FAMILY rule, valid for a type of graph, and must never be reported as a universal one.
# Nothing here is refitted. The cut is Module 10's to four decimals; the envelope is a mechanical min/max description of
# the discovery nine, rounded outward, and is a DEFINITION, not a fit - it contains all nine by construction.
Candidate = namedtuple("Candidate", "name signal predictor op point interval fitted_on record")
FROZEN_ENTROPY = Candidate("entropy1", "degree", "nbr_label_entropy", "<", 0.6724, (0.6296, 0.7151), "DISCOVERY_9",
                           "falsified as a universal rule: 1/3 on DEGREE_RULE_VALIDATION, a constant 'always psi' scores 3/3")

# The nine cells the cut was fitted on (Module 10, paired band, degree-or-psi sole winners). A FIXED LITERAL: any graph
# here can never test the rule, and the envelope below is derived from these nine and from nothing else.
DISCOVERY_9 = ["actor", "airports_europe", "amazon_computers", "questions", "twitch_engb",     # degree wins
               "blogcatalog", "tolokers", "twitch_de", "twitch_es"]                             # psi wins

# Cells already SPENT testing this cut - Module 10's single held-out graph and Module 11's pre-registered six. Scoring
# them again would recycle a used test as if it were new evidence.
ENTROPY_SPENT = ["twitch_ptbr"] + ["wiki_attr", "crocodile", "cora_full", "penn94", "genius", "twitch_gamers"]

# THE FAMILY, declared before any candidate graph was measured: the discovery nine's own envelope on seven axes, all
# read off the ORIGINAL graph. The axes are the ones the nine are TIGHT on - every one of them is single-mode, near
# connected, disassortative, sparse and few-class - so the envelope describes what they are, not what separates them.
# `nbr_label_entropy` is deliberately EXCLUDED: it is the predictor, and a family defined on it would hand the rule its
# own answer instead of testing it on graphs that straddle the cut naturally.
FAMILY = {"largest_component_frac": (0.97, 1.0),        # 9/9 in 0.973-1.000; 8 of the 9 are a single component
          "degree_assortativity":   (-0.30, 0.00),      # 9/9 in -0.2252..-0.0200 - disassortative WITHOUT exception
          "avg_degree":             (5.0, 100.0),       # 9/9 in 6.28-88.28
          "density":                (0.0, 0.10),        # 9/9 in 0.0001-0.0755
          "n_classes":              (2, 10),            # 9/9 in 2-10, median 2
          "nodes":                  (300, 60000)}       # 9/9 in 399-48921
# One negative condition, and the reason it is needed: the 14 Netzschleuder graphs ingested for Module 12 are BIPARTITE,
# a class absent from the discovery nine, and the study already knows degree wins there for a family reason of its own.
# Letting them in would answer a different question with this rule's name on it.
FAMILY_EXCLUDE_BIPARTITE = True


def in_family(props):
    '''Which FAMILY axes a graph satisfies, and whether it is in the family at all: (bool, list of failing axes).'''
    bad = [k for k, (lo, hi) in FAMILY.items()
           if props.get(k) is None or props.get(k) != props.get(k) or not lo <= float(props[k]) <= hi]
    if FAMILY_EXCLUDE_BIPARTITE and props.get("bipartite"):
        bad.append("bipartite")
    return not bad, bad


def predict_entropy(props):
    '''Which of degree or psi the FROZEN_ENTROPY cut names, or "n/a" when the property is missing.'''
    v = props.get(FROZEN_ENTROPY.predictor)
    if v is None or v != v:
        return "n/a"
    return FROZEN_ENTROPY.signal if v < FROZEN_ENTROPY.point else "psi"


# --- Module 14: the ONE-SIDED degree rule, frozen 2026-09-08 -----------------------------------------------------------
# The user's decision after the family test: freeze the DEGREE side and claim nothing above the cut. FROZEN_ENTROPY was
# a two-sided rule ("below => degree, else psi") and its psi half failed repeatedly; this artifact keeps only the half
# the evidence supports. Above the cut it returns "no claim", NOT psi - that is the whole point of the change.
#     nbr_label_entropy < 0.6724  =>  degree
#     nbr_label_entropy >= 0.6724 =>  no claim
# The cut is Module 10's, unchanged to four decimals. Nothing here was refitted.
#
# HOW A ONE-SIDED RULE MUST BE SCORED, because the obvious way is wrong: below the cut it always answers "degree", so on
# below-cut cells alone it IS the constant "always degree" and can never beat it. Its content is entirely in WHICH graphs
# fall below the cut, so the evidence is ENRICHMENT - the degree win rate below the cut against the rate above it.
#   all 40 augmenting cells (argmax labels)   below 17/29 = 0.586   above 2/11 = 0.182   base 0.475   Fisher p = 0.0248
#   out-of-sample (discovery 9 removed, n=31) below 12/24 = 0.500   above  2/7 = 0.286   base 0.452   Fisher p = 0.2874
# So the enrichment is significant over the whole pool and is NOT significant once the cells it was fitted on are
# removed. That is why this is frozen as a CANDIDATE to be tested, exactly as FROZEN_DEGREE was, and never reported as
# an established selector.
OneSided = namedtuple("OneSided", "name signal predictor op point interval fitted_on claims record")
FROZEN_DEGREE_ONESIDED = OneSided(
    "degree_onesided", "degree", "nbr_label_entropy", "<", 0.6724, (0.6296, 0.7151), "DISCOVERY_9",
    "below the cut only - above it the rule makes NO CLAIM",
    "below-cut calls out of sample 6/9; enrichment p=0.0248 over all 40 augmenting cells, p=0.2874 out of sample")


def predict_degree_onesided(props):
    '''"degree" below the cut, "no claim" above it, "n/a" when the property is missing. Never returns psi.'''
    v = props.get(FROZEN_DEGREE_ONESIDED.predictor)
    if v is None or v != v:
        return "n/a"
    return FROZEN_DEGREE_ONESIDED.signal if v < FROZEN_DEGREE_ONESIDED.point else "no claim"
