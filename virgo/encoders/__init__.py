'''Swappable encoders over the virtual graph. THE extension point: the graph is the variable under study, the encoder is a plug.'''
# Add an encoder in three lines: (1) new module subclassing GNNEncoder with one build_convs(), (2) import it here,
# (3) add its study id to ENCODERS. Every driver, CLI and scoreboard row picks it up with no further edit.

from virgo.encoders.base import GNNEncoder, feature_cache, WALKS
from virgo.encoders.sage import SageEncoder
from virgo.encoders.gin import GinEncoder
from virgo.encoders.gatv2 import GATv2Encoder

# Study id (used verbatim in .emb filenames and the scoreboard "encoder" column) -> encoder class.
# The id carries the ablation-A positives setting; the value itself is read from config.GNN_PARAMS["positives"].
ENCODERS = {
    "graphsage_edge": SageEncoder,
    "gin_edge": GinEncoder,
    "gatv2_edge": GATv2Encoder,
}

__all__ = ["GNNEncoder", "SageEncoder", "GinEncoder", "GATv2Encoder", "ENCODERS", "feature_cache", "WALKS"]
