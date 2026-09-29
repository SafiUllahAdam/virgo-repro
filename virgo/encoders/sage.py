'''GraphSAGE over the role graph - the locked encoder (Phase 3).'''
# Ablation B decided the aggregation: "mean" won; "weighted" swaps SAGEConv for GraphConv (same root+neighbour form, edge-weight-aware).

import torch
from torch_geometric.nn import SAGEConv, GraphConv

from virgo.encoders.base import GNNEncoder


class SageEncoder(GNNEncoder):
    '''2-layer GraphSAGE; neighbor aggregation mean/weighted/sum/max (ablation B).'''

    name = "graphsage"

    def build_convs(self, dims, agg):
        '''SAGEConv per layer; GraphConv when edge weights are used (SAGEConv ignores them).'''
        Conv = GraphConv if agg == "weighted" else SAGEConv
        return torch.nn.ModuleList(Conv(dims[i], dims[i + 1], aggr='add' if agg == "weighted" else agg)
                                   for i in range(len(dims) - 1))
