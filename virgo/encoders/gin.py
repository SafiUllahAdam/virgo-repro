'''GIN over the virtual graph - the expressive alternative (CLAUDE.md phase 4, step 3).'''
# GIN is sum-aggregation by definition (that is where its isomorphism power comes from), so ablation B's agg knob does not apply.
# Per Xu et al. 2019: each layer is a 2-layer MLP over the summed neighbourhood, eps learned (train_eps=True).
# SCAFFOLD: wired and registered, but no results have been produced with it yet.

import torch
from torch_geometric.nn import GINConv

from virgo.encoders.base import GNNEncoder


class GinEncoder(GNNEncoder):
    '''GIN; sum aggregation + learned eps, everything else identical to the GraphSAGE arm.'''

    name = "gin"

    def build_convs(self, dims, agg):
        '''One GINConv per layer, each wrapping a Linear-ReLU-Linear MLP.'''
        if agg not in ("mean", "sum"):     # "mean"/"sum" both accepted so the locked default config runs unchanged; GIN sums regardless
            raise ValueError(f"GIN aggregates by sum; agg='{agg}' is a GraphSAGE-only option (ablation B).")
        return torch.nn.ModuleList(
            GINConv(torch.nn.Sequential(torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU(),
                                        torch.nn.Linear(dims[i + 1], dims[i + 1])), train_eps=True)
            for i in range(len(dims) - 1))
