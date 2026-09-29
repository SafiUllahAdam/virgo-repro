'''GATv2 over the role graph - the attention alternative (CLAUDE.md phase 4, step 4).'''
# GATv2 (Brody et al. 2022) fixes GAT's static attention: the scoring MLP is applied AFTER the concatenation, so the
# ranking of neighbours can depend on the query node. Chosen over GAT deliberately - it is the harder baseline, so a
# surviving augmentation win cannot be dismissed as beating a weak attention model.
# Aggregation is attention-weighted by definition, so ablation B's agg knob does not apply (same situation as GIN).
# Multi-head convention follows the paper: hidden layers CONCAT their heads, the output layer AVERAGES them, so the
# embedding dimension is exactly GNN_PARAMS["dimensions"] and every eval script reads it unchanged.

import torch
from torch_geometric.nn import GATv2Conv

from virgo.encoders.base import GNNEncoder

HEADS = 4          # hidden width must divide by this: dims[i+1] // HEADS channels per head, concatenated back to dims[i+1]


class GATv2Encoder(GNNEncoder):
    '''GATv2; attention-weighted aggregation, 4 heads concatenated in hidden layers and averaged at the output.'''

    name = "gatv2"

    def build_convs(self, dims, agg):
        '''One GATv2Conv per layer; hidden layers concat HEADS heads, the last averages them to keep the output dimension.'''
        if agg != "mean":                  # "mean" is the locked default and means "leave aggregation to the encoder"
            raise ValueError(f"GATv2 aggregates by attention; agg='{agg}' is a GraphSAGE-only option (ablation B).")
        convs = []
        for i in range(len(dims) - 1):
            last = i == len(dims) - 2
            assert last or dims[i + 1] % HEADS == 0, f"hidden width {dims[i + 1]} must divide by HEADS={HEADS}"
            convs.append(GATv2Conv(dims[i], dims[i + 1] if last else dims[i + 1] // HEADS,
                                   heads=HEADS, concat=not last, add_self_loops=True))
        return torch.nn.ModuleList(convs)
