from torch import nn

from src.modules import MLP


class HeteroGNNNodeHead(nn.Module):
    """
    Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    """
    def __init__(self, dim_in, dim_out):
        super(HeteroGNNNodeHead, self).__init__()

        self.layer_post_mp = MLP(dim_in, dim_out, num_layers=2, bias=True, final_act=True)

    def forward(self, batch):
        return self.layer_post_mp(batch)

class HeteroGNNNodeRegressionHead(nn.Module):
    """
    Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    """
    def __init__(self, dim_in, dim_out):
        super(HeteroGNNNodeRegressionHead, self).__init__()

        self.layer_post_mp = MLP(dim_in, dim_out, num_layers=2, bias=True, final_act=True, has_l2norm=True)

    def forward(self, batch):
        return self.layer_post_mp(batch)
