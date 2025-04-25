from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import HeteroConv, LayerNorm, GINConv, MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import HeteroData

from .base_model import BaseModel
from relbench.modeling.nn import HeteroGraphSAGE as HeteroGraphSAGE_RelBench

class HeteroGraphSage(BaseModel):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict,
        channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: str = "sum",
        norm: str = "batch_norm",
        torch_frame_model_kwargs: Dict[str, Any] = {},
    ):
        # Call parent class constructor first
        super().__init__(
            data=data,
            col_stats_dict=col_stats_dict,
            channels=channels,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )

        self.gnn = HeteroGraphSAGE_RelBench(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def post_forward(
        self,
        x_dict: Dict[str, Tensor],
        batch: HeteroData,
        entity_table: NodeType,
        seed_time: Tensor,
    ) -> Tensor:
        
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[entity_table][: seed_time.size(0)])