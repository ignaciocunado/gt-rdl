from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import HeteroConv, LayerNorm, GINConv, MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import HeteroData

from .base_model import BaseModel

class HeteroGraphGIN(BaseModel):
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

        # Initialize GIN-specific components
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GINConv(
                        nn.Sequential(
                            nn.Linear(channels, channels), 
                            nn.ReLU(), 
                            nn.Linear(channels, channels)
                        ), 
                        aggr=aggr
                    )
                    for edge_type in data.edge_types
                },
                aggr="sum",
            )            
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)
        
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        self.head.reset_parameters()

    def post_forward(
        self,
        x_dict: Dict[str, Tensor],
        batch: HeteroData,
        entity_table: NodeType,
        seed_time: Tensor,
    ) -> Tensor:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        

        return self.head(x_dict[entity_table][: seed_time.size(0)])