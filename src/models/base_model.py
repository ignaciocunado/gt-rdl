from abc import ABC, abstractmethod
import torch
from typing import Dict, Any
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


class BaseModel(torch.nn.Module, ABC):
    """Base RelBench model with common encoder components.
    
    This abstract class provides the basic encoder structure and temporal processing,
    while leaving the rest of the architecture to be defined by child classes.
    
    Args:
        data (HeteroData): Input graph data
        col_stats_dict (Dict): Column statistics dictionary
        channels (int): Number of hidden channels
        torch_frame_model_kwargs (Dict[str, Any], optional): Additional encoder kwargs
    """
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict,
        channels: int,
        torch_frame_model_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        
        # Initialize base encoders
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )
        
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types 
                if "time" in data[node_type]
            ],
            channels=channels,
        )
    
    def reset_base_parameters(self):
        """Reset parameters of base encoders."""
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
    
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        """Forward pass with base encoding operations.
        
        Args:
            batch (HeteroData): Input batch
            entity_table (NodeType): Target entity type
            
        Returns:
            Tensor: Model output
        """
        # Get basic encodings
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        
        # Add temporal information
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
            
        # Let child class handle the rest
        return self.post_forward(x_dict, batch, entity_table, seed_time)
    
    @abstractmethod
    def post_forward(
        self,
        x_dict: Dict[str, Tensor],
        batch: HeteroData,
        entity_table: NodeType,
        seed_time: Tensor,
    ) -> Tensor:
        """Process encoded features to produce final output.
        
        This method should be implemented by child classes to define
        custom processing of encoded features (e.g., GNN layers, MLPs, etc.)
        
        Args:
            x_dict (Dict[str, Tensor]): Encoded node features
            batch (HeteroData): Input batch
            entity_table (NodeType): Target entity type
            seed_time (Tensor): Temporal information
            
        Returns:
            Tensor: Model output
        """
        raise NotImplementedError