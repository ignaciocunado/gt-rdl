from typing import Dict, Any

import math
import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import NodeType
from torch_scatter import scatter_max

from src.models.base_model import BaseModel

class FraudGT(BaseModel):

    def __init__(
            self,
            data: HeteroData,
            col_stats_dict: Dict,
            channels: int,
            out_channels: int,
            dropouts: list,
            num_layers: int = 2,
            num_layers_pre_gt: int = 0,
            head: str = 'None',
            edge_features: bool = False,
            torch_frame_model_kwargs: Dict[str, Any] = {},
    ):

        super().__init__(
            data=data,
            col_stats_dict=col_stats_dict,
            channels=channels,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )

        self.num_layers = num_layers
        self.metadata = data.metadata()
        self.dim_h = channels
        self.input_drop = nn.Dropout(0.0)
        self.activation = nn.GELU
        self.batch_norm = False
        self.layer_norm = True
        self.l2_norm = False
        self.layers_pre_gt = num_layers_pre_gt
        self.edge_features = edge_features
        dim_edge_in = self.dim_h * 2 if self.edge_features else self.dim_h

        if self.layers_pre_gt > 0:
            self.pre_gt_dict = torch.nn.ModuleDict()
            for node_type in self.metadata[0]:
                self.pre_gt_dict[node_type] = GeneralMultiLayer(self.layers_pre_gt, self.dim_h, self.dim_h, dim_inner=self.dim_h, final_act=True, has_bn=self.batch_norm, has_ln=self.layer_norm, has_l2norm=self.l2_norm)

        local_gnn_type, global_model_type = 'None', 'SparseNodeTransformer'

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() # Never used?
        dim_h_total = self.dim_h
        for i in range(self.num_layers):
            conv = FraudGTLayer(self.dim_h, dim_edge_in, self.dim_h, self.dim_h, self.metadata,
                                local_gnn_type, global_model_type, i,
                                dropouts,
                                8,
                                layer_norm=self.layer_norm,
                                batch_norm=self.batch_norm,
                                return_attention=False)
            self.convs.append(conv)

            if self.layer_norm or self.batch_norm:
                self.norms.append(nn.ModuleDict())
                for node_type in self.metadata[0]:
                    if self.layer_norm:
                        self.norms[-1][node_type] = nn.LayerNorm(self.dim_h)
                    elif self.batch_norm:
                        self.norms[-1][node_type] = nn.BatchNorm1d(self.dim_h)

            self.residual = 'Fixed'

            dim_h_total = self.dim_h

        if head == 'HeteroGNNNodeHead':
            self.post_gt = HeteroGNNNodeHead(dim_h_total, out_channels)
        else:
            raise ValueError(f'Attention head {head} not supported.')

    def post_forward(self, x_dict: Dict[str, Tensor], batch: HeteroData, entity_table: NodeType, seed_time: Tensor):
        x_dict = {
            node_type: self.input_drop(x_dict[node_type]) for node_type in x_dict
        }

        if self.layers_pre_gt > 0:
            x_dict = {
                node_type: self.pre_gt_dict[node_type](x_dict[node_type]) for node_type in x_dict
            }

        # Concat source and destination node features for edges
        if self.edge_features:
            for edge_type in batch.edge_types:
                src, rel, dst = edge_type
                edge_index = batch[edge_type].edge_index
                row, col = edge_index   # each of shape [E]

                src_feats = x_dict[src][row]   # shape [E, D]
                dst_feats = x_dict[dst][col]   # shape [E, D]
                new_feat = torch.cat([src_feats, dst_feats], dim=1)

                batch[edge_type].edge_attr = new_feat

        # Forward through transformer layers
        for i in range(self.num_layers):
            batch, x_dict = self.convs[i](batch, x_dict)

        return self.post_gt(x_dict[entity_table][: seed_time.size(0)])

class FraudGTLayer(nn.Module):
    """
    FraudGT layer
    """
    def __init__(self, dim_in, dim_edge_in, dim_h, dim_out, metadata, local_gnn_type, global_model_type, index, dropouts, num_heads=1,
                 layer_norm=False, batch_norm=False, return_attention=False, **kwargs):
        super(FraudGTLayer, self).__init__()

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.index = index
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = nn.GELU()
        self.metadata = metadata
        self.return_attention = return_attention
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.kHop = 2
        self.bias = Parameter(torch.Tensor(self.kHop))
        self.attn_bi = Parameter(torch.empty(self.num_heads, self.kHop))

        # Residual connection
        self.skip_local = torch.nn.ParameterDict()
        self.skip_global = torch.nn.ParameterDict()
        for node_type in metadata[0]:
            self.skip_local[node_type] = Parameter(torch.Tensor(1))
            self.skip_global[node_type] = Parameter(torch.Tensor(1))


        # Global Attention
        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.e_lin = torch.nn.ModuleDict()
        self.g_lin = torch.nn.ModuleDict()
        self.oe_lin = torch.nn.ModuleDict()
        self.o_lin = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            # Different node type have a different projection matrix
            self.k_lin[node_type] = Linear(dim_in, dim_h)
            self.q_lin[node_type] = Linear(dim_in, dim_h)
            self.v_lin[node_type] = Linear(dim_in, dim_h)
            self.o_lin[node_type] = Linear(dim_h, dim_out)
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.e_lin[edge_type] = Linear(dim_edge_in, dim_h)
            self.g_lin[edge_type] = Linear(dim_edge_in, dim_h)
            self.oe_lin[edge_type] = Linear(dim_h, dim_out * 2)

        H, D = self.num_heads, self.dim_h // self.num_heads

        edge_weight = True # TODO: Do we want edge weights
        if edge_weight:
            self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
            self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
            nn.init.xavier_uniform_(self.edge_weights)
            nn.init.xavier_uniform_(self.msg_weights)

        self.norm1_local = torch.nn.ModuleDict()
        self.norm1_global = torch.nn.ModuleDict()
        self.norm2_ffn = torch.nn.ModuleDict()
        self.project = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.project[node_type] = Linear(dim_h * 2, dim_h)
            if self.layer_norm:
                self.norm1_local[node_type] = nn.LayerNorm(dim_h)
                self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_local[node_type] = nn.BatchNorm1d(dim_h)
                self.norm1_global[node_type] = nn.BatchNorm1d(dim_h)
        self.norm1_edge_local = torch.nn.ModuleDict()
        self.norm1_edge_global = torch.nn.ModuleDict()
        self.norm2_edge_ffn = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            if self.layer_norm:
                self.norm1_edge_local[edge_type] = nn.LayerNorm(dim_edge_in)
                self.norm1_edge_global[edge_type] = nn.LayerNorm(dim_edge_in)
            if self.batch_norm:
                self.norm1_edge_local[edge_type] = nn.BatchNorm1d(dim_edge_in)
                self.norm1_edge_global[edge_type] = nn.BatchNorm1d(dim_edge_in)

        self.dropout_local = nn.Dropout(dropouts[0]) # Lower dropout as graph size increases - 0.0 for large graphs, 0.2 for small
        self.dropout_global = nn.Dropout(dropouts[1]) # Lower dropout as graph size increases - 0.0 for large graphs, 0.2 for small
        self.dropout_attn = nn.Dropout(dropouts[1]) # Large 0.2, medium and small 0.3

        for node_type in metadata[0]:
            # Different node type have a different projection matrix
            if self.layer_norm:
                self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm2_ffn[node_type] = nn.BatchNorm1d(dim_h)

        # Feed Forward block.
        self.ff_linear1_type = torch.nn.ModuleDict()
        self.ff_linear2_type = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.ff_linear1_type[node_type] = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2_type[node_type] = nn.Linear(dim_h * 2, dim_h)
        self.ff_linear1_edge_type = torch.nn.ModuleDict()
        self.ff_linear2_edge_type = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            self.ff_linear1_edge_type[edge_type] = nn.Linear(dim_edge_in, dim_edge_in * 2)
            self.ff_linear2_edge_type[edge_type] = nn.Linear(dim_edge_in * 2, dim_edge_in)

        self.ff_dropout1 = nn.Dropout(0.2)
        self.ff_dropout2 = nn.Dropout(0.2)
        self.reset_parameters()


    def reset_parameters(self):
        zeros(self.attn_bi)


    def forward(self, batch, x_dict):
        has_edge_attr = False
        h_dict = x_dict.copy()

        if sum(batch.num_edge_features.values()) > len(batch.edge_types):
            edge_attr_dict = batch.collect('edge_attr')
            has_edge_attr = True

        h_in_dict = h_dict
        if has_edge_attr:
            edge_attr_in_dict = edge_attr_dict.copy()

        h_out_dict_list = {node_type: [] for node_type in h_dict}

        # Pre-normalization
        if self.layer_norm or self.batch_norm:
            h_dict = {
                node_type: self.norm1_global[node_type](h_dict[node_type])
                for node_type in batch.node_types
            }
            if has_edge_attr:
                edge_attr_dict = {
                    edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
                    for edge_type in batch.edge_types
                }

            h_attn_dict_list = {node_type: [] for node_type in h_dict}

            # We need to modify attention the attention mechanism for heterogeneous graphs
            # Test if Signed attention is beneficial
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            H, D = self.num_heads, self.dim_h // self.num_heads
            homo_data = batch.to_homogeneous()
            edge_index = homo_data.edge_index
            node_type_tensor = homo_data.node_type
            edge_type_tensor = homo_data.edge_type
            q = torch.empty((homo_data.num_nodes, self.dim_h), device=device)
            k = torch.empty((homo_data.num_nodes, self.dim_h), device=device)
            v = torch.empty((homo_data.num_nodes, self.dim_h), device=device)
            edge_attr = torch.empty((homo_data.num_edges, self.dim_h), device=device)
            edge_gate = torch.empty((homo_data.num_edges, self.dim_h), device=device)
            for idx, node_type in enumerate(batch.node_types):
                mask = node_type_tensor == idx
                q[mask] = self.q_lin[node_type](h_dict[node_type])
                k[mask] = self.k_lin[node_type](h_dict[node_type])
                v[mask] = self.v_lin[node_type](h_dict[node_type])

            if has_edge_attr:
                for idx, edge_type_tuple in enumerate(batch.edge_types):
                    edge_type = '__'.join(edge_type_tuple)
                    mask = edge_type_tensor == idx
                    edge_attr[mask] = self.e_lin[edge_type](edge_attr_dict[edge_type_tuple])
                    edge_gate[mask] = self.g_lin[edge_type](edge_attr_dict[edge_type_tuple])
            src_nodes, dst_nodes = edge_index
            num_edges = edge_index.shape[1]
            L = batch.num_nodes
            S = batch.num_nodes

            if has_edge_attr:
                edge_attr = edge_attr.view(-1, H, D)
                edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)

                edge_gate = edge_gate.view(-1, H, D)
                edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

            q = q.view(-1, H, D)
            k = k.view(-1, H, D)
            v = v.view(-1, H, D)

            # transpose to get dimensions h * sl * d_model
            q = q.transpose(0,1)
            k = k.transpose(0,1)
            v = v.transpose(0,1)

            src_nodes, dst_nodes = homo_data.edge_index
            # Compute query and key for each edge
            edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # num_heads * num_edges * d_k
            edge_k = k[:, src_nodes, :]  # Keys for source nodes
            edge_v = v[:, src_nodes, :]

            if hasattr(self, 'edge_weights'):
                edge_weight = self.edge_weights[edge_type_tensor]  # (num_edges, num_heads, d_k, d_k)

                edge_weight = edge_weight.transpose(0, 1)  # Transpose for batch matrix multiplication: (num_heads, num_edges, d_k, d_k)
                edge_k = edge_k.unsqueeze(-1) # Add dimension for matrix multiplication (num_heads, num_edges, d_k, 1)

                edge_k = torch.matmul(edge_weight, edge_k)  # (num_heads, num_edges, d_k, 1)
                edge_k = edge_k.squeeze(-1)  # Remove the extra dimension (num_heads, num_edges, d_k)

            # Compute attention scores
            edge_scores = edge_q * edge_k
            if has_edge_attr:
                edge_scores = edge_scores + edge_attr
                edge_v = edge_v * F.sigmoid(edge_gate)
                edge_attr = edge_scores

            edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # num_heads * num_edges
            edge_scores = torch.clamp(edge_scores, min=-5, max=5)

            expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head

            # Step 2: Calculate max for each destination node per head using scatter_max
            max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
            max_scores = max_scores.gather(1, expanded_dst_nodes)

            # Step 3: Exponentiate scores and sum
            exp_scores = torch.exp(edge_scores - max_scores)
            sum_exp_scores = torch.zeros((H, L), device=device)
            sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
            # sum_exp_scores.clamp_(min=1e-9)

            # Step 4: Apply softmax
            edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
            edge_scores = edge_scores.unsqueeze(-1)
            edge_scores = self.dropout_attn(edge_scores)

            out = torch.zeros((H, L, D), device=device)
            out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

            out = out.transpose(0,1).contiguous().view(-1, H * D)

            for idx, node_type in enumerate(batch.node_types):
                mask = node_type_tensor == idx
                out_type = self.o_lin[node_type](out[mask, :])
                h_attn_dict_list[node_type].append(out_type.squeeze())
            if has_edge_attr:
                edge_attr = edge_attr.transpose(0,1).contiguous().view(-1, H * D)
                for idx, edge_type_tuple in enumerate(batch.edge_types):
                    edge_type = '__'.join(edge_type_tuple)
                    mask = edge_type_tensor == idx
                    out_type = self.oe_lin[edge_type](edge_attr[mask, :])
                    edge_attr_dict[edge_type_tuple] = out_type

            h_attn_dict = {}
            for node_type in h_attn_dict_list:
                h_attn_dict[node_type] = torch.sum(torch.stack(h_attn_dict_list[node_type], dim=0), dim=0)
                h_attn_dict[node_type] = self.dropout_global(h_attn_dict[node_type])

            h_attn_dict = {
                node_type: h_attn_dict[node_type] + h_in_dict[node_type]
                for node_type in batch.node_types
            }

            if has_edge_attr:
                edge_attr_dict = {
                    edge_type: edge_attr_dict[edge_type] + edge_attr_in_dict[edge_type]
                    for edge_type in batch.edge_types
                }


            # Concat output
            h_out_dict_list = {
                node_type: h_out_dict_list[node_type] + [h_attn_dict[node_type]] for node_type in batch.node_types
            }

        # Combine global information
        h_dict = {
            node_type: sum(h_out_dict_list[node_type]) for node_type in batch.node_types
        }

        # Pre-normalization
        if self.layer_norm or self.batch_norm:
            h_dict = {
                node_type: self.norm2_ffn[node_type](h_dict[node_type])
                for node_type in batch.node_types
            }

        h_dict = {
            node_type: h_dict[node_type] + self._ff_block_type(h_dict[node_type], node_type)
            for node_type in batch.node_types
        }
        if has_edge_attr:
            edge_attr_dict = {
                edge_type: edge_attr_dict[edge_type] + self._ff_block_edge_type(edge_attr_dict[edge_type], edge_type)
                for edge_type in batch.edge_types
            }

        if has_edge_attr:
            for edge_type in batch.edge_types:
                batch[edge_type].edge_attr = edge_attr_dict[edge_type]

        return batch, h_dict

    def _ff_block_type(self, x, node_type):
        """
        Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_type[node_type](x)))
        return self.ff_dropout2(self.ff_linear2_type[node_type](x))

    def _ff_block_edge_type(self, x, edge_type):
        """
        Feed Forward block.
        """
        edge_type = "__".join(edge_type)
        x = self.ff_dropout1(self.activation(self.ff_linear1_edge_type[edge_type](x)))
        return self.ff_dropout2(self.ff_linear2_edge_type[edge_type](x))

class HeteroGNNNodeHead(nn.Module):
    """
    Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    """
    def __init__(self, dim_in, dim_out):
        super(HeteroGNNNodeHead, self).__init__()

        self.layer_post_mp = MLP(dim_in, dim_out, num_layers=2, bias=True)


    def forward(self, batch):
        return self.layer_post_mp(batch)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None, num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        layers.append(GeneralMultiLayer(num_layers - 1, dim_in, dim_inner, dim_inner, final_act=True))
        layers.append(Linear(dim_inner, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch

class GeneralMultiLayer(nn.Module):
    """
    General wrapper for stack of layers
    """
    def __init__(self, num_layers, dim_in, dim_out, dim_inner=None, final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch

class GeneralLayer(nn.Module):
    """General wrapper for layers"""
    def __init__(self, dim_in, dim_out, has_act=True, has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = CustomLinear(dim_in, dim_out, bias=True, **kwargs)
        layer_wrapper = []
        gnn_dropout = 0.2 # TODO: COULD CHANGE?

        if gnn_dropout > 0:
            layer_wrapper.append(nn.Dropout(p=gnn_dropout))
        if has_act:
            layer_wrapper.append(nn.ReLU())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class CustomLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(CustomLinear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch