from matplotlib.cbook import is_math_text
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import (HeteroData, Batch)

from fraudGT.graphgym.models import head  # noqa, register module
from fraudGT.graphgym import register as register
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.gnn import GNNPreMP
from fraudGT.graphgym.register import register_network
from torch_geometric.nn import (BatchNorm, GINEConv, GATConv, PNAConv, RGCNConv)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.utils import degree
from fraudGT.graphgym.models.layer import BatchNorm1dNode
from fraudGT.layer.gatedgcn_layer import GatedGCNLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, data):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = isinstance(data, HeteroData)
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, data)
            if cfg.dataset.node_encoder_bn:
                # self.node_encoder_bn = BatchNorm1dNode(
                #     new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                #                      has_bias=False, cfg=cfg))
                self.node_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension of the node features
            if self.is_hetero:
                self.dim_in = {node_type: cfg.gnn.dim_inner for node_type in dim_in}
            else:
                self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, data)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

@register_network('HomoGNNEdgeModel')
class HomoGNNEdgeModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata = dataset[0].metadata()
        data = dataset[0]
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.activation = register.act_dict[cfg.gnn.act]
        self.layer_norm = cfg.gnn.layer_norm
        self.batch_norm = cfg.gnn.batch_norm
        task_entity = cfg.dataset.task_entity
        self.edge_updates = cfg.gnn.edge_updates

        self.encoder = FeatureEncoder(dim_in, dataset)
        dim_in = self.encoder.dim_in
        dim_h_total = cfg.gnn.dim_inner

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                cfg.gnn.dim_inner, cfg.gnn.dim_inner,
                has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
            )

        self.model = None
        # Following the PyG implementation whenver possible
        norm = None
        if self.layer_norm or self.batch_norm:
            if self.layer_norm:
                norm = nn.LayerNorm(cfg.gnn.dim_inner)
            elif self.batch_norm:
                norm = nn.BatchNorm1d(cfg.gnn.dim_inner)
        # Official PyG GAT implementation doesn't reproduce correct results
        # elif cfg.gnn.layer_type == 'GAT':
        #     self.model = GAT(
        #         in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
        #         out_channels=cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
        #         jk='cat' if cfg.gnn.jumping_knowledge else None
        #     )
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        if self.layer_norm or self.batch_norm:
            self.norms = nn.ModuleList()
        for i in range(cfg.gnn.layers_mp):
            norm_dim = cfg.gnn.dim_inner
            if cfg.gnn.layer_type == 'GatedGCN':
                    conv = GatedGCNLayer(
                        in_dim=cfg.gnn.dim_inner, out_dim=cfg.gnn.dim_inner,
                        dropout=cfg.gnn.dropout, residual=True, act=cfg.gnn.act
                    )

            elif cfg.gnn.layer_type == 'GINE':
                mlp = nn.Sequential(
                        nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                        nn.ReLU(), 
                        nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                    )
                conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner)

            elif cfg.gnn.layer_type == 'GATE':
                if i == 0:
                    conv = GATConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads, 
                                    concat=True, add_self_loops=True, dropout=cfg.gnn.attn_dropout,
                                    edge_dim=cfg.gnn.dim_inner)
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                elif i < cfg.gnn.layers_mp - 1:
                    conv = GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads,
                                    concat=True, add_self_loops=True, dropout=cfg.gnn.attn_dropout,
                                    edge_dim=cfg.gnn.dim_inner)
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                else:
                    conv = GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads,
                                    concat=False, add_self_loops=True, dropout=cfg.gnn.attn_dropout,
                                    edge_dim=cfg.gnn.dim_inner)
                    norm_dim = cfg.gnn.dim_inner
                
            elif cfg.gnn.layer_type == 'PNA':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                if not isinstance(data, HeteroData):
                    d = degree(data.edge_index[1], dtype=torch.long)
                else:
                    d = degree(data.to_homogeneous().edge_index[1], dtype=torch.long)
                    # index = torch.cat((data['node', 'to', 'node'].edge_index[1], data['node', 'rev_to', 'node'].edge_index[1]), 0)
                    # d = degree(index, dtype=torch.long)
                deg = torch.bincount(d, minlength=1)
                conv = PNAConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=cfg.gnn.dim_inner, towers=5, pre_layers=1, post_layers=1,
                        divide_input=False)
                
            elif cfg.gnn.layer_type == 'RGCN':
                conv = RGCNConv(
                    in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, 
                    num_relations=len(self.metadata[1]),
                )
            
            else:
                raise NotImplementedError(f"{cfg.gnn.layer_type} is not implemented!")
            self.convs.append(conv)

            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                    nn.ReLU(),
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                ))

            if self.layer_norm:
                self.norms.append(nn.LayerNorm(norm_dim))
            elif self.batch_norm:
                self.norms.append(nn.BatchNorm1d(norm_dim))

        if cfg.gnn.jumping_knowledge:
            self.jk = JumpingKnowledge('cat', cfg.gnn.dim_inner, cfg.gnn.layers_mp)
            if cfg.gnn.layer_type == 'GAT':
                dim_h_total = cfg.gnn.dim_inner * (cfg.gnn.attn_heads * (cfg.gnn.layers_mp - 1) + 1)
            else:
                dim_h_total = cfg.gnn.layers_mp * cfg.gnn.dim_inner

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h_total, dim_out, dataset)


    def forward(self, batch):
        batch = self.encoder(batch)

        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, edge_index, edge_attr = homo.x, homo.edge_index, homo.edge_attr
            node_type_tensor = homo.node_type
            edge_type_tensor = homo.edge_type
        else:
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.input_drop(x)
        if cfg.gnn.layers_pre_mp > 0:
            x = self.pre_mp(x) 

        src, dst = edge_index
        xs = []
        for i in range(cfg.gnn.layers_mp):
            if cfg.gnn.layer_type == 'RGCN':
                x = self.convs[i](x, edge_index, homo.edge_type)
            elif cfg.gnn.layer_type == 'GatedGCN':
                out = self.convs[i](Batch(batch=batch,
                                          x=x,
                                          edge_index=edge_index,
                                          edge_attr=edge_attr))
                x = out.x
                edge_attr = out.edge_attr
            else:
                x = self.convs[i](x, edge_index, edge_attr)

            if self.layer_norm or self.batch_norm:
                x = self.norms[i](x)
            x = self.drop(self.activation(x))

            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x

        # Write back
        if isinstance(batch, HeteroData):
            for idx, node_type in enumerate(batch.node_types):
                node_mask = node_type_tensor == idx
                batch[node_type].x = x[node_mask]
            for idx, edge_type in enumerate(batch.edge_types):
                edge_mask = edge_type_tensor == idx
                batch[edge_type].edge_attr = edge_attr[edge_mask]
        else:
            batch.x = x
            batch.edge_attr = edge_attr
        return self.post_mp(batch)
