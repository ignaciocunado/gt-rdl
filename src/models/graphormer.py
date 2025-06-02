from typing import Dict, Any, Callable, Optional, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, GELU, Linear
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from src.heads import HeteroGNNNodeRegressionHead
from src.models.base_model import BaseModel
from src.models.fraudgt import HeteroGNNNodeHead
from src.utils import quant_noise, graphormer_softmax, concat_node_features_to_edge


class Graphormer(BaseModel):
    """An implementation of the Graphormer model to work with RelBench datasets

        Args:
            data: the graph object
            col_stats_dict: column statistics
            channels: number of hidden dimensions
            out_channels: output channels (1)
            dropouts: dropout values
            num_layers: number of Graphormer layers
            head: prediction head to use
            edge_features: whether to use edge features
            torch_frame_model_kwargs: other args for torch frame
    """
    def __init__(
            self,
            data: HeteroData,
            col_stats_dict: Dict,
            channels: int,
            out_channels: int,
            dropouts: list,
            num_layers: int = 2,
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

        self.edge_features = edge_features
        multi_hop_max_dist = 5
        embedding_dim = channels
        num_attention_heads = 8
        ffn_embedding_dim = 80
        encoder_normalize_before = True
        apply_graphormer_init = True
        share_encoder_input_output_embed = False
        pre_layernorm = False
        dropout = dropouts[0]
        attention_dropout = dropouts[1]
        activation_dropout = dropouts[2]

        max_d = 700 # TODO: What value do we want? Can cause errors
        num_spatial = (max_d + 1) + 1

        num_edge_dis = 128
        edge_type = 'edge_type' # or multi-hop

        max_in_deg = 0
        max_out_deg = 0
        for ntype in data.node_types:
            max_in_deg = max(max_in_deg, data[ntype].in_degree.max().item())
            max_out_deg = max(max_out_deg, data[ntype].out_degree.max().item())

        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            node_types=data.node_types,
            num_in_degree=max_in_deg + 1,
            num_out_degree=max_out_deg + 1,
            num_edges=len(data.edge_types),
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis, # Only used when edge_type is multi-hop
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            # >
            num_encoder_layers=num_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_normalize_before=encoder_normalize_before,
            pre_layernorm=pre_layernorm,
            apply_graphormer_init=apply_graphormer_init,
        )

        if apply_graphormer_init:
            self.apply(init_graphormer_params)

        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        self.load_softmax = True

        self.lm_head_transform_weight = nn.Linear(embedding_dim, embedding_dim)
        self.activation_fn = nn.GELU()
        self.layer_norm = LayerNorm(embedding_dim)

        self.lm_output_learned_bias = None

        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                if head == 'HeteroGNNNodeHead':
                    self.embed_out = HeteroGNNNodeHead(embedding_dim, out_channels)
                elif head == 'HeteroGNNNodeRegressionHead':
                    self.embed_out = HeteroGNNNodeRegressionHead(embedding_dim, out_channels)
                elif head == 'Linear':
                    self.embed_out = Linear(embedding_dim, out_channels)
                else:
                    raise ValueError(f'Attention head {head} not supported.')
                # self.embed_out = nn.Linear(embedding_dim, out_channels, bias=True)
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        """
        Resets parameters
        """
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def post_forward(self, x_dict: Dict[str, Tensor], batch: HeteroData, entity_table: NodeType, seed_time: Tensor):
        """Overrides the post_forward method of the base class.
        Args:
            x_dict: encoded features
            batch: current batch
            entity_table: type of node
            seed_time: seed time

        Returns:
            Prediction
        """
        if self.edge_features:
            batch, x_dict = concat_node_features_to_edge(batch, x_dict)

        batch = self.concat_x_dict(batch, x_dict)

        inner_states, graph_rep = self.graph_encoder(batch, perturb=None)

        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)

        x_nodes = x[:, 1:, :]
        x_dict = self.rebuild_x_dict(x_nodes, x_dict)

        return self.embed_out(x_dict[entity_table][: seed_time.size(0)])

    def concat_x_dict(self, batch: HeteroData, x_dict: Dict[str, Tensor]) -> HeteroData:
        """Concatenates the per-type encoded node features into a single tensor
        Args:
            batch: the current batch
            x_dict: encoded node features per node type

        Returns:
            The modified batch
        """
        device = x_dict[next(iter(x_dict))].device
        batch.x = torch.cat([x_dict[node_type] for node_type in batch.node_types], dim=0).to(device)
        batch.x = batch.x.unsqueeze(0)
        return batch

    def rebuild_x_dict(self, x, x_dict):
        """Rebuilds x_dict from the single tensor, does the opposite of concat_x_dict.
        Args:
            x: tensor storing the result of the forward pass
            x_dict: old encoded node features

        Returns:
            Updated x_dict
        """
        x = x[0]
        index = 0
        for type in x_dict:
            x_dict[type] = x[index : index + x_dict[type].size(0)]
            index += x_dict[type].size(0)
        return x_dict


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model. https://github.com/microsoft/Graphormer
    """
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    """Graphormer logic from https://github.com/microsoft/Graphormer with small tweaks"""
    def __init__(
            self,
            node_types: List[str],
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            apply_graphormer_init: bool = False,
            embed_scale: float = None,
            n_trans_layers_to_freeze: int = 0,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
    ) -> None:

        super().__init__()

        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            node_types=node_types,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = quant_noise(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),q_noise,qn_block_size)
        else:
            self.quant_noise = None

        self.emb_layer_norm = LayerNorm(self.embedding_dim) if encoder_normalize_before else None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.layers = LayerDropModuleList(p=self.layerdrop) if self.layerdrop > 0.0 else nn.ModuleList()

        self.layers.extend(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def forward(
            self,
            batch,
            perturb=None,
            last_state_only: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        data_x = batch["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batch)

        if perturb is not None:
            # ic(torch.mean(torch.abs(perturb)))
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.graph_attn_bias(batch)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep

class GraphNodeFeature(nn.Module):
    """
    Compute some node features for each node in the graph. https://github.com/microsoft/Graphormer
    """

    def __init__(self, node_types, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super(GraphNodeFeature, self).__init__()
        self.node_types = node_types
        # self.type_embed = nn.Embedding(len(node_types), channels) # TODO: Optional, learn node type bias?
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token

        node_feature = x + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature

class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head. https://github.com/microsoft/Graphormer
    """

    def __init__(
            self,
            num_heads,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batch):
        # attn_bias, spatial_pos, edge_input, attn_edge_type = (
        #     batch["attn_bias"],
        #     batch["spatial_pos"],
        #     batch["edge_input"],
        #     batch["attn_edge_type"],
        # )

        attn_bias, edge_input, attn_edge_type, x = (
            batch["attn_bias"],
            batch["edge_input"],
            batch["attn_edge_type"],
            batch["x"],
        )

        # spatial_pos = batch["spatial_pos]

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop": # TODO: Fix if needed
            pass
            # spatial_pos_ = spatial_pos.clone()
            # spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # # set 1 to 1, x > 1 to x - 1
            # spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            # if self.multi_hop_max_dist > 0:
            #     spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            #     edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # # [n_graph, n_node, n_node, max_dist, n_head]
            # edge_input = self.edge_encoder(edge_input).mean(-2)
            # max_dist = edge_input.size(-2)
            # edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
            #     max_dist, -1, self.num_heads
            # )
            # edge_input_flat = torch.bmm(
            #     edge_input_flat,
            #     self.edge_dis_encoder.weight.reshape(
            #         -1, self.num_heads, self.num_heads
            #     )[:max_dist, :, :],
            # )
            # edge_input = edge_input_flat.reshape(
            #     max_dist, n_graph, n_node, n_node, self.num_heads
            # ).permute(1, 2, 3, 0, 4)
            # edge_input = (
            #         edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            # ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias

class GraphormerGraphEncoderLayer(nn.Module):
    """Encoder layer block. https://github.com/microsoft/Graphormer"""
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            init_fn: Callable = None,
            pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = Dropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = GELU()

        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        self.fc1 = quant_noise(nn.Linear(self.embedding_dim, ffn_embedding_dim), q_noise, qn_block_size)
        self.fc2 = quant_noise(nn.Linear(ffn_embedding_dim, self.embedding_dim), q_noise, qn_block_size)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_bias: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn

class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`. https://github.com/microsoft/Graphormer

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class MultiheadAttention(nn.Module):
    """
    Multi-headed attention from https://github.com/microsoft/Graphormer.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            self_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = Dropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        self.reset_parameters()

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = graphormer_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0) # average attention weights over heads

        return attn, attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

class Dropout(nn.Module):
    """Custom dropout implementation from https://github.com/microsoft/Graphormer"""
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
            self,
            name: str,
            retain_dropout: bool = False,
            retain_dropout_modules: Optional[List[str]] = None,
            **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is None or self.module_name in retain_dropout_modules:
                self.apply_during_inference = True
