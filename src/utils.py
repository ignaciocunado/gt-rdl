from typing import Dict, List, Tuple, Set
import numpy as np
import logging
import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from collections import defaultdict

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .. import algos


def logger_setup(log_dir: str = "logs"):
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(os.path.join(log_dir, f"run_{timestamp}.log"))),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )



def analyze_multi_edges(data: HeteroData) -> List[EdgeType]:
    """Analyzes a heterogeneous graph for multi-edges.
    
    For each edge type, checks if there are multiple edges between the same node pairs
    and prints statistics about edge counts.
    
    Args:
        data (HeteroData): The heterogeneous graph to analyze
        
    Returns:
        List[EdgeType]: List of edge types that contain multi-edges
    """
    multi_edge_types = []
    
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        
        # Create unique node pairs
        node_pairs = tuple(map(tuple, edge_index.t().tolist()))
        unique_pairs = set(node_pairs)
        
        total_edges = len(node_pairs)
        unique_edges = len(unique_pairs)
        
        if total_edges > unique_edges:
            multi_edge_types.append(edge_type)
            
            # Count frequency of each node pair
            pair_counts = defaultdict(int)
            for pair in node_pairs:
                pair_counts[pair] += 1
            
            # Find maximum number of edges between any node pair
            max_edges = max(pair_counts.values())
            multi_edge_pairs = sum(1 for count in pair_counts.values() if count > 1)
            
            print(f"\nEdge Type: {edge_type}")
            print(f"Total edges: {total_edges}")
            print(f"Unique node pairs: {unique_edges}")
            print(f"Node pairs with multiple edges: {multi_edge_pairs}")
            print(f"Maximum edges between any node pair: {max_edges}")
    
    if multi_edge_types:
        print("\nEdge types with multi-edges:")
        for edge_type in multi_edge_types:
            print(f"- {edge_type}")
    else:
        print("\nNo multi-edges found in the graph.")
    
    return multi_edge_types

def graphormer_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
                module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                    module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
