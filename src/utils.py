from typing import Dict, List, Tuple, Set
import numpy as np
import logging
import os
import sys
from datetime import datetime 
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from collections import defaultdict


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