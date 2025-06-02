import os


import numpy as np
import torch
from typing import Dict, List, Optional, Any

from torch_frame.data import StatType
from torch_frame._stype import stype
from torch_geometric.loader import NeighborLoader
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer
from torch import Tensor
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from torch_geometric.transforms import Compose

from src.utils import preprocess_batch, add_centrality_encoding_info


class GloveTextEmbedding:
    """Text embedding class using GloVe embeddings via sentence-transformers.
    
    Args:
        device (torch.device, optional): Device to run the embedding model on.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        """Convert input sentences to embeddings.
        
        Args:
            sentences (List[str]): List of input sentences
            
        Returns:
            Tensor: Tensor containing sentence embeddings
        """
        return torch.from_numpy(self.model.encode(sentences))


class RelBenchDataLoader:
    """Data loader class for RelBench datasets.
    
    This class handles:
    - Loading and processing RelBench datasets
    - Creating heterogeneous graphs
    - Setting up train/val/test data loaders
    - Managing text embeddings
    
    Args:
        data_name (str): Name of the RelBench dataset (e.g., 'f1')
        task_name (str): Name of the task (e.g., 'driver-position')
        device (torch.device): Device to run computations on
        root_dir (str, optional): Directory for cached data. Defaults to "./data"
        batch_size (int, optional): Batch size for dataloaders. Defaults to 128
        num_neighbors (List[int], optional): Number of neighbors to sample. Defaults to [32, 32]
        num_workers (int, optional): Number of worker processes. Defaults to 0
        temporal_strategy (str, optional): Strategy for temporal neighbor sampling. Defaults to "uniform"
        reverse_mp (bool, optional): Whether to use reverse message passing. Defaults to False
        add_ports (bool, optional): Whether to use port numbering. Defaults to False
        ego_ids (bool, optional): Whether to add IDs to the centre nodes of the batch. Defaults to False
        preprocess_graph (bool, optional): Whether to apply Graphormer preprocessing steps the graph. Defaults to False
    """
    def __init__(
        self,
        data_name: str,
        task_name: str,
        device: torch.device,
        root_dir: str = "./data",
        batch_size: int = 128,
        num_neighbors: List[int] = [32, 32],
        num_workers: int = 0,
        temporal_strategy: str = "uniform",
        reverse_mp: bool = False,
        add_ports: bool = False,
        ego_ids: bool = False,
        preprocess_graph: bool = False,
    ):
        self.data_name = data_name
        self.task_name = task_name
        self.device = device
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
        self.entity_table = None
        self.temporal_strategy = temporal_strategy
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        self.ego_ids = ego_ids
        self.preprocess_graph = preprocess_graph

        # Load dataset and task
        self.dataset = self._load_dataset()
        self.task = self._load_task()
        
        # Load data tables
        self.tables = self._load_tables()
        
        # Initialize data structures
        self.db = self.dataset.get_db()
        self.col_to_stype_dict = get_stype_proposal(self.db)
        
        # Setup text embedder
        self.text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=256
        )
        
        # Create graph data
        self.graph, self.col_stats_dict = self._create_graph()
        
        # Initialize loaders
        self.loader_dict = self._create_loaders()
        
    def _load_dataset(self):
        """Load the RelBench dataset.
        
        Returns:
            Dataset: The loaded RelBench dataset
        """
        return get_dataset(f"rel-{self.data_name}", download=True)
    
    def _load_task(self):
        """Load the RelBench task.
        
        Returns:
            Task: The loaded RelBench task
        """
        return get_task(f"rel-{self.data_name}", f"{self.task_name}", download=True)
    
    def _load_tables(self) -> Dict[str, Any]:
        """Load train/val/test tables.
        
        Returns:
            Dict[str, Any]: Dictionary containing data tables for each split
        """
        return {
            "train": self.task.get_table("train"),
            "val": self.task.get_table("val"),
            "test": self.task.get_table("test")
        }
    
    @property
    def train_table(self):
        """Get training data table."""
        return self.tables["train"]
    
    @property
    def val_table(self):
        """Get validation data table."""
        return self.tables["val"]
    
    @property
    def test_table(self):
        """Get test data table."""
        return self.tables["test"]
    
    def _create_graph(self):
        """Create or load the cached heterogeneous graph from the database.
        
        Returns:
            tuple: (graph, column statistics dictionary)
        """
        graph_path = os.path.join(self.root_dir, f"rel-{self.data_name}", f"rel-{self.data_name}-graph.pth")
        col_stats_dict_path = os.path.join(self.root_dir, f"rel-{self.data_name}", f"rel-{self.data_name}-col_stats_dict.pth")
        
        if os.path.exists(graph_path) and os.path.exists(col_stats_dict_path):
            print(f"Loading graph from {graph_path}")
            graph = torch.load(graph_path)
            print(f"Loading column stats dictionary from {col_stats_dict_path}")
            col_stats_dict = torch.load(col_stats_dict_path)
        else:
            graph, col_stats_dict = make_pkey_fkey_graph(
                self.db,
                col_to_stype_dict=self.col_to_stype_dict,
                text_embedder_cfg=self.text_embedder_cfg,
                cache_dir=os.path.join(self.root_dir, f"rel-{self.data_name}", f"rel-{self.data_name}_materialized_cache"),
            )

            if not self.reverse_mp: # Reverse MP is implemented by default in RelBench but we can remove it
                for source, rel, destination in graph.edge_types:
                    if 'rev' in rel:
                        del graph[source, rel, destination]

            if self.add_ports:
                graph = self._compute_and_add_ports(graph)

            if self.ego_ids:
                for ntype in graph.node_types:
                    N = graph[ntype].num_nodes
                    ego_id = torch.arange(N, dtype=torch.long)

                    orig = graph[ntype].tf.feat_dict.get(
                        stype.numerical,
                        torch.empty((N, 0), dtype=torch.float)
                    )
                    graph[ntype].tf.feat_dict[stype.numerical] = torch.cat(
                        [orig, ego_id.unsqueeze(1).float()],
                        dim=1
                    )
                    names = graph[ntype].tf.col_names_dict.setdefault(stype.numerical, [])
                    names.append('ego_id')

                    vals =ego_id.numpy()

                    # compute the same stats PyG uses:
                    mean = float(vals.mean())
                    std  = float(vals.std(ddof=0))       # population std
                    quantiles = list(np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist())

                    # insert under the same ntype → column → { StatType: value } hierarchy
                    col_stats_dict.setdefault(ntype, {})['ego_id'] = {
                        StatType.MEAN:      mean,
                        StatType.STD:       std,
                        StatType.QUANTILES: quantiles,
                    }

                # for ntype in graph.node_types:
                #     num_nodes = graph[ntype].num_nodes
                #
                #     ids = torch.zeros((num_nodes, 1),
                #                       dtype=torch.float,
                #                       device=self.device)
                #
                #     ego_idx = self.ego_ids[ntype] if isinstance(self.ego_ids, dict) else self.ego_ids
                #
                #     if isinstance(ego_idx, torch.BoolTensor) or type(ego_idx) == torch.bool:
                #         ids[ego_idx.view(-1), 0] = 1.
                #     else:
                #         ids[ego_idx, 0] = 1.
                #
                #     if hasattr(graph[ntype], 'x') and graph[ntype].x is not None:
                #         graph[ntype].x = torch.cat([graph[ntype].x, ids], dim=-1)
                #     else:
                #         graph[ntype].x = ids

                # x = data['node'].x
                # device = x.device
                # ids = torch.zeros((x.shape[0], 1), device=device)
                # nodes = torch.arange(data['node'].batch_size, device=device)
                # ids[nodes] = 1
                # data['node'].x = torch.cat([x, ids], dim=1)
                #
                # return data

        return graph, col_stats_dict
    
    def _create_loaders(self) -> Dict[str, NeighborLoader]:
        """Create train/validation/test data loaders.
        
        Returns:
            Dict[str, NeighborLoader]: Dictionary containing data loaders for each split
        """
        loader_dict = {}
        if self.preprocess_graph:
            self.graph = add_centrality_encoding_info(self.graph, self.device)

        for split, table in self.tables.items():
            table_input = get_node_train_table_input(
                table=table,
                task=self.task,
            )
            base_t = table_input.transform
            tf_list = []
            if base_t is not None:
                tf_list.append(base_t)
            if self.preprocess_graph:
                tf_list.append(preprocess_batch)

            transform = Compose(tf_list) if tf_list else None

            self.entity_table = table_input.nodes[0]
            loader_dict[split] = NeighborLoader(
                self.graph,
                num_neighbors=self.num_neighbors,
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=transform,
                batch_size=self.batch_size,
                temporal_strategy=self.temporal_strategy,
                shuffle=split == "train",
                num_workers=self.num_workers,
                persistent_workers=False,
            )
        
        return loader_dict
    
    def get_loader(self, split: str) -> NeighborLoader:
        """Get data loader for a specific split.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            NeighborLoader: The requested data loader
        """
        return self.loader_dict[split]

    def _compute_and_add_ports(self, graph):
        """Implements port numbering for the graph
        Args:
            graph: the graph to add port numbering to

        Returns: the modified graph

        """
        for edge_type in graph.edge_types:
            adj_list_in, adj_list_out = to_adj_simple(graph, edge_type)
            ei = graph[edge_type].edge_index

            in_ports = ports(ei, adj_list_in)
            out_ports = ports(ei.flipud(), adj_list_out)
            graph = self.__add_ports_to_edge_features(graph, edge_type, (in_ports, out_ports))

        return graph

    def __add_ports_to_edge_features(self, graph, edge_type, ports):
        """Utils function for port numbering
        Args:
            graph: the graph
            edge_type: the type of edge to add ports to
            ports: the ports to add
        """
        if hasattr(graph[edge_type], 'edge_attr') and graph[edge_type].edge_attr is not None:
            base_feat = graph[edge_type].edge_attr
        else:
            base_feat = torch.zeros((graph[edge_type].num_edges, 0), dtype=torch.long, device=self.device)

        in_ports, out_ports = ports

        if not self.reverse_mp:
            new_feat = torch.cat([base_feat, in_ports, out_ports], dim=1).to(self.device)
        else:
            source, rel, destination = edge_type
            if 'rev' in rel:
                new_feat = torch.cat([base_feat, out_ports], dim=1)
            else:
                new_feat = torch.cat([base_feat, in_ports],  dim=1)

        graph[edge_type].edge_attr = new_feat

        return graph

def to_adj_simple(graph, edge_type):
    """
    Build adjacency lists
    Returns two dicts:
      adj_in[v]  = [u1, u2, ...] for edges (u->v)
      adj_out[u] = [v1, v2, ...] for edges (u->v)
    """
    src_type, _, dst_type = edge_type
    num_nodes_src = graph[src_type].num_nodes
    num_nodes_dst = graph[dst_type].num_nodes
    ei = graph[edge_type].edge_index            # [2, E_rel]

    adj_in  = {i: [] for i in range(num_nodes_dst)}
    adj_out = {i: [] for i in range(num_nodes_src)}

    for u, v in ei.t().tolist():
        adj_out[u].append(v)
        adj_in [v].append(u)

    return adj_in, adj_out


def ports(edge_index, adj_list):
    """
    For each edge (u->v), assign a port ID based on sorting all unique
    neighbours u of v in ascending order.
    Args:
        edge_index: edge index
        adj_list: adjacency list

    Returns:

    """
    E = edge_index.size(1)
    ports = torch.zeros((E,1), dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ports_dict = {}

    for v, nbs in adj_list.items():
        unique_nbs = sorted(set(nbs))
        for rank, u in enumerate(unique_nbs):
            ports_dict[(u, v)] = rank

    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        ports[idx] = ports_dict[(u, v)]

    return ports