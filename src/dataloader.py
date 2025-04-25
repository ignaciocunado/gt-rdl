import os
import torch
from typing import Dict, List, Optional, Any, Tuple
from torch_geometric.loader import NeighborLoader
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer
from torch import Tensor
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.datasets import get_dataset
from relbench.tasks import get_task


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
        return graph, col_stats_dict
    
    def _create_loaders(self) -> Dict[str, NeighborLoader]:
        """Create train/validation/test data loaders.
        
        Returns:
            Dict[str, NeighborLoader]: Dictionary containing data loaders for each split
        """
        loader_dict = {}
        
        for split, table in self.tables.items():
            table_input = get_node_train_table_input(
                table=table,
                task=self.task,
            )
            self.entity_table = table_input.nodes[0]
            loader_dict[split] = NeighborLoader(
                self.graph,
                num_neighbors=self.num_neighbors,
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
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