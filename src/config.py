from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
import logging
import sys
import yaml
from datetime import datetime 
import torch


def logger_setup(log_dir: str):
    """Setup logging to file and stdout"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "logs.log")),  # Log to run-specific log file
            logging.StreamHandler(sys.stdout)                        # Log also to stdout
        ]
    )


@dataclass
class CustomConfig:
    """Unified configuration class for RelBench project.
    
    Attributes:
        # Model Configuration
        channels (int): Number of hidden channels in the model
        num_layers (int): Number of GNN layers
        out_channels (int): Number of output channels
        aggr (str): Aggregation method for GNN
        norm (str): Normalization type
        
        # Training Configuration
        learning_rate (float): Learning rate for optimizer
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        device (torch.device): Device to run training on
        
        # Model Selection
        higher_is_better (bool): Whether higher metric value is better
        tune_metric (str): Metric to use for model selection
        early_stopping (bool): Whether to use early stopping
        patience (int): Number of epochs to wait before early stopping
        
        # Data and File Paths
        data_dir (str): Directory for data storage
        output_dir (str): Root directory for all run outputs
        
        # Evaluation parameters
        evaluation_freq (int): Frequency of evaluation (every N epochs)
    """
    # Data and File Paths
    data_name: str = 'f1'
    task_name: str = 'driver-position'
    data_dir: str = './data'
    output_dir: str = './runs'  # Root directory for all run outputs

    # Model Configuration
    channels: int = 32
    num_layers: int = 2
    num_layers_pre_gt: int = 0
    out_channels: int = 1
    num_neighbors: List[int] = field(default_factory=lambda: [32, 32])
    temporal_strategy: str = "uniform"
    aggr: str = "sum"
    norm: str = "batch_norm"
    reverse_mp: bool = True
    port_numbering: bool = True
    dropouts: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    head: str = 'HeteroGNNNodeHead'
    edge_features: bool = False
    
    # Training Configuration
    learning_rate: float = 0.005
    epochs: int = 10
    optimiser: str = "Adam"
    max_steps_per_epoch: int = 2000
    batch_size: int = 128
    num_workers: int = 2
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ego_ids: bool = True
    save_artifacts: bool = True
    
    # Model Selection
    higher_is_better: bool = True
    tune_metric: str = "f1"
    early_stopping: bool = False
    patience: int = 5
    
    # Evaluation parameters
    evaluation_freq: int = 4
    
    # Run-specific paths, set during initialization
    run_dir: str = None
    log_dir: str = None
    checkpoint_dir: str = None
    config_path: str = None
    
    def __post_init__(self):
        # Create a unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{self.data_name}_{self.task_name}_{timestamp}")
        
        # Set up paths for logs, checkpoints, and config
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.config_path = os.path.join(self.run_dir, "config.yaml")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up logging
        logger_setup(self.log_dir)
        
        # Save config to YAML file
        self.save_config()
    
    def save_config(self):
        """Save configuration to YAML file"""
        # Create a dictionary of serializable config values
        config_dict = {k: v for k, v in vars(self).items() 
                      if not k.startswith('_') and not callable(v) and k != 'device'}
        
        # Handle non-serializable types
        config_dict['device'] = str(self.device)
        
        # Write to YAML file
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logging.info(f"Configuration saved to {self.config_path}")
    
    def print_config(self):
        """Prints and logs all configuration settings."""
        config_str = "\n=== RelBench Configuration ===\n"
        
        # Model Configuration
        config_str += "\nModel Configuration:\n"
        config_str += f"Channels: {self.channels}\n"
        config_str += f"Number of Layers: {self.num_layers}\n"
        config_str += f"Number of Layers Pre-GT: {self.num_layers_pre_gt}\n"
        config_str += f"Output Channels: {self.out_channels}\n"
        config_str += f"Aggregation: {self.aggr}\n"
        config_str += f"Normalization: {self.norm}\n"
        config_str += f"Reverse Message Passing: {self.reverse_mp}\n"
        config_str += f"Port Numbering: {self.port_numbering}\n"
        config_str += f"Dropouts: {self.dropouts}\n"
        config_str += f"Head: {self.head}\n"
        
        # Training Configuration
        config_str += "\nTraining Configuration:\n"
        config_str += f"Learning Rate: {self.learning_rate}\n"
        config_str += f"Epochs: {self.epochs}\n"
        config_str += f"Optimiser: {self.optimiser}\n"
        config_str += f"Batch Size: {self.batch_size}\n"
        config_str += f"Number of Workers: {self.num_workers}\n"
        config_str += f"Device: {self.device}\n"
        config_str += f"Ego IDs: {self.ego_ids}\n"
        
        # Model Selection
        config_str += "\nModel Selection:\n"
        config_str += f"Higher is Better: {self.higher_is_better}\n"
        config_str += f"Tune Metric: {self.tune_metric}\n"
        config_str += f"Early Stopping: {self.early_stopping}\n"
        config_str += f"Patience: {self.patience}\n"
        
        # Paths
        config_str += "\nPaths:\n"
        config_str += f"Data Directory: {self.data_dir}\n"
        config_str += f"Run Directory: {self.run_dir}\n"
        config_str += f"Log Directory: {self.log_dir}\n"
        config_str += f"Checkpoint Directory: {self.checkpoint_dir}\n"
        
        # Log configuration
        logging.info("Configuration:\n%s", config_str)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Returns model configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Model configuration parameters
        """
        return {
            "channels": self.channels,
            "num_layers": self.num_layers,
            "out_channels": self.out_channels,
            "aggr": self.aggr,
            "norm": self.norm
        }
    
    def get_training_kwargs(self) -> Dict[str, Any]:
        """Returns training configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Training configuration parameters
        """
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "higher_is_better": self.higher_is_better,
            "tune_metric": self.tune_metric,
            "early_stopping": self.early_stopping,
            "patience": self.patience
        }