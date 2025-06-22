import argparse
import random

import numpy as np
import torch

import wandb
import os

os.environ['XDG_CACHE_HOME'] = '/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio/data'

from src.config import CustomConfig
from src.dataloader import RelBenchDataLoader
from src.models.hetero_sage import HeteroGraphSage
from src.models.fraudgt import FraudGT
from src.models.graphormer import Graphormer

from src.train import train
from src.utils import analyze_multi_edges

from torch.optim import Adam, AdamW
from torch.nn import L1Loss, BCEWithLogitsLoss
import logging

from relbench.base import TaskType

def set_seed(seed):
    """Sets the seed for all random number generators.
    Args:
        seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use (local vs global)", required=True)
    parser.add_argument("--dataset", type=str, help="The dataset to use", required=True)
    parser.add_argument("--task", type=str, help="The task to solve", required=True)
    parser.add_argument("--save_artifacts", action='store_true', help="Whether to save artifacts")
    parser.add_argument("--num_workers", type=int, default=8, help="How many workers to use for data loading. Default: 8.")
    parser.add_argument("--eval_freq", type=int, default=2, help="Evaluate every x epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument('--optimiser', type=str, default='adam', help='Optimizer to use')
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_layers_pre_gt", type=int, default=0, help="Number of layers pre graph transformer")
    parser.add_argument("--num_neighbors", type=int, nargs='*', default=[128, 128], help="Number of neighbors")
    parser.add_argument("--temporal_strategy", type=str, default="uniform", help="Temporal strategy")
    parser.add_argument("--rev_mp", action='store_true', help="Use Reverse Message Passing")
    parser.add_argument("--port_numbering", action='store_true', help="Add Port Numbering")
    parser.add_argument("--ego_ids", action='store_true', help="Use Ego IDs")
    parser.add_argument("--edge_features", action='store_true', help="Concat node features into edge features")
    parser.add_argument("--dropouts", type=float, nargs='*', help="Local, global and attention dropout rates")
    parser.add_argument("--head", type=str, default="HeteroGNNNodeHead", help="Attention Head for the transformer")
    parser.add_argument("--early_stopping", action='store_true', help="Use early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Override default configuration
    config = CustomConfig(
        data_name = args.dataset ,
        task_name = args.task,
        evaluation_freq = args.eval_freq,
        learning_rate = args.lr,
        epochs = args.epochs,
        optimiser = args.optimiser,
        batch_size = args.batch_size,
        channels = args.channels,
        aggr = args.aggr,
        num_layers = args.num_layers,
        num_layers_pre_gt = args.num_layers_pre_gt,
        num_neighbors = args.num_neighbors,
        temporal_strategy = args.temporal_strategy,
        reverse_mp = args.rev_mp,
        port_numbering = args.port_numbering,
        ego_ids = args.ego_ids,
        dropouts = args.dropouts,
        head = args.head,
        edge_features = args.edge_features,
        save_artifacts=args.save_artifacts,
        early_stopping=args.early_stopping,
    )

    config.print_config()


    data_loader = RelBenchDataLoader(
        data_name=config.data_name,
        task_name=config.task_name,
        device=config.device,
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        num_neighbors=config.num_neighbors,
        num_workers=config.num_workers,
        temporal_strategy=config.temporal_strategy,
        reverse_mp=config.reverse_mp,
        add_ports=config.port_numbering,
        ego_ids=config.ego_ids,
        preprocess_graph=args.model=='graphormer',
    )

    if data_loader.task.task_type == TaskType.BINARY_CLASSIFICATION:
        loss_fn = BCEWithLogitsLoss()
        config.tune_metric = "roc_auc"
        config.higher_is_better = True
    elif data_loader.task.task_type == TaskType.REGRESSION:
        loss_fn = L1Loss()
        config.tune_metric = "mae"
        config.higher_is_better = False

    multi_edge_types = analyze_multi_edges(data_loader.graph)
    logging.info(f"\nFound {len(multi_edge_types)} edge types with multi-edges")

    model = None

    if args.model == 'graphormer':
        model = Graphormer(
            data=data_loader.graph,
            col_stats_dict=data_loader.col_stats_dict,
            channels=config.channels,
            out_channels=config.out_channels,
            dropouts=config.dropouts,
            num_layers=config.num_layers,
            head=config.head,
            edge_features=config.edge_features,
            torch_frame_model_kwargs={"channels": config.channels, "num_layers": config.num_layers},
        ).to(config.device)
    elif args.model == 'local':
        model = FraudGT(
            data=data_loader.graph,
            col_stats_dict=data_loader.col_stats_dict,
            channels=config.channels,
            out_channels=config.out_channels,
            dropouts=config.dropouts,
            num_layers=config.num_layers,
            num_layers_pre_gt=config.num_layers_pre_gt,
            head=config.head,
            edge_features=config.edge_features,
            torch_frame_model_kwargs={"channels": config.channels, "num_layers": config.num_layers},
        ).to(config.device)
    else:
        model = HeteroGraphSage(
            data=data_loader.graph,
            col_stats_dict=data_loader.col_stats_dict,
            channels=config.channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            aggr=config.aggr,
            norm=config.norm,
            torch_frame_model_kwargs={"channels": config.channels, "num_layers": config.num_layers},
        ).to(config.device)

    logging.info(f"Model: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total model parameters: {total_params}")

    # Initialize optimizer and loss function
    optimiser = None
    if config.optimiser == 'adam':
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimiser == 'adamW':
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    else:
        raise ValueError("Invalid optimizer specified")


    wandb.init(
        project="Graph Learning",
        config={
                   "model": 'Graphormer' if args.model == 'graphormer' else 'FraudGT' if args.model == 'local' else 'GlobalHGT' if args.model == 'globalhgt' else 'Arbitrary Model',
               } | config.__dict__
    )

    best_metrics, best_model = train(
        model=model,
        loaders=data_loader.loader_dict,
        optimizer=optimizer,
        loss_fn=loss_fn,
        task=data_loader.task,
        config=config,
    )
