import argparse

import wandb
import os
os.environ['XDG_CACHE_HOME'] = '/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/data'

from src.config import CustomConfig
from src.dataloader import RelBenchDataLoader 
from src.models.hetero_gin import HeteroGraphGIN
from src.models.hetero_sage import HeteroGraphSage

from src.train import train
from src.utils import analyze_multi_edges

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import L1Loss, BCELoss, BCEWithLogitsLoss
import logging

from relbench.base import TaskType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use (local vs global)", required=True)
    parser.add_argument("--dataset", type=str, help="The dataset to use", required=True)
    parser.add_argument("--task", type=str, help="The task to solve", required=True)
    parser.add_argument("--eval_freq", type=int, default=2, help="Evaluate every x epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_neighbors", type=int, nargs='*', default=[128, 128], help="Number of neighbors")
    parser.add_argument("--temporal_strategy", type=str, default="uniform", help="Temporal strategy")
    parser.add_argument("--rev_mp", type=bool, default=True, help="Use Reverse Message Passing")
    parser.add_argument("--port_numbering", type=bool, default=True, help="Add Port Numbering")
    parser.add_argument("--ego_ids", type=bool, default=True, help="Use Ego IDs")

    args = parser.parse_args()

    # Override default configuration
    config = CustomConfig(
        data_name = args.dataset ,
        task_name = args.task,
        evaluation_freq = args.eval_freq,
        learning_rate = args.lr,
        epochs = args.epochs,
        batch_size = args.batch_size,
        channels = args.channels,
        aggr = args.aggr,
        num_layers = args.num_layers,
        num_neighbors = args.num_neighbors,
        temporal_strategy = args.temporal_strategy,
        reverse_mp = args.rev_mp,
        port_numbering = args.port_numbering,
        ego_ids = args.ego_ids,
    )

    config.print_config()


    data_loader = RelBenchDataLoader(
        data_name=config.data_name,
        task_name=config.task_name,
        device=config.device,
        root_dir=config.data_dir,
        batch_size=config.batch_size,
        num_neighbors=config.num_neighbors,
        num_workers=2,
        temporal_strategy=config.temporal_strategy,
        reverse_mp=config.reverse_mp,
        add_ports=config.port_numbering,
        ego_ids=config.ego_ids,
    )

    wandb.init(
        project="Graph Learning",
        config={
            "model": 'Global MP Transformer' if args.model == 'global' else 'FraudGT' if args.model == 'local' else 'Arbitrary Model',
        } | config.__dict__
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

    if args.model == 'global':
        # Global MP transformer
        pass
    elif args.model == 'local':
        # Gocal MP transformer (FraudGT)
        pass
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

        # Initialize optimizer and loss function
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        best_metrics, best_model = train(
            model=model,
            loaders=data_loader.loader_dict,
            optimizer=optimizer,
            loss_fn=loss_fn,
            task=data_loader.task,
            config=config,
        )

        # model = HeteroGraphGIN(
        #     data=data_loader.graph,
        #     col_stats_dict=data_loader.col_stats_dict,
        #     channels=config.channels,
        #     out_channels=config.out_channels,
        #     num_layers=config.num_layers,
        #     aggr=config.aggr,
        #     norm=config.norm,
        #     torch_frame_model_kwargs={"channels": config.channels, "num_layers": config.num_layers},
        # ).to(config.device)

