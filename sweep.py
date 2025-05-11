import yaml

import wandb
import os
os.environ['XDG_CACHE_HOME'] = '/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio/data'

from src.models.fraudgt import FraudGT

from src.config import CustomConfig
from src.dataloader import RelBenchDataLoader

from src.train import train
from src.utils import analyze_multi_edges

from torch.optim import Adam, AdamW
from torch.nn import L1Loss, BCELoss, BCEWithLogitsLoss
import logging

from relbench.base import TaskType

def outside_train():
    with wandb.init() as run:
        # Override default configuration
        config = CustomConfig(
            data_name = 'f1' ,
            task_name = 'driver-top3',
            evaluation_freq = run.config.evaluation_freq,
            learning_rate = run.config.learning_rate,
            epochs = run.config.epochs,
            optimiser = run.config.optimiser,
            batch_size = run.config.batch_size,
            channels = run.config.channels,
            aggr = run.config.aggr,
            num_layers = run.config.num_layers,
            num_layers_pre_gt = run.config.num_layers_pre_gt,
            num_neighbors = [run.config.num_neighbors, run.config.num_neighbors],
            temporal_strategy = 'uniform',
            reverse_mp = run.config.reverse_mp,
            port_numbering = run.config.port_numbering,
            ego_ids = run.config.ego_ids,
            dropouts = [run.config.local_dropout, run.config.global_dropout, run.config.attention_dropout],
            head = run.config.head,
            edge_features = run.config.edge_features,
            save_artifacts=False,
        )

        config.print_config()

        data_loader = RelBenchDataLoader(
            data_name=config.data_name,
            task_name=config.task_name,
            device=config.device,
            root_dir=config.data_dir,
            batch_size=config.batch_size,
            num_neighbors=config.num_neighbors,
            num_workers=6,
            temporal_strategy=config.temporal_strategy,
            reverse_mp=config.reverse_mp,
            add_ports=config.port_numbering,
            ego_ids=config.ego_ids,
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

        logging.info(f"Model: {model}")

        # Initialize optimizer and loss function
        optimiser = None
        if config.optimiser == 'adam':
            optimizer = Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimiser == 'adamW':
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        else:
            raise ValueError("Invalid optimizer specified")

        train(
            model=model,
            loaders=data_loader.loader_dict,
            optimizer=optimizer,
            loss_fn=loss_fn,
            task=data_loader.task,
            config=config,
        )

        run.finish()

if __name__ == "__main__":
    outside_train()