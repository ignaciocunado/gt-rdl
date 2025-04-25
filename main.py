import os
os.environ['XDG_CACHE_HOME'] = '/tudelft.net/staff-umbrella/ScalableGraphLearning/cagri/data'

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


# Override default configuration
config = CustomConfig(
    data_name='f1',
    task_name='driver-top3',
    data_dir=os.path.join(os.environ['XDG_CACHE_HOME'], 'relbench'),
    output_dir= '/home/cbilgi/projects/relbench/runs',
    evaluation_freq=2,  # Evaluate every 4 epochs
    learning_rate = 0.005,
    epochs = 10,
    batch_size=512,    
    channels=128,
    aggr='sum',
    num_layers=2,
    num_neighbors=[128, 128],
    temporal_strategy='uniform',
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
    temporal_strategy=config.temporal_strategy
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
    config=config
)

