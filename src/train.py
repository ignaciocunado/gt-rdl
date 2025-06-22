from typing import Dict, Any, Tuple
import torch
import copy
from tqdm import tqdm
from torch.optim import Optimizer
from torch.nn import Module
from torch_geometric.loader import NeighborLoader
import numpy as np
from .config import CustomConfig
import logging
from relbench.base import TaskType
import os
import wandb



def train_epoch(
    loader: NeighborLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    task: Any,
    config: CustomConfig
) -> float:
    """Trains the model for one epoch.
    
    Args:
        loader: Data loader for training
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        task: RelBench task
        config: Training configuration
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader), config.max_steps_per_epoch)
    for batch in tqdm(loader, desc="Training", total=total_steps):
        batch = batch.to(config.device)
        optimizer.zero_grad()

        pred = model(
            batch,
            task.entity_table
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        
        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > config.max_steps_per_epoch:
            break
    
    return loss_accum / count_accum


@torch.no_grad()
def eval_epoch(
    loader: NeighborLoader,
    model: Module,
    task: Any,
    config: CustomConfig
) -> np.ndarray:
    """Evaluates the model on the given loader.
    
    Args:
        loader: Data loader for evaluation
        model: Model to evaluate
        task: RelBench task
        config: Training configuration
        
    Returns:
        np.ndarray: Model predictions
    """
    model.eval()
    pred_list = []
    
    for batch in tqdm(loader):
        batch = batch.to(config.device)
        pred = model(batch, task.entity_table)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)
        
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        
    return torch.cat(pred_list, dim=0).numpy()


def train(
    model: Module,
    loaders: Dict[str, NeighborLoader],
    optimizer: Optimizer,
    loss_fn: Module,
    task: Any,
    config: CustomConfig
) -> Tuple[Dict[str, float], Module]:
    """Main training loop with validation and model selection.
    
    Args:
        model: Model to train
        loaders: Dictionary containing train/val/test dataloaders
        optimizer: Optimizer for training
        loss_fn: Loss function
        task: RelBench task
        config: Training configuration

    Returns:
        Tuple containing:
            - Dictionary of best metrics
            - Best model state
    """
    state_dict = None
    best_val_metric = float('-inf') if config.higher_is_better else float('inf')
    no_improve_count = 0
    best_metrics = {}
    
    for epoch in range(1, config.epochs + 1):
        # Training phase
        train_loss = train_epoch(loaders['train'], model, optimizer, loss_fn, task, config)

        wandb.log({"epoch": epoch, "train_loss": train_loss})
        # Perform validation based on the evaluation frequency setting
        if epoch % config.evaluation_freq == 0 or epoch == 1 or epoch == config.epochs:
            # Validation phase
            val_pred = eval_epoch(loaders['val'], model, task, config)
            test_pred = eval_epoch(loaders['test'], model, task, config)

            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            test_metrics = task.evaluate(test_pred)

            # Logging
            logging.info(f"Epoch: {epoch:02d}, Train loss: {train_loss:.4f}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics
            })

            # Model selection
            current_metric = val_metrics[config.tune_metric]
            improved = (config.higher_is_better and current_metric > best_val_metric) or \
                      (not config.higher_is_better and current_metric < best_val_metric)
            
            if improved:
                best_val_metric = current_metric
                best_test_metric = test_metrics[config.tune_metric]
                wandb.log({
                    "epoch": epoch,
                    "best_val_metric": best_val_metric,
                    "best_test_metric": best_test_metric,
                })
                state_dict = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics
                no_improve_count = 0

                
                # Save the model with comprehensive metadata
                if config.save_artifacts:
                    # Save checkpoint in the run-specific checkpoint directory
                    # Create a unique filename with metric value
                    metric_value = f"{current_metric:.4f}".replace(".", "-")
                    checkpoint_filename = f"epoch_{epoch:03d}_{config.tune_metric}_{metric_value}.pt"
                    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_filename)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    artifact = wandb.Artifact('model_checkpoint', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
                    logging.info(f"Saved model checkpoint to {checkpoint_path}")

            else:
                no_improve_count += 1
            
            # Early stopping
            if config.early_stopping and no_improve_count >= config.patience:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break
        else:
            logging.info(f"Epoch: {epoch:02d}, Train loss: {train_loss:.4f}")
    
    # Load best model
    if state_dict is not None:
        model.load_state_dict(state_dict)
    
    # Final evaluation on test set
    test_pred = eval_epoch(loaders['test'], model, task, config)
  
    test_metrics = task.evaluate(test_pred)
    logging.info(f"Best validation metrics: {best_metrics}")
    logging.info(f"Test metrics: {test_metrics}")
    wandb.finish()

    return best_metrics, model