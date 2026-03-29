"""
Utility functions for the Real-time Emotion Detection project.
"""

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
from pathlib import Path


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics for beauty score prediction.
    
    Args:
        predictions: Predicted scores
        targets: Ground-truth scores
        
    Returns:
        Dictionary containing various metrics
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    
    # Correlation coefficients
    pearson_corr, pearson_p = pearsonr(predictions.flatten(), targets.flatten())
    spearman_corr, spearman_p = spearmanr(predictions.flatten(), targets.flatten())
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p)
    }


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate
    ax2.plot(epochs, history['learning_rates'], marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """
    Plot predicted vs actual scores.
    
    Args:
        predictions: Predicted scores
        targets: Ground-truth scores
        save_path: Optional path to save the plot
    """
    metrics = compute_metrics(predictions, targets)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
             'r--', lw=2, label='Perfect prediction')
    
    plt.xlabel('Ground Truth Score')
    plt.ylabel('Predicted Score')
    plt.title(f"Predictions vs Ground Truth\n"
              f"Pearson: {metrics['pearson_correlation']:.3f}, "
              f"RMSE: {metrics['rmse']:.3f}")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_predictions(predictions: np.ndarray, 
                     targets: np.ndarray, 
                     image_paths: List[str],
                     save_path: str):
    """
    Save predictions to a file.
    
    Args:
        predictions: Predicted scores
        targets: Ground-truth scores
        image_paths: List of image paths
        save_path: Path to save the predictions
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'image_path': image_paths,
        'ground_truth': targets.flatten(),
        'prediction': predictions.flatten(),
        'error': np.abs(targets.flatten() - predictions.flatten())
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


class EarlyStopping:
    """
    Early stopping handler.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.should_stop


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test metrics
    dummy_preds = np.random.rand(100, 1) * 5
    dummy_targets = np.random.rand(100, 1) * 5
    
    metrics = compute_metrics(dummy_preds, dummy_targets)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
    
    print("\nTesting early stopping:")
    for epoch, loss in enumerate(losses, 1):
        should_stop = early_stop(loss)
        print(f"  Epoch {epoch}, Loss: {loss:.2f}, Stop: {should_stop}")
        if should_stop:
            break
