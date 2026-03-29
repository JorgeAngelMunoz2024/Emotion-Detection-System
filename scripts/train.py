"""
Training script for Scale-Interaction Transformer (SIT) model.
Implements the training protocol described in Algorithm 2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time
from tqdm import tqdm

from models.transformer import ScaleInteractionTransformer


class BeautyScoreDataset(Dataset):
    """
    Dataset class for beauty score prediction.
    Replace this with your actual dataset implementation.
    """
    def __init__(self, image_paths, scores, transform=None):
        """
        Args:
            image_paths: List of paths to images
            scores: List of ground-truth beauty scores
            transform: Optional transforms to apply to images
        """
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # TODO: Implement actual image loading
        # For now, return random data for testing
        image = torch.randn(3, 224, 224)  # Replace with actual image loading
        score = torch.tensor([self.scores[idx]], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, score


class Trainer:
    """
    Trainer class implementing Algorithm 2: Scale-Interaction Transformer Training Procedure
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 10,
        save_dir: str = "checkpoints"
    ):
        """
        Args:
            model: The SIT model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on (CPU or GPU)
            learning_rate: Initial learning rate (10^-4 as specified)
            max_epochs: Maximum number of epochs (300 as specified)
            patience: Early stopping patience (10 epochs as specified)
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Line 2: Initialize Adam optimizer with learning rate η = 10^-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # MSE loss function (Equation 12)
        self.criterion = nn.MSELoss()
        
        # ReduceLROnPlateau scheduler: reduce learning rate by factor 0.5
        # if validation loss stagnates for 5 epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        # Line 4: for each batch (X, y) in the training set
        for images, scores in progress_bar:
            images = images.to(self.device)
            scores = scores.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Lines 5-10: Forward pass through the model
            # F_base ← MobileNetV2(X)
            # S ← MultiScaleFeatureModule(F_base)
            # S_proj ← LinearProjection(S)
            # S_trans ← TransformerBlocks(S_proj)
            # v ← GlobalAveragePooling(S_trans)
            # ŷ ← RegressionHead(v)
            predictions = self.model(images)
            
            # Line 11: Compute loss: L ← MSE(y, ŷ)
            loss = self.criterion(predictions, scores)
            
            # Line 12: Update parameters: θ ← AdamUpdate(θ, ∇_θ L)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, scores in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                scores = scores.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, scores)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            print(f"Saved best model with validation loss: {self.best_val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, resume_from: Optional[str] = None):
        """
        Complete training procedure implementing Algorithm 2.
        
        Args:
            resume_from: Optional path to checkpoint to resume training from
        """
        start_epoch = 1
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        # Line 3: for epoch = 1 to 300
        for epoch in range(start_epoch, self.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            print("-" * 50)
            
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Line 14: Evaluate model on the validation set
            val_loss = self.validate()
            
            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Line 15: Update learning rate scheduler and check early stopping criterion
            self.scheduler.step(val_loss)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping check (patience = 10 epochs)
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
                break
        
        # Line 16: end for
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size (32 as specified in the paper)
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """
    Main training function.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy datasets (replace with actual data)
    # TODO: Replace with actual dataset loading
    dummy_images = [f"image_{i}.jpg" for i in range(1000)]
    dummy_scores = np.random.uniform(1, 5, 1000).tolist()
    
    train_dataset = BeautyScoreDataset(dummy_images[:800], dummy_scores[:800])
    val_dataset = BeautyScoreDataset(dummy_images[800:], dummy_scores[800:])
    
    # Create data loaders with batch_size=32 as specified
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        batch_size=32
    )
    
    # Create model
    model = ScaleInteractionTransformer(
        scales=[3, 10, 13],
        d_proj=128,
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,  # 10^-4 as specified
        max_epochs=300,
        patience=10
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
