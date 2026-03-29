"""
Configuration file for the Real-time Emotion Detection project.
Centralized configuration management.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the Scale-Interaction Transformer model."""
    # Architecture
    scales: List[int] = None
    d_proj: int = 128
    num_transformer_blocks: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [3, 10, 13]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 300
    
    # Regularization
    weight_decay: float = 0.0
    dropout: float = 0.1
    
    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    lr_factor: float = 0.5
    lr_patience: int = 5
    
    # Early stopping
    early_stop_patience: int = 10
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_frequency: int = 5  # Save every N epochs
    
    # Random seed
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data."""
    # Paths
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Image preprocessing
    image_size: int = 224
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    rotation_degrees: int = 10
    color_jitter: bool = True
    
    def __post_init__(self):
        if self.normalize_mean is None:
            # ImageNet normalization
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]


@dataclass
class Config:
    """Main configuration class combining all configs."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    # Project paths
    project_root: Path = Path(__file__).parent
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("\nModel:")
        for key, value in self.model.__dict__.items():
            lines.append(f"  {key}: {value}")
        lines.append("\nTraining:")
        for key, value in self.training.__dict__.items():
            lines.append(f"  {key}: {value}")
        lines.append("\nData:")
        for key, value in self.data.__dict__.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# Default configuration
default_config = Config()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print(config)
    
    # Save to YAML
    config.to_yaml("config_example.yaml")
    print("\nSaved configuration to config_example.yaml")
    
    # Load from YAML
    loaded_config = Config.from_yaml("config_example.yaml")
    print("\nLoaded configuration:")
    print(loaded_config)
