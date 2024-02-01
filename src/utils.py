"""
Utility functions for training and evaluation.
Includes early stopping, model checkpointing, and metrics tracking.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Check if training should stop."""
        return self.should_stop(val_loss, model)
    
    def should_stop(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss: Current validation loss
            model: Model to save best weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            # Save best model weights
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_model(self, model: torch.nn.Module):
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(self, save_dir: str, filename: str = "best_model.pt", 
                 save_best_only: bool = True, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            filename: Filename for the checkpoint
            save_best_only: Only save when the model improves
            monitor: Metric to monitor for improvement
            mode: 'min' for metrics that should be minimized, 'max' for maximized
        """
        self.save_dir = save_dir
        self.filename = filename
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        os.makedirs(save_dir, exist_ok=True)
        
        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
        else:
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
    
    def save_checkpoint(self, state_dict: Dict[str, Any], score: Optional[float] = None):
        """
        Save model checkpoint.
        
        Args:
            state_dict: Dictionary containing model state and metadata
            score: Current score for the monitored metric
        """
        filepath = os.path.join(self.save_dir, self.filename)
        
        # Add timestamp to state dict
        state_dict['timestamp'] = datetime.now().isoformat()
        
        if not self.save_best_only or score is None:
            torch.save(state_dict, filepath)
            logger.info(f"Checkpoint saved to {filepath}")
            return
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            torch.save(state_dict, filepath)
            logger.info(f"Best checkpoint saved to {filepath} (score: {score:.4f})")
    
    def load_checkpoint(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        if filepath is None:
            filepath = os.path.join(self.save_dir, self.filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self):
        self.history = {}
        self.current_epoch = 0
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get complete metrics history."""
        return self.history
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a specific metric."""
        if metric_name in self.history and self.history[metric_name]:
            return self.history[metric_name][-1]
        return None
    
    def get_best(self, metric_name: str, mode: str = 'max') -> tuple:
        """
        Get best value and epoch for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.history:
            return None, None
        
        values = self.history[metric_name]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return values[best_idx], best_idx + 1
    
    def save_to_csv(self, filepath: str):
        """Save metrics history to CSV file."""
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics saved to {filepath}")
    
    def save_to_json(self, filepath: str):
        """Save metrics history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")


class ConfigManager:
    """Configuration management utility."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add default values if missing
        config = ConfigManager._add_defaults(config)
        return config
    
    @staticmethod
    def _add_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Add default configuration values."""
        defaults = {
            'model': {
                'type': 'bert',
                'num_classes': 2,
                'dropout_rate': 0.3
            },
            'data': {
                'dataset': 'sst2',
                'tokenizer': 'bert-base-uncased',
                'max_length': 512
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'epochs': 10,
                'patience': 3,
                'min_delta': 0.001,
                'checkpoint_dir': './checkpoints',
                'results_dir': './results'
            },
            'logging': {
                'use_wandb': False,
                'project_name': 'sentiment-analysis',
                'log_interval': 100
            }
        }
        
        # Recursively merge defaults with provided config
        def merge_dicts(default, provided):
            result = default.copy()
            for key, value in provided.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dicts(defaults, config)
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """Save configuration to YAML file."""
        import yaml
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


class Timer:
    """Simple timer utility for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop the timer."""
        self.end_time = datetime.now()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()
    
    def elapsed_str(self) -> str:
        """Get elapsed time as formatted string."""
        elapsed = self.elapsed()
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def save_predictions(predictions: List[int], true_labels: List[int], 
                    texts: List[str], filepath: str):
    """Save predictions to CSV file for analysis."""
    df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predictions,
        'correct': [t == p for t, p in zip(true_labels, predictions)]
    })
    
    df.to_csv(filepath, index=False)
    logger.info(f"Predictions saved to {filepath}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test timer
    timer = Timer()
    timer.start()
    import time
    time.sleep(1)
    timer.stop()
    print(f"Timer test: {timer.elapsed_str()}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({'loss': 0.5, 'accuracy': 0.8})
    tracker.update({'loss': 0.3, 'accuracy': 0.9})
    print(f"Best accuracy: {tracker.get_best('accuracy')}")
    
    print("All tests passed!") 