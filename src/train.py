"""
Training script for sentiment analysis models.
Supports BERT, RoBERTa, DistilBERT, and ensemble models.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import wandb
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import argparse
import logging
import yaml
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from models import get_model
from data_loader import DataProcessor
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for sentiment analysis models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(
            config['model']['type'],
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        ).to(self.device)
        
        # Initialize data processor
        self.data_processor = DataProcessor(config['data']['tokenizer'])
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize tracking utilities
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        self.checkpoint = ModelCheckpoint(
            save_dir=config['training']['checkpoint_dir'],
            save_best_only=True
        )
        
        self.metrics_tracker = MetricsTracker()
        
        # Initialize wandb if enabled
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                config=config,
                name=f"{config['model']['type']}_{config['data']['dataset']}"
            )
    
    def load_data(self):
        """Load and prepare datasets."""
        dataset_name = self.config['data']['dataset']
        
        if dataset_name == 'sst2':
            train_dataset, val_dataset, test_dataset = self.data_processor.load_sst2()
        elif dataset_name == 'imdb':
            train_dataset, val_dataset, test_dataset = self.data_processor.load_imdb()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_processor.create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config['training']['batch_size']
        )
        
        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * self.config['training']['epochs']
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(true_labels, predictions)
        epoch_f1 = f1_score(true_labels, predictions, average='weighted')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs['logits'], dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'f1': val_f1
        }
    
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs['logits'], dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_acc = accuracy_score(true_labels, predictions)
        test_f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Generate classification report
        class_report = classification_report(
            true_labels, predictions,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        return {
            'accuracy': test_acc,
            'f1': test_f1,
            'classification_report': class_report,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.load_data()
        best_val_acc = 0
        
        for epoch in range(self.config['training']['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Track metrics
            self.metrics_tracker.update({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            })
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                })
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.checkpoint.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                    'config': self.config
                })
            
            # Early stopping
            if self.early_stopping.should_stop(val_metrics['loss']):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Test final model
        logger.info("Testing final model...")
        test_metrics = self.test()
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        
        # Save final results
        self.save_results(test_metrics)
        
        return test_metrics
    
    def save_results(self, test_metrics: Dict):
        """Save training results and plots."""
        results_dir = self.config['training']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics history
        self.metrics_tracker.save_to_csv(
            os.path.join(results_dir, 'training_metrics.csv')
        )
        
        # Plot training curves
        self.plot_training_curves(results_dir)
        
        # Save test results
        with open(os.path.join(results_dir, 'test_results.yaml'), 'w') as f:
            yaml.dump({
                'test_accuracy': float(test_metrics['accuracy']),
                'test_f1': float(test_metrics['f1']),
                'classification_report': test_metrics['classification_report']
            }, f)
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves."""
        history = self.metrics_tracker.get_history()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
        ax1.plot(history['epoch'], history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(history['epoch'], history['train_acc'], label='Train Acc')
        ax2.plot(history['epoch'], history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 curves
        ax3.plot(history['epoch'], history['train_f1'], label='Train F1')
        ax3.plot(history['epoch'], history['val_f1'], label='Val F1')
        ax3.set_title('Training and Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate
        ax4.plot(history['epoch'], [self.scheduler.get_last_lr()[0]] * len(history['epoch']))
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['bert', 'roberta', 'distilbert', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, choices=['sst2', 'imdb'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.model:
        config['model']['type'] = args.model
    if args.dataset:
        config['data']['dataset'] = args.dataset
    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 