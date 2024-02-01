"""
Evaluation script for sentiment analysis models.
Includes comprehensive evaluation metrics and visualizations.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.manifold import TSNE
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import yaml

from models import get_model, AttentionVisualizer
from data_loader import DataProcessor
from utils import ConfigManager, Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utility."""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize evaluator with model and configuration.
        
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to configuration file
        """
        self.config = ConfigManager.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.config['data']['tokenizer'])
        
        # Initialize attention visualizer
        self.attention_viz = AttentionVisualizer()
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = get_model(
            self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def evaluate_dataset(self, dataset_name: str = None) -> Dict[str, any]:
        """
        Evaluate model on specified dataset.
        
        Args:
            dataset_name: Dataset to evaluate on ('sst2', 'imdb', or None for config default)
            
        Returns:
            Dictionary containing evaluation results
        """
        if dataset_name is None:
            dataset_name = self.config['data']['dataset']
        
        # Load dataset
        if dataset_name == 'sst2':
            _, _, test_dataset = self.data_processor.load_sst2()
        elif dataset_name == 'imdb':
            _, _, test_dataset = self.data_processor.load_imdb()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Create data loader
        _, _, test_loader = self.data_processor.create_data_loaders(
            test_dataset, test_dataset, test_dataset,  # Only need test loader
            batch_size=self.config['training']['batch_size']
        )
        
        # Run evaluation
        results = self._evaluate_loader(test_loader, test_dataset)
        results['dataset'] = dataset_name
        
        return results
    
    def _evaluate_loader(self, data_loader, dataset) -> Dict[str, any]:
        """Evaluate model on data loader."""
        timer = Timer()
        timer.start()
        
        predictions = []
        true_labels = []
        prediction_probs = []
        attention_weights = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                prediction_probs.extend(probs.cpu().numpy())
                
                # Collect attention weights (for first few samples)
                if len(attention_weights) < 10 and 'attentions' in outputs:
                    attention_weights.extend(outputs['attentions'])
        
        timer.stop()
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, predictions, prediction_probs)
        
        # Add timing information
        metrics['evaluation_time'] = timer.elapsed()
        metrics['samples_per_second'] = len(predictions) / timer.elapsed()
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'true_labels': true_labels,
            'prediction_probs': prediction_probs,
            'attention_weights': attention_weights[:10],  # Keep first 10 samples
            'texts': [dataset.texts[i] for i in range(min(len(dataset), 100))]  # First 100 texts
        }
    
    def _calculate_metrics(self, true_labels: List[int], predictions: List[int], 
                          probs: List[List[float]]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        metrics['precision'] = precision_score(true_labels, predictions, average='weighted')
        metrics['recall'] = recall_score(true_labels, predictions, average='weighted')
        metrics['f1_score'] = f1_score(true_labels, predictions, average='weighted')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(true_labels, predictions, average=None).tolist()
        metrics['recall_per_class'] = recall_score(true_labels, predictions, average=None).tolist()
        metrics['f1_per_class'] = f1_score(true_labels, predictions, average=None).tolist()
        
        # ROC AUC (for binary classification)
        if len(set(true_labels)) == 2:
            pos_probs = [p[1] for p in probs]  # Probability of positive class
            fpr, tpr, _ = roc_curve(true_labels, pos_probs)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr.tolist()
            metrics['tpr'] = tpr.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_names = ['Negative', 'Positive'] if len(set(true_labels)) == 2 else None
        report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def visualize_results(self, results: Dict[str, any], save_dir: str):
        """Create and save visualization plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        metrics = results['metrics']
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        # Plot confusion matrix
        self._plot_confusion_matrix(metrics['confusion_matrix'], save_dir)
        
        # Plot ROC curve (if binary classification)
        if 'roc_auc' in metrics:
            self._plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], save_dir)
        
        # Plot prediction distribution
        self._plot_prediction_distribution(predictions, true_labels, save_dir)
        
        # Plot attention visualizations
        if results['attention_weights'] and results['texts']:
            self._plot_attention_examples(results['attention_weights'], results['texts'][:5], save_dir)
        
        # Plot metrics summary
        self._plot_metrics_summary(metrics, save_dir)
    
    def _plot_confusion_matrix(self, cm: List[List[int]], save_dir: str):
        """Plot confusion matrix."""
        cm_array = np.array(cm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, fpr: List[float], tpr: List[float], auc_score: float, save_dir: str):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, predictions: List[int], true_labels: List[int], save_dir: str):
        """Plot prediction distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True label distribution
        unique, counts = np.unique(true_labels, return_counts=True)
        ax1.bar(['Negative', 'Positive'], counts, color=['red', 'green'], alpha=0.7)
        ax1.set_title('True Label Distribution')
        ax1.set_ylabel('Count')
        
        # Predicted label distribution
        unique, counts = np.unique(predictions, return_counts=True)
        ax2.bar(['Negative', 'Positive'], counts, color=['red', 'green'], alpha=0.7)
        ax2.set_title('Predicted Label Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_examples(self, attention_weights: List, texts: List[str], save_dir: str):
        """Plot attention visualization examples."""
        # This would require tokenizer to convert back to tokens
        # Simplified version - just log that attention data is available
        logger.info(f"Attention weights available for {len(attention_weights)} examples")
        logger.info("Attention visualization would require tokenizer integration")
    
    def _plot_metrics_summary(self, metrics: Dict[str, float], save_dir: str):
        """Plot summary of key metrics."""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(metric, 0) for metric in key_metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(key_metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Summary')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, any], save_path: str):
        """Generate comprehensive evaluation report."""
        metrics = results['metrics']
        
        report = f"""
# Model Evaluation Report

## Dataset: {results['dataset']}

## Performance Metrics

### Overall Performance
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}

### Per-Class Performance
- **Negative Class**:
  - Precision: {metrics['precision_per_class'][0]:.4f}
  - Recall: {metrics['recall_per_class'][0]:.4f}
  - F1 Score: {metrics['f1_per_class'][0]:.4f}

- **Positive Class**:
  - Precision: {metrics['precision_per_class'][1]:.4f}
  - Recall: {metrics['recall_per_class'][1]:.4f}
  - F1 Score: {metrics['f1_per_class'][1]:.4f}

### Additional Metrics
"""
        
        if 'roc_auc' in metrics:
            report += f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n"
        
        report += f"""
### Performance Statistics
- **Evaluation Time**: {metrics['evaluation_time']:.2f} seconds
- **Samples per Second**: {metrics['samples_per_second']:.1f}

### Confusion Matrix
```
{np.array(metrics['confusion_matrix'])}
```

## Analysis

The model shows {'strong' if metrics['accuracy'] > 0.9 else 'good' if metrics['accuracy'] > 0.8 else 'moderate'} performance on the {results['dataset']} dataset.

Key observations:
- Accuracy of {metrics['accuracy']:.1%} indicates {'excellent' if metrics['accuracy'] > 0.95 else 'good' if metrics['accuracy'] > 0.85 else 'acceptable'} overall performance
- F1 score of {metrics['f1_score']:.4f} shows {'well-balanced' if abs(metrics['precision'] - metrics['recall']) < 0.05 else 'some imbalance in'} precision and recall
"""
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['sst2', 'imdb'],
                       help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_dataset(args.dataset)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    logger.info("Creating visualizations...")
    evaluator.visualize_results(results, args.output_dir)
    
    # Generate report
    logger.info("Generating report...")
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    evaluator.generate_report(results, report_path)
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, 'detailed_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results['metrics'], f, default_flow_style=False)
    
    # Print summary
    metrics = results['metrics']
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Dataset: {results['dataset']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Evaluation Time: {metrics['evaluation_time']:.2f}s")
    print(f"{'='*50}")
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 