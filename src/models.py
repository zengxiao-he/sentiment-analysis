"""
Model architectures for sentiment analysis.
Includes BERT, RoBERTa, and ensemble implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, BertConfig,
    RobertaModel, RobertaConfig,
    DistilBertModel, DistilBertConfig
)
from typing import Dict, List, Optional, Tuple


class BERTSentimentClassifier(nn.Module):
    """BERT-based sentiment classifier with custom head."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(BERTSentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Freeze early layers for better generalization
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BERT model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing logits and attention weights
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                           output_attentions=True)
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'attentions': outputs.attentions,
            'hidden_states': outputs.last_hidden_state
        }


class RoBERTaSentimentClassifier(nn.Module):
    """RoBERTa-based sentiment classifier with enhanced attention."""
    
    def __init__(self, model_name: str = 'roberta-base', 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(RoBERTaSentimentClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Custom attention head
        hidden_size = self.roberta.config.hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through RoBERTa model with custom attention."""
        
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                              output_attentions=True)
        
        # Apply custom attention
        sequence_output = outputs.last_hidden_state
        attended_output, attention_weights = self.attention(
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        
        attended_output = attended_output.transpose(0, 1)
        attended_output = self.layer_norm(attended_output + sequence_output)
        
        # Global average pooling
        pooled_output = attended_output.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'attentions': outputs.attentions,
            'custom_attention': attention_weights,
            'hidden_states': attended_output
        }


class EnsembleSentimentClassifier(nn.Module):
    """Ensemble model combining BERT, RoBERTa, and DistilBERT."""
    
    def __init__(self, num_classes: int = 2):
        super(EnsembleSentimentClassifier, self).__init__()
        
        self.bert = BERTSentimentClassifier('bert-base-uncased', num_classes)
        self.roberta = RoBERTaSentimentClassifier('roberta-base', num_classes)
        self.distilbert = DistilBERTSentimentClassifier('distilbert-base-uncased', num_classes)
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble of models."""
        
        # Get predictions from all models
        bert_outputs = self.bert(input_ids, attention_mask)
        roberta_outputs = self.roberta(input_ids, attention_mask)
        distilbert_outputs = self.distilbert(input_ids, attention_mask)
        
        # Apply temperature scaling and ensemble weights
        bert_probs = F.softmax(bert_outputs['logits'] / self.temperature, dim=-1)
        roberta_probs = F.softmax(roberta_outputs['logits'] / self.temperature, dim=-1)
        distilbert_probs = F.softmax(distilbert_outputs['logits'] / self.temperature, dim=-1)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_probs = (weights[0] * bert_probs + 
                         weights[1] * roberta_probs + 
                         weights[2] * distilbert_probs)
        
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return {
            'logits': ensemble_logits,
            'individual_logits': {
                'bert': bert_outputs['logits'],
                'roberta': roberta_outputs['logits'],
                'distilbert': distilbert_outputs['logits']
            },
            'ensemble_weights': weights,
            'temperature': self.temperature
        }


class DistilBERTSentimentClassifier(nn.Module):
    """DistilBERT-based sentiment classifier for fast inference."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(DistilBERTSentimentClassifier, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(self.distilbert.config.dim, self.distilbert.config.dim)
        self.classifier = nn.Linear(self.distilbert.config.dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through DistilBERT model."""
        
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask,
                                 output_attentions=True)
        
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # Take [CLS] token
        
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'attentions': outputs.attentions,
            'hidden_states': hidden_state
        }


class AttentionVisualizer:
    """Utility class for visualizing attention weights."""
    
    @staticmethod
    def extract_attention_weights(model_outputs: Dict, layer: int = -1, head: int = 0) -> torch.Tensor:
        """Extract attention weights from model outputs."""
        attentions = model_outputs['attentions']
        return attentions[layer][:, head, :, :].detach().cpu()
    
    @staticmethod
    def visualize_attention(tokens: List[str], attention_weights: torch.Tensor, 
                          save_path: Optional[str] = None):
        """Create attention heatmap visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.numpy(), 
                   xticklabels=tokens, yticklabels=tokens,
                   cmap='Blues', annot=True, fmt='.2f')
        plt.title('Attention Weights Visualization')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def get_model(model_type: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory function to create models."""
    
    model_registry = {
        'bert': BERTSentimentClassifier,
        'roberta': RoBERTaSentimentClassifier,
        'distilbert': DistilBERTSentimentClassifier,
        'ensemble': EnsembleSentimentClassifier
    }
    
    if model_type not in model_registry:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_registry[model_type](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test model instantiation
    model = get_model('bert', num_classes=2)
    print(f"BERT model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    ensemble = get_model('ensemble', num_classes=2)
    print(f"Ensemble model created with {sum(p.numel() for p in ensemble.parameters())} parameters") 