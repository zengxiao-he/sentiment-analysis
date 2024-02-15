"""
CS 224N Final Project: Advanced Sentiment Analysis with Transformer Models

This package contains implementations of various transformer-based models
for sentiment analysis, including BERT, RoBERTa, DistilBERT, and ensemble methods.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "zengxiao@stanford.edu"

from .models import (
    BERTSentimentClassifier,
    RoBERTaSentimentClassifier,
    DistilBERTSentimentClassifier,
    EnsembleSentimentClassifier,
    get_model
)

from .data_loader import DataProcessor, SentimentDataset
from .train import Trainer
from .evaluate import ModelEvaluator
from .utils import (
    EarlyStopping,
    ModelCheckpoint,
    MetricsTracker,
    set_seed,
    get_device
)

__all__ = [
    'BERTSentimentClassifier',
    'RoBERTaSentimentClassifier', 
    'DistilBERTSentimentClassifier',
    'EnsembleSentimentClassifier',
    'get_model',
    'DataProcessor',
    'SentimentDataset',
    'Trainer',
    'ModelEvaluator',
    'EarlyStopping',
    'ModelCheckpoint',
    'MetricsTracker',
    'set_seed',
    'get_device'
] 