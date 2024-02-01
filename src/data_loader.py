"""
Data loading and preprocessing utilities for sentiment analysis.
Supports SST-2, IMDB, and Amazon reviews datasets.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Custom dataset class for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """Handles data loading and preprocessing for different datasets."""
    
    def __init__(self, tokenizer_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_sst2(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess SST-2 dataset."""
        logger.info("Loading SST-2 dataset...")
        
        # Load from HuggingFace datasets
        dataset = load_dataset("glue", "sst2")
        
        train_texts = [item['sentence'] for item in dataset['train']]
        train_labels = [item['label'] for item in dataset['train']]
        
        val_texts = [item['sentence'] for item in dataset['validation']]
        val_labels = [item['label'] for item in dataset['validation']]
        
        # Create test split from validation (since test labels are not available)
        val_size = len(val_texts) // 2
        test_texts = val_texts[val_size:]
        test_labels = val_labels[val_size:]
        val_texts = val_texts[:val_size]
        val_labels = val_labels[:val_size]
        
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer)
        
        logger.info(f"SST-2 loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset
    
    def load_imdb(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess IMDB dataset."""
        logger.info("Loading IMDB dataset...")
        
        dataset = load_dataset("imdb")
        
        train_texts = [item['text'] for item in dataset['train']]
        train_labels = [item['label'] for item in dataset['train']]
        
        test_texts = [item['text'] for item in dataset['test']]
        test_labels = [item['label'] for item in dataset['test']]
        
        # Create validation split from training data
        val_size = len(train_texts) // 10  # 10% for validation
        val_texts = train_texts[:val_size]
        val_labels = train_labels[:val_size]
        train_texts = train_texts[val_size:]
        train_labels = train_labels[val_size:]
        
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer)
        
        logger.info(f"IMDB loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset
    
    def load_amazon_reviews(self, subset: str = "electronics") -> Dataset:
        """Load Amazon reviews for domain adaptation."""
        logger.info(f"Loading Amazon {subset} reviews...")
        
        try:
            # Try to load Amazon reviews dataset
            dataset = load_dataset("amazon_reviews_multi", "en", split="train[:100000]")
            
            texts = [item['review_title'] + " " + item['review_body'] for item in dataset]
            # Convert 5-star ratings to binary sentiment (1-2 stars = negative, 4-5 stars = positive)
            labels = [1 if item['stars'] >= 4 else 0 for item in dataset]
            
            # Filter out neutral reviews (3 stars)
            filtered_data = [(text, label) for text, label, stars in 
                           zip(texts, labels, [item['stars'] for item in dataset]) 
                           if stars != 3]
            
            texts, labels = zip(*filtered_data)
            
            amazon_dataset = SentimentDataset(list(texts), list(labels), self.tokenizer)
            logger.info(f"Amazon {subset} loaded: {len(amazon_dataset)} samples")
            return amazon_dataset
            
        except Exception as e:
            logger.warning(f"Could not load Amazon reviews: {e}")
            return None
    
    def create_data_loaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                           test_dataset: Dataset, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    import re
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very long sequences of repeated characters
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    
    return text


def analyze_dataset_statistics(dataset: Dataset) -> Dict:
    """Analyze dataset statistics."""
    texts = [dataset.texts[i] for i in range(len(dataset))]
    labels = [dataset.labels[i] for i in range(len(dataset))]
    
    text_lengths = [len(text.split()) for text in texts]
    
    stats = {
        'total_samples': len(dataset),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels),
        'avg_text_length': np.mean(text_lengths),
        'max_text_length': max(text_lengths),
        'min_text_length': min(text_lengths),
        'median_text_length': np.median(text_lengths)
    }
    
    return stats


def main():
    """Main function for data loading and preprocessing."""
    parser = argparse.ArgumentParser(description='Data loading and preprocessing')
    parser.add_argument('--dataset', type=str, choices=['sst2', 'imdb', 'amazon'], 
                       default='sst2', help='Dataset to load')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                       help='Tokenizer to use')
    parser.add_argument('--download', action='store_true',
                       help='Download datasets')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess datasets')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset statistics')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.tokenizer)
    
    if args.download or args.preprocess:
        if args.dataset == 'sst2':
            train_dataset, val_dataset, test_dataset = processor.load_sst2()
        elif args.dataset == 'imdb':
            train_dataset, val_dataset, test_dataset = processor.load_imdb()
        elif args.dataset == 'amazon':
            train_dataset = processor.load_amazon_reviews()
            val_dataset = test_dataset = None
        
        if args.analyze:
            logger.info("Dataset Statistics:")
            if train_dataset:
                train_stats = analyze_dataset_statistics(train_dataset)
                logger.info(f"Training set: {train_stats}")
            
            if val_dataset:
                val_stats = analyze_dataset_statistics(val_dataset)
                logger.info(f"Validation set: {val_stats}")
            
            if test_dataset:
                test_stats = analyze_dataset_statistics(test_dataset)
                logger.info(f"Test set: {test_stats}")


if __name__ == "__main__":
    main() 