# CS 224N Final Project: Advanced Sentiment Analysis with Transformer Models

**Course**: CS 224N - Natural Language Processing with Deep Learning  
**Quarter**: Winter 2024  

## Abstract

This project explores advanced techniques for sentiment analysis using transformer-based models. We implement and compare multiple approaches including fine-tuned BERT, RoBERTa, and a custom attention mechanism on the Stanford Sentiment Treebank (SST) and IMDB movie reviews datasets. Our best model achieves 94.2% accuracy on SST-2 and 92.8% on IMDB, demonstrating the effectiveness of domain-adaptive pre-training and ensemble methods.

## Project Structure

```
cs224n-final-project/
├── data/                   # Dataset files and preprocessing scripts
├── models/                 # Model implementations and saved checkpoints
├── src/                    # Source code
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── models.py          # Model architectures
│   ├── train.py           # Training scripts
│   ├── evaluate.py        # Evaluation scripts
│   └── utils.py           # Utility functions
├── experiments/            # Experiment configurations and results
├── notebooks/             # Jupyter notebooks for analysis
├── results/               # Output files, plots, and metrics
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Key Contributions

1. **Multi-Dataset Training**: Developed a unified framework for training on multiple sentiment analysis datasets
2. **Attention Visualization**: Implemented attention visualization techniques to understand model behavior
3. **Domain Adaptation**: Applied domain-adaptive pre-training for better cross-domain performance
4. **Ensemble Methods**: Combined multiple models using weighted voting and stacking approaches

## Datasets

- **Stanford Sentiment Treebank (SST-2)**: 67,349 movie review sentences with binary sentiment labels
- **IMDB Movie Reviews**: 50,000 movie reviews with binary sentiment labels
- **Amazon Product Reviews**: 400,000 product reviews for domain adaptation experiments

## Model Architectures

### 1. Fine-tuned BERT
- Base BERT model fine-tuned on sentiment data
- Added dropout and classification head
- Learning rate: 2e-5, Batch size: 16

### 2. RoBERTa with Custom Head
- RoBERTa-base with additional attention layers
- Multi-head attention mechanism for feature extraction
- Gradient clipping and weight decay regularization

### 3. Ensemble Model
- Combines predictions from BERT, RoBERTa, and DistilBERT
- Weighted voting based on validation performance
- Confidence-based prediction selection

## Results

| Model | SST-2 Accuracy | IMDB Accuracy | F1-Score (SST-2) |
|-------|----------------|---------------|------------------|
| BERT-base | 91.8% | 90.1% | 0.917 |
| RoBERTa-base | 93.1% | 91.4% | 0.929 |
| Custom Ensemble | **94.2%** | **92.8%** | **0.941** |

## Key Findings

1. **Domain Adaptation Helps**: Pre-training on Amazon reviews improved IMDB performance by 1.3%
2. **Attention Patterns**: Models focus heavily on adjectives and sentiment-bearing phrases
3. **Ensemble Benefits**: Combining models reduces overfitting and improves generalization
4. **Data Quality Matters**: Cleaning and preprocessing improved accuracy by 0.8-1.2%

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/[username]/cs224n-final-project.git
cd cs224n-final-project

# Install dependencies
pip install -r requirements.txt

# Download and preprocess data
python src/data_loader.py --download --preprocess

# Train models
python src/train.py --model bert --dataset sst2
python src/train.py --model roberta --dataset imdb

# Evaluate models
python src/evaluate.py --model ensemble --dataset sst2
```

## File Descriptions

- `src/models.py`: Contains all model architectures including BERT, RoBERTa, and ensemble implementations
- `src/train.py`: Training loop with logging, checkpointing, and early stopping
- `src/evaluate.py`: Evaluation metrics, confusion matrices, and attention visualization
- `experiments/configs/`: YAML configuration files for different experimental setups
- `notebooks/analysis.ipynb`: Data exploration and result analysis

## Future Work

1. **Multi-label Classification**: Extend to emotion detection beyond binary sentiment
2. **Cross-lingual Analysis**: Apply techniques to non-English sentiment analysis
3. **Real-time Applications**: Optimize models for production deployment
4. **Explainability**: Develop better interpretability methods for transformer predictions

## References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
3. Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.
4. Maas, A., et al. (2011). Learning Word Vectors for Sentiment Analysis.

## Acknowledgments

This project was completed as part of CS 224N: Natural Language Processing with Deep Learning at Stanford University. Special thanks to the teaching staff for their guidance and feedback throughout the course.

## Contact

For questions about this project, please contact [zengxiao@stanford.edu]. 