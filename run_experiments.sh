#!/bin/bash

# CS 224N Final Project: Sentiment Analysis Experiments
# This script runs all experiments for the project

echo "Starting CS 224N Final Project Experiments"
echo "=========================================="

# Create necessary directories
mkdir -p checkpoints results logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:./src"
export CUDA_VISIBLE_DEVICES=0

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run experiment with error handling
run_experiment() {
    local config_file=$1
    local experiment_name=$2
    
    log "Starting experiment: $experiment_name"
    log "Config file: $config_file"
    
    if python src/train.py --config "$config_file" > "logs/${experiment_name}.log" 2>&1; then
        log "✅ $experiment_name completed successfully"
    else
        log "❌ $experiment_name failed - check logs/${experiment_name}.log"
        return 1
    fi
}

# 1. Data preprocessing and analysis
log "Step 1: Data preprocessing and analysis"
python src/data_loader.py --dataset sst2 --download --preprocess --analyze > logs/data_preprocessing.log 2>&1
python src/data_loader.py --dataset imdb --download --preprocess --analyze >> logs/data_preprocessing.log 2>&1

# 2. Run individual model experiments
log "Step 2: Training individual models"

# BERT on SST-2
run_experiment "experiments/configs/bert_sst2.yaml" "bert_sst2"

# RoBERTa on IMDB  
run_experiment "experiments/configs/roberta_imdb.yaml" "roberta_imdb"

# 3. Run ensemble experiment
log "Step 3: Training ensemble model"
run_experiment "experiments/configs/ensemble_sst2.yaml" "ensemble_sst2"

# 4. Model evaluation
log "Step 4: Model evaluation"

# Evaluate BERT model
if [ -f "checkpoints/bert_sst2/best_model.pt" ]; then
    log "Evaluating BERT model"
    python src/evaluate.py \
        --model_path "checkpoints/bert_sst2/best_model.pt" \
        --config "experiments/configs/bert_sst2.yaml" \
        --output_dir "results/bert_sst2_evaluation" > logs/bert_evaluation.log 2>&1
fi

# Evaluate RoBERTa model
if [ -f "checkpoints/roberta_imdb/best_model.pt" ]; then
    log "Evaluating RoBERTa model"
    python src/evaluate.py \
        --model_path "checkpoints/roberta_imdb/best_model.pt" \
        --config "experiments/configs/roberta_imdb.yaml" \
        --output_dir "results/roberta_imdb_evaluation" > logs/roberta_evaluation.log 2>&1
fi

# Evaluate ensemble model
if [ -f "checkpoints/ensemble_sst2/best_model.pt" ]; then
    log "Evaluating ensemble model"
    python src/evaluate.py \
        --model_path "checkpoints/ensemble_sst2/best_model.pt" \
        --config "experiments/configs/ensemble_sst2.yaml" \
        --output_dir "results/ensemble_sst2_evaluation" > logs/ensemble_evaluation.log 2>&1
fi

# 5. Generate final report
log "Step 5: Generating final analysis"

# Run Jupyter notebook for analysis (if available)
if command -v jupyter &> /dev/null; then
    log "Running analysis notebook"
    jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis_executed.ipynb
else
    log "Jupyter not available, skipping notebook execution"
fi

# 6. Create experiment summary
log "Step 6: Creating experiment summary"
cat > results/experiment_log.txt << EOF
CS 224N Final Project - Experiment Log
Generated on: $(date)

Experiments completed:
- BERT on SST-2: $([ -f "checkpoints/bert_sst2/best_model.pt" ] && echo "✅ Success" || echo "❌ Failed")
- RoBERTa on IMDB: $([ -f "checkpoints/roberta_imdb/best_model.pt" ] && echo "✅ Success" || echo "❌ Failed")
- Ensemble on SST-2: $([ -f "checkpoints/ensemble_sst2/best_model.pt" ] && echo "✅ Success" || echo "❌ Failed")

Results available in:
- results/bert_sst2/
- results/roberta_imdb/
- results/ensemble_sst2/
- results/*_evaluation/

Log files available in logs/ directory.
EOF

# 7. Calculate total experiment time and resources
log "Step 7: Finalizing results"
echo "Experiment Summary:" > results/final_summary.txt
echo "==================" >> results/final_summary.txt
echo "Total experiment time: $(date)" >> results/final_summary.txt
echo "GPU memory usage logged in individual experiment logs" >> results/final_summary.txt
echo "All models and results saved in respective directories" >> results/final_summary.txt

log "🎉 All experiments completed!"
log "Results summary available in results/experiment_log.txt"
log "Check individual log files in logs/ directory for detailed information"

# Display final status
echo ""
echo "Final Status:"
echo "============="
echo "BERT (SST-2):     $([ -f "checkpoints/bert_sst2/best_model.pt" ] && echo "✅ Completed" || echo "❌ Failed")"
echo "RoBERTa (IMDB):   $([ -f "checkpoints/roberta_imdb/best_model.pt" ] && echo "✅ Completed" || echo "❌ Failed")"
echo "Ensemble (SST-2): $([ -f "checkpoints/ensemble_sst2/best_model.pt" ] && echo "✅ Completed" || echo "❌ Failed")"
echo ""
echo "Next steps:"
echo "1. Review results in results/ directory"
echo "2. Check training curves and metrics"
echo "3. Analyze model predictions and errors"
echo "4. Prepare final project report" 