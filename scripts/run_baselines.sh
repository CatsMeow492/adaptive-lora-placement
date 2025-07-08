#!/bin/bash
# Batch script to run baseline experiments with ranks 8 and 16

echo "🚀 Starting baseline experiments..."

# Create results directory if it doesn't exist
mkdir -p results

# Run baseline with rank 8
echo "📊 Running baseline experiment with rank 8..."
python scripts/train_baseline.py \
    --lora_rank 8 \
    --dataset_size 200 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --seed 42 \
    --experiment_name "baseline_rank8_official"

# Run baseline with rank 16
echo "📊 Running baseline experiment with rank 16..."
python scripts/train_baseline.py \
    --lora_rank 16 \
    --dataset_size 200 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --seed 42 \
    --experiment_name "baseline_rank16_official"

echo "✅ Baseline experiments completed!"
echo "📁 Results saved in: results/"
echo "📈 Check the results.json files for detailed metrics" 