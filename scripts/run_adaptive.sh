#!/bin/bash
# Batch script to run adaptive LoRA experiments with different strategies

echo "🚀 Starting adaptive LoRA experiments..."

# Create results directory if it doesn't exist
mkdir -p results

# Common parameters
DATASET_SIZE=200
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=3e-4
SEED=42
BASE_RANK=16
MIN_RANK=4
MAX_RANK=32

# Run linear decay strategy
echo "📊 Running linear decay strategy..."
python scripts/train_adaptive.py \
    --strategy linear_decay \
    --base_rank $BASE_RANK \
    --min_rank $MIN_RANK \
    --max_rank $MAX_RANK \
    --dataset_size $DATASET_SIZE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --experiment_name "adaptive_linear_decay_official"

# Run attention-heavy strategy
echo "📊 Running attention-heavy strategy..."
python scripts/train_adaptive.py \
    --strategy attention_heavy \
    --base_rank $BASE_RANK \
    --min_rank $MIN_RANK \
    --max_rank $MAX_RANK \
    --dataset_size $DATASET_SIZE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --experiment_name "adaptive_attention_heavy_official"

# Run empirical strategy
echo "📊 Running empirical strategy..."
python scripts/train_adaptive.py \
    --strategy empirical \
    --base_rank $BASE_RANK \
    --min_rank $MIN_RANK \
    --max_rank $MAX_RANK \
    --dataset_size $DATASET_SIZE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --experiment_name "adaptive_empirical_official"

echo "✅ Adaptive experiments completed!"
echo "📁 Results saved in: results/"
echo "📈 Check the results.json and allocation.json files for detailed metrics" 