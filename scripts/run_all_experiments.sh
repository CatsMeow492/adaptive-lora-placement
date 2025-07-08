#!/bin/bash
# Comprehensive script to run all experiments: baselines and adaptive strategies

echo "🔬 Starting comprehensive Adaptive LoRA experiments..."
echo "This will run baseline experiments (rank 8, 16) and adaptive strategies."
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# Common parameters
DATASET_SIZE=200
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=3e-4
SEED=42

echo "📋 Experiment Parameters:"
echo "  Dataset Size: $DATASET_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo ""

# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

echo "🏁 Phase 1: Baseline Experiments"
echo "================================="

# Run baseline with rank 8
echo "📊 Running baseline experiment with rank 8..."
python scripts/train_baseline.py \
    --lora_rank 8 \
    --dataset_size $DATASET_SIZE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --experiment_name "baseline_rank8_official"

if [ $? -eq 0 ]; then
    echo "✅ Baseline rank 8 completed successfully"
else
    echo "❌ Baseline rank 8 failed"
    exit 1
fi

# Run baseline with rank 16
echo "📊 Running baseline experiment with rank 16..."
python scripts/train_baseline.py \
    --lora_rank 16 \
    --dataset_size $DATASET_SIZE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --experiment_name "baseline_rank16_official"

if [ $? -eq 0 ]; then
    echo "✅ Baseline rank 16 completed successfully"
else
    echo "❌ Baseline rank 16 failed"
    exit 1
fi

echo "🎯 Baseline experiments completed!"
echo ""

# ============================================================================
# ADAPTIVE EXPERIMENTS
# ============================================================================

echo "🧠 Phase 2: Adaptive Strategy Experiments"
echo "========================================="

# Adaptive parameters
BASE_RANK=16
MIN_RANK=4
MAX_RANK=32

echo "📋 Adaptive Parameters:"
echo "  Base Rank: $BASE_RANK"
echo "  Min Rank: $MIN_RANK"
echo "  Max Rank: $MAX_RANK"
echo ""

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

if [ $? -eq 0 ]; then
    echo "✅ Linear decay strategy completed successfully"
else
    echo "❌ Linear decay strategy failed"
    exit 1
fi

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

if [ $? -eq 0 ]; then
    echo "✅ Attention-heavy strategy completed successfully"
else
    echo "❌ Attention-heavy strategy failed"
    exit 1
fi

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

if [ $? -eq 0 ]; then
    echo "✅ Empirical strategy completed successfully"
else
    echo "❌ Empirical strategy failed"
    exit 1
fi

echo "🧠 Adaptive experiments completed!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "🎉 All experiments completed successfully!"
echo "========================================="
echo ""
echo "📁 Results saved in: results/"
echo "📊 Experiments run:"
echo "  • Baseline rank 8"
echo "  • Baseline rank 16"
echo "  • Adaptive linear decay"
echo "  • Adaptive attention-heavy"
echo "  • Adaptive empirical"
echo ""
echo "📈 Next steps:"
echo "  1. Run analysis: python scripts/analyze_results.py"
echo "  2. Generate figures: python scripts/generate_figures.py"
echo "  3. Review results in each experiment directory"
echo ""
echo "✨ Ready for analysis and paper writing!" 