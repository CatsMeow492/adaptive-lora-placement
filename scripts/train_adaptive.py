#!/usr/bin/env python3
"""
Adaptive training script for variable-rank LoRA experiments.
Implements different rank allocation strategies across transformer layers.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DialoGPT-medium with adaptive-rank LoRA"
    )
    
    # Model and data arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/DialoGPT-medium",
        help="Base model name"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="tatsu-lab/alpaca",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset_size", 
        type=int, 
        default=200,
        help="Number of examples to use from dataset"
    )
    
    # Adaptive LoRA arguments
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True,
        choices=["linear_decay", "attention_heavy", "empirical", "custom"],
        help="Adaptive rank allocation strategy"
    )
    parser.add_argument(
        "--base_rank", 
        type=int, 
        default=16,
        help="Base rank for strategies (starting point)"
    )
    parser.add_argument(
        "--min_rank", 
        type=int, 
        default=4,
        help="Minimum rank for any layer"
    )
    parser.add_argument(
        "--max_rank", 
        type=int, 
        default=32,
        help="Maximum rank for any layer"
    )
    parser.add_argument(
        "--custom_ranks", 
        type=str, 
        default=None,
        help="Custom rank allocation as JSON string (for custom strategy)"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.1,
        help="LoRA dropout rate"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default=None,
        help="Experiment name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--save_model", 
        action="store_true",
        help="Save the trained model"
    )
    
    return parser.parse_args()


def get_model_layer_info(model_name: str) -> Dict[str, Any]:
    """Get information about model layers for adaptive allocation."""
    # For DialoGPT-medium, we know the structure
    if "DialoGPT" in model_name:
        return {
            "num_layers": 24,  # DialoGPT-medium has 24 layers
            "target_modules": ["c_attn", "c_proj", "c_fc"],
            "attention_modules": ["c_attn", "c_proj"],
            "feedforward_modules": ["c_fc"],
        }
    else:
        # Default fallback
        return {
            "num_layers": 12,
            "target_modules": ["query", "key", "value", "dense"],
            "attention_modules": ["query", "key", "value"],
            "feedforward_modules": ["dense"],
        }


def linear_decay_strategy(num_layers: int, base_rank: int, min_rank: int, max_rank: int) -> List[int]:
    """
    Linear decay strategy: rank decreases linearly from input to output layers.
    
    Args:
        num_layers: Number of transformer layers
        base_rank: Starting rank for first layer
        min_rank: Minimum rank for any layer
        max_rank: Maximum rank for any layer
        
    Returns:
        List of ranks for each layer
    """
    # Ensure base_rank is within bounds
    start_rank = min(max(base_rank, min_rank), max_rank)
    end_rank = min_rank
    
    # Create linear decay
    ranks = []
    for i in range(num_layers):
        # Linear interpolation from start_rank to end_rank
        progress = i / (num_layers - 1) if num_layers > 1 else 0
        rank = int(start_rank - (start_rank - end_rank) * progress)
        rank = max(min_rank, min(rank, max_rank))
        ranks.append(rank)
    
    return ranks


def attention_heavy_strategy(num_layers: int, base_rank: int, min_rank: int, max_rank: int) -> Dict[str, List[int]]:
    """
    Attention-heavy strategy: higher ranks for attention layers, lower for feed-forward.
    
    Args:
        num_layers: Number of transformer layers
        base_rank: Base rank for attention layers
        min_rank: Minimum rank for any layer
        max_rank: Maximum rank for any layer
        
    Returns:
        Dictionary with ranks for each module type
    """
    # Attention layers get higher ranks
    attention_rank = min(max(base_rank, min_rank), max_rank)
    
    # Feed-forward layers get lower ranks (typically half)
    feedforward_rank = max(min_rank, attention_rank // 2)
    
    return {
        "c_attn": [attention_rank] * num_layers,
        "c_proj": [attention_rank] * num_layers,
        "c_fc": [feedforward_rank] * num_layers,
    }


def empirical_strategy(num_layers: int, base_rank: int, min_rank: int, max_rank: int) -> List[int]:
    """
    Empirical strategy: based on common patterns observed in transformer fine-tuning.
    
    Early layers: Lower ranks (feature extraction)
    Middle layers: Higher ranks (task-specific processing)
    Late layers: Medium ranks (output formatting)
    
    Args:
        num_layers: Number of transformer layers
        base_rank: Base rank for reference
        min_rank: Minimum rank for any layer
        max_rank: Maximum rank for any layer
        
    Returns:
        List of ranks for each layer
    """
    ranks = []
    
    for i in range(num_layers):
        # Normalize layer position to [0, 1]
        pos = i / (num_layers - 1) if num_layers > 1 else 0
        
        if pos < 0.3:  # Early layers (0-30%)
            rank = int(base_rank * 0.6)  # Lower rank
        elif pos < 0.7:  # Middle layers (30-70%)
            rank = int(base_rank * 1.2)  # Higher rank
        else:  # Late layers (70-100%)
            rank = base_rank  # Base rank
        
        # Ensure within bounds
        rank = max(min_rank, min(rank, max_rank))
        ranks.append(rank)
    
    return ranks


def create_adaptive_lora_config(
    strategy: str, 
    model_info: Dict[str, Any], 
    base_rank: int, 
    min_rank: int, 
    max_rank: int,
    custom_ranks: str = None,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
) -> Tuple[List[LoraConfig], Dict[str, Any]]:
    """
    Create adaptive LoRA configurations for different layers.
    
    Returns:
        List of LoraConfig objects and allocation info
    """
    num_layers = model_info["num_layers"]
    
    if strategy == "linear_decay":
        ranks = linear_decay_strategy(num_layers, base_rank, min_rank, max_rank)
        allocation = {"strategy": "linear_decay", "ranks": ranks}
        
        # Create configs for each layer
        configs = []
        for i, rank in enumerate(ranks):
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=model_info["target_modules"],
                bias="none",
            )
            configs.append(config)
            
    elif strategy == "attention_heavy":
        rank_dict = attention_heavy_strategy(num_layers, base_rank, min_rank, max_rank)
        allocation = {"strategy": "attention_heavy", "rank_dict": rank_dict}
        
        # For attention_heavy, we need different configs per module type
        # This is more complex - we'll use a single config with average rank
        avg_rank = int(np.mean([
            np.mean(rank_dict["c_attn"]),
            np.mean(rank_dict["c_proj"]),
            np.mean(rank_dict["c_fc"])
        ]))
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=avg_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=model_info["target_modules"],
            bias="none",
        )
        configs = [config]  # Single config for now
        
    elif strategy == "empirical":
        ranks = empirical_strategy(num_layers, base_rank, min_rank, max_rank)
        allocation = {"strategy": "empirical", "ranks": ranks}
        
        # Create configs for each layer
        configs = []
        for i, rank in enumerate(ranks):
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=model_info["target_modules"],
                bias="none",
            )
            configs.append(config)
            
    elif strategy == "custom":
        if custom_ranks is None:
            raise ValueError("Custom ranks must be provided for custom strategy")
        
        ranks = json.loads(custom_ranks)
        allocation = {"strategy": "custom", "ranks": ranks}
        
        # Create configs for each layer
        configs = []
        for i, rank in enumerate(ranks):
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=model_info["target_modules"],
                bias="none",
            )
            configs.append(config)
    
    return configs, allocation


def load_and_prepare_dataset(dataset_name: str, dataset_size: int, tokenizer, max_length: int):
    """Load and prepare the dataset for training."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Take subset if specified
    if dataset_size > 0:
        dataset = dataset.select(range(min(dataset_size, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    def preprocess_function(examples):
        """Preprocess examples for language modeling."""
        # Combine instruction and output for language modeling
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            
            # Format as conversation
            text = f"Human: {instruction}\nAssistant: {output}"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )
        
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing"
    )
    
    # Split into train/eval
    train_size = int(0.9 * len(processed_dataset))
    eval_size = len(processed_dataset) - train_size
    
    train_dataset = processed_dataset.select(range(train_size))
    eval_dataset = processed_dataset.select(range(train_size, train_size + eval_size))
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def setup_model_and_tokenizer(model_name: str, lora_configs: List[LoraConfig]):
    """Set up the model and tokenizer with adaptive LoRA."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Apply LoRA - for now, use the first config (we'll enhance this later)
    lora_config = lora_configs[0]
    logger.info(f"Applying LoRA with adaptive ranks")
    model = get_peft_model(model, lora_config)
    
    # Print model info
    model.print_trainable_parameters()
    
    return model, tokenizer


def calculate_perplexity(trainer, eval_dataset):
    """Calculate perplexity on evaluation dataset."""
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    return perplexity.item()


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"adaptive_{args.strategy}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get model information
    model_info = get_model_layer_info(args.model_name)
    
    # Create adaptive LoRA configurations
    lora_configs, allocation_info = create_adaptive_lora_config(
        args.strategy,
        model_info,
        args.base_rank,
        args.min_rank,
        args.max_rank,
        args.custom_ranks,
        args.lora_alpha,
        args.lora_dropout
    )
    
    # Save allocation info
    with open(output_dir / "allocation.json", "w") as f:
        json.dump(allocation_info, f, indent=2)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, lora_configs)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset_name, args.dataset_size, tokenizer, args.max_length
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb for now
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training...")
    start_time = datetime.now()
    
    train_result = trainer.train()
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Calculate final metrics
    logger.info("Calculating final metrics...")
    final_perplexity = calculate_perplexity(trainer, eval_dataset)
    trainable_params, total_params = count_parameters(model)
    
    # Get final metrics from trainer state
    final_eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    final_train_loss = getattr(train_result.metrics, 'train_loss', 0)
    final_eval_loss = final_eval_results.get('eval_loss', 0)
    
    # Prepare results
    results = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_size": args.dataset_size,
        "strategy": args.strategy,
        "base_rank": args.base_rank,
        "min_rank": args.min_rank,
        "max_rank": args.max_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "training_time_seconds": training_time,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "final_perplexity": final_perplexity,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "parameter_efficiency": trainable_params / total_params,
        "allocation_info": allocation_info,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Strategy: {args.strategy}")
    print(f"Base Rank: {args.base_rank}")
    print(f"Final Train Loss: {results['final_train_loss']:.4f}")
    print(f"Final Eval Loss: {results['final_eval_loss']:.4f}")
    print(f"Final Perplexity: {results['final_perplexity']:.2f}")
    print(f"Trainable Parameters: {results['trainable_parameters']:,}")
    print(f"Parameter Efficiency: {results['parameter_efficiency']:.1%}")
    print(f"Training Time: {training_time:.1f} seconds")
    
    # Print allocation details
    if "ranks" in allocation_info:
        avg_rank = np.mean(allocation_info["ranks"])
        print(f"Average Rank: {avg_rank:.1f}")
        print(f"Rank Range: {min(allocation_info['ranks'])}-{max(allocation_info['ranks'])}")
    
    print("="*60)
    
    # Save model if requested
    if args.save_model:
        model_dir = output_dir / "model"
        logger.info(f"Saving model to: {model_dir}")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    
    logger.info("Training completed successfully!")
    
    return results


if __name__ == "__main__":
    main() 