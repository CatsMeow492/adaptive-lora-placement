#!/usr/bin/env python3
"""
Baseline training script for fixed-rank LoRA experiments.
Trains DialoGPT-medium with fixed LoRA rank across all layers.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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
        description="Train DialoGPT-medium with fixed-rank LoRA"
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
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        required=True,
        help="LoRA rank (e.g., 8, 16)"
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


def setup_model_and_tokenizer(model_name: str, lora_config: LoraConfig):
    """Set up the model and tokenizer with LoRA."""
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
    
    # Apply LoRA
    logger.info(f"Applying LoRA with rank {lora_config.r}")
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
        args.experiment_name = f"baseline_rank{args.lora_rank}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],  # DialoGPT specific
        bias="none",
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, lora_config)
    
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
        "lora_rank": args.lora_rank,
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
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Final Train Loss: {results['final_train_loss']:.4f}")
    print(f"Final Eval Loss: {results['final_eval_loss']:.4f}")
    print(f"Final Perplexity: {results['final_perplexity']:.2f}")
    print(f"Trainable Parameters: {results['trainable_parameters']:,}")
    print(f"Parameter Efficiency: {results['parameter_efficiency']:.1%}")
    print(f"Training Time: {training_time:.1f} seconds")
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