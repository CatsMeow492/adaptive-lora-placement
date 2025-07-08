#!/usr/bin/env python3
"""
Data preprocessing script for the Alpaca dataset.
Handles loading, preprocessing, and saving the dataset for experiments.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Alpaca dataset for training"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name to load"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=200,
        help="Size of subset to create (0 for full dataset)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Tokenizer to use for preprocessing"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    return parser.parse_args()


def load_alpaca_dataset(dataset_name: str, subset_size: int = 0) -> Dataset:
    """Load the Alpaca dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Take subset if specified
    if subset_size > 0:
        logger.info(f"Taking subset of size {subset_size}")
        dataset = dataset.select(range(min(subset_size, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)}")
    return dataset


def analyze_dataset(dataset: Dataset) -> Dict[str, Any]:
    """Analyze the dataset and return statistics."""
    logger.info("Analyzing dataset...")
    
    # Convert to pandas for analysis
    df = dataset.to_pandas()
    
    # Basic statistics
    stats = {
        "total_examples": len(df),
        "columns": list(df.columns),
        "instruction_lengths": {
            "mean": df["instruction"].str.len().mean(),
            "median": df["instruction"].str.len().median(),
            "max": df["instruction"].str.len().max(),
            "min": df["instruction"].str.len().min(),
        },
        "output_lengths": {
            "mean": df["output"].str.len().mean(),
            "median": df["output"].str.len().median(),
            "max": df["output"].str.len().max(),
            "min": df["output"].str.len().min(),
        },
    }
    
    # Input field analysis (if present)
    if "input" in df.columns:
        stats["input_lengths"] = {
            "mean": df["input"].str.len().mean(),
            "median": df["input"].str.len().median(),
            "max": df["input"].str.len().max(),
            "min": df["input"].str.len().min(),
        }
        stats["has_input_proportion"] = (df["input"].str.len() > 0).mean()
    
    return stats


def format_examples(dataset: Dataset) -> List[str]:
    """Format examples for language modeling."""
    logger.info("Formatting examples...")
    
    formatted_texts = []
    for example in dataset:
        instruction = example["instruction"]
        output = example["output"]
        
        # Handle input field if present
        if "input" in example and example["input"].strip():
            text = f"Human: {instruction}\n\nContext: {example['input']}\n\nAssistant: {output}"
        else:
            text = f"Human: {instruction}\n\nAssistant: {output}"
        
        formatted_texts.append(text)
    
    return formatted_texts


def tokenize_dataset(texts: List[str], tokenizer, max_length: int) -> Dict[str, List]:
    """Tokenize the formatted texts."""
    logger.info("Tokenizing dataset...")
    
    # Tokenize all texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    # For language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def split_dataset(dataset: Dataset, train_split: float, seed: int) -> tuple:
    """Split dataset into train and validation sets."""
    logger.info(f"Splitting dataset: {train_split:.1%} train, {1-train_split:.1%} validation")
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    
    # Split
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def save_dataset_info(output_dir: Path, stats: Dict[str, Any], args: argparse.Namespace):
    """Save dataset information and statistics."""
    info = {
        "dataset_name": args.dataset_name,
        "subset_size": args.subset_size,
        "train_split": args.train_split,
        "seed": args.seed,
        "tokenizer_name": args.tokenizer_name,
        "max_length": args.max_length,
        "statistics": stats,
        "preprocessing_args": vars(args),
    }
    
    info_file = output_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to: {info_file}")


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_alpaca_dataset(args.dataset_name, args.subset_size)
    
    # Analyze dataset
    stats = analyze_dataset(dataset)
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Columns: {stats['columns']}")
    print(f"Instruction length - Mean: {stats['instruction_lengths']['mean']:.1f}, Max: {stats['instruction_lengths']['max']}")
    print(f"Output length - Mean: {stats['output_lengths']['mean']:.1f}, Max: {stats['output_lengths']['max']}")
    if "has_input_proportion" in stats:
        print(f"Examples with input: {stats['has_input_proportion']:.1%}")
    print("="*50)
    
    # Format examples
    formatted_texts = format_examples(dataset)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    tokenized_data = tokenize_dataset(formatted_texts, tokenizer, args.max_length)
    
    # Create dataset from tokenized data
    tokenized_dataset = Dataset.from_dict(tokenized_data)
    
    # Split dataset
    train_dataset, eval_dataset = split_dataset(tokenized_dataset, args.train_split, args.seed)
    
    # Save datasets
    train_file = output_dir / "train_dataset.json"
    eval_file = output_dir / "eval_dataset.json"
    
    logger.info(f"Saving train dataset to: {train_file}")
    train_dataset.to_json(train_file)
    
    logger.info(f"Saving eval dataset to: {eval_file}")
    eval_dataset.to_json(eval_file)
    
    # Save raw formatted texts for reference
    formatted_file = output_dir / "formatted_texts.json"
    with open(formatted_file, "w") as f:
        json.dump(formatted_texts, f, indent=2)
    
    logger.info(f"Formatted texts saved to: {formatted_file}")
    
    # Save dataset info
    save_dataset_info(output_dir, stats, args)
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    logger.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main() 