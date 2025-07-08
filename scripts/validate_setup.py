#!/usr/bin/env python3
"""
Validation script to test all components of the adaptive LoRA system.
Checks dependencies, validates configurations, and tests core functionality.
"""

import sys
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any

def check_imports() -> bool:
    """Check if all required packages are available."""
    print("🔍 Checking package imports...")
    
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully!")
    return True


def validate_adaptive_strategies() -> bool:
    """Validate the adaptive strategy implementations."""
    print("\n🧠 Validating adaptive strategies...")
    
    try:
        # Import strategy functions
        sys.path.append('scripts')
        from train_adaptive import (
            linear_decay_strategy,
            attention_heavy_strategy,
            empirical_strategy,
            get_model_layer_info
        )
        
        # Test parameters
        num_layers = 24  # DialoGPT-medium
        base_rank = 16
        min_rank = 4
        max_rank = 32
        
        # Test linear decay
        linear_ranks = linear_decay_strategy(num_layers, base_rank, min_rank, max_rank)
        assert len(linear_ranks) == num_layers
        assert all(min_rank <= r <= max_rank for r in linear_ranks)
        assert linear_ranks[0] >= linear_ranks[-1]  # Should decay
        print(f"  ✅ Linear decay: {linear_ranks[:3]}...{linear_ranks[-3:]} (range: {min(linear_ranks)}-{max(linear_ranks)})")
        
        # Test attention heavy
        attention_ranks = attention_heavy_strategy(num_layers, base_rank, min_rank, max_rank)
        assert all(len(ranks) == num_layers for ranks in attention_ranks.values())
        print(f"  ✅ Attention heavy: c_attn={attention_ranks['c_attn'][0]}, c_fc={attention_ranks['c_fc'][0]}")
        
        # Test empirical
        empirical_ranks = empirical_strategy(num_layers, base_rank, min_rank, max_rank)
        assert len(empirical_ranks) == num_layers
        assert all(min_rank <= r <= max_rank for r in empirical_ranks)
        print(f"  ✅ Empirical: {empirical_ranks[:3]}...{empirical_ranks[-3:]} (range: {min(empirical_ranks)}-{max(empirical_ranks)})")
        
        # Test model info
        model_info = get_model_layer_info("microsoft/DialoGPT-medium")
        assert model_info['num_layers'] == 24
        assert 'c_attn' in model_info['target_modules']
        print(f"  ✅ Model info: {model_info['num_layers']} layers, modules: {model_info['target_modules']}")
        
        print("✅ All adaptive strategies validated!")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive strategy validation failed: {e}")
        return False


def validate_directory_structure() -> bool:
    """Validate the project directory structure."""
    print("\n📁 Validating directory structure...")
    
    required_dirs = [
        'data',
        'models', 
        'results',
        'scripts',
        'paper',
        'paper/figures',
        '.memory'
    ]
    
    required_files = [
        'requirements.txt',
        'setup.py',
        'install.sh',
        'scripts/train_baseline.py',
        'scripts/train_adaptive.py',
        'scripts/prepare_data.py',
        'scripts/analyze_results.py',
        'scripts/run_baselines.sh',
        'scripts/run_adaptive.sh',
        'scripts/run_all_experiments.sh'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_items.append(f"Directory: {dir_path}")
        else:
            print(f"  ✅ {dir_path}/")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_items.append(f"File: {file_path}")
        else:
            print(f"  ✅ {file_path}")
    
    if missing_items:
        print(f"\n❌ Missing items:")
        for item in missing_items:
            print(f"    - {item}")
        return False
    
    print("✅ Directory structure validated!")
    return True


def test_baseline_script() -> bool:
    """Test baseline script argument parsing."""
    print("\n🏁 Testing baseline script...")
    
    try:
        sys.path.append('scripts')
        from train_baseline import parse_args
        
        # Test with minimal args
        test_args = ['--lora_rank', '8']
        import argparse
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = ['train_baseline.py'] + test_args
        
        try:
            args = parse_args()
            assert args.lora_rank == 8
            assert args.model_name == "microsoft/DialoGPT-medium"
            print(f"  ✅ Argument parsing: rank={args.lora_rank}, model={args.model_name}")
        finally:
            sys.argv = original_argv
        
        print("✅ Baseline script validated!")
        return True
        
    except Exception as e:
        print(f"❌ Baseline script validation failed: {e}")
        return False


def test_adaptive_script() -> bool:
    """Test adaptive script argument parsing."""
    print("\n🧠 Testing adaptive script...")
    
    try:
        sys.path.append('scripts')
        from train_adaptive import parse_args
        
        # Test with minimal args
        test_args = ['--strategy', 'linear_decay']
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = ['train_adaptive.py'] + test_args
        
        try:
            args = parse_args()
            assert args.strategy == 'linear_decay'
            assert args.base_rank == 16
            print(f"  ✅ Argument parsing: strategy={args.strategy}, base_rank={args.base_rank}")
        finally:
            sys.argv = original_argv
        
        print("✅ Adaptive script validated!")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive script validation failed: {e}")
        return False


def check_memory_bank() -> bool:
    """Check memory bank completeness."""
    print("\n🧠 Checking memory bank...")
    
    memory_files = [
        '.memory/01-brief.md',
        '.memory/10-product.md', 
        '.memory/20-system.md',
        '.memory/30-tech.md',
        '.memory/40-active.md',
        '.memory/50-progress.md',
        '.memory/60-decisions.md',
        '.memory/70-knowledge.md'
    ]
    
    for file_path in memory_files:
        if not Path(file_path).exists():
            print(f"  ❌ Missing: {file_path}")
            return False
        else:
            print(f"  ✅ {file_path}")
    
    print("✅ Memory bank complete!")
    return True


def main():
    """Run all validation tests."""
    print("🔬 Adaptive LoRA Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Package imports", check_imports),
        ("Directory structure", validate_directory_structure),
        ("Memory bank", check_memory_bank),
        ("Adaptive strategies", validate_adaptive_strategies),
        ("Baseline script", test_baseline_script),
        ("Adaptive script", test_adaptive_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validations passed! System ready for experiments.")
        print("\nNext steps:")
        print("  1. Run experiments: ./scripts/run_all_experiments.sh")
        print("  2. Analyze results: python scripts/analyze_results.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please fix issues before running experiments.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 