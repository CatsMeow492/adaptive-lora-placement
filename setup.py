#!/usr/bin/env python3
"""
Setup script for Adaptive LoRA research project.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-lora",
    version="0.1.0",
    description="Adaptive LoRA: Layerwise Rank Allocation for Parameter-Efficient Fine-Tuning",
    author="Taylor Mohney",
    author_email="mohney@unlv.nevada.edu",
    url="https://github.com/taylormohney/adaptive-lora-placement",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest>=7.0.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 