# Adaptive LoRA: Layerwise Rank Allocation for Parameter-Efficient Fine-Tuning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents a comprehensive empirical study of **adaptive rank allocation** in LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. We investigate whether varying LoRA rank per transformer layer can improve the trade-off between fine-tuning performance and parameter efficiency compared to fixed-rank baselines.

## 🔬 Research Question & Key Findings

**Research Question**: *Can adaptive (non-uniform) LoRA rank allocation across layers outperform fixed-rank LoRA in efficiency or performance?*

### 📊 **Key Results**

| Strategy | Eval Loss | Perplexity | Parameters (M) | Efficiency | Training Time |
|----------|-----------|------------|----------------|------------|---------------|
| **Baseline Rank 16** | **4.90** | **134.0** | **6.3** | **1.7%** | **81.8s** |
| Linear Decay | 5.11 | 165.1 | 6.3 | 1.7% | 83.5s |
| Baseline Rank 8 | 5.31 | 203.3 | 3.1 | 0.9% | 87.5s |
| Attention-Heavy | NaN | NaN | 5.1 | 1.4% | 90.2s ⚠️ |
| Empirical | NaN | NaN | 3.5 | 1.0% | 88.8s ⚠️ |

### 🎯 **Main Findings**
1. **Fixed-rank LoRA (rank 16) achieved best performance** on DialoGPT-medium
2. **Linear decay adaptive strategy was stable** but performed 23% worse than best baseline
3. **Complex adaptive strategies experienced training instability** (NaN gradients)
4. **Gradual rank variation** (linear decay) more stable than dramatic changes
5. **Adaptive strategies require specialized optimization techniques** for practical deployment

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/TaylorMohney/adaptive-lora-placement.git
cd adaptive-lora-placement

# Install dependencies
pip install -r requirements.txt

# Run all experiments (takes ~30 minutes)
bash scripts/run_all_experiments.sh

# Generate analysis and figures
python scripts/analyze_results.py

# View results
open results/analysis/analysis_report.md
```

## 📋 Background & Previous Work

This work builds on our previous research on **selective LoRA placement**, which demonstrated that layer type matters for efficient adaptation. We extend this by exploring whether individual layers should have different adaptation capacities.

> 📄 **Previous Work**: [Selective LoRA: Systematic Placement Strategies for Parameter-Efficient Fine-Tuning](https://github.com/CatsMeow492/parameter-efficient-fine-tuning-of-large-models/blob/master/papers/arxiv_draft.md)

### Research Motivation
- **Layer Hierarchy**: Different transformer layers capture different types of representations
- **Efficiency Challenge**: Fixed-rank LoRA treats all layers equally
- **Optimization Opportunity**: Can we allocate adaptation capacity more strategically?

## 🔧 Installation & Setup

### Requirements
- Python 3.11+
- PyTorch 2.0+
- CUDA compatible GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/TaylorMohney/adaptive-lora-placement.git
cd adaptive-lora-placement

# Install dependencies
pip install -r requirements.txt

# Validate installation
python scripts/validate_setup.py
```

### Alternative Installation (with conda)
```bash
# Create conda environment
conda create -n adaptive-lora python=3.11
conda activate adaptive-lora

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🧪 Experimental Setup

### Model & Dataset
- **Model**: DialoGPT-medium (361M parameters, 24 layers)
- **Dataset**: Alpaca instruction-following dataset (200 examples)
- **Task**: Conversational response generation
- **Evaluation**: Perplexity, loss, parameter efficiency

### Adaptive Strategies Tested

#### 1. **Linear Decay** 📉
Gradually reduces rank from input to output layers (16→4)
```python
rank_i = max(4, 16 - (16-4) * i / (24-1))
```

#### 2. **Attention-Heavy** 🎯
Higher ranks for attention layers, lower for feed-forward
```python
rank_attention = 16
rank_feedforward = 8
```

#### 3. **Empirical** 🔬
Based on transformer learning patterns (early: 10, middle: 19, late: 16)
```python
# Early layers: 60% of base rank
# Middle layers: 120% of base rank  
# Late layers: 100% of base rank
```

## 🏃‍♂️ Running Experiments

### Run Individual Experiments
```bash
# Baseline experiments
python scripts/train_baseline.py --rank 8 --output_dir results/baseline_rank8
python scripts/train_baseline.py --rank 16 --output_dir results/baseline_rank16

# Adaptive experiments  
python scripts/train_adaptive.py --strategy linear_decay --output_dir results/adaptive_linear_decay
python scripts/train_adaptive.py --strategy attention_heavy --output_dir results/adaptive_attention_heavy
python scripts/train_adaptive.py --strategy empirical --output_dir results/adaptive_empirical
```

### Run All Experiments
```bash
# Run everything (takes ~30 minutes)
bash scripts/run_all_experiments.sh

# Generate analysis and figures
python scripts/analyze_results.py

# View comprehensive report
open results/analysis/analysis_report.md
```

### Custom Strategies
```bash
# Create custom rank allocation
python scripts/train_adaptive.py --strategy custom --config_file my_strategy.json
```

Example `my_strategy.json`:
```json
{
  "strategy_name": "custom",
  "layer_ranks": [16, 16, 14, 12, 10, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
}
```

## 📁 Project Structure

```
adaptive-lora-placement/
├── data/                    # Preprocessed Alpaca samples
├── models/                  # LoRA configurations per strategy
├── results/                 # Experimental results and analysis
│   ├── baseline_rank8/      # Baseline rank 8 results
│   ├── baseline_rank16/     # Baseline rank 16 results
│   ├── adaptive_*/          # Adaptive strategy results
│   └── analysis/            # Comprehensive analysis and figures
├── scripts/                 # Training and evaluation scripts
│   ├── train_baseline.py    # Fixed-rank LoRA training
│   ├── train_adaptive.py    # Adaptive rank training
│   ├── prepare_data.py      # Data preprocessing
│   ├── analyze_results.py   # Results analysis
│   └── validate_setup.py    # Installation validation
├── paper/                   # Research paper and figures
│   ├── draft.md            # Complete research paper
│   └── figures/            # Publication-quality figures
├── .memory/                # Memory bank system (internal)
└── requirements.txt        # Python dependencies
```

## 📊 Results & Analysis

### Performance Summary
- **Best Overall**: Baseline Rank 16 (Perplexity: 134.0)
- **Best Adaptive**: Linear Decay (Perplexity: 165.1, stable training)
- **Training Issues**: 2 of 3 adaptive strategies failed with NaN gradients
- **Parameter Efficiency**: All strategies used 0.9-1.7% of total parameters

### Generated Outputs
After running experiments, you'll find:
- **Detailed Results**: `results/analysis/analysis_report.md`
- **Performance Charts**: `results/analysis/performance_comparison.png`
- **Efficiency Plots**: `results/analysis/efficiency_analysis.png`
- **Strategy Comparison**: `results/analysis/strategy_comparison.png`
- **Raw Data**: `results/analysis/combined_results.csv`

### Key Insights for Practitioners
1. **Use fixed-rank LoRA (rank 16)** for production deployments
2. **Linear decay shows promise** but needs optimization improvements
3. **Avoid complex adaptive strategies** without specialized training techniques
4. **Monitor gradient behavior** carefully with adaptive allocation

## 📄 Research Paper

The complete research paper is available in `paper/draft.md` and includes:
- **Abstract & Introduction**: Research motivation and background
- **Methodology**: Detailed description of adaptive strategies
- **Experimental Results**: Comprehensive analysis of all strategies
- **Discussion**: Implications for adaptive LoRA research
- **Conclusion**: Practical recommendations and future work

## 🔗 Related Work & References

### Key Papers
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2022)
- **AdaLoRA**: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) (Zhang et al., 2023)
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- **Selective LoRA**: [Our previous work](https://github.com/CatsMeow492/parameter-efficient-fine-tuning-of-large-models/blob/master/papers/arxiv_draft.md)

### Comparison with Previous Work
| Strategy (Previous) | Loss | Params (M) | Reduction | Perplexity |
|---------------------|------|------------|-----------|------------|
| Full LoRA           | 3.089| 6.29       | 0%        | 4,283      |
| Attention Only      | 3.481| 4.33       | 31.2%     | 2,272      |
| Feed-Forward Only   | 4.252| 1.97       | 68.8%     | 3,391      |

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your adaptive strategy or improvements
4. Submit a pull request

### Ideas for Contributions
- New adaptive rank allocation strategies
- Training stabilization techniques
- Extension to other models (GPT, BERT, etc.)
- Advanced analysis methods
- Performance optimizations

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{mohney2024adaptive,
  title={Adaptive LoRA: Layerwise Rank Allocation for Parameter-Efficient Fine-Tuning},
  author={Mohney, Taylor},
  journal={arXiv preprint arXiv:TBD},
  year={2024}
}
```

## 📫 Contact & Support

- **Lead Researcher**: Taylor Mohney
- **Affiliation**: University of Nevada, Las Vegas
- **Email**: mohney@unlv.nevada.edu
- **GitHub Issues**: For technical questions and bug reports

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository** if you find it useful for your research!