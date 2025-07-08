# Adaptive LoRA: Layerwise Rank Allocation for Parameter-Efficient Fine-Tuning

### Repository Purpose  
This repository continues our previous research on **selective LoRA placement** by exploring a new direction in parameter-efficient fine-tuning (PEFT): **adaptive rank allocation**. The goal is to investigate whether varying LoRA rank *per transformer layer* can improve the trade-off between fine-tuning performance and parameter efficiency, relative to fixed-rank baselines.

The final deliverable will be a **ready-to-submit arXiv paper**, including experimental results, visualizations, and conclusions. This repo is structured to support both human researchers and autonomous agents tasked with executing the work.

---

## 🔁 Background and Continuity

### Previous Work
Our earlier paper — *Selective LoRA: Systematic Placement Strategies for Parameter-Efficient Fine-Tuning* — demonstrated that applying LoRA only to attention layers preserved performance while reducing parameter count, and that feed-forward-only LoRA yielded the highest parameter savings at a greater performance cost. The conclusion: **layer type matters** for efficient adaptation.

> 📄 [Selective LoRA Paper](https://github.com/CatsMeow492/parameter-efficient-fine-tuning-of-large-models/blob/master/papers/arxiv_draft.md)

### Current Focus
This project builds directly on that foundation by asking:  
**"Can we go beyond layer *type* and strategically allocate *different LoRA ranks* to each individual layer to further optimize efficiency-performance trade-offs?"**

We aim to:
- Systematically compare adaptive rank allocation strategies to fixed-rank baselines
- Understand how per-layer capacity impacts learning dynamics
- Produce actionable guidelines for practitioners deploying LoRA in constrained environments

---

## 📋 Research Plan (Agent-Friendly 10 Step Guide)

This is the action plan an agent or human assistant should follow.

### 1. Define Research Question
> Can adaptive (non-uniform) LoRA rank allocation across layers outperform fixed-rank LoRA in efficiency or performance?

### 2. Choose Model and Dataset
- Model: `DialoGPT-medium` (361M)  
- Dataset: `Alpaca` (or Dolly, ~100–200 example subset)  
- Goal: Match prior experimental conditions for comparability

### 3. Set Up Fixed-Rank Baseline Experiments
- Ranks: 8, 16
- Apply to all layers (`c_attn`, `c_proj`, `c_fc`)
- Measure: loss, perplexity, parameter count, training time

### 4. Design Adaptive Rank Strategies
Example strategies:
- Linear decay from input to output layers
- High-rank for attention layers, low-rank for feed-forward
- Empirical rank allocation from activation magnitudes or parameter norms

### 5. Run Adaptive Experiments
- Repeat training with adaptive ranks
- Log results in consistent format (e.g., JSON, CSV, Weights & Biases)

### 6. Analyze and Compare
- Compare performance and efficiency across strategies
- Chart: rank distribution vs loss
- Highlight trade-offs in parameter count vs perplexity

### 7. Write the Paper
Outline based on prior structure:
1. Introduction
2. Related Work (AdaLoRA, PEFT, LoRA, Selective LoRA)
3. Methodology (adaptive rank heuristics)
4. Experiments (setup, baselines, results)
5. Conclusion (findings, limitations, future work)

### 8. Create Figures
- Bar chart: trainable parameters vs evaluation loss
- Line chart: rank allocation per layer
- Table: strategy comparison

### 9. Polish and Validate
- Ensure reproducibility
- Run grammar and spell checks
- Ensure all figures and tables match text references

### 10. Prepare for arXiv Submission
- Convert paper to Overleaf (arXiv LaTeX template)
- Upload code snapshot to Zenodo or GitHub archive
- Verify formatting and compliance

---

## 📁 Suggested Directory Structure
adaptive-lora/
├── data/                # Preprocessed Alpaca samples
├── models/              # LoRA configurations per strategy
├── results/             # Evaluation logs and plots
├── scripts/             # Training and evaluation scripts
├── paper/               # Draft, figures, LaTeX files
│   ├── draft.md
│   ├── figures/
│   └── main.tex
├── README.md            # You’re here
└── requirements.txt     # Reproducible environment
---

## 🔍 Reference Baseline (from Previous Work)

| Strategy          | Loss   | Params (M) | Reduction | Perplexity |
|------------------|--------|------------|-----------|------------|
| Full LoRA        | 3.089  | 6.29       | 0%        | 4,283      |
| Attention Only   | 3.481  | 4.33       | 31.2%     | 2,272      |
| Feed-Forward Only| 4.252  | 1.97       | 68.8%     | 3,391      |

---

## 📌 Notes for Agent Operation

- Maintain reproducibility: fixed seeds, same hardware if possible
- Save checkpoints and evaluation logs
- Use the `paper/` directory to incrementally write as results become available
- Document any heuristic or design choices with a short justification in `EXPERIMENTS.md`

---

## 🧠 Suggested Papers to Read
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2022)  
- **AdaLoRA: Adaptive Budget Allocation** (Zhang et al., 2023)  
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)  
- **Selective LoRA** (our prior work)  

---

## 📤 End Goal
Produce a self-contained, reproducible paper titled something like:

> **"Adaptive LoRA: Layerwise Rank Allocation for Efficient Transformer Fine-Tuning"**

with an arXiv-ready draft, figures, and repo structure ready for open publication.

---

## 📫 Contact
**Lead Researcher:** Taylor Mohney  
**Affiliation:** University of Nevada, Las Vegas  
**Email:** mohney@unlv.nevada.edu