# Project Brief: Adaptive LoRA Placement

## Project Outline
This research project investigates **adaptive rank allocation** for LoRA (Low-Rank Adaptation) in parameter-efficient fine-tuning of large language models. Building on our previous work on "Selective LoRA," we explore whether varying LoRA rank *per transformer layer* can improve the trade-off between fine-tuning performance and parameter efficiency.

## Core Requirements
1. **Systematic Comparison**: Compare adaptive rank allocation strategies to fixed-rank baselines
2. **Reproducible Experiments**: Use DialoGPT-medium (361M) with Alpaca dataset subset
3. **Multiple Strategies**: Implement and evaluate various adaptive rank allocation methods
4. **Academic Paper**: Produce a ready-to-submit arXiv paper with experimental results
5. **Code Repository**: Maintain clean, reproducible codebase structure

## Success Criteria
- Completion of baseline experiments (fixed ranks 8, 16)
- Implementation of at least 3 adaptive rank strategies
- Comprehensive analysis with visualizations
- Draft paper ready for arXiv submission
- Reproducible code with proper documentation

## Stakeholders
- **Lead Researcher**: Taylor Mohney (University of Nevada, Las Vegas)
- **Primary Contact**: mohney@unlv.nevada.edu
- **Target Audience**: PEFT research community, practitioners using LoRA

## Constraints
- **Model Size**: DialoGPT-medium (361M parameters) for computational feasibility
- **Dataset**: Alpaca subset (100-200 examples) for comparison with prior work
- **Computational**: Limited to reasonable training times and resources
- **Academic Timeline**: Target arXiv submission within research cycle

## Timeline
- **Phase 1**: Environment setup and baseline experiments
- **Phase 2**: Adaptive strategy implementation and testing
- **Phase 3**: Analysis, visualization, and paper writing
- **Phase 4**: Final review and arXiv submission preparation

## Research Question
**"Can adaptive (non-uniform) LoRA rank allocation across layers outperform fixed-rank LoRA in efficiency or performance?"** 