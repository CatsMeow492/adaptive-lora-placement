# Adaptive LoRA: Experimental Results Analysis

**Generated:** 2025-07-07 21:09:54

## Overview

- **Total Experiments:** 5
- **Strategies Tested:** empirical, linear_decay, attention_heavy, baseline
- **Dataset Size:** 200 examples
- **Training Epochs:** 3

## Summary Statistics

### Performance Metrics by Strategy

                final_eval_loss         final_perplexity          trainable_parameters               parameter_efficiency         training_time_seconds        
                           mean     std             mean      std                 mean           std                 mean     std                  mean     std
strategy                                                                                                                                                       
attention_heavy             NaN     NaN              NaN      NaN            5111808.0           NaN               0.0142     NaN               90.2106     NaN
baseline                 5.1063  0.2947         168.6552  49.0029            4718592.0  2.224366e+06               0.0131  0.0061               84.6257  4.0122
empirical                   NaN     NaN              NaN      NaN            3538944.0           NaN               0.0099     NaN               88.7866     NaN
linear_decay             5.1065     NaN         165.0961      NaN            6291456.0           NaN               0.0174     NaN               83.5101     NaN

## Best Performers

### Lowest Evaluation Loss
- **Strategy:** baseline
- **Loss:** 4.8979
- **Perplexity:** 134.00
- **Parameters:** 6,291,456

### Most Parameter Efficient
- **Strategy:** baseline
- **Efficiency:** 0.9%
- **Loss:** 5.3147
- **Parameters:** 3,145,728

## Strategy Rankings

### Final Eval Loss
1. **Baseline**: 5.1063
2. **Linear Decay**: 5.1065
3. **Attention Heavy**: nan
4. **Empirical**: nan

### Final Perplexity
1. **Linear Decay**: 165.0961
2. **Baseline**: 168.6552
3. **Attention Heavy**: nan
4. **Empirical**: nan

### Parameter Efficiency
1. **Empirical**: 0.0099
2. **Baseline**: 0.0131
3. **Attention Heavy**: 0.0142
4. **Linear Decay**: 0.0174

### Training Time Seconds
1. **Linear Decay**: 83.5101
2. **Baseline**: 84.6257
3. **Empirical**: 88.7866
4. **Attention Heavy**: 90.2106

## Detailed Results

| experiment_name                   | strategy        |   final_eval_loss |   final_perplexity |   trainable_parameters | parameter_efficiency   | training_time_seconds   |
|:----------------------------------|:----------------|------------------:|-------------------:|-----------------------:|:-----------------------|:------------------------|
| adaptive_empirical_official       | empirical       |         nan       |            nan     |                3538944 | 1.0%                   | 88.8s                   |
| adaptive_linear_decay_official    | linear_decay    |           5.10653 |            165.096 |                6291456 | 1.7%                   | 83.5s                   |
| adaptive_attention_heavy_official | attention_heavy |         nan       |            nan     |                5111808 | 1.4%                   | 90.2s                   |
| baseline_rank8_official           | baseline        |           5.31471 |            203.305 |                3145728 | 0.9%                   | 87.5s                   |
| baseline_rank16_official          | baseline        |           4.89788 |            134.005 |                6291456 | 1.7%                   | 81.8s                   |

## Key Findings

1. **Best Performance:** Baseline achieved the lowest evaluation loss (5.1063)
2. **Best Efficiency:** Empirical achieved the highest parameter efficiency (1.0%)
3. **Adaptive vs Baseline:** Adaptive strategies show -0.0% degradation in average loss

