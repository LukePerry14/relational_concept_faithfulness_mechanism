# Concept Bottleneck Model for Relational Deep Learning

This package implements an interpretable concept bottleneck model for temporal relational graphs, designed to work with the RelBench benchmark.

## Overview

The model learns interpretable **concept prototypes** that correspond to meta-path patterns in relational data. Each concept captures:
- **Relation sequence**: Which node types appear in the path (e.g., driver → results → race)
- **Time pattern**: Expected temporal relationships between nodes
- **Feature pattern**: Expected feature vectors at each position
- **Tolerance bounds**: How much variation is allowed (gamma values)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Relational Entity Graph                   │
│  (from RelBench: rel-f1, rel-amazon, rel-stack, etc.)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Meta-Path Sampling                         │
│  • Schema-guided: only valid transitions                     │
│  • Temporal-aware: respects seed time constraints           │
│  • Biased: ensures rare path types represented              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Concept Bottleneck                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐   │
│  │ ConceptDecoder │→ │ EvidenceScorer │→ │  LogicHead   │   │
│  │  (prototypes)  │  │  (similarity)  │  │ (fuzzy DNF)  │   │
│  └────────────────┘  └────────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        Prediction
```

## Files

```
cbm_relbench/
├── data_pipeline.py          # Schema extraction, meta-path enumeration, sampling
├── cbm_dataset.py            # PyTorch Dataset for CBM training
├── train_cbm.py              # Training script with model definitions
├── test_mock_relf1.py        # Mock test without RelBench dependency
├── test_relf1_integration.py # Integration test with real RelBench data
└── README.md                 # This file
```


## Model Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_hops` | 2 | Maximum path length (hops from root) |
| `num_concepts` | 8 | Number of learnable concept prototypes |
| `concept_dim` | 64 | Dimension of concept embeddings |
| `feature_dim` | 128 | Dimension of node features |
| `samples_per_schema` | 4 | Paths sampled per meta-path type |
| `max_paths` | 64 | Maximum paths per seed node |


## References

- [RelBench Paper](https://arxiv.org/abs/2407.XXXXX)
- [RelGT Paper](https://arxiv.org/abs/2505.10960)
- [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612)
