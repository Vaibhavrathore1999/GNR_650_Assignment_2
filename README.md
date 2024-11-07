# GNR_650_Assignment_2
# Zero-Shot Learning (ZSL) with Attribute-Based Embeddings

This repository implements a Zero-Shot Learning (ZSL) model for datasets such as AWA1, AWA2, APY, CUB, and SUN. The approach leverages attribute-based embeddings and model regularization techniques to classify unseen classes by learning from seen classes.

## Overview

Zero-Shot Learning (ZSL) allows a model to predict classes it hasn't seen during training by leveraging semantic information about the classes. This project uses attribute embeddings for ZSL and evaluates both standard ZSL accuracy and generalized ZSL (GZSL) performance.

### Key Features

- **Datasets Supported**: AWA1, AWA2, APY, CUB, and SUN
- **Attribute Embedding**: Uses features extracted from ViT (Vision Transformer) models (`vitL14.mat`).
- **Model Regularization**: Includes spectral normalization and class standardization to improve model robustness.
- **Optimizer & Scheduler**: Uses Adam with StepLR learning rate scheduling.

## Setup

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.7
- tqdm
- scipy

### Install Dependencies

```bash
pip install torch tqdm scipy

project-root/
├── IAB-GZSL/
│   └── data/
│       ├── AWA2/
│       │   ├── vitL14.mat
│       │   ├── att_splits.mat
│       └── (other datasets)
├── main.py
└── README.md

python main.py

Model Description
The model architecture includes:

Linear Layers: MLP with hidden dimensions.
Class Standardization: Applies normalization to ensure class attributes are comparable.
Spectral Normalization: Stabilizes weights, reducing sensitivity to outliers.
Attribute Scaling: Scales embeddings to balance ZSL and GZSL performances.

Results
Best Results (for vitL14 features on AWA2):

Hidden dim = 1024, epochs = 100, learning rate = 0.0005
ZSL accuracy for unseen classes = 32.76%