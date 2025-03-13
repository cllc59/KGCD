# KGCD: Knowledge Enhanced Global Graph Contrastive Denoising for Recommendation

![Paper](https://img.shields.io/badge/Paper-ICIC%202025-blue)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)

This repository contains the official implementation of ​**KGCD**, a novel knowledge-aware recommendation framework that addresses sparse interaction signals and KG noise through contrastive denoising and global graph learning.


## Key Features
- 🧠 ​**Neighbor-aware Smoothing**: Dynamically aggregates semantic features from KG
- 🌐 ​**Path-aware GNN**: Captures global topological patterns in user-item interactions
- 🔍 ​**Contrastive Denoising**: Reduces KG noise through multi-view contrastive learning
- 🚀 ​**Knowledge Enhancement**: Improves tail node representations via relation aggregation


## Requirements
The code has been tested running under Python 3.8.0. Required packages:
- numpy == 1.22.4
- pandas == 1.4.3
- scikit-learn == 1.1.1
- scipy == 1.7.0
- networkx == 2.5.1
- tqdm == 4.64.1  
- torch == 1.10.1+cu113
- torch-cluster == 1.5.9+pt110cu113
- torch-scatter == 2.0.9+pt110cu113
- torch-sparse == 0.6.12+pt110cu113
- torch-geometric == 1.7.2 

