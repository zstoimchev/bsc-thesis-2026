# **A Reproducible Evaluation and Deployment-Oriented Re-Implementation of Modern IDS/DDoS Detection Models Under Realistic Feature, Split, and Dataset Constraints**

This repository contains the code for my thesis experiments on IDS/DDoS ML models.

Obsius link for hot-reload: [https://obsius.site/126j5t6m2q324f3f666y](https://obsius.site/126j5t6m2q324f3f666y)

## Overview

Existing IDS/DDoS models often report very high accuracy, but these results are strongly affected by dataset choice, preprocessing, feature representation, class balance, and train/test splitting. Therefore, I will evaluates whether representative top-performing IDS architectures remain reliable under a unified, reproducible, deployment-oriented evaluation pipeline. In a nutshell:

## Abstract

Intrusion Detection Systems and Distributed Denial-of-Service detection models often report very high accuracy in the literature, especially when evaluated on benchmark datasets such as CIC-IDS2018, CIC-DDoS2019, NSL-KDD, and KDD99. However, these results are frequently influenced by dataset choice, preprocessing decisions, feature representation, class imbalance, and train/test split strategy. This thesis focuses on the reproducible and deployment-oriented re-implementation of selected IDS/DDoS detection models under realistic experimental constraints.

The work begins with an analysis of existing IDS/DDoS research papers and their corresponding implementations. Based on this review, representative models are selected from different levels of complexity: a simple baseline model, a strong classical machine learning model such as XGBoost or LightGBM, and a neural network baseline such as an MLP/DNN. A GRU-based model is considered as an optional advanced architecture if a meaningful temporal flow representation is used.

The selected models are re-implemented in a unified codebase and evaluated using consistent preprocessing, feature handling, train/test splitting, and metrics. The main focus is placed on CIC-like flow-based datasets, since they are more suitable for realistic firewall or IDS deployment than legacy NSL/KDD-style datasets. The evaluation investigates whether high reported performance remains stable when models are trained and tested under the same conditions, and whether reduced deployment-oriented feature sets affect detection performance.

The goal of this thesis is not only to compare model accuracy, but also to examine reproducibility, generalization, dataset bias, and practical suitability for real-world IDS/DDoS detection.

I will re-implement representative IDS/DDoS detection models from the given literature in an unified pipeline and evaluates whether their reported high accuracy remains stable under fair train/test splits, common preprocessing, feature-space constraints, and deployment-oriented CIC-like flow data. The work focuses primarily on CIC-like flow features suitable for firewall/IDS deployment, while NSL-like data is used as a secondary comparison to demonstrate feature-space incompatibility and dataset bias.

