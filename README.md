# **Modeling and Understanding of Modern Network Intrusion Detection Systems**

This repository contains the code for my thesis experiments on IDS/DDoS ML models.

Obsius link for hot-reload: [https://obsius.site/1q5t2k0m4k4c011z1651](https://obsius.site/1q5t2k0m4k4c011z1651)

## Overview

Existing IDS/DDoS models often report very high accuracy, but these results are strongly affected by dataset choice, preprocessing, feature representation, class balance, and train/test splitting. Therefore, I will evaluates whether representative top-performing IDS architectures remain reliable under a unified, reproducible, deployment-oriented evaluation pipeline. In a nutshell:

## Abstract (outline)

Intrusion Detection Systems and Distributed Denial-of-Service detection models often report very high accuracy in the literature, especially when evaluated on benchmark datasets such as the CIC-IDS2018, CIC-DDoS2019, NSL-KDD, and KDD99. However, most of the time, these results can be strongly influenced by dataset choice, preprocessing decisions, feature representation, class imbalance, and train/test split strategy. Therefore, my goal is to examine how selected models perform when they are and evaluated under the same experimental conditions, through the design and implementation of a unified modeling and evaluation pipeline.

The work will begin with an analysis of existing IDS/DDoS research papers and their corresponding implementations. Based on this review, representative models with highest reported performance will be selected from different levels of complexity. The experimental setup will include a simple majority-class baseline model, used to verify the pipeline and expose the effect of class imbalance, a strong classical machine learning model such as XGBoost, and a neural network model such as an MLP/DNN. In addition, an advanced transformer-style model will be considered as a proposed modeling component for flow-based intrusion detection. A GRU-based model may also be included if a meaningful temporal or sequential flow representation is prepared.

The selected models will be implemented in a unified codebase with consistent data loading, preprocessing, feature selection, train/test splitting, and evaluation metrics. The experiments will focus mainly on CIC-like datasets, since they are more suitable for realistic IDS/DDoS detection scenarios than older legacy datasets. The evaluation will use accuracy, precision, recall, F1-score, balanced accuracy, and confusion matrices, with special attention to cases where high accuracy does not correspond to effective attack detection.

The main aim is not only to compare model accuracy, but also to study reproducibility, generalization, dataset bias, feature dependence, and practical suitability of different IDS/DDoS detection models under a unified modeling framework.
