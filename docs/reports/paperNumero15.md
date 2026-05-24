# 15 - Deep Structured Energy Based Models for Anomaly Detection

> EBM = Energy Based Model  
> DSEBM = Deep Structured Energy Based Model

This paper proposes a deep learning approach for **anomaly detection** using **Energy Based Models (EBMs)**. The main idea is that normal data should have **low energy values**, while anomalous or abnormal data should produce **high energy values**.

The authors extend traditional EBMs into deep architectures, creating **Deep Structured Energy Based Models (DSEBMs)**.

The paper introduces three types of deep architectures for anomaly detection:
- fully connected networks
- recurrent neural networks (RNNs)
- convolutional neural networks (CNNs)

This allows the model to work with:
- static/tabular data
- sequential/time-series data
- image/spatial data

Instead of using Maximum Likelihood Estimation (MLE), the models are trained using **score matching**, which simplifies the training process significantly.

The authors evaluate two anomaly detection criteria: **energy score** and **reconstruction error**

The experiments show that the **energy score consistently performs better** than reconstruction error for detecting anomalies.

The intuition is simple: even if an anomaly looks visually similar to normal data, its internal energy representation may still be abnormal.

The models are tested on multiple datasets, including:
- sequential datasets (CUAVE, NATOPS, FITNESS)
- image datasets (MNIST, CIFAR-10, Caltech-101)

The proposed models are compared with several traditional anomaly detection approaches such as:
- Kernel PCA
- One-Class SVM
- Robust Kernel Density Estimation
- Hidden Markov Models

The best-performing model is **DSEBM-e** (energy-based criterion).

On image datasets, it achieves:
- **96.89% F1-score on MNIST**
- **87.84% F1-score on CIFAR-10**

The model also outperforms previous methods on sequential datasets, achieving the highest overall F1-scores in most experiments.

An important result in the paper is that DSEBM performs well across very different data types, showing that the approach is highly flexible.

The authors conclude that deep energy-based models are effective for anomaly detection because they combine:
- strong representation learning from deep learning
- probabilistic modeling from energy-based methods
