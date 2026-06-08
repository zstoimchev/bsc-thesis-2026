# 15 - Deep Structured Energy Based Models for Anomaly Detection

> EBM = Energy Based Model  
> DSEBM = Deep Structured Energy Based Model  
> DSEBM-e = DSEBM using energy score  
> DSEBM-r = DSEBM using reconstruction error


This paper proposes Deep Structured Energy Based Models for anomaly detection. The main idea is to model the distribution of normal data using an energy function. Normal samples should have low energy, while anomalous samples should have high energy.

The paper is not specifically an intrusion detection or DDoS detection paper. It is a general anomaly detection paper. It is still relevant to this thesis because it includes KDD99 as one benchmark dataset and because intrusion detection can be viewed as an anomaly detection problem.

Energy Based Models define a probability distribution through an energy function. Samples with lower energy correspond to higher probability, while samples with higher energy correspond to lower probability. Therefore, anomaly detection can be performed by flagging samples whose energy is above a chosen threshold.

The authors extend traditional Energy Based Models into deep architectures. They propose three types of DSEBM depending on the structure of the input data. For static or tabular data, they use fully connected networks. For sequential data, they use recurrent neural networks. For spatial/image data, they use convolutional neural networks.

Training traditional EBMs can be difficult because maximum likelihood estimation requires dealing with the partition function, which is usually intractable. To avoid this, the authors use score matching. This connects EBMs with regularized denoising autoencoders and allows the model to be trained using standard stochastic gradient descent instead of complex sampling procedures.

The paper evaluates two anomaly detection criteria. The first is the energy score, where a sample is treated as anomalous if its energy is higher than a threshold. The second is reconstruction error, where a sample is treated as anomalous if the reconstruction error is higher than a threshold.

The energy-score criterion is usually stronger. The paper explains that reconstruction error can fail in some situations because an outlier can lie near a local maximum of the energy surface, where the gradient is small. In such cases, reconstruction error may incorrectly treat the outlier as normal, while the energy score can still detect it.

The experiments are performed on three groups of datasets. Static/tabular experiments use KDD99 10%, Thyroid, and Usenet. Sequential experiments use CUAVE, NATOPS, and FITNESS. Image/spatial experiments use Caltech-101, MNIST, and CIFAR-10.

For the static datasets, DSEBM is compared with PCA, Kernel PCA, KDE, RKDE, One-Class SVM, and autoencoder-based outlier detection. On KDD99, DSEBM-e achieves the best F1-score among the compared methods, with precision 0.8619, recall 0.6446, and F1-score 0.7399.

The results show that DSEBM-e often performs better than DSEBM-r and is usually competitive with or better than traditional anomaly detection baselines. However, it is not the best method on every dataset. For example, on Thyroid, kernel-based methods such as One-Class SVM and RKDE are very competitive.

The main contribution of the paper is showing that deep energy-based models can be used for anomaly detection across different data structures: tabular, sequential, and image data. The paper also shows that energy score is often a better anomaly criterion than reconstruction error.

However, the paper has important limitations for this thesis. It is not DDoS-specific and does not evaluate modern DDoS datasets such as CICDDoS2019. It does not propose an IDS deployment pipeline and does not focus on practical network traffic preprocessing. The KDD99 result is useful but limited because KDD99 is old and the reported F1-score is much lower than the headline accuracies reported in many IDS papers.

For my research, Paper 15 should be grouped under Deep Unsupervised Anomaly Detection / Energy-Based Models. It is useful as theoretical background, but it should not be one of the main representative implementations. Its main value is showing that anomaly detection can be approached through deep density/energy modeling, not only through supervised classification or clustering.
