# 05 - A Deep Learning Approach for Network Intrusion Detection System

> NIDS = Network Intrusion Detection System  
> STL = Self-Taught Learning  
> SAE = Sparse Autoencoder  
> SMR = Softmax Regression


This paper proposes a deep learning-based approach for network intrusion detection using Self-Taught Learning. The main idea is to improve anomaly-based intrusion detection by automatically learning useful feature representations instead of relying only on manually selected features.

The proposed method uses a two-stage process. In the first stage, unsupervised feature learning is performed using a sparse autoencoder. The autoencoder learns a compressed representation of the input data and extracts useful patterns from the network traffic features. In the second stage, the learned representation is used for supervised classification using softmax regression.

The authors focus on anomaly-based NIDS because signature-based systems are effective for known attacks but cannot reliably detect new or unknown attacks. Anomaly-based systems are more flexible because they learn patterns from data, although they can also produce higher false positives.

The paper uses the NSL-KDD dataset, which is an improved version of KDD’99. The dataset contains normal traffic and attack traffic grouped into DoS, Probe, R2L, and U2R categories. Each record originally contains 41 features. Three nominal features are converted using 1-to-n encoding, one feature that always has value zero is removed, and after preprocessing the dataset has 121 attributes. The values are then normalized using min-max normalization.

The models are evaluated on three classification tasks. The first is 2-class classification, where traffic is classified as normal or attack. The second is 5-class classification, where the classes are normal, DoS, Probe, R2L, and U2R. The third is 23-class classification, where the model predicts normal traffic or one of the individual attack types.

The authors evaluate the model in two ways. First, they use 10-fold cross-validation on the training data for 2-class, 5-class, and 23-class classification. In this setting, STL achieves more than 98% accuracy for all classification tasks. However, this result is less realistic because it uses only the training data.

The more important evaluation uses separate training and test data. In this setting, the paper reports results for 2-class and 5-class classification. For 2-class classification, STL achieves 88.39% accuracy, compared with 78.06% for softmax regression without feature learning. For 5-class classification, STL achieves 79.10% accuracy, compared with 75.23% for softmax regression.

An important observation is the tradeoff between precision and recall. For 2-class classification, STL has lower precision than softmax regression, but much higher recall. This means that STL detects more attacks, although it may also produce more false positives. For intrusion detection, this can still be useful because missing attacks is often more dangerous than raising extra alarms.

The main contribution of the paper is showing that automatic feature learning with a sparse autoencoder can improve intrusion detection performance compared with directly applying softmax regression to the raw features.

However, the paper has limitations. Although Self-Taught Learning can theoretically use large amounts of unlabeled data, this experiment uses the same NSL-KDD training data as unlabeled data during the feature-learning phase. The paper also relies on NSL-KDD, which is an older benchmark dataset and may not represent modern DDoS traffic well. The paper is therefore more useful as an autoencoder/representation-learning IDS study than as a direct DDoS detection representative.

For my research, Paper 05 should be grouped under Autoencoder / Representation Learning. It is useful background for understanding feature learning, but it is probably not one of the main papers unless an autoencoder-based model is selected for implementation.
