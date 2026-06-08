# 01 - Towards Detecting and Classifying Network Intrusion Traffic Using Deep Learning Frameworks

> NIDS = Network Intrusion Detection System  
> ANIDS = Anomaly-based Network Intrusion Detection System  
> SNIDS = Signature-based Network Intrusion Detection System


This paper studies the use of deep learning frameworks for detecting and classifying network intrusion traffic. The authors compare several popular deep learning frameworks, including Keras, TensorFlow, Theano, fast.ai, and PyTorch, using the CSE-CIC-IDS2018 dataset.

The main goal of the paper is not to propose a completely new neural network architecture, but to show that modern deep learning frameworks can be used effectively for building anomaly-based intrusion detection models. The models are mainly simple feedforward deep neural networks / multilayer perceptrons.

The paper explains the difference between signature-based and anomaly-based NIDS. Signature-based systems detect known attacks by matching traffic against predefined rules, but they usually fail to detect new or unknown attacks. Anomaly-based systems learn normal and malicious behavior from data, which gives them better potential for detecting new attacks, but their performance depends heavily on the quality of the training dataset.

The authors use the CSE-CIC-IDS2018 dataset because many older IDS studies rely on outdated datasets such as KDD’99, which contain redundant records, outdated attacks, and traffic that may not represent modern networks well.

The dataset contains benign traffic and several attack categories, including brute force attacks, DoS/DDoS attacks, bot attacks, web attacks, SQL injection, infiltration, and benign traffic. The authors use the labeled CSV files generated from CICFlowMeter features.

Before training, the dataset is cleaned. Samples with missing, NaN, or infinite values are removed, timestamps are converted to numeric Unix epoch values, repeated headers are removed, and around 20,000 problematic samples are dropped. The cleaned datasets contain 79 features. Destination Port and Protocol are treated as categorical features, while the remaining features are numeric.

The authors train models for two tasks. The first task is binary classification, where traffic is classified as benign or malicious. The second task is multi-class classification, where the model predicts the specific attack type.

The models are evaluated using accuracy, false positive rate, false negative rate, and confusion matrices. The authors also compare CPU and GPU training times.

The results show that fast.ai performs best overall. It achieves around 98.68% accuracy for binary classification and around 98.31% accuracy for multi-class classification. Some individual datasets reach nearly 100% accuracy.

However, the results are not equally strong for all attack categories. The infiltration dataset is more difficult, with lower attack detection and higher false positives compared to the other datasets. This shows that the headline “99% accuracy” should be interpreted carefully, because overall accuracy can hide weaker performance on specific attack types.

The paper also reports practical training challenges. Some experiments were computationally expensive, and GPU acceleration did not always help as much as expected because of a possible data pipeline bottleneck.

For my research, Paper 01 is useful as a representative of the DNN / MLP / feedforward neural network group. It is a good baseline deep learning paper, but it is not the strongest DDoS-specific representative because it studies general intrusion detection, not only DDoS detection.
