# 01 - Towards Detecting and Classifying Network Intrusion Traffic Using Deep Learning Frameworks

> NIDS = Network Intrusion Detection System

This paper uses deep learning algorithms to detect network intrusion. The authors used various state-of-the-art deep learning frameworks like **Keras**, **TensorFlow**, **Theano**, **fast\.ai**, and **PyTorch**. They reported accuracy of about 99% using **fast\.ai** with low false positive and negative rates in both detecting and classifying various intrusion types. 

> NIDS differs from firewall, since it monitors the incoming traffic and alarms if a thread is detected, whereas a firewall only allows/blocks traffic from trusted/unknown devices.

### Two types of NIDS presented
- **SNIDS (signature-based NIDS)**: they look for patterns in network traffic and compares them to preinstalled rules. They are good for detecting already known attacks, meaning they fail to detect new, previously unrecognized attacks. 
- **ANIDS (anomaly-based NIDS)**: they rely on various statistical machine learning models to automatically learn patterns and create rules that distinguish normal traffic from malicious attacks. 

**ANIDS** are well suited for detecting new or previously unknown attacks, but they are limited by the dataset used to train its model.

The authors are interested in how well a **hypothetical ANIDS** model can be trained using popular, high level ML frameworks (mentioned above)...

**Goal: show that it is possible to train deep neural networks using the frameworks mentioned above and that further progression can be made without a lot of effort owing to the rapid progress and advancement of ML APIs.**

Since many related works used the old **KDD '99** dataset, the authors used newly published **CSE-CIC-IDS2018** dataset, that contains seven of the most common attack classes along with benign (normal) traffic, all labeled. The dataset contained the following traffic:
- **Brute force attacks**
- **Denial of Service (DoS) attacks**
- **Bot attacks**
- **Web attacks**
- **Infiltration attacks**
- **Benign traffic**

They trained models to perform two main tasks:
1. **Binary classification** (benign vs malicious traffic). Attack traffic was labeled as 1, and benign traffic as 0
2. **Multi-class classification** (identifying specific attack types). The labels were left as they were (text), or converted into one-hot encoding depending on framework

They used two common approaches to evaluate the models:
1. **n-fold cross-validation** (normally 10-fold) and
2. **train-test split** (normally 70-30 or 80-20)

The models were evaluated using standard metrics like *accuracy*, *false positive rate*, and *false negative rate*. 

The results showed that deep learning models perform extremely well in both detecting and classifying network intrusions. **fast\.ai** **achieved the best performance** with **99% accuracy**, low false positives and low false negatives.
