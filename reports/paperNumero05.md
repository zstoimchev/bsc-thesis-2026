# 05 - A Deep Learning Approach for Network Intrusion Detection System

This paper proposes a **deep learning-based approach** for detecting network intrusions using a method called **Self-taught Learning (STL)**. The idea is to improve anomaly-based intrusion detection by automatically learning useful features from data instead of manually selecting them.

> The main advantage of this approach is that it can use large amounts of **unlabeled data**, which is easier to obtain than labeled data.

As in previous work, the focus is on **ANIDS (anomaly-based systems)**, since they are capable of detecting **unknown and new attacks**, unlike signature-based systems.

The core of the paper is the **Self-taught Learning approach**, which works in two stages.

First, the model performs **unsupervised feature learning** using a **sparse auto-encoder**. In this stage, the network learns to represent input data in a compressed way, extracting important patterns automatically. Then, in the second stage, these learned features are used for **classification** using **softmax regression**. This means that instead of training directly on raw features, the model first transforms the data into a better representation, which improves classification performance.

The authors use the **NSL-KDD dataset**, which is an improved version of the older KDD’99 dataset. It contains labeled network traffic with both normal and attack records, grouped into categories such as:
- **DoS**
- **Probe**
- **R2L**
- **U2R**
- and **normal traffic**

The models are trained to perform three types of classification:
- **Binary classification** (normal vs attack)
- **5-class classification** (normal + 4 attack categories)
- **23-class classification** (normal + all attack types)

The evaluation is done in two ways, but the important one is using **separate training and test data**, since the test set contains **previously unseen attacks**, making the evaluation more realistic.

The models are evaluated using standard metrics like *accuracy*, *precision*, *recall*, and *F-measure*.

The results show that the deep learning approach performs very well, especially for **binary classification**.

For **2-class classification (normal vs attack)**:
- Accuracy is around **88% on test data**
- This is significantly better than basic classifiers like softmax without feature learning (~78%)

For **5-class classification**:
- Accuracy is around **79%**
- Performance is lower than binary classification, as expected

On training data (cross-validation), accuracy exceeds **98%**, but this is less realistic.

An important observation is the tradeoff between **precision and recall**.

The STL model achieves:
- **Higher recall** → detects more attacks  
- Slightly **lower precision** → more false positives  

Because of this, the overall **F-measure is better**, meaning the model is more effective at detecting intrusions even if it raises more alarms.

The authors conclude that **automatic feature learning using deep learning** significantly improves intrusion detection. The model is especially effective when labeled data is limited, since it can learn from unlabeled data first.
