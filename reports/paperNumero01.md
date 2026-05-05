# 01 - Towards Detecting and Classifying Network Intrusion Traffic Using Deep Learning Frameworks

> NIDS = Network Intrusion Detection System

This paper uses deep learning algorithms to detect network intrusion. The authors used various state-of-the-art deep learning frameworks like **Keras**, **TensorFlow**, **Theano**, **fast\.ai**, and **PyTorch**. They reported accuracy of about 99% using **fast\.ai** with low false positive and negative rates in both detecting and classifying various intrusion types. 

> NIDS differs from firewall, since it monitors the incoming traffic and alarms if a thread is detected, whereas a firewall only allows/blocks traffic from trusted/unknown devices.
### Two types of NIDS presented
- **SNIDS (signature-based NIDS)**: they look for patterns in network traffic and compares them to preinstalled rules. They are good for detecting already known attacks, meaning they fail to detect new, previously unrecognized attacks. 
- **ANIDS (anomaly-based NIDS)**: they rely on various statistical machine learning models to automatically learn patterns and create rules that distinguish normal traffic from malicious attacks. 

**ANIDS** are well suited for detecting new or previously unknown attacks, but they are limited by the dataset used to train its model.







---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---







## **Methodology**

The study uses a modern and realistic dataset called **CSE-CIC-IDS2018**, which contains labeled network traffic including both normal behavior and multiple types of attacks. This dataset is important because it reflects real-world traffic conditions better than older datasets, making the results more relevant.

The methodology involves training and evaluating deep learning models across several popular frameworks:
- Keras (with TensorFlow backend)
- TensorFlow (standalone)
- Theano
- PyTorch
- fast.ai (a high-level wrapper over PyTorch)

The models are trained to perform two main tasks:
1. Binary classification (benign vs malicious traffic)
2. Multi-class classification (identifying specific attack types)

Before training, the dataset undergoes preprocessing steps such as normalization and formatting into suitable input structures for neural networks. The models are evaluated using standard metrics like accuracy, false positive rate, and false negative rate, which are critical in security contexts.

To highlight the most impactful aspects of the methodology:
- **Dataset choice matters**  
	The use of CSE-CIC-IDS2018 is a major strength because it includes modern attack types and realistic traffic patterns. This directly improves the reliability of the results compared to outdated datasets like KDD99.
- **Framework comparison is central**  
	Instead of proposing a new algorithm, the study compares implementations across frameworks. This is useful because performance differences may come from optimization, usability, and training efficiency rather than model design alone.
- **Focus on both detection and classification**  
	Many works focus only on detecting attacks, but this paper also classifies them. This adds practical value since knowing the type of attack is essential for response strategies.
	
## **Results**

The results show that deep learning models perform extremely well in both detecting and classifying network intrusions. Among all tested frameworks, **fast.ai achieved the best performance**, reaching around **99% accuracy** with low false positives and false negatives.

Key observations include:
- Deep learning models are highly effective at identifying malicious traffic.
- Performance differences exist between frameworks, even when similar models are used.
- fast.ai stands out due to its simplicity and strong performance, likely because it builds on PyTorch while providing optimized training workflows.

To emphasize the most important findings:
- **High accuracy across the board**  
    All frameworks achieved strong results, confirming that deep learning is well-suited for intrusion detection tasks.
- **fast.ai performs best**  
    Its combination of ease-of-use and optimized training pipelines likely contributed to superior results.
- **Low error rates are critical**  
    The low false positive and false negative rates make these models practical, since high false alarms would reduce usability in real systems.

## **Conclusion and Future Work**

The paper demonstrates that deep learning is a powerful tool for network intrusion detection and classification. It also shows that the choice of framework can impact performance, not just the model itself. The success of fast.ai suggests that higher-level tools can simplify development while still achieving excellent results.

From a critical perspective, the paper is strong in its experimental comparison and use of a modern dataset. However, it focuses heavily on accuracy metrics and less on deployment challenges such as real-time processing, scalability, and interpretability. In real-world cybersecurity systems, these factors are just as important as raw performance.

Some areas that could be improved or expanded:
- **Real-world deployment considerations**  
    The paper does not deeply address how these models would perform in live network environments with streaming data.
- **Model interpretability**  
    Deep learning models are often black boxes. Understanding why a decision is made is important in cybersecurity but not explored here.
- **Comparison with non-deep learning methods**  
    While the focus is on deep learning frameworks, comparing against traditional machine learning models would provide more context on the actual improvement gained.

For future work, research could explore lightweight models for real-time intrusion detection, hybrid systems combining deep learning with rule-based approaches, and explainable AI techniques to make decisions more transparent.
