# 11.1 - On the Analysis of Open Source Datasets: Validating IDS Implementation for Well-Known and Zero-Day Attack Detection

This paper extends the previous ANIDINR work by focusing on open-source IDS datasets, implementation details, prediction performance, and validation against well-known and zero-day-style attacks.

The proposed system is an anomaly-based IDS designed to detect well-known attacks and possible zero-day attacks. The system is divided into two main phases. The first phase is the Training phase, where machine learning predictors are generated from public datasets. The second phase consists of Extraction and Prediction. The Extraction process converts PCAP files into CSV files containing statistical network traffic metrics. The Prediction process loads trained predictors into memory and classifies new CSV records.

The paper uses KDD99, NSL-KDD, and CIC-IDS2018. Compared with the previous ANIDINR paper, this version adds CIC-IDS2018 and compares Python and R implementations. The authors argue that many IDS studies do not describe dataset cleaning and training methodology clearly enough, making comparison and reproduction difficult.

The tested machine learning models include Support Vector Machine, Random Forest, and XGBoost. The paper also compares prediction time and shows that Python implementations are generally faster than R implementations.

The evaluation does not rely only on accuracy. The paper emphasizes False Alarm Rate, Undetected Attack, and prediction time. FAR measures benign traffic incorrectly detected as an anomaly, while UA measures malicious traffic that is not detected as an attack. These metrics are important because high accuracy alone can hide serious IDS problems.

The CIC-IDS2018 results show very high accuracy for several sub-datasets, especially using XGBoost. However, the paper also highlights that accuracy alone is not enough for evaluating IDS performance.

A major contribution of the paper is the validation platform for zero-day-style attacks. The authors execute six attacks that were not recorded in the initial datasets, including DoS and scanning attacks. The results show that zero-day traffic can be detected as anomalous, but attack classification can still be wrong. This is important because wrong classification can lead to incorrect automatic mitigation decisions.

The main limitation is that zero-day detection is still uncertain. If a zero-day attack produces traffic that looks similar to legitimate traffic, the IDS may fail to detect it. Also, if the system misclassifies the attack type, automatic response may be unsafe.

For my research, Paper 11.1 is useful because it focuses on reproducible IDS implementation, dataset cleaning, prediction time, FAR, UA, and practical deployment issues. It is not mainly a new model architecture paper, but it is important for methodology.

---

# 11.2 - Training Guidance with KDD Cup 1999 and NSL-KDD Data Sets of ANIDINR: Anomaly-Based Network Intrusion Detection System

This paper presents the earlier ANIDINR training methodology for an anomaly-based network intrusion detection system. The goal is to provide step-by-step guidance for training Machine and Deep Learning models on KDD99 and NSL-KDD.

ANIDINR is designed as a passive defense system for detecting well-known and zero-day computer network attacks. The architecture contains a Packet Sniffer module, Training phase, Prediction module, Notification module, and Decision Making module.

The paper compares five models: Support Vector Machine, Random Forest, XGBoost, Neuralnet, and Keras. The main goal is not only to maximize accuracy, but also to minimize Undetected Attack, False Alarm Rate, and testing time.

The authors argue that accuracy is not a sufficient IDS metric because the datasets are imbalanced. A model can have high accuracy while still producing many false alarms or missing important attacks. Therefore, FAR, UA, and testing time are more useful for practical IDS evaluation.

The training process includes dataset cleaning, encoding, normalization, handling null values, correlation analysis, Random Forest feature importance analysis, and random undersampling to reduce class imbalance. The authors also note that random undersampling has drawbacks, such as information loss and possible distortion of the original distribution.

The results show that Random Forest and XGBoost perform strongly. On KDD99, Random Forest achieves the highest accuracy, but XGBoost has much faster testing time. On NSL-KDD, Random Forest again has slightly higher accuracy, while XGBoost remains much faster and has strong UA performance.

The paper concludes that XGBoost trained on KDD99 gives the best overall practical performance when considering accuracy, FAR, UA, and testing time together.

The paper also discusses limitations of zero-day detection. A zero-day attack is not present in the training dataset, so the IDS can only detect it if its traffic profile differs from legitimate traffic. If the zero-day traffic looks similar to normal traffic, the system may classify it as legitimate. If the system detects it but assigns the wrong attack class, the mitigation response may also be wrong.

For my research, Paper 11.2 is useful as methodology background. It supports the idea that IDS evaluation should include FAR, UA, and testing time, not only accuracy. It also supports the choice of XGBoost and Random Forest as strong classical ML baselines.
