# 02 - Detection and Characterization of DDoS Attacks Using Time-Based Features

This paper focuses on detecting and classifying Distributed Denial-of-Service (DDoS) attacks using machine learning and deep learning models. The main idea is that DDoS attacks have strong temporal patterns, so a reduced set of time-based flow features may be enough to detect attacks efficiently.

The authors use the CICDDoS2019 dataset, which contains benign traffic and multiple modern DDoS attack types. Although the original dataset contains more than 70 million samples, it is highly imbalanced, with a very small amount of benign traffic compared to attack traffic. Because of this, the authors clean, down-sample, and balance the data before training.

The preprocessing includes removing unnecessary or problematic features, such as features containing only zero values, identifier features that could cause overfitting, duplicate/unlabeled features, and samples with missing or infinite values. Normalization is applied after the train-test split to avoid introducing bias.

The paper compares two feature sets. The first is the control feature set, which contains 70 cleaned features. The second is the reduced time-based feature set, which contains only 25 features. These features include timing-related flow statistics such as flow duration, forward and backward inter-arrival times, flow bytes per second, and flow packets per second.

The experiments are divided into two scenarios. Scenario A is binary classification, where the model predicts whether traffic is benign or DDoS. Scenario B is multi-class classification, where the model identifies the specific DDoS attack type. Scenario B does not include benign traffic; it only classifies among DDoS attack types such as SYN Flood, UDP Flood, LDAP, DNS, NetBIOS, SNMP, NTP, SSDP, MSSQL, TFTP, Portmap, and UDP-Lag.

The authors compare nine classifiers. These include Random Forest, K-Nearest Neighbors, Support Vector Machine, Gaussian Naive Bayes, Linear Discriminant, LightGBM, XGBoost, AdaBoost, and a deep neural network implemented with fast.ai.

The models are evaluated using accuracy, precision, F1-score, AUC, and training time. Training time is especially important because the paper focuses on whether reduced time-based features can support near-real-time or continuous learning systems.

The results show that time-based features are very effective for binary DDoS detection. In Scenario A, the best models achieve more than 99% accuracy using all 70 features and more than 98% accuracy using only the 25 time-based features. The accuracy drop is small, while the median training time is reduced by around 36%.

For multi-class DDoS attack characterization, the task is harder. In Scenario B, XGBoost achieves the best result, with around 74% accuracy using all features and around 69% accuracy using only time-based features. This shows that the reduced feature set is still useful, but identifying the exact DDoS attack type is more difficult than detecting whether traffic is benign or malicious.

The most important conclusion is that time-based features provide a good tradeoff between accuracy and efficiency. XGBoost gives the best overall accuracy, while LightGBM and Linear Discriminant provide a strong speed-performance balance. For binary DDoS detection, time-based features are especially promising because they reduce training time while keeping high accuracy.

The paper also has limitations. The dataset was heavily downsampled, some attack types such as low-rate DDoS were not tested, the original flow interval settings of CICDDoS2019 were not fully clear, and only limited hyperparameter tuning was performed. Because of this, the results should be interpreted carefully and re-evaluated on a common dataset.

For my research, this paper is highly relevant. It should be used as a strong representative of classical machine learning and boosting-based DDoS detection. It is especially useful as a baseline for comparing deep learning models against XGBoost, LightGBM, or Random Forest on the same prepared DDoS superset.
