# 14 - Neural Network-Based Voting System with High Capacity and Low Computation for Intrusion Detection in SIEM/IDS Systems

> SIEM = Security Information and Event Management  
> IDS = Intrusion Detection System  
> MLP = Multilayer Perceptron  
> PCA = Principal Component Analysis


This paper proposes a lightweight neural-network-based intrusion detection model for SIEM and IDS systems. The main motivation is that SIEM/IDS platforms already perform resource-intensive tasks such as collecting, normalizing, analyzing, detecting, and sending notifications in real time. Because of this, adding a heavy deep learning model can increase computational cost.

Instead of using one large neural network, the authors combine several small feedforward neural networks using an ensemble voting system. The goal is to increase detection capability without a proportional increase in computational complexity.

The proposed system has three main parts. The first part is data preparation. The second part is a set of weak neural networks. The third part is the ensemble module, which combines the outputs of the weak neural networks using a reliability-based majority mechanism.

The paper uses the NSL-KDD dataset. The task is mainly binary classification, where the model predicts whether a network record is normal or an attack. The paper does not focus on multi-class attack classification; the authors mention multiclassification as future work.

The preprocessing stage is important. The original NSL-KDD records contain 41 features. The authors remove features that previous studies considered unimportant, leaving 29 features. Since the dataset contains categorical features such as protocol type, service, and flag, the authors use one-hot encoding instead of integer encoding. This is important because these categorical values do not have a natural order. After one-hot encoding, each record is represented by 110 coordinates. The authors then apply Z-score normalization and PCA, reducing the representation to 90 dimensions while keeping 99% of the variance.

The weak learners are small feedforward neural networks. The idea is that each network should be simple and should make somewhat different predictions. This is important because ensemble voting only helps if the individual learners are not fully correlated. If all learners make the same mistakes, voting does not improve the final result.

The paper compares several ensemble strategies, including hard voting, weighted average, mixture of experts, and majority function. The best result is obtained using a modified majority function with a reliability mechanism.

The reliability mechanism is used when the classifiers disagree or when the decision is uncertain. Instead of treating every weak learner equally, the system considers which learner was more reliable during training and uses this information to produce the final decision.

The individual neural networks are not very strong on their own. Their reported accuracies are approximately 73%, 59%, and 70%. However, when combined using the reliability-based majority system, the final model improves significantly.

The final ensemble achieves around 89% accuracy, 91% precision, 90% F-score, and 88% recall. The authors compare this with other NSL-KDD approaches and argue that the proposed model is competitive while using lower computational resources. The largest neural network used in the system has only one hidden layer with 25 neurons.

The main contribution of this paper is showing that lightweight neural networks can be combined into a stronger IDS model using ensemble voting. This is useful for SIEM/IDS environments where computational cost matters.

However, the paper has limitations. It uses NSL-KDD, which is an older benchmark dataset and may not represent modern DDoS traffic. The model is evaluated mainly for binary classification, not for multi-class attack-type detection. The paper is also not specifically focused on DDoS detection. In addition, the low-computation claim is reasonable because the neural networks are small, but the paper does not provide as detailed a runtime analysis as some other DDoS-focused studies.

For my research, Paper 14 should be grouped under DNN / MLP / Feedforward and Ensemble Learning. It is useful as background for lightweight neural IDS models, but it should not be one of the main DDoS representative papers.
