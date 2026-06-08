# 06 - A GRU deep learning system against attacks in software defined networks

> SDN = Software-Defined Networking  
> GRU = Gated Recurrent Unit  
> IDS = Intrusion Detection System  
> DDoS = Distributed Denial-of-Service


This paper proposes an SDN defense system for detecting and mitigating DDoS and intrusion attacks. The main idea is to analyze individual IP flow records instead of using larger sampled time windows. This allows the system to detect attacks faster and respond more directly.

The paper focuses on Software-Defined Networking because SDN centralizes network control in a controller. This makes network management more flexible, but it also creates a critical failure point. If the SDN controller is attacked, the whole network can be affected. Because of this, the authors propose a defense system that runs close to the SDN controller and protects it from DDoS and intrusion attacks.

The proposed system has two main modules. The Detection module analyzes IP flow records and classifies each flow as normal or abnormal. The Mitigation module uses the detection result to generate a directed drop policy, which blocks traffic from the detected attacker.

The key contribution is the use of single-flow analysis. Many anomaly detection systems analyze traffic in time windows, such as several seconds or minutes. This can hide stealthier attacks and delay the response. In contrast, this paper analyzes each IP flow separately, which allows faster detection and makes it possible to identify the attacker’s source information.

The Detection module uses a GRU deep learning model. GRU is a recurrent neural network architecture that can learn dependencies in sequential data while being simpler and faster than LSTM. The model architecture contains a GRU layer with 32 cells, a dropout layer with rate 0.5, a fully connected layer with 10 neurons, and a final sigmoid output neuron for binary classification.

The task is binary classification. The model does not identify the specific attack type; it only classifies each flow as normal or abnormal.

The paper evaluates the system on two datasets. The first is CICDDoS2019, which contains multiple DDoS attack types. The second is CICIDS2018, which contains broader intrusion scenarios and is more difficult because the attacks are more diverse and stealthier.

For CICDDoS2019, the dataset originally contains 87 IP flow features. The authors use 83 features after removing source IP address, destination IP address, source port, and Flow ID to avoid bias. Destination port is kept because many network services use default ports, which can be useful for detection. Qualitative features such as protocol are converted into numerical values before classification.

The GRU model is compared with several other methods, including DNN, CNN, LSTM, SVM, Logistic Regression, kNN, and Gradient Descent. The evaluation uses accuracy, precision, recall, F-measure, and the number of flows per second each method can process. Runtime speed is important because the system is intended for near-real-time defense.

The results show that most methods perform very well on CICDDoS2019 because many DDoS attacks are flooding-based and easier to distinguish from normal traffic. GRU performs slightly better and gives one of the most balanced results across accuracy, precision, recall, and F-measure.

On CICIDS2018, the task is harder because the network is more complex and the attacks are more stealthy. GRU achieves the best overall balance of detection metrics and also performs well in terms of flow-processing speed.

The Mitigation module uses the detected attacker information to generate a direct drop policy for the SDN controller. This is lightweight because it does not require probabilistic estimation; it directly blocks the source associated with the malicious flow.

The paper’s main conclusion is that GRU is a strong candidate for SDN-based DDoS and intrusion detection because it provides a good balance between detection performance, implementation simplicity, and runtime feasibility.

However, the paper also has limitations. The system performs binary detection only and does not classify the exact attack type. The mitigation module depends heavily on detection quality, because false positives could block legitimate traffic. The approach also assumes that blocking based on attacker IP information is practical, which may be more difficult in real-world DDoS scenarios with spoofed or distributed sources. The comparison also does not include some strong tabular baselines such as Random Forest, XGBoost, or LightGBM.

For my research, Paper 06 is highly relevant. It should be considered one of the main candidate papers for the GRU / recurrent deep learning representative. Compared with older GRU papers based on KDD99 or NSL-KDD, this paper is more directly connected to modern DDoS detection because it uses CICDDoS2019 and CICIDS2018.
