# 06 - A GRU deep learning system against attacks in software defined networks

This paper proposes an SDN defense system that detects DDoS and intrusion attacks from individual IP flow records using a GRU deep learning model, then immediately applies a directed mitigation action to block the attacker’s traffic. Its main contribution is showing that **single-flow analysis** can give fast detection and practical mitigation with good accuracy and feasible processing speed.

The paper addresses a weakness in Software-Defined Networking: the centralized controller is a high-value target, so attacks on it can disrupt the whole network. The authors want to detect attacks earlier and more precisely by analyzing each flow separately instead of using coarse time-window sampling.

> The goal is not only detection, but also fast response. That is why the system has two parts: a Detection module and a Mitigation module.

The Detection module uses a GRU recurrent neural network to classify each IP flow as normal or abnormal. GRU is chosen because it can learn sequence-like dependencies while remaining simpler and faster than LSTM in many cases.

The Mitigation module uses the detection result to identify the attacker’s IP address and generate a direct drop policy for the SDN controller. This makes the mitigation lightweight because it does not rely on probabilistic guessing; it blocks the source that was detected as malicious.

The authors evaluated the method on two public datasets: CICDDoS 2019 and CICIDS 2018. CICDDoS 2019 contains several DDoS attack types, while CICIDS 2018 contains intrusion scenarios and is harder because the malicious traffic is more stealthy and mixed with more legitimate activity.

They compared GRU against several other methods: DNN, CNN, LSTM, SVM, Logistic Regression, kNN, and Gradient Descent. The evaluation used accuracy, precision, recall, F-measure, and also the number of flows per second each model can process, because speed is critical for real-time defense.

On both datasets, GRU gave the best overall balance between detection quality and practicality. In the CICDDoS 2019 scenario, most methods performed very well, but GRU was slightly more balanced across metrics and also strong at classifying normal traffic correctly.

On CICIDS 2018, the task was harder, but GRU still achieved the best overall average performance, with strong accuracy and F-measure while keeping good precision and recall. The authors also found that GRU was feasible for real-time deployment because it could process enough flows per second for a large network environment.

The paper’s practical message is that GRU is a strong choice when you need a good tradeoff between detection quality, implementation simplicity, and runtime speed.