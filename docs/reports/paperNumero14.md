# 14 - Neural Network-Based Voting System with High Capacity and Low Computation for Intrusion Detection in SIEM/IDS Systems

> SIEM = Security Information and Event Management

This paper proposes an IDS based on multiple small neural networks combined using a voting system. The main goal is to achieve high detection performance while keeping computational cost low.

Instead of using one large and complex deep neural network, the authors use several simple feedforward neural networks, where each network specializes in a different subset of features from the **NSL-KDD dataset**. The final prediction is then produced using ensemble voting methods.

The paper compares several voting approaches such as Hard Voting, Weighted Average, Mixture of Experts (MoE), and Majority Function.

The best results are achieved using a modified **Majority Function with a reliability mechanism**.

The reliability system is used when one classifier is uncertain (close to probability 0.5) and the others disagree. In that case, the system selects the classifier that proved most reliable during training.

The individual neural networks achieve only moderate results on their own:
- NN1 → 73% accuracy
- NN2 → 59%
- NN3 → 70%

However, combining them improves the performance significantly.

The final ensemble model achieves:
- **89% accuracy**
- **91% precision**
- **90% F-score**
- **88% recall**

An important point in the paper is that the proposed system achieves these results using very small neural networks with low computational requirements. The largest model uses only one hidden layer with 25 neurons.

Compared to more complex approaches such as deep CNNs, autoencoders, or recurrent neural networks, the proposed model is much simpler and faster to train while still achieving competitive performance.
