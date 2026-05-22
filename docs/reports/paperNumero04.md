# 04 - An Intrusion Detection System Using a Deep Neural Network With Gated Recurrent Units

> IDS = Intrusion Detection System  
> GRU = Gated Recurrent Unit  
> RNN = Recurrent Neural Network

This paper proposes a **deep learning-based IDS** that uses **Recurrent Neural Networks (RNNs)** with **Gated Recurrent Units (GRUs)** for detecting network intrusions.

This paper proposes a **deep learning-based IDS** using **Recurrent Neural Networks (RNNs)** with **Gated Recurrent Units (GRUs)**. The main idea is that network traffic has **temporal behavior** (time-related patterns), meaning that previous events and packet sequences are important for detecting attacks. Because of this, the authors use **RNNs**, since they can remember information from previous time steps.

> Traditional machine learning models usually treat records independently, while RNNs can “remember” previous information and use it for future predictions (can process sequential/time-series data).

The authors aim to improve intrusion detection performance while also reducing the need for **manual feature engineering**.

The proposed IDS consists of three main components:
- **GRU module**: which extracts and stores temporal features from network traffic
- **MLP (Multilayer Perceptron)**: performs non-linear classification using the GRU output
- **Softmax layer**: outputs final class probabilities
> The GRU is the core part of the system.

Compared to **LSTM**, GRU has *fewer parameters*, *simpler structure*, *faster convergence* and *lower computational cost*.

The authors also experiment with **Bidirectional GRU (BGRU)**, which processes sequences both forward and backward in time.

> The experiments later show that GRU also achieves better performance for intrusion detection.

The authors evaluate the system using the **KDD99** and **NSL-KDD** datasets, which contain both normal traffic and attack traffic. The attacks belong to four main categories: **DoS**, **Probe**, **R2L**, **U2R**.

The datasets also include attack types that appear only in the test set, which helps evaluate how well the model generalizes to unseen attacks.

The IDS is trained as a **multi-class classifier**, meaning that it predicts the exact traffic category instead of only determining whether traffic is malicious or benign.  

The evaluation uses metrics such as ***accuracy***, ***detection rate***, ***false positive rate***, ***precision***, and ***F-measure***. The authors place strong emphasis on **Detection Rate** and **False Positive Rate**, since intrusion detection systems must detect attacks while minimizing false alarms.

The results show that the proposed model performs extremely well on both datasets.

- On **KDD99**, the model achieved overall detection rate of **99.42%**, and false positive rate of **0.05%**
- On **NSL-KDD**, the model achieved overall detection rate of **99.31%**, and false positive rate of **0.84%**
- The performance for **DoS attacks** is especially strong, with detection rates close to **100%** on both datasets.

The experiments also compares several architectures, including, *LSTM*, *GRU*, *Bidirectional GRU*, *MLP alone*, *combinations of RNN + MLP*.

The best results are achieved using **Bidirectional GRU combined with MLP (BGRU + MLP)**.

#### The experiments show that:
- GRU performs better than LSTM
- Bidirectional GRU performs better than normal GRU
- Combining RNN with MLP performs better than using either individually

The models also converge quickly, requiring only around **20 epochs**.

The system performs much better on **DoS** and **Probe** attacks than on **R2L** and **U2R** attacks. The authors explain this by saying that DoS and Probe attacks have clearer **time-series behavior**, making them easier for RNNs to learn. R2L and U2R attacks are also underrepresented in the datasets, which reduces learning quality.

The authors conclude that **GRU-based deep neural networks are highly effective for intrusion detection**, especially for attacks with temporal patterns. The proposed system achieves very high detection rates with very low false positives, while also being simpler and faster than LSTM-based systems.