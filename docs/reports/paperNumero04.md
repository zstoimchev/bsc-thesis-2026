# 04 - An Intrusion Detection System Using a Deep Neural Network With Gated Recurrent Units

> IDS = Intrusion Detection System  
> RNN = Recurrent Neural Network  
> GRU = Gated Recurrent Unit  
> BGRU = Bidirectional Gated Recurrent Unit  
> MLP = Multilayer Perceptron


This paper proposes a deep learning-based intrusion detection system using Recurrent Neural Networks with Gated Recurrent Units. The main idea is that some network attacks have temporal behavior, meaning that previous network events can help classify later traffic. Because of this, the authors use GRU-based recurrent networks, which are able to keep information from previous time steps.

The proposed system is composed of four main parts: a preprocessing module, a GRU or bidirectional GRU module, an MLP module, and a softmax output layer. The preprocessing module transforms and normalizes the input data. The GRU module extracts temporal features from the traffic records. The MLP performs nonlinear classification based on the GRU output, and the softmax layer produces the final class probabilities.

The authors choose GRU because it is simpler than LSTM. GRU has fewer gates and fewer parameters, while still being able to learn sequential dependencies. This makes GRU potentially faster and easier to train than LSTM, which is important for intrusion detection systems.

The paper also evaluates Bidirectional GRU. A bidirectional recurrent model processes the sequence in both forward and backward directions, allowing the model to use both previous and later context when learning patterns.

The experiments are performed on the KDD 99 and NSL-KDD datasets. The model performs multi-class classification using the high-level categories Normal, DoS, Probe, R2L, and U2R. The evaluation uses metrics such as accuracy, detection rate, false positive rate, precision, and F-measure.

The authors compare several architectures, including LSTM, GRU, Bidirectional GRU, MLP alone, and combinations of recurrent networks with MLP. The best results are achieved by the BGRU + MLP model.

The reported results are very strong. The proposed system achieves a detection rate of around 99.42% on KDD 99 and 99.31% on NSL-KDD, with false positive rates of 0.05% and 0.84%. The detection of DoS attacks is especially strong, reaching almost 100% detection rate.

The experiments show three important findings. First, GRU performs better than LSTM in this setting. Second, Bidirectional GRU performs better than normal GRU. Third, combining GRU/BGRU with MLP performs better than using either recurrent layers or MLP alone.

However, the results are not equally strong for all attack categories. DoS and Probe attacks are detected much better than R2L and U2R attacks. The main reasons are that R2L and U2R have very few samples in the datasets and their behavior is less clearly time-based than DoS and Probe attacks.

The paper also has limitations. KDD 99 and NSL-KDD are old benchmark datasets and may not represent modern real network traffic. The authors also state that the system still needs more engineering work before it can be applied to real network environments.

For my research, Paper 04 is useful as a representative of the RNN / GRU-based architecture group. It is especially useful for understanding why GRU can be applied to intrusion detection. However, because the datasets are old and the paper is not focused specifically on modern DDoS detection, it should probably be used as background for GRU-based IDS rather than as the main DDoS representative.
