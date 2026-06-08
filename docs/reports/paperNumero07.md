# 07.1 - Applying Convolutional Neural Network for Network Intrusion Detection

This paper studies whether convolutional neural networks and CNN-based hybrid models can improve network intrusion detection. The authors evaluate MLP, CNN, CNN-RNN, CNN-LSTM, and CNN-GRU models on the KDDCup’99 dataset.

The main idea is to treat each TCP/IP connection record as a one-dimensional sequence of features and apply 1D convolution over the feature vector. CNN is used to learn higher-level feature representations from lower-level traffic features. The hybrid models first use CNN to extract feature maps and then pass those learned representations to recurrent layers such as RNN, LSTM, or GRU.

The dataset used is KDDCup’99, which contains connection records grouped into Normal, DoS, Probe, R2L, and U2R categories. The authors also discuss NSL-KDD as a related improved benchmark, but the main CNN experiments are focused on KDDCup’99.

Because the dataset is imbalanced, the authors modify the training data by adding more R2L and U2R records and removing some Normal and DoS records. This is important because R2L and U2R are rare classes and are usually difficult to classify correctly.

The authors tune several hyperparameters, including the number of filters, filter length, number of layers, learning rate, and epochs. The best CNN configuration uses 64 filters with filter length 5.

The results show that CNN performs very well for intrusion detection. In binary classification, CNN reaches very high accuracy, close to 99%. In multi-class attack categorization, CNN with two or three layers reaches around 97% accuracy. CNN-LSTM with three layers achieves the best reported multi-class result, around 98.7% accuracy, while CNN-GRU also performs strongly.

However, the hybrid CNN-recurrent models do not always improve over plain CNN. The paper concludes that CNN is already very strong on this dataset, and recurrent extensions are often only comparable rather than clearly better.

The difficult classes remain R2L and U2R. These classes have fewer samples and more subtle behavior than DoS and Probe attacks, so they are harder for the models to learn.

The main limitation is that KDDCup’99 is old and does not represent modern real-world network traffic well. Therefore, the results should be interpreted as useful architecture evidence, not as proof that the same performance would be achieved on modern DDoS datasets.

For my research, 07.1 is useful as a CNN / CNN-hybrid architecture reference. It should not be treated as the main DDoS representative, but it can support the discussion of CNN, CNN-LSTM, and CNN-GRU models for intrusion detection.

---

# 07.2 - Evaluating Effectiveness of Shallow and Deep Networks to Intrusion Detection System

This paper compares shallow machine learning models and deep neural networks for network intrusion detection. The goal is to evaluate whether deep models can learn better representations of normal and attack traffic than classical classifiers.

The experiments are performed on KDDCup’99 and NSL-KDD. Both datasets contain Normal, DoS, Probe, R2L, and U2R traffic categories. KDDCup’99 is older and larger, while NSL-KDD is a cleaned-up version designed to reduce some redundancy and bias.

The paper evaluates both binary classification and multi-class classification. Binary classification detects whether a record is normal or malicious. Multi-class classification predicts the attack category.

For shallow learning, the paper tests Logistic Regression, Naive Bayes, KNN, Decision Tree, AdaBoost, Random Forest, Support Vector Machine, and Extreme Learning Machine. For deep learning, it evaluates MLP and DBN architectures with different numbers of layers and neurons.

The best binary classification result is achieved by a DBN with four layers and 350 neurons per layer. It reaches around 99.7% accuracy on KDDCup’99 and around 97.3% accuracy on NSL-KDD.

The paper also evaluates minimal feature sets of 4, 8, and 12 features. Some reduced feature sets perform reasonably well, especially for classical classifiers, but deep models usually perform better when they are trained on richer feature inputs.

The main conclusion is that deep networks generally outperform shallow models, especially on the harder NSL-KDD dataset. The authors argue that deep models perform better because they learn hierarchical and nonlinear feature representations across multiple layers.

However, the paper has important limitations. KDDCup’99 and NSL-KDD are old datasets and may not represent modern network traffic. The authors also note that real-time NIDS datasets are needed for stronger validation. In addition, the models require careful tuning and long training, with experiments running up to 1000 epochs.

For my research, 07.2 should be treated as a background comparison paper. It is useful for explaining why deep learning became popular in IDS research, but it is not directly DDoS-specific and should not be one of the main representative papers unless a DBN/MLP comparison is selected.
