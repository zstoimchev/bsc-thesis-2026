# 07.1 - Applying Convolutional Neural Network for Network Intrusion Detection

This paper studies whether CNN and CNN-based hybrid models can improve network intrusion detection, and the main result is that CNNs perform very well, often better than classical machine learning methods, especially for the KDDCup ’99 dataset. The strongest conclusion is that modeling network traffic as a time series and learning hierarchical features with CNNs is effective for both binary and multi-class intrusion detection.

The paper addresses the need for better intrusion detection systems that can handle both known and new attack patterns. Traditional signature-based and shallow machine learning methods often struggle with modern, complex attacks or produce high false positives.

The authors want to test whether deep learning, especially CNN and hybrid CNN-RNN/LSTM/GRU models, can learn better representations of network traffic than classical classifiers. They also want to see which architecture and topology works best for intrusion detection.

The core idea is to treat TCP/IP connection records as time-series data. Instead of only using handcrafted statistics, the model learns patterns directly from the sequence of network features, which allows CNN to extract higher-level representations from low-level traffic information.

This matters because CNNs are good at finding local patterns and combining them into more abstract features. The paper argues that this makes CNNs a natural fit for intrusion detection, especially when the input is a structured connection record rather than raw packet bytes.

The paper compares several architectures: plain CNN, CNN-RNN, CNN-LSTM, CNN-GRU, and also MLP as a baseline. The models are trained and tested on the KDDCup ’99 dataset, and the paper also discusses NSL-KDD in the broader dataset description.

They tune the number of filters, filter size, learning rate, layers, and epochs. The best CNN configuration used 64 filters with filter length 5, and the authors settled on a learning rate of 0.1 after trying values in the 0.01–0.5 range.

The paper uses the standard KDDCup ’99 intrusion dataset, which contains 41 features per connection and four attack categories: DOS, Probe, R2L, and U2R. It also refers to the 10% version of KDDCup ’99 and the NSL-KDD dataset as the improved benchmark version.

Because the original data is imbalanced, the authors modify the training data by adding more R2L and U2R samples and removing some normal and DOS records. This helps reduce bias toward the majority classes and makes training more useful for rare attacks.

For binary classification, CNN performs extremely well. The best plain CNN model reaches 0.999 accuracy, while CNN with two layers also performs nearly as well at 0.998.

Among the hybrid models, CNN-LSTM with two layers performs best overall in this setting, reaching 0.997 accuracy. CNN-GRU and CNN-RNN are also strong, but in general the hybrids do not consistently beat the simpler CNN.

For attack categorization, the best results again come from CNN-based models. CNN with two layers and CNN with three layers both reach 0.970 accuracy in the multi-class setting.

CNN-LSTM with three layers performs especially well, reaching 0.987 accuracy, which is the best multi-class result in the paper. CNN-GRU also performs strongly, with 0.977 accuracy for the three-layer version.

The paper shows that the difficult classes are U2R and R2L, which remain much harder to detect than normal traffic, DOS, or Probe. This is a common problem in IDS research because those classes have far fewer samples and more subtle patterns.

The main conclusion is that CNN and CNN-based hybrid networks are suitable for intrusion detection because they can learn high-level features from network traffic and outperform many classical methods. The authors also conclude that adding recurrent layers does not always improve results, since plain CNN is already very strong for this dataset.

They further note that deep learning is promising for IDS, but training complex models is expensive and requires substantial computation. Another limitation is that the benchmark datasets are old, so future work should evaluate these methods on newer, real-world traffic such as UNSW-NB15 or live network data.

---
# 07.2 - Evaluating Effectiveness of Shallow and Deep Networks to Intrusion Detection System

This paper evaluates how well shallow and deep neural networks work for network intrusion detection, and its main conclusion is that deeper models usually perform better, especially on the harder NSL-KDD dataset. The key takeaway is that deep networks capture more complex attack patterns than classical shallow classifiers, but they also need more careful tuning and longer training.

The paper studies Network Intrusion Detection Systems, which are used to detect and classify malicious network activity in ICT environments. The authors want to compare classical machine learning methods with deep learning methods and see which ones detect normal and attack traffic more effectively.

The motivation is that modern attacks are diverse and dynamic, while many existing IDS approaches rely on hand-crafted statistics or shallow models that may not generalize well. The authors specifically test both binary classification and multi-class classification, so they examine not only whether a record is normal or malicious, but also what kind of attack it is.

The paper uses two classic IDS benchmarks: KDDCup ’99 and NSL-KDD. KDDCup ’99 is the older and larger dataset, while NSL-KDD is a cleaned-up version designed to reduce redundancy and class imbalance problems.

Both datasets contain four attack groups: DOS, Probe, R2L, and U2R. The paper notes that these datasets are widely used in IDS research, but also that they are outdated and do not fully represent modern real-world network traffic.

For shallow learning, the paper tests Logistic Regression, Naive Bayes, kNN, Decision Tree, AdaBoost, Random Forest, Support Vector Machine, and Extreme Learning Machine. For deep learning, it tests multilayer perceptrons and deep belief networks with different depths and neuron counts.

The deep models are built with increasing numbers of hidden layers and neurons, and the authors tune them using learning rates in the range 0.01 to 0.5 and training up to 1000 epochs. They also examine minimal feature sets of 4, 8, and 12 features to see whether smaller input representations can still support good detection.

In binary classification, the shallow methods already perform reasonably well on KDDCup ’99, but the deep models still improve results overall, especially DBN. On NSL-KDD, the advantage of deep networks becomes clearer because the dataset is harder and less biased.

The best deep result is DBN with 4 layers and 350 neurons per layer, which reaches 0.997 accuracy on KDDCup ’99 and 0.973 accuracy on NSL-KDD in binary classification. Among shallow models, ELM performs best in multi-class classification, but deep networks still outperform most shallow methods overall.

The paper argues that deep networks work better because they learn hierarchical and non-linear feature representations across multiple layers. In other words, instead of relying only on local TCP/IP patterns, they can combine information from several layers to separate normal traffic from different attack types more effectively.

Another important result is that minimal feature sets can still work fairly well, especially the 8-feature set for classical classifiers. However, the deep models generally achieve better performance when given richer feature inputs, showing that representation depth and feature quality both matter.

A major limitation is that the datasets are old and not fully representative of current network traffic. The authors explicitly say that real-time NIDS datasets are still needed for stronger validation.

Another limitation is that some rare attacks, especially U2R and R2L, are difficult to learn and often produce weaker results than common DOS or Probe traffic. This is typical in intrusion detection because rare classes have fewer samples and are harder to model.[