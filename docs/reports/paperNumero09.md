# 09 - A Subset Feature Elimination Mechanism for Intrusion Detection System

This paper focuses on improving intrusion detection systems by selecting only the most relevant features from network traffic data. The authors argue that using all available features is inefficient, since many of them are redundant or irrelevant and may decrease classifier performance while increasing computation time.

The proposed approach combines a **Decision Tree classifier** with **Recursive Feature Elimination (RFE)** and feature ranking techniques based on information gain. The goal is to improve both detection accuracy and execution speed by removing unnecessary features from the dataset.

The experiments are performed using the **NSL-KDD dataset**, which is an improved version of the older KDD’99 dataset. The dataset contains 41 traffic features and includes multiple attack categories such as DoS, Probe, R2L, U2R, and normal traffic.

> Instead of training the classifier on all 41 features, the authors attempt to identify a smaller subset of features that best represents each attack class.

The feature selection process starts with feature scoring using an **ANOVA F-test**. After that, **Recursive Feature Elimination** is applied.

The RFE process repeatedly trains the classifier, ranks the features according to importance, removes the least important feature, and retrains the model again. This process continues until the most relevant subset of features is found.

The classification itself is performed using a **Decision Tree**, since decision trees naturally rank features using entropy and information gain.

An important observation in the paper is that different attacks depend on different features. For example, the most relevant features for DoS, Probe, R2L, and U2R attacks are different, meaning that no single subset works equally well for every attack category.

After the elimination process, the final feature subsets are much smaller than the original dataset. Depending on the attack category, the classifier uses only around 11–15 features instead of all 41.

The models are evaluated using metrics such as *accuracy*, *precision*, *recall*, *F-measure*, and confusion matrices. A 10-fold cross-validation is also performed during evaluation.

The results show that feature elimination significantly improves performance.

For example:
- DoS accuracy reaches ~99.90%
- Probe reaches ~99.80%
- R2L improves from ~97% to ~99.88%
- U2R achieves ~99.95%

The paper especially highlights the improvement for **R2L attacks**, since they are usually difficult to classify correctly.

Another important result is the reduction in execution time. After feature selection, training becomes much faster because the classifier processes far fewer attributes.

For example, the DoS classifier training time decreases from about 15.5 seconds to less than 1 second after feature reduction.

This demonstrates that removing irrelevant features improves both efficiency and classifier quality.

The authors compare their method with several previous feature selection approaches and show that their recursive feature elimination approach achieves some of the best overall accuracies among the compared methods.

The paper concludes that feature selection plays a major role in intrusion detection systems. Removing redundant and irrelevant features makes IDS models faster, simpler, and more accurate.
