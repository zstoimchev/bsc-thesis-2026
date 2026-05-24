# 12 - Ant Colony Induced Decision Trees for Intrusion Detection

This is a conference proceedings, not a single paper. The relevant paper in it is about the **Ant Tree Miner (ATM) classifier**, which uses ant colony optimization to build decision trees for intrusion detection. The main idea is to improve intrusion detection by tuning ATM parameters, using the NSL-KDD dataset, and comparing ATM with standard classifiers like J48, Random Forest, SVM, and MLP.

The authors focused on intrusion detection as a classification problem and tested ATM in a more rigorous way than earlier studies. They used NSL-KDD, applied feature selection, split the data into training, validation, and test sets, and measured not only accuracy but also false alarm rate, detection rate, runtime, and tree quality.

ATM performed well, especially after parameter tuning and when combined with per-attack ensemble models. The ensemble version, called ATMa, reached about 99% accuracy and reduced false alarms, showing that ant-colony-based decision tree learning can be competitive for intrusion detection.

The paper’s message is that decision-tree-based methods are still very strong for IDS, and ant colony optimization can improve them further. It also stresses that reliable evaluation matters: using transparent datasets, clear parameter settings, and reproducible experiments is essential in intrusion detection research.