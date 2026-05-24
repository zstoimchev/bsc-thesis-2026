# 10 - Optimizing Preprocessing & Parameters for Cluster-Based Anomaly Detection

This study is about improving **cluster-based anomaly detection** by tuning preprocessing and algorithm parameters, mainly for K-Means and DBSCAN. The key finding is that good preprocessing matters a lot, and on the tested datasets K-Means slightly outperformed DBSCAN in F-score and speed.

The study compares unsupervised methods such as K-Means, DBSCAN, Isolation Forest, and Local Outlier Factor on IDS datasets like NSL-KDD and IDS 2017. It also evaluates how handling missing values, categorical encoding, and feature reduction affect detection quality.

The main conclusion is that K-Means and DBSCAN are both promising for anomaly detection, but K-Means was faster while DBSCAN sometimes achieved slightly better clustering quality depending on the parameter settings.