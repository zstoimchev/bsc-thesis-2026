# 02 - Detection and Characterization of DDoS Attacks Using Time-Based Features

This paper focuses on detecting and classifying **DDoS attacks** using machine learning and deep learning models. The main idea is that instead of using all available network traffic features, it is possible to use only a **small subset of time-based features** and still achieve very high performance, while significantly reducing training time.

DDoS attacks are designed to overwhelm a target system using a large number of requests coming from multiple machines (botnets), effectively denying service to legitimate users.

The authors are interested in whether **time-related statistics in network traffic flows** (such as packet timing and frequency) are sufficient for detecting and classifying attacks.

> **Goal: show that a reduced feature set (only time-based features) can achieve comparable accuracy to full feature sets, while improving efficiency and training speed.**

### Dataset used
The dataset used is **CICDDoS2019**, which contains both benign traffic and 12 types of DDoS attacks. It originally includes ~70 millions of samples, so the authors **downsampled and balanced the data** to make training feasible and avoid bias.
#### Types of attacks included:
- **SYN Flood**
- **UDP Flood**
- **LDAP**
- **DNS**
- **NetBIOS**
- **SNMP**
- **NTP**
- **SSDP**
- **MSSQL**
- **TFTP**
- **Portmap**
- **UDP-Lag**

The authors focus on **time-based features**, which are statistics computed over network traffic flows within a time window. These include things like packet arrival times, flow duration, and packet rate statistics (mean, min, max, standard deviation).

These features are important because DDoS attacks typically generate **large volumes of traffic in short time intervals**, meaning their temporal patterns differ from normal traffic.

Compared to the full feature set (~70 features), the time-based subset contains only **25 features**, which:
- reduces dimensionality
- reduces noise
- speeds up training
- makes real-time applications more feasible

The models are trained to perform two main tasks:
1. **Binary classification** (benign vs DDoS traffic): The model learns to distinguish normal traffic from attack traffic.
2. **Multi-class classification** (specific attack type): The model identifies which of the 12 DDoS attacks is occurring.

The authors compare **9 different classifiers**, including traditional machine learning, boosting methods, and one deep learning model:
- **Traditional ML:**  
	- Random Forest (RF)  
	- K-Nearest Neighbors (KNN)  
	- Support Vector Machine (SVM)  
	- Gaussian Naive Bayes (GNB)  
	- Linear Discriminant (LD)  
- **Boosting methods:**  
	- LightGBM (LGBM)  
	- XGBoost (XGB)  
	- AdaBoost (ADA)  
- **Deep Learning:**  
	- Deep Neural Network (DNN, using fast.ai)

The models are evaluated using standard metrics such as ***accuracy***, ***precision***, ***F1-score***, ***AUC***, and also **training time**, which is important for practical deployment.

The results show that models perform extremely well for **binary classification**. Using all features, accuracy is around **99%**, and when using only time-based features, the accuracy remains almost the same (~98–99%), with only a very small drop.

For **multi-class classification**, the task is harder. The best models achieve around **70–74% accuracy** with all features, and slightly lower (~65–70%) when using only time-based features.

However, the key result is that **training time is significantly reduced** when using time-based features:
- about **36% faster** for binary classification
- about **25% faster** for multi-class classification

Among the models, **XGBoost achieved the best overall accuracy**, while **LightGBM and Linear Discriminant** provided a very good balance between speed and performance. Random Forest also performed consistently well, while some models like Naive Bayes and SVM performed worse, especially with reduced features.

The authors conclude that **time-based features alone are sufficient** for effective DDoS detection and classification. Even though there is a small drop in accuracy, the improvement in efficiency makes this approach very useful for real-world systems, especially those requiring **near real-time detection or continuous learning**.
