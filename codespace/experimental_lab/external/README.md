# External repositories

This directory is reserved for local copies of original repositories used during the audit phase of the thesis implementation.

The repositories are **not committed to this Git repository** because they may contain large files, notebooks, datasets, trained models, generated outputs, or third-party code with separate licenses. They should be downloaded locally only when needed for inspection, auditing, or comparison.

The clean re-implementation used for the research is located in:

```text
common/
runners/
registries/
orchestrate.py
```

## Repository credits

The following repositories were used as references for architecture, preprocessing, hyperparameters, training scripts, or evaluation methodology:

| Paper | Local folder name                       | Original repository / source      | Notes                                                |
|-------|-----------------------------------------|-----------------------------------|------------------------------------------------------|
| 01    | `01_DeepLearning-IDS`                   | TODO: add original GitHub URL     | DNN / MLP IDS reference implementation               |
| 02    | `02_DDoS_Traffic_Research`              | TODO: add original GitHub URL     | Classical ML and DNN models for CIC-DDoS2019         |
| 03    | `03_Anomaly-Detection-KDD99-CNNLSTM`    | TODO: add original GitHub URL     | Repository-only audit / KDD-based reference          |
| 04    | `04_Intrusion-Detection-on-NSL-KDD`     | TODO: add original GitHub URL     | GRU / LSTM / BiGRU / BiLSTM NSL-KDD reference        |
| 05    | `05_Intrusion_detection_system`         | TODO: add original GitHub URL     | Self-taught learning / softmax-style reference       |
| 06    | `06_CIC-DDoS2019-DeepLearning`          | TODO: add original GitHub URL     | GRU-based CIC-DDoS2019 reference                     |
| 07    | `07_A-Novel-Approach-for-IDS`           | TODO: add original GitHub URL     | CNN / CNN-LSTM / classical IDS reference             |
| 08    | `08_Network-Intrusion-Detection`        | TODO: add original GitHub URL     | Deep/shallow network variants                        |
| 09    | `09_Network-Intrusion-Detection`        | TODO: add original GitHub URL     | Feature selection / decision-tree-style reference    |
| 10    | `10_clustering-based-anomaly-detection` | TODO: add original GitHub URL     | K-means / clustering anomaly detection reference     |
| 11    | `11_Network-Intrusion-Detection-System` | TODO: add original GitHub URL     | XGBoost / RF / SVM reference                         |
| 12    | `12_NSLKDD-Dataset`                     | TODO: add original GitHub URL     | Ant-colony / decision-tree-related NSL-KDD reference |
| 13    | `13_python-fire`                        | Repository deleted or unavailable | Recorded as audit-only                               |
| 14    | `14_NSL-KDD-Neural-Networks-Pytorch`    | TODO: add original GitHub URL     | Neural-network IDS reference                         |
| 15    | `15_NSL-KDD-ADS`                        | TODO: add original GitHub URL     | Deep energy / anomaly detection reference            |

## License note

Each external repository remains the property of its original author(s). Any license terms from the original repositories should be respected. This research repository does not claim ownership of third-party code and does not redistribute the full external repositories by default.
