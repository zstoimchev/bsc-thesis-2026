# 13.1 - Fuzzy Intrusion Detection

> FIRE = Fuzzy Intrusion Recognition Engine  
> IDS = Intrusion Detection System  
> NID = Network Intrusion Detection  
> AAFID = Autonomous Agents for Intrusion Detection  
> FCM = Fuzzy C-Means

This paper introduces the Fuzzy Intrusion Recognition Engine, called FIRE. FIRE is a fuzzy logic-based intrusion detection system that uses fuzzy rules to evaluate whether network activity is malicious.

The main motivation is that many intrusion indicators are not clearly binary. For example, a number of connections may be normal in one context but suspicious in another. Because of this, fuzzy logic is useful because it can express alert levels by degree instead of only true or false.

FIRE uses an agent-based architecture based on AAFID. The system separates monitoring tasks into different agents. Each agent monitors a specific type of network behavior and produces fuzzy inputs. A monitor then combines these fuzzy inputs using fuzzy rules and produces alerts.

The main FIRE agents include TCPconn, UDPconn, ICMPconn, and PortAgent. TCPconn monitors TCP connection patterns, UDPconn monitors UDP traffic, ICMPconn monitors ICMP traffic, and PortAgent monitors unusual services and service-host combinations.

The system focuses mainly on packet header information instead of packet payloads. It monitors TCP, UDP, and ICMP traffic and collects information such as source, destination, service used, connection duration, number of packets, and connection completion status.

FIRE uses a two-week sliding observation window to understand typical network behavior. Metrics are calculated in discrete time intervals, such as 10-minute intervals. The system monitors multiple traffic metrics and uses them to build fuzzy input sets.

The fuzzy membership functions are generated using fuzzy C-means clustering. Inputs are usually represented using five fuzzy categories: LOW, MED-LOW, MEDIUM, MED-HIGH, and HIGH. Fuzzy rules are then created to detect specific attack scenarios.

The paper discusses three main intrusion scenarios: host and port scanning, denial of service, and unauthorized servers or backdoor-like behavior.

For port scanning, the fuzzy rules look for unusual combinations of source, destination, and service ports. For denial of service, the paper tests an ICMP pingflood scenario. The fuzzy rule checks whether the number of unusual ICMP identifiers and echo requests is high. The system was able to detect this pingflood behavior from network traces.

The paper also discusses unauthorized server detection, where unusual service ports or unusual traffic patterns may indicate a backdoor or Trojan horse.

The main contribution of this paper is showing that fuzzy logic can be used as a correlation engine for intrusion detection. It allows the system to combine multiple weak indicators into a single alert level.

However, the paper has limitations. The evaluation is mostly qualitative and does not use modern metrics such as accuracy, F1-score, AUC, or confusion matrices. It also does not use modern benchmark datasets such as CICDDoS2019 or CICIDS2018. Because of this, the paper is useful as fuzzy IDS background, but it should not be used as a main implementation representative.

---

# 13.2 - Fuzzy Network Profiling for Intrusion Detection

This paper continues the FIRE idea but focuses more on network profiling and data mining. It presents FIRE as an anomaly-based intrusion detection system that compares current network behavior with historical network profiles.

The architecture contains three main components: the Network Data Collector, the Network Data Processor, and the Fuzzy Threat Analyzer. The Network Data Collector captures raw network packets. The Network Data Processor summarizes and mines the raw packet data into meaningful metrics. The Fuzzy Threat Analyzer combines fuzzy inputs and applies fuzzy rules to generate alerts.

The paper explains that raw network traffic is too large and too difficult to use directly. Therefore, FIRE performs data mining to extract important traffic features. These features include the number of observed packets, unique source-destination-port combinations, new connections, well-known ports used, packet count variance, foreign host connections, and successful TCP connections.

This data mining step is important because it reduces the amount of stored data and produces features that are more meaningful for anomaly detection than raw packets alone.

The fuzzy rules use characteristics such as COUNT, UNIQUENESS, and VARIANCE. These values are generated from historical network behavior and are transformed into fuzzy input sets. As network behavior changes over time, the fuzzy profiles can also adapt.

The paper evaluates FIRE on production local area networks at Iowa State University. The system collected data for three weeks. During testing, FIRE detected nine TCP port scans and four ICMP ping scans. It also detected local non-malicious scans and triggered high alerts when seldom-seen traffic appeared.

The main contribution of this paper is showing how fuzzy logic can be combined with network profiling and simple data mining to detect anomalous activity.

However, the paper has important limitations. It does not provide a modern benchmark evaluation, does not compare against machine learning or deep learning models, and does not report standard classification metrics. The detected attacks are mainly scans and unusual traffic patterns, not modern large-scale DDoS datasets.

For my research, Paper 13 should be grouped under Fuzzy Logic / Rule-Based Anomaly Detection. It is useful for historical and conceptual background, but it should not be chosen as a main model for implementation. Its main value is showing that anomaly detection can also be built with interpretable fuzzy rules and network behavior profiles.
