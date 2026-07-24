# Model 05: Rule-Based Detector

## Overview

This model implements a deterministic rule-based detector for binary network-intrusion detection.

Instead of learning a classification function, it applies simple thresholds to flow-level features such as packet rate,
byte rate, TCP SYN activity, flow duration, and packet counts.

## Role in the Comparison

The rule-based detector represents a simple and explainable alternative to the machine-learning and deep-learning
models.

It provides a comparison with the majority-class baseline, XGBoost, MLP, GRU, and Transformer-style model under the same
data preparation and evaluation conditions.

## Implementation

The detector uses four rules:

- unusually high packet rate;
- unusually high byte rate;
- unusually high SYN activity with little backward traffic;
- short flow duration combined with an unusually high forward-packet count.

The thresholds are calculated only from benign records in the training split. Packet-rate and byte-rate thresholds use
the 99.5th percentile, while the remaining thresholds use selected 99th or 10th percentiles.

A flow is classified as malicious when at least two rules are satisfied.

The detector requires the following common features:

- `flow_duration`;
- `total_fwd_packets`;
- `total_bwd_packets`;
- `flow_packets_s`;
- `flow_bytes_s`;
- `syn_flag_count`.

## Usage

```bash
python run.py train-evaluate --model mdl05_rule_based --split-id cic_random_80_20_v1
```

## Notes

The detector classifies individual network flows and does not aggregate traffic across hosts or time windows. It should
therefore not be interpreted as a complete operational DDoS-detection system.

Its thresholds depend on the benign training distribution and may not transfer successfully to traffic originating from
a substantially different dataset.
