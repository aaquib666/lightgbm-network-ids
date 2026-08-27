# LightGBM DoS/DDoS Network Intrusion Detection System

> **Aaquib Rizwan (CB012956)**
> Staffordshire University — BSc Cyber Security (Hons), 2026
> Dissertation: *"Design and Evaluation of a Lightweight, Real-Time DoS/DDoS Detection Model Trained on Recent, Imbalanced Network Traffic Using Practical Feature Sets"*

A lightweight, real-time Network Intrusion Detection System (NIDS) that captures live network traffic, aggregates it into flows, extracts a compact 9-feature vector per flow, and classifies each flow using a pre-trained **LightGBM** binary classifier. Detected attacks trigger an automated, safeguarded response layer (`iptables` blocking + SELinux advisory logging).

This is a **Linux-based, terminal-driven** detection and automated response framework with no GUI, as described in the dissertation (Chapter 5, Section 5.5).

> ⚠️ **Disclaimer**: This is an academic research prototype. It modifies live firewall rules (`iptables`) when run outside dry-run/simulation mode. Run only on networks/hosts you own or have explicit authorization to monitor and modify. See [Safety & Deployment Notes](#safety--deployment-notes).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Detection Pipeline](#detection-pipeline)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Feature Vector](#feature-vector)
- [Feature Alignment Check](#feature-alignment-check)
- [Automated Response](#automated-response)
- [Testing / Attack Simulation](#testing--attack-simulation)
- [Constraints](#constraints)
- [Safety & Deployment Notes](#safety--deployment-notes)
- [Dissertation Reference Map](#dissertation-reference-map)
- [License](#license)

---

## Overview

Traditional signature-based IDS solutions struggle to keep pace with evolving DoS/DDoS attack patterns. This system instead uses a supervised machine learning model — **LightGBM**, trained on the **CSE-CIC-IDS2018** and **CICDDoS2019** datasets — to classify network flows in real time as **NORMAL**, **SUSPICIOUS**, or **ATTACK**, based on a compact, low-latency 9-feature representation of each flow.

Key design goals:

- **Lightweight** — a 9-feature vector keeps per-flow inference latency in the microsecond range (~6 µs/flow benchmark), suitable for real-time operation on modest hardware.
- **Stateful flow tracking** — bidirectional flow aggregation (5-tuple keyed, with reverse-key matching) rather than raw per-packet classification.
- **Hybrid detection policy** — combines model confidence with volumetric thresholds to reduce false positives on real-world (non-benchmark) traffic.
- **Safe-by-default response** — whitelisting, dry-run mode, and a "safe response mode" that only blocks high-volume flows even at high model confidence.

## Architecture

```
                ┌────────────────────┐
 Live Traffic → │  packet_capture.py │  (Scapy AsyncSniffer)
                └─────────┬──────────┘
                          │ raw packets
                          ▼
                ┌────────────────────┐
                │   flow_tracker.py  │  Stateful 5-tuple flow aggregation
                └─────────┬──────────┘
                          │ expired flows
                          ▼
                ┌────────────────────┐
                │ feature_extractor  │  9-feature vector per flow
                └─────────┬──────────┘
                          │ feature DataFrame
                          ▼
                ┌────────────────────┐
                │  model_inference.py│  LightGBM predict_proba()
                └─────────┬──────────┘
                          │ confidence score
                          ▼
                ┌────────────────────┐
                │  ids_main.py        │  Hybrid classification policy
                │  (NORMAL/SUSPICIOUS/│  (confidence + volumetric gate)
                │   ATTACK)           │
                └─────────┬──────────┘
                          │ ATTACK
                          ▼
                ┌────────────────────┐
                │ response_engine.py │  iptables DROP + SELinux advisory
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │     logger.py      │  Dual output: colour terminal + file log
                └────────────────────┘
```

## Detection Pipeline

1. **Capture** — `packet_capture.py` sniffs TCP/UDP/ICMP packets non-blockingly via Scapy's `AsyncSniffer`.
2. **Flow aggregation** — `flow_tracker.py` groups packets into bidirectional flows keyed by `(src_ip, dst_ip, src_port, dst_port, protocol)`, tracking packet lengths, timestamps, and SYN/ACK counts. Flows are flushed once idle for `FLOW_IDLE_TIMEOUT` (default 5s).
3. **Feature extraction** — `feature_extractor.py` converts each completed flow into a 1×9 feature vector, validating bounds to reject implausible/incomplete flows before inference.
4. **Inference** — `model_inference.py` loads the pre-trained LightGBM model, verifies feature-name alignment against the live feature set at startup, and returns an attack-class probability with microsecond-level latency.
5. **Hybrid classification** — `ids_main.py` applies a three-tier decision policy combining the model's confidence score with volumetric indicators (packets/sec, packet count, bytes/sec) to classify each flow as `NORMAL`, `SUSPICIOUS`, or `ATTACK`.
6. **Response** — `response_engine.py` inserts an `iptables DROP` rule for confirmed attack sources (subject to whitelist and safe-response-mode checks), and logs a non-destructive SELinux advisory.
7. **Logging** — `logger.py` writes colour-coded terminal output and a structured, timestamped plain-text log file simultaneously.

## Project Structure

```
ids_prototype/
├── ids_main.py               # Entry point — CLI launcher and main loop
├── packet_capture.py         # Live packet sniffing (Scapy AsyncSniffer)
├── feature_extractor.py      # 9-feature flow vector builder (Table 5.2)
├── model_inference.py        # LightGBM .pkl model loader and predictor
├── response_engine.py        # iptables rule insertion and SELinux advisory
├── logger.py                 # Dual terminal + file logger with ANSI colours
├── flow_tracker.py           # Stateful per-flow aggregation (5-tuple key)
├── requirements.txt          # Python dependencies
├── config/
│   ├── ids_config.yaml        # Detection thresholds & response settings
│   └── whitelist.txt          # Never-block IP list
├── model/
│   └── lgbm_model.pkl         # Pre-trained LightGBM binary classifier (user-supplied)
└── logs/
    └── ids_detections.log     # Auto-created at runtime
```

## Requirements

- **Linux** (Ubuntu 20.04+ or Kali Linux)
- **Python 3.10+**
- Must run as **root** for live packet capture and `iptables` (except `--simulate` mode)
- `hping3` on an attacker machine, if you intend to generate live test traffic

### Dependencies (`requirements.txt`)

```
scapy>=2.5.0
lightgbm>=4.0.0
scikit-learn>=1.6.1
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
```

PyYAML is optional but recommended for full YAML config support (a minimal fallback parser is used if it isn't installed).

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pyyaml   # optional but recommended
```

## Model Setup

The system expects a pre-trained LightGBM `.pkl` file saved via `joblib.dump()` from a scikit-learn-compatible `LGBMClassifier`.

```bash
mkdir -p model
cp /path/to/your/lgbm_model.pkl model/lgbm_model.pkl
```

### Expected Training Configuration

```python
LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Training Feature Columns (in order)

```
['Flow Bytes/s', 'Flow Packets/s', 'Avg Packet Size',
 'Flow Duration', 'Flow IAT Mean', 'Flow IAT Max',
 'Protocol', 'SYN Flag Count', 'ACK Flag Count']
```

> Column names must match `FEATURE_COLUMNS` in `feature_extractor.py`. See [Feature Alignment Check](#feature-alignment-check) below for automatic mismatch handling.

## Configuration

All detection and response thresholds are controlled via `config/ids_config.yaml`:

| Key | Description | Default |
|---|---|---|
| `suspicious_threshold` | Confidence above which a flow is flagged `SUSPICIOUS` | `0.70` |
| `detection_threshold` | Confidence above which a flow is eligible for `ATTACK` classification | `0.90` |
| `min_packets_per_second` | Volumetric gate — packets/sec | `1000` |
| `min_packet_count` | Volumetric gate — total packet count | `50` |
| `min_bytes_per_second` | Volumetric gate — bytes/sec | `1000000` |
| `safe_response_mode` | If `true`, only blocks flows that also exceed volumetric thresholds | `true` |
| `whitelist_file` | Path to IPs that are never blocked | `config/whitelist.txt` |

`config/whitelist.txt` contains one IP per line (`#` for comments) — pre-populated with common public DNS resolvers and localhost as examples.

## Usage

### Live Mode

```bash
sudo python3 ids_main.py --interface ens33 --threshold 0.7
```

### Dry Run (no iptables changes)

```bash
sudo python3 ids_main.py --interface ens33 --dry-run
```

### Simulation Demo (no root, no live traffic needed)

```bash
python3 ids_main.py --simulate
```

### Debug Mode (per-flow feature vectors and confidence)

```bash
sudo python3 ids_main.py --interface ens33 --debug
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--interface` | `ens33` | Network interface to sniff on |
| `--model` | `model/lgbm_model.pkl` | Path to the LightGBM `.pkl` file |
| `--threshold` | *(from config)* | Override `detection_threshold` from config |
| `--window` | `2.0` | Time window in seconds to poll expired flows |
| `--log` | `logs/ids_detections.log` | Path to the detection log file |
| `--config` | `config/ids_config.yaml` | Path to the YAML configuration file |
| `--dry-run` | `False` | Detect only — do NOT insert iptables rules |
| `--simulate` | `False` | Run with synthetic traffic (no root required) |
| `--debug` | `False` | Print per-flow feature vectors and prediction confidence |

Stop the IDS at any time with `Ctrl+C`. On shutdown, a summary is printed and all `iptables` rules inserted during the session are automatically flushed.

## Feature Vector

Each flow is reduced to 9 features (matching the model's training schema):

| # | Feature | Description |
|---|---|---|
| 1 | Flow Bytes/s | Total bytes ÷ flow duration |
| 2 | Flow Packets/s | Packet count ÷ flow duration |
| 3 | Avg Packet Size | Mean packet length |
| 4 | Flow Duration | Last − first timestamp (µs) |
| 5 | Flow IAT Mean | Mean inter-arrival time |
| 6 | Flow IAT Max | Max inter-arrival time |
| 7 | Protocol | 6 (TCP), 17 (UDP), 0 (ICMP, CICFlowMeter convention) |
| 8 | SYN Flag Count | Count of SYN flags observed |
| 9 | ACK Flag Count | Count of ACK flags observed |

Feature vectors are bounds-checked (`feature_extractor.py`) to discard implausible or malformed values (e.g. NaN/Inf, negative duration, throughput exceeding physically realistic bounds) before inference.

## Feature Alignment Check

On startup, `model_inference.py` prints the model's expected feature names via `model.feature_name_` alongside the live `FEATURE_COLUMNS` set. If they match exactly, alignment is confirmed. If they differ only by whitespace/underscore formatting, an automatic column-rename mapping is applied. If they cannot be reconciled, startup aborts with an error — update `FEATURE_COLUMNS` in `feature_extractor.py` to match your model and restart.

## Automated Response

On an `ATTACK` classification, `response_engine.py` applies a layered response:

- **Layer 1 — iptables**: inserts a `DROP` rule for the source IP (`iptables -I INPUT -s <ip> -j DROP`), unless suppressed by `--dry-run`.
- **Layer 2 — SELinux advisory**: logs a non-destructive advisory recommending `audit2allow`/`semanage` review — no policy changes are made automatically.

Before any block is applied:

- **Whitelist check** — whitelisted IPs are never blocked.
- **Safe response mode** — even a high-confidence prediction is only acted on if the flow also exceeds volumetric thresholds (packets/sec, packet count, or bytes/sec), preventing containment of low-volume, high-confidence false positives.
- **Duplicate check** — an already-blocked IP is not re-blocked.

All blocked IPs are tracked for the session and automatically unblocked on graceful shutdown.

## Testing / Attack Simulation

To exercise live detection, generate attack traffic from a separate attacker machine (e.g. a Kali Linux VM) on the same network, targeting the host running the IDS. Replace `10.45.197.129` with your actual target IP.

**SYN flood — baseline**

```bash
sudo hping3 -S -p 80 -c 100 -i u10000 10.45.197.129
```

**SYN flood — randomized source**

```bash
sudo hping3 -S -p 80 -c 100 -i u10000 --rand-source 10.45.197.129
```

**UDP flood**

```bash
sudo hping3 --udp -p 80 -c 100 -i u10000 10.45.197.129
```

**ICMP flood**

```bash
sudo hping3 --icmp -c 100 -i u10000 10.45.197.129
```

Run the IDS in `--dry-run` mode while testing to observe classification behaviour without inserting live `iptables` rules.

## Constraints

- **No SMOTE** — the model uses `class_weight='balanced'` for imbalance handling (Section 5.4.3)
- **No GUI** — terminal-only interaction (Section 5.5)
- **SELinux advisory only** — no `semanage` or `audit2allow` execution
- **All iptables rules are cleaned up** on Ctrl+C or normal exit
- **`--dry-run` suppresses all `subprocess` calls** completely
- **Flow duration in microseconds** to match CIC dataset format
- **Binary labels**: `Benign = 0`, `Attack = 1` (Section 5.3.4)
- **`--simulate` does not require root**

## Safety & Deployment Notes

- Always test with `--dry-run` or `--simulate` before running with live `iptables` enforcement.
- The model was trained on benchmark datasets (CSE-CIC-IDS2018, CICDDoS2019) and may not generalize perfectly to arbitrary real-world traffic profiles (e.g. QUIC/UDP-heavy streaming traffic) — the hybrid confidence + volumetric policy is designed to mitigate this, but review logs regularly.
- Run only on networks/hosts you own or have explicit authorization to monitor and modify.
- SELinux logging is advisory only and does not alter system policy.

## Dissertation Reference Map

| Module | Dissertation Reference |
|---|---|
| `flow_tracker.py` | Chapter 2, Section 2.3.4 — Flow-based traffic preprocessing |
| `packet_capture.py` | Chapter 3, Section 3.4.3.2 — Scapy for live packet sniffing |
| `feature_extractor.py` | Chapter 5, Section 5.4.2 — Feature Selection Module (Table 5.2) |
| `model_inference.py` | Chapter 5, Section 5.4.4 — Model Training Module (LGBMClassifier) |
| `response_engine.py` | Chapter 5, Section 5.4.5 — Automated Response and Security Enforcement Module |
| `logger.py` | Chapter 5, Section 5.5 — User Interface (terminal-based reporting) |
| `ids_main.py` | Chapter 5, Section 5.5 — Linux background detection process |

## License

Add a license of your choice (e.g. MIT) before making the repository public, or state that this is academic/coursework code not licensed for reuse.

---

*Developed as the practical component of a BSc Cyber Security (Hons) Final Year Project — Staffordshire University, 2026.*
