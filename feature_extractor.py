"""
feature_extractor.py — 9-Feature Flow Vector Builder

Dissertation Reference:
    Chapter 5, Section 5.4.2 — Feature Selection Module (Table 5.2)

Converts each aggregated FlowRecord into exactly the 9 features used to
train the LightGBM model, in the same column order as the training data:

    1. Flow Bytes/s         — sum(pkt_lengths) / flow_duration
    2. Flow Packets/s       — packet_count / flow_duration
    3. Avg Packet Size      — mean(pkt_lengths)
    4. Flow Duration        — last_ts - first_ts  (microseconds)
    5. Flow IAT Mean        — mean(inter-arrival times)
    6. Flow IAT Max         — max(inter-arrival times)
    7. Protocol             — 6 (TCP) or 17 (UDP)
    8. SYN Flag Count       — count of SYN flags observed
    9. ACK Flag Count       — count of ACK flags observed
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flow_tracker import FlowKey, FlowRecord

# ── Feature column names — MUST match training data exactly ──────────────────
# Updated to match the actual model training feature set (Issue 5).
FEATURE_COLUMNS: List[str] = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Avg Packet Size",
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Protocol",
    "SYN Flag Count",
    "ACK Flag Count",
]

# ── Feature validation thresholds (Issue 4) ──────────────────────────────────
# Reject feature vectors that exceed physically plausible bounds.
_MAX_FLOW_BYTES_PER_SEC = 100_000_000    # 100 MB/s — above this is implausible
_MAX_FLOW_PACKETS_PER_SEC = 1_000_000    # 1 M pkt/s — above this is implausible


def _compute_iat(timestamps: List[float]) -> Tuple[float, float]:
    """Compute mean and max inter-arrival times (in microseconds)."""
    if len(timestamps) < 2:
        return 0.0, 0.0
    iats = np.diff(sorted(timestamps)) * 1e6  # seconds → microseconds
    return float(np.mean(iats)), float(np.max(iats))


def extract_features(flow: FlowRecord, *,
                     debug: bool = False) -> Optional[pd.DataFrame]:
    """
    Convert a single FlowRecord into a 1-row DataFrame with the 9
    training-aligned feature columns.

    Returns:
        pd.DataFrame with shape (1, 9) and column names matching
        FEATURE_COLUMNS, or **None** if the flow is incomplete or
        produces invalid features.
    """
    packet_count = len(flow.pkt_lengths)

    # ── Issue 2: skip single-packet / zero-duration flows ────────────────
    duration_sec = flow.last_time - flow.start_time
    if packet_count < 2 or duration_sec <= 0:
        return None  # caller logs DEBUG skip message

    total_bytes = sum(flow.pkt_lengths)

    # Flow duration in microseconds (CIC dataset format)
    duration_us = duration_sec * 1e6

    # ── Issue 1: safe throughput — use real duration, never near-zero ────
    flow_bytes_per_sec = total_bytes / duration_sec
    flow_packets_per_sec = packet_count / duration_sec

    avg_pkt_size = float(np.mean(flow.pkt_lengths))

    iat_mean, iat_max = _compute_iat(flow.timestamps)

    row = {
        "Flow Bytes/s":         flow_bytes_per_sec,
        "Flow Packets/s":       flow_packets_per_sec,
        "Avg Packet Size":      avg_pkt_size,
        "Flow Duration":        duration_us,
        "Flow IAT Mean":        iat_mean,
        "Flow IAT Max":         iat_max,
        "Protocol":             flow.protocol,
        "SYN Flag Count":       flow.syn_count,
        "ACK Flag Count":       flow.ack_count,
    }

    # ── Issue 4: validate feature bounds before returning ────────────────
    if not _validate_features(row):
        return None  # caller logs WARNING

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # ── Issue 6: debug dump ──────────────────────────────────────────────
    if debug:
        _print_debug(flow.src_ip, row)

    return df


def _validate_features(row: dict) -> bool:
    """Return False if any feature value is NaN, Inf, negative duration,
    or exceeds plausible physical bounds (Issue 4)."""
    for col, val in row.items():
        if val is None or (isinstance(val, float) and (
                np.isnan(val) or np.isinf(val))):
            return False
    if row["Flow Duration"] < 0:
        return False
    if row["Flow Bytes/s"] > _MAX_FLOW_BYTES_PER_SEC:
        return False
    if row["Flow Packets/s"] > _MAX_FLOW_PACKETS_PER_SEC:
        return False
    return True


def _print_debug(src_ip: str, row: dict) -> None:
    """Print pre-inference feature dump (Issue 6 — --debug mode)."""
    print(
        f"  [DEBUG] Flow → {src_ip}\n"
        f"          Packet Count   : {int(row['Flow Packets/s'] * (row['Flow Duration'] / 1e6))}\n"
        f"          Flow Duration  : {row['Flow Duration']:.1f} µs\n"
        f"          Flow Bytes/s   : {row['Flow Bytes/s']:.2f}\n"
        f"          Flow Packets/s : {row['Flow Packets/s']:.2f}\n"
        f"          Flow IAT Mean  : {row['Flow IAT Mean']:.2f} µs\n"
        f"          Flow IAT Max   : {row['Flow IAT Max']:.2f} µs\n"
        f"          SYN Count      : {int(row['SYN Flag Count'])}\n"
        f"          ACK Count      : {int(row['ACK Flag Count'])}",
        flush=True,
    )


def extract_all_features(
    flows: Dict[FlowKey, FlowRecord],
    *,
    debug: bool = False,
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Extract features for every flow in a flushed flow dict.

    Flows that are incomplete (single-packet) or produce invalid features
    are silently skipped (Issues 2 & 4).

    Returns:
        List of (feature_df, src_ip) tuples — one per valid flow.
    """
    results = []
    skipped = 0
    invalid = 0
    for _key, record in flows.items():
        packet_count = len(record.pkt_lengths)
        duration_sec = record.last_time - record.start_time
        if packet_count < 2 or duration_sec <= 0:
            skipped += 1
            if debug:
                print(f"  [DEBUG] Skipping incomplete flow: {record.src_ip} "
                      f"(packets={packet_count}, duration={duration_sec:.6f}s)",
                      flush=True)
            continue

        df = extract_features(record, debug=debug)
        if df is None:
            invalid += 1
            if debug:
                print(f"  [WARNING] Invalid feature vector discarded: "
                      f"{record.src_ip}", flush=True)
            continue

        results.append((df, record.src_ip))

    if skipped > 0 and debug:
        print(f"  [DEBUG] Skipped {skipped} incomplete flow(s)", flush=True)
    if invalid > 0:
        print(f"  [WARNING] Discarded {invalid} invalid feature vector(s)",
              flush=True)

    return results
