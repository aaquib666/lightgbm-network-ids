#!/usr/bin/env python3
"""
ids_main.py — Entry Point & CLI Launcher

Dissertation Reference:
    Chapter 5, Section 5.5 — Linux background detection process

Main loop: capture packets → aggregate flows → extract features →
run LightGBM inference → hybrid detection policy → automated response.

Supports three operating modes:
    1. Live mode   — real packet capture with Scapy (requires root)
    2. Dry-run     — live capture but no iptables modifications
    3. Simulation  — synthetic flow generation, no root needed

Deployment hardening (post-training mitigation):
    - Three-tier classification: NORMAL / SUSPICIOUS / ATTACK
    - Hybrid detection policy: confidence + volumetric thresholds
    - Safe response mode: blocks only high-volume attack traffic
    - Whitelist support: never blocks whitelisted IPs
    - All thresholds loaded from config/ids_config.yaml
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import pandas as pd

from logger import IDSLogger, COLOURS
from response_engine import ResponseEngine, load_whitelist
from feature_extractor import FEATURE_COLUMNS


# ── Configuration Loader ─────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    """Load YAML configuration file.  Falls back to defaults if missing."""
    defaults = {
        "suspicious_threshold": 0.70,
        "detection_threshold": 0.90,
        "min_packets_per_second": 1000,
        "min_packet_count": 50,
        "min_bytes_per_second": 1_000_000,
        "min_scoring_packet_count": 40,      # new
        "min_scoring_duration_sec": 1.0,     # new
        "connection_flood_threshold": 30,    # new
        "distributed_flood_min_sources": 10,   # new
        "safe_response_mode": True,
        "whitelist_file": "config/whitelist.txt",
    }
    if not os.path.isfile(config_path):
        return defaults
    try:
        # Use PyYAML if available, otherwise fall back to simple parser
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        merged = {**defaults, **user_cfg}
        return merged
    except ImportError:
        # Minimal YAML-subset parser (key: value lines only)
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = {}
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    # Type coercion
                    if val.lower() in ("true", "yes"):
                        val = True
                    elif val.lower() in ("false", "no"):
                        val = False
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass  # keep as string
                    user_cfg[key] = val
        merged = {**defaults, **user_cfg}
        return merged


# ── Hybrid Decision Policy ───────────────────────────────────────────────────
# Issues 1, 2, 3, 4 — Three-tier classification with volumetric gates

def _classify_flow(confidence: float, flow_row: pd.Series,
                   packet_count: int, config: dict) -> str:
    """
    Apply the hybrid detection policy.

    Returns one of: "NORMAL", "SUSPICIOUS", "ATTACK"

    Decision tree:
        confidence < suspicious_threshold           → NORMAL
        suspicious_threshold ≤ confidence < detection_threshold → SUSPICIOUS
        confidence ≥ detection_threshold AND volumetric gate met → ATTACK
        confidence ≥ detection_threshold AND volumetric gate NOT met → SUSPICIOUS
    """
    susp_thresh = config["suspicious_threshold"]
    det_thresh = config["detection_threshold"]

    if confidence < susp_thresh:
        return "NORMAL"

    if confidence < det_thresh:
        return "SUSPICIOUS"

    # High confidence — check volumetric thresholds
    volume_exceeded = (
        flow_row["Flow Packets/s"] >= config["min_packets_per_second"]
        or packet_count >= config["min_packet_count"]
        or flow_row["Flow Bytes/s"] >= config["min_bytes_per_second"]
    )

    if volume_exceeded:
        return "ATTACK"
    else:
        return "SUSPICIOUS"


def _decision_reason(classification: str, confidence: float,
                     flow_row: pd.Series, packet_count: int,
                     config: dict) -> str:
    """Generate a human-readable reason string for the classification."""
    if classification == "NORMAL":
        return f"Confidence {confidence:.4f} below suspicious threshold {config['suspicious_threshold']:.2f}."

    if classification == "ATTACK":
        reasons = []
        if flow_row["Flow Packets/s"] >= config["min_packets_per_second"]:
            reasons.append(f"packets/sec={flow_row['Flow Packets/s']:.0f} ≥ {config['min_packets_per_second']}")
        if packet_count >= config["min_packet_count"]:
            reasons.append(f"packet_count={packet_count} ≥ {config['min_packet_count']}")
        if flow_row["Flow Bytes/s"] >= config["min_bytes_per_second"]:
            reasons.append(f"bytes/sec={flow_row['Flow Bytes/s']:.0f} ≥ {config['min_bytes_per_second']}")
        return (f"Confidence {confidence:.4f} ≥ {config['detection_threshold']:.2f} "
                f"AND volumetric threshold exceeded: {'; '.join(reasons)}.")

    # SUSPICIOUS
    if confidence >= config["detection_threshold"]:
        return ("High confidence but traffic volume below attack threshold. "
                "Logged only — no containment.")
    return (f"Confidence {confidence:.4f} between suspicious "
            f"({config['suspicious_threshold']:.2f}) and attack "
            f"({config['detection_threshold']:.2f}) thresholds. "
            f"Logged only — no containment.")


def _log_classified_flow(logger: IDSLogger, classification: str,
                         src_ip: str, confidence: float,
                         flow_row: pd.Series, packet_count: int,
                         latency_us: float, reason: str,
                         *, simulation: bool = False, debug: bool = False):
    """Log a SUSPICIOUS or ATTACK flow with full detail (Issue 3)."""
    protocol = int(flow_row["Protocol"])
    proto_name = "TCP" if protocol == 6 else ("UDP" if protocol == 17 else str(protocol))

    if classification == "ATTACK":
        # Use the existing rich log_detection for attacks
        logger.log_detection(
            src_ip=src_ip,
            confidence=confidence,
            packet_count=packet_count,
            bytes_per_sec=flow_row["Flow Bytes/s"],
            syn_count=int(flow_row["SYN Flag Count"]),
            ack_count=int(flow_row["ACK Flag Count"]),
            latency_us=latency_us,
            blocked=True,
            simulation=simulation,
        )
        logger.log("ATTACK", f"Reason: {reason}", simulation=simulation)

    elif classification == "SUSPICIOUS":
        block = (
            f"\n{COLOURS.YELLOW}[SUSPICIOUS]{COLOURS.RESET} Elevated threat score — not blocked.\n"
            f"         Source IP    : {src_ip}\n"
            f"         Protocol     : {proto_name}\n"
            f"         Confidence   : {confidence * 100:.2f}%\n"
            f"         Packet Count : {packet_count}\n"
            f"         Flow Duration: {flow_row['Flow Duration']:.0f} µs\n"
            f"         Flow Bytes/s : {flow_row['Flow Bytes/s']:.2f}\n"
            f"         Flow Pkts/s  : {flow_row['Flow Packets/s']:.2f}\n"
            f"         SYN Count    : {int(flow_row['SYN Flag Count'])}\n"
            f"         ACK Count    : {int(flow_row['ACK Flag Count'])}\n"
            f"         Latency      : {latency_us:.1f} µs/flow\n"
            f"         Reason       : {reason}\n"
        )
        sim_prefix = "[SIMULATION MODE] " if simulation else ""
        print(f"{sim_prefix}{block}", flush=True)

        # Also write to log file
        logger.log("SUSPICIOUS",
                   f"SRC: {src_ip} | CONFIDENCE: {confidence:.4f} | "
                   f"PKTS: {packet_count} | REASON: {reason}",
                   simulation=simulation)

    # Issue 6 / Issue 3 — debug feature dump for non-normal flows
    if debug and classification != "NORMAL":
        print(
            f"  [DEBUG] Classification detail for {src_ip}:\n"
            f"          Protocol        : {proto_name}\n"
            f"          Packet Count    : {packet_count}\n"
            f"          Flow Duration   : {flow_row['Flow Duration']:.0f} µs\n"
            f"          Flow Bytes/s    : {flow_row['Flow Bytes/s']:.2f}\n"
            f"          Flow Packets/s  : {flow_row['Flow Packets/s']:.2f}\n"
            f"          Flow IAT Mean   : {flow_row['Flow IAT Mean']:.2f} µs\n"
            f"          Flow IAT Max    : {flow_row['Flow IAT Max']:.2f} µs\n"
            f"          SYN Count       : {int(flow_row['SYN Flag Count'])}\n"
            f"          ACK Count       : {int(flow_row['ACK Flag Count'])}\n"
            f"          Confidence      : {confidence:.4f}\n"
            f"          Decision        : {classification}\n"
            f"          Decision Reason : {reason}",
            flush=True,
        )


# ── Banner ───────────────────────────────────────────────────────────────────

def _print_banner(args, config: dict, logger: IDSLogger) -> None:
    mode_tag = ""
    if args.simulate:
        mode_tag = "  |  Mode: SIMULATION"
    elif args.dry_run:
        mode_tag = "  |  Mode: DRY-RUN"

    banner = f"""
{COLOURS.CYAN}╔══════════════════════════════════════════════════════════════╗
║   LightGBM DoS/DDoS IDS — Aaquib Rizwan (CB012956)          ║
║   Staffordshire University — BSc Cyber Security (Hons)       ║
║   Interface: {args.interface:<8s}|  Model: {args.model:<26s}║
║   Suspicious: {config['suspicious_threshold']:.2f}  |  Attack: {config['detection_threshold']:.2f}  |  Window: {args.window:.1f}s{mode_tag:<10s}║
║   Safe Response: {'ON' if config['safe_response_mode'] else 'OFF':<4s}|  Whitelist: {args.config:<27s}║
╚══════════════════════════════════════════════════════════════╝{COLOURS.RESET}"""
    logger.log_raw(banner)


# ── Simulation Flow Generator ────────────────────────────────────────────────

_SIM_NORMAL_TEMPLATE = {
    "Flow Bytes/s":         lambda: random.uniform(500, 50_000),
    "Flow Packets/s":       lambda: random.uniform(5, 200),
    "Avg Packet Size":      lambda: random.uniform(60, 1500),
    "Flow Duration":        lambda: random.uniform(500_000, 5_000_000),
    "Flow IAT Mean":        lambda: random.uniform(5_000, 500_000),
    "Flow IAT Max":         lambda: random.uniform(50_000, 2_000_000),
    "Protocol":             lambda: random.choice([6, 17]),
    "SYN Flag Count":       lambda: random.randint(1, 5),
    "ACK Flag Count":       lambda: random.randint(5, 50),
}

_SIM_ATTACK_TEMPLATE = {
    "Flow Bytes/s":         lambda: random.uniform(5_000_000, 50_000_000),
    "Flow Packets/s":       lambda: random.uniform(5_000, 150_000),
    "Avg Packet Size":      lambda: random.uniform(40, 80),
    "Flow Duration":        lambda: random.uniform(100, 10_000),
    "Flow IAT Mean":        lambda: random.uniform(1, 100),
    "Flow IAT Max":         lambda: random.uniform(5, 500),
    "Protocol":             lambda: 6,
    "SYN Flag Count":       lambda: random.randint(4500, 5000),
    "ACK Flag Count":       lambda: random.randint(5, 20),
}


def _generate_sim_flows(template, count: int):
    """Generate `count` synthetic flow feature vectors from a template."""
    rows = []
    for _ in range(count):
        row = {col: gen() for col, gen in template.items()}
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

# ── Connection-Rate Heuristics (catches what per-flow ML misses) ────────────

def _check_connection_flood(flows, own_ip, response, logger, config, *,
                             window_seconds, simulation=False):
    """
    Volume/rate-based supplements to the ML classifier. Three checks, each
    for a flood shape the per-flow LightGBM path can miss on its own:

    1. Per-source connection flood — many small flows from ONE source IP
       (hping3's default port-churn behaviour on SYN/UDP floods).
    2. Per-destination distributed flood — many small flows targeting ONE
       destination from MANY distinct source IPs (hping3 --rand-source,
       where spoofing defeats check 1).
    3. Single-flow volumetric backstop — one merged flow whose own volume
       already crosses the project's attack-scale thresholds, regardless
       of ML confidence (e.g. ICMP floods: the training data has ~no
       labelled ICMP attacks, and ICMP has no ports, so it collapses into
       one large flow rather than many small ones).
    """
    threshold = config.get("connection_flood_threshold", 30)
    min_sources = config.get("distributed_flood_min_sources", 10)

    src_counts, src_packets = {}, {}
    dst_counts, dst_sources = {}, {}

    for record in flows.values():
        if own_ip is not None and record.src_ip == own_ip:
            continue
        src_counts[record.src_ip] = src_counts.get(record.src_ip, 0) + 1
        src_packets[record.src_ip] = src_packets.get(record.src_ip, 0) + len(record.pkt_lengths)
        dst_counts[record.dst_ip] = dst_counts.get(record.dst_ip, 0) + 1
        dst_sources.setdefault(record.dst_ip, set()).add(record.src_ip)

    already_flagged = set()

    # ── 1. Per-source connection flood ────────────────────────────────────
    for src_ip, conn_count in src_counts.items():
        if conn_count < threshold:
            continue
        total_packets = src_packets[src_ip]
        pkt_rate = total_packets / window_seconds

        logger.log("ATTACK",
                   f"Connection-flood pattern detected from {src_ip}: "
                   f"{conn_count} distinct connections in {window_seconds:.1f}s "
                   f"(threshold: {threshold})", simulation=simulation)

        blocked = response.block_ip(
            src_ip, flow_packets_per_sec=pkt_rate, packet_count=total_packets,
            flow_bytes_per_sec=0.0, simulation=simulation,
        )
        if blocked:
            logger.log("BLOCKED",
                       f"Confirmed: {src_ip} contained after connection-flood "
                       f"detection ({conn_count} connections in "
                       f"{window_seconds:.1f}s).", simulation=simulation)
        already_flagged.add(src_ip)

    # ── 2. Per-destination distributed / spoofed flood ────────────────────
    for dst_ip, conn_count in dst_counts.items():
        distinct_sources = dst_sources[dst_ip]
        if conn_count < threshold or len(distinct_sources) < min_sources:
            continue

        logger.log("ATTACK",
                   f"Distributed flood pattern detected targeting {dst_ip}: "
                   f"{conn_count} connections from {len(distinct_sources)} "
                   f"distinct source IPs in {window_seconds:.1f}s — no single "
                   f"source crossed the per-source threshold, consistent "
                   f"with IP spoofing.", simulation=simulation)

        blocked_any = False
        for src_ip in distinct_sources:
            if src_ip in already_flagged:
                continue
            blocked = response.block_ip(
                src_ip,
                flow_packets_per_sec=src_packets.get(src_ip, 0) / window_seconds,
                packet_count=src_packets.get(src_ip, 0),
                flow_bytes_per_sec=0.0, simulation=simulation,
            )
            blocked_any = blocked_any or blocked

        if blocked_any:
            logger.log("BLOCKED",
                       f"Contained {len(distinct_sources)} source IP(s) "
                       f"observed attacking {dst_ip}. Note: with spoofed "
                       f"sources, per-IP blocking has limited durable "
                       f"effect — rate-limiting at {dst_ip} is the "
                       f"stronger mitigation.", simulation=simulation)

    # ── 3. Single-flow volumetric backstop ─────────────────────────────────
    min_pkts = config.get("min_packet_count", 50)
    min_pps = config.get("min_packets_per_second", 1000)
    min_bps = config.get("min_bytes_per_second", 1_000_000)

    for record in flows.values():
        if own_ip is not None and record.src_ip == own_ip:
            continue
        if record.src_ip in already_flagged:
            continue

        packet_count = len(record.pkt_lengths)
        duration_sec = record.last_time - record.start_time
        if duration_sec <= 0:
            continue
        total_bytes = sum(record.pkt_lengths)
        pkt_rate = packet_count / duration_sec
        byte_rate = total_bytes / duration_sec

        if not (packet_count >= min_pkts or pkt_rate >= min_pps or byte_rate >= min_bps):
            continue

        logger.log("ATTACK",
                   f"Volumetric flood detected from {record.src_ip} "
                   f"(protocol {record.protocol}): {packet_count} packets, "
                   f"{byte_rate/1e6:.2f} MB/s in a single flow — exceeds "
                   f"attack-scale volume regardless of model confidence.",
                   simulation=simulation)

        blocked = response.block_ip(
            record.src_ip, flow_packets_per_sec=pkt_rate,
            packet_count=packet_count, flow_bytes_per_sec=byte_rate,
            simulation=simulation,
        )
        if blocked:
            logger.log("BLOCKED",
                       f"Confirmed: {record.src_ip} contained after "
                       f"volumetric-flood detection.", simulation=simulation)

# ── Shared Per-Flow Processing ───────────────────────────────────────────────

def _process_flow(feature_df, src_ip, model_engine, response, logger, config,
                  *, debug=False, simulation=False, dry_run=False, own_ip=None):
    """
    Run inference → classify → respond for a single flow.

    Returns (classification, latency_us) tuple.
    """
    # ── Self-traffic gate ───────────────────────────────────────────────
    # If this flow's initiator is the monitoring host itself, it is by
    # definition not an incoming attack — skip scoring entirely so it
    # never appears as SUSPICIOUS/ATTACK. A flood aimed AT this host has
    # the remote attacker as src_ip, so genuine attacks are unaffected.
    if own_ip is not None and src_ip == own_ip:
        return "NORMAL", 0.0

    row = feature_df.iloc[0]
    duration_us = row["Flow Duration"]
    duration_sec = duration_us / 1e6 if duration_us > 0 else 1.0
    packet_count = int(row["Flow Packets/s"] * duration_sec)

    # ── Minimum flow-maturity gate ────────────────────────────────────
    # Rate features (Flow Bytes/s, Flow Packets/s) are unreliable on
    # very short, low-packet flows: a tiny denominator (duration) makes
    # trivial background traffic look like a high-rate burst. The model
    # was trained on full completed flows and was never shown enough
    # examples this small to judge them safely, so skip scoring rather
    # than trust a confidence figure built on a handful of packets.
    min_pkts = config.get("min_scoring_packet_count", 40)
    min_dur = config.get("min_scoring_duration_sec", 1.0)
    if packet_count < min_pkts and duration_sec < min_dur:
        return "NORMAL", 0.0

    confidence, latency_us = model_engine.predict(feature_df, debug=debug)

    classification = _classify_flow(confidence, row, packet_count, config)
    reason = _decision_reason(classification, confidence, row, packet_count, config)

    if classification in ("SUSPICIOUS", "ATTACK"):
        _log_classified_flow(
            logger, classification, src_ip, confidence,
            row, packet_count, latency_us, reason,
            simulation=simulation, debug=debug,
        )

    if classification == "ATTACK":
        response.block_ip(
            src_ip,
            flow_packets_per_sec=row["Flow Packets/s"],
            packet_count=packet_count,
            flow_bytes_per_sec=row["Flow Bytes/s"],
            simulation=simulation,
        )

    return classification, latency_us


# ── Simulation Mode Main Loop ───────────────────────────────────────────────

def _run_simulation(args, logger, model_engine, response, config):
    """Full demo pipeline without root access or live traffic."""
    debug = getattr(args, "debug", False)
    logger.log("SIMULATION", "Simulation mode active — no live traffic, no root required.",
               simulation=True)
    logger.log("INFO", "Sniffing simulated traffic... (Ctrl+C to stop)", simulation=True)

    total_flows = 0
    total_attacks = 0
    total_suspicious = 0
    iteration = 0

    # Pattern: 5 normal → 3 attack → repeat
    SIM_PATTERN = (
        [("normal", random.randint(8, 15)) for _ in range(5)] +
        [("attack", random.randint(3, 6)) for _ in range(3)]
    )

    try:
        while True:
            phase, flow_count = SIM_PATTERN[iteration % len(SIM_PATTERN)]

            if phase == "normal":
                df = _generate_sim_flows(_SIM_NORMAL_TEMPLATE, flow_count)
                sim_src_ip = "10.0.0.50"
            else:
                df = _generate_sim_flows(_SIM_ATTACK_TEMPLATE, flow_count)
                sim_src_ip = "192.168.1.105"

            window_latencies = []
            window_attacks = 0
            window_suspicious = 0

            for i in range(len(df)):
                feature_row = df.iloc[[i]]
                classification, latency_us = _process_flow(
                    feature_row, sim_src_ip, model_engine, response,
                    logger, config, debug=debug, simulation=True,
                    dry_run=args.dry_run,
                )
                window_latencies.append(latency_us)
                if classification == "ATTACK":
                    window_attacks += 1
                    total_attacks += 1
                elif classification == "SUSPICIOUS":
                    window_suspicious += 1
                    total_suspicious += 1

            total_flows += flow_count
            avg_lat = np.mean(window_latencies) if window_latencies else 0

            if window_attacks == 0 and window_suspicious == 0:
                logger.log("INFO",
                           f"Flow window flushed. Flows analysed: {flow_count:<4d}| "
                           f"Latency: ~{avg_lat:.1f} µs/flow",
                           simulation=True)

            time.sleep(args.window)
            iteration += 1

    except KeyboardInterrupt:
        pass

    return total_flows, total_attacks, total_suspicious


# ── Live Mode Main Loop ─────────────────────────────────────────────────────

def _run_live(args, logger, model_engine, response, config):
    """Real packet capture and inference loop."""
    from scapy.all import get_if_addr
    from flow_tracker import FlowTracker, FLOW_IDLE_TIMEOUT
    from packet_capture import PacketCapture
    from feature_extractor import extract_all_features

    debug = getattr(args, "debug", False)
    tracker = FlowTracker()
    capture = PacketCapture(args.interface, tracker)

    # ── Own-IP detection ──────────────────────────────────────────────
    # Flows this host itself initiates (DNS, updates, keepalives, etc.)
    # should never be scored as suspicious — only traffic where a REMOTE
    # host is the initiator matters for DoS/DDoS detection. Resolved
    # dynamically from the interface so it isn't hardcoded per-VM.
    try:
        own_ip = get_if_addr(args.interface)
        if not own_ip or own_ip == "0.0.0.0":
            own_ip = None
    except Exception:
        own_ip = None

    if own_ip:
        logger.log("INFO", f"Local address on {args.interface}: {own_ip} "
                            f"(self-originated traffic will be ignored)")
    else:
        logger.log("WARNING", f"Could not resolve local address on "
                               f"{args.interface} — self-traffic filtering disabled")

    logger.log("INFO", f"IDS engine started. Monitoring interface: {args.interface}")
    logger.log("INFO", f"Flow idle timeout: {FLOW_IDLE_TIMEOUT}s")
    logger.log("INFO", "Sniffing live traffic... (Ctrl+C to stop)")

    capture.start()

    total_flows = 0
    total_attacks = 0
    total_suspicious = 0

    try:
        while True:
            time.sleep(args.window)

            flows = tracker.flush_expired_flows()
            if not flows:
                continue

            _check_connection_flood(flows, own_ip, response, logger, config,
                                     window_seconds=args.window,
                                     simulation=False)

            feature_list = extract_all_features(flows, debug=debug)
            window_latencies = []

            for feature_df, src_ip in feature_list:
                classification, latency_us = _process_flow(
                    feature_df, src_ip, model_engine, response,
                    logger, config, debug=debug, simulation=False,
                    dry_run=args.dry_run, own_ip=own_ip,
                )
                window_latencies.append(latency_us)
                total_flows += 1
                if classification == "ATTACK":
                    total_attacks += 1
                elif classification == "SUSPICIOUS":
                    total_suspicious += 1

            avg_lat = np.mean(window_latencies) if window_latencies else 0
            logger.log("INFO",
                       f"Flow window flushed. Flows analysed: {len(feature_list):<4d}| "
                       f"Latency: ~{avg_lat:.1f} µs/flow")

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()

    return total_flows, total_attacks, total_suspicious


# ── Shutdown Summary ─────────────────────────────────────────────────────────

def _print_summary(logger, total_flows, total_attacks, total_suspicious,
                   response, *, simulation=False):
    blocked = response.blocked_ips
    blocked_list = ", ".join(blocked) if blocked else "none"

    logger.log_raw("")
    logger.log("SHUTDOWN", "IDS stopped by user.", simulation=simulation)
    summary = (
        f"\n{COLOURS.BLUE}[SUMMARY]{COLOURS.RESET} ─────────────────────────────────────\n"
        f"           Total flows analysed : {total_flows}\n"
        f"           Attacks detected     : {total_attacks}\n"
        f"           Suspicious flows     : {total_suspicious}\n"
        f"           IPs blocked          : {response.blocked_count}  ({blocked_list})\n"
    )
    logger.log_raw(summary)

    flushed = response.unblock_all(simulation=simulation)
    logger.log_raw(f"           Rules flushed        : {flushed}")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LightGBM DoS/DDoS IDS — Real-time detection and automated response"
    )
    parser.add_argument("--interface", default="ens33",
                        help="Network interface to sniff on (default: ens33)")
    parser.add_argument("--model", default="model/lgbm_model.pkl",
                        help="Path to the LightGBM .pkl model file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override detection_threshold from config")
    parser.add_argument("--window", type=float, default=2.0,
                        help="Time window in seconds to poll expired flows")
    parser.add_argument("--log", default="logs/ids_detections.log",
                        help="Path to the detection log file")
    parser.add_argument("--config", default="config/ids_config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect only — do NOT insert iptables rules")
    parser.add_argument("--simulate", action="store_true",
                        help="Run with synthetic traffic (no root, no live capture)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-flow feature vectors and prediction "
                             "confidence before inference")

    args = parser.parse_args()

    # ── Load configuration ───────────────────────────────────────────────
    config = _load_config(args.config)

    # CLI --threshold overrides config file
    if args.threshold is not None:
        config["detection_threshold"] = args.threshold

    # ── Initialise components ────────────────────────────────────────────
    logger = IDSLogger(log_path=args.log)
    _print_banner(args, config, logger)

    # Load whitelist
    whitelist_path = config.get("whitelist_file", "config/whitelist.txt")
    whitelist = load_whitelist(whitelist_path)
    if whitelist:
        logger.log("INFO", f"Whitelist loaded: {len(whitelist)} IP(s) from {whitelist_path}")
    else:
        logger.log("INFO", f"No whitelist loaded (file: {whitelist_path})")

    # Log config summary
    logger.log("INFO", f"Suspicious threshold : {config['suspicious_threshold']:.2f}")
    logger.log("INFO", f"Attack threshold     : {config['detection_threshold']:.2f}")
    logger.log("INFO", f"Min packets/sec      : {config['min_packets_per_second']}")
    logger.log("INFO", f"Min packet count     : {config['min_packet_count']}")
    logger.log("INFO", f"Min bytes/sec        : {config['min_bytes_per_second']}")
    logger.log("INFO", f"Safe response mode   : {'ON' if config['safe_response_mode'] else 'OFF'}")

    # In simulation mode, skip iptables entirely
    is_dry = args.dry_run or args.simulate
    response = ResponseEngine(
        logger, dry_run=is_dry, whitelist=whitelist,
        safe_response_mode=config["safe_response_mode"],
        config=config,
    )

    # Load model
    from model_inference import ModelInference
    try:
        model_engine = ModelInference(
            args.model, config["detection_threshold"], logger)
    except FileNotFoundError:
        logger.log("INFO", f"Model file not found: {args.model}")
        logger.log("INFO", "Place your trained lgbm_model.pkl in the model/ directory.")
        sys.exit(1)

    logger.log("STARTED", "IDS engine running. Monitoring for DoS/DDoS attacks...",
               simulation=args.simulate)

    # ── Dispatch to appropriate mode ─────────────────────────────────────
    if args.simulate:
        total_flows, total_attacks, total_suspicious = _run_simulation(
            args, logger, model_engine, response, config)
    else:
        total_flows, total_attacks, total_suspicious = _run_live(
            args, logger, model_engine, response, config)

    # ── Graceful shutdown ────────────────────────────────────────────────
    _print_summary(logger, total_flows, total_attacks, total_suspicious,
                   response, simulation=args.simulate)


if __name__ == "__main__":
    main()
