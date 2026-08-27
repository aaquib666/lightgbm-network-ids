"""
flow_tracker.py — Stateful Per-Flow Aggregation

Dissertation Reference:
    Chapter 2, Section 2.3.4 — Flow-based traffic preprocessing

Maintains a dictionary of active flows keyed by
(src_ip, dst_ip, src_port, dst_port, protocol).

For each packet the matching flow's running statistics are updated:
    - packet lengths
    - packet arrival timestamps
    - SYN / ACK flag counts
    - flow start time and last-seen time
"""

import time
from typing import Dict, Tuple

from scapy.layers.inet import IP, TCP, UDP, ICMP


# Type alias for the 5-tuple flow key
FlowKey = Tuple[str, str, int, int, int]


class FlowRecord:
    """Mutable accumulator for a single network flow."""

    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "pkt_lengths", "timestamps", "syn_count", "ack_count",
        "start_time", "last_time",
    )

    def __init__(self, src_ip: str, dst_ip: str, src_port: int,
                 dst_port: int, protocol: int, timestamp: float,
                 pkt_len: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.pkt_lengths = [pkt_len]
        self.timestamps = [timestamp]
        self.syn_count = 0
        self.ack_count = 0
        self.start_time = timestamp
        self.last_time = timestamp

    def update(self, pkt_len: int, timestamp: float,
               is_syn: bool, is_ack: bool) -> None:
        self.pkt_lengths.append(pkt_len)
        self.timestamps.append(timestamp)
        self.last_time = timestamp
        if is_syn:
            self.syn_count += 1
        if is_ack:
            self.ack_count += 1


# Default idle timeout — a flow is considered complete when no packets
# have been seen for this many seconds.  Prevents premature flushing of
# active flows that would otherwise produce single-packet / zero-duration
# feature vectors and trigger false positives.
FLOW_IDLE_TIMEOUT: float = 5.0


class FlowTracker:
    """Maintains a buffer of active flows and flushes idle ones on demand."""

    def __init__(self):
        self._flows: Dict[FlowKey, FlowRecord] = {}

    def process_packet(self, packet) -> None:
        """Ingest a single Scapy packet and update the appropriate flow."""
        if not packet.haslayer(IP):
            return

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        pkt_len = len(packet)
        ts = float(packet.time)

        # Determine protocol, ports, and TCP flags
        is_syn = False
        is_ack = False

        if packet.haslayer(TCP):
            tcp = packet[TCP]
            protocol = 6
            src_port = tcp.sport
            dst_port = tcp.dport
            flags = tcp.flags
            is_syn = bool(flags & 0x02)
            is_ack = bool(flags & 0x10)
        elif packet.haslayer(UDP):
            udp = packet[UDP]
            protocol = 17
            src_port = udp.sport
            dst_port = udp.dport
        elif packet.haslayer(ICMP):
            # CICFlowMeter encodes non-TCP/UDP traffic as protocol 0 in the
            # training data (not the raw IP protocol number 1) — match that
            # convention so the model sees the same category it was trained
            # on. ICMP has no ports, so 0/0 naturally merges all ICMP
            # packets between the same two hosts into one flow.
            protocol = 0
            src_port = 0
            dst_port = 0
        else:
            return  # ignore anything else

        fwd_key: FlowKey = (src_ip, dst_ip, src_port, dst_port, protocol)
        rev_key: FlowKey = (dst_ip, src_ip, dst_port, src_port, protocol)

        if fwd_key in self._flows:
            self._flows[fwd_key].update(pkt_len, ts, is_syn, is_ack)
        elif rev_key in self._flows:
            self._flows[rev_key].update(pkt_len, ts, is_syn, is_ack)
        else:
            record = FlowRecord(src_ip, dst_ip, src_port, dst_port,
                                protocol, ts, pkt_len)
            if is_syn:
                record.syn_count += 1
            if is_ack:
                record.ack_count += 1
            self._flows[fwd_key] = record

    def flush_expired_flows(
        self, timeout_seconds: float = FLOW_IDLE_TIMEOUT,
    ) -> Dict[FlowKey, FlowRecord]:
        """Return only completed (idle) flows and preserve active ones.

        A flow is considered expired when:
            current_time - flow.last_time > timeout_seconds

        Expired flows are removed from the internal buffer and returned.
        Active flows remain untouched.
        """
        now = time.time()
        expired: Dict[FlowKey, FlowRecord] = {}
        active: Dict[FlowKey, FlowRecord] = {}

        for key, record in self._flows.items():
            if now - record.last_time > timeout_seconds:
                expired[key] = record
            else:
                active[key] = record

        self._flows = active
        return expired

    def flush_flows(self) -> Dict[FlowKey, FlowRecord]:
        """Return ALL flows and reset the buffer (used by simulation mode)."""
        flushed = self._flows
        self._flows = {}
        return flushed
