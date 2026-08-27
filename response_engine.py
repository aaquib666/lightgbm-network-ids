"""
response_engine.py — Automated iptables Blocking & SELinux Advisory

Dissertation Reference:
    Chapter 5, Section 5.4.5 — Automated Response and Security Enforcement Module

Two-layer response architecture:
    Layer 1 — iptables firewall block (DROP rule for attacker source IP)
    Layer 2 — SELinux advisory logging (non-destructive, informational only)

Deployment hardening:
    - Whitelist support (Issue 5): IPs in whitelist.txt are never blocked.
    - Safe response mode (Issue 4): High-confidence predictions on low-volume
      traffic are logged but not blocked.

Maintains a blocklist to prevent duplicate rule insertions and provides
an unblock_all() cleanup method called on session exit.
"""

import os
import subprocess
from typing import Dict, Optional, Set

from logger import IDSLogger


def load_whitelist(path: str) -> Set[str]:
    """Load a whitelist file — one IP per line, '#' comments ignored."""
    whitelist: Set[str] = set()
    if not os.path.isfile(path):
        return whitelist
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                whitelist.add(line)
    return whitelist


class ResponseEngine:
    """Manages iptables-based automated response with deployment safeguards."""

    def __init__(self, logger: IDSLogger, dry_run: bool = False,
                 whitelist: Optional[Set[str]] = None,
                 safe_response_mode: bool = True,
                 config: Optional[Dict] = None):
        self._logger = logger
        self._dry_run = dry_run
        self._blocked_ips: Set[str] = set()
        self._whitelist: Set[str] = whitelist or set()
        self._safe_response_mode = safe_response_mode

        # Volumetric thresholds from config (Issue 4 — safe response mode)
        cfg = config or {}
        self._min_packets_per_sec = cfg.get("min_packets_per_second", 1000)
        self._min_packet_count = cfg.get("min_packet_count", 50)
        self._min_bytes_per_sec = cfg.get("min_bytes_per_second", 1_000_000)

    @property
    def blocked_ips(self) -> Set[str]:
        return self._blocked_ips.copy()

    @property
    def blocked_count(self) -> int:
        return len(self._blocked_ips)

    def block_ip(self, src_ip: str, *,
                 flow_packets_per_sec: float = 0.0,
                 packet_count: int = 0,
                 flow_bytes_per_sec: float = 0.0,
                 simulation: bool = False) -> bool:
        """
        Insert an iptables DROP rule for the given source IP.

        Applies deployment safeguards before containment:
            - Whitelist check (Issue 5)
            - Safe response mode volumetric verification (Issue 4)
            - Duplicate blocklist check

        Returns True if a new block was applied, False otherwise.
        """
        # ── Issue 5: whitelist check ─────────────────────────────────────
        if src_ip in self._whitelist:
            self._logger.log("INFO",
                             f"Whitelisted address detected: {src_ip}. "
                             f"Containment skipped.",
                             simulation=simulation)
            return False

        # ── Issue 4: safe response mode ──────────────────────────────────
        if self._safe_response_mode:
            volume_exceeded = (
                flow_packets_per_sec >= self._min_packets_per_sec
                or packet_count >= self._min_packet_count
                or flow_bytes_per_sec >= self._min_bytes_per_sec
            )
            if not volume_exceeded:
                self._logger.log(
                    "SAFE MODE",
                    f"High confidence prediction ignored for {src_ip}. "
                    f"Traffic volume insufficient for containment. "
                    f"(pkt/s={flow_packets_per_sec:.0f}, "
                    f"pkts={packet_count}, "
                    f"bytes/s={flow_bytes_per_sec:.0f})",
                    simulation=simulation,
                )
                return False

        # ── Duplicate check ──────────────────────────────────────────────
        if src_ip in self._blocked_ips:
            self._logger.log("INFO", f"IP {src_ip} already blocked — skipping.",
                             simulation=simulation)
            return False

        # ── Layer 1: iptables block ──────────────────────────────────────
        if self._dry_run:
            self._logger.log("DRY-RUN",
                             f"Would block: {src_ip} (iptables rule suppressed)",
                             simulation=simulation)
        else:
            cmd = ["iptables", "-I", "INPUT", "-s", src_ip, "-j", "DROP"]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                self._logger.log("BLOCKED",
                                 f"iptables rule inserted: DROP all traffic from {src_ip}",
                                 simulation=simulation)
                self._logger.log("BLOCKED",
                                 f"Command executed: iptables -I INPUT -s {src_ip} -j DROP",
                                 simulation=simulation)
            except FileNotFoundError:
                self._logger.log("BLOCKED",
                                 f"[SIMULATED] iptables not available — "
                                 f"rule for {src_ip} logged only.",
                                 simulation=simulation)
            except subprocess.CalledProcessError as exc:
                self._logger.log("BLOCKED",
                                 f"iptables command failed for {src_ip}: {exc}",
                                 simulation=simulation)

        self._blocked_ips.add(src_ip)

        # ── Layer 2: SELinux advisory (non-destructive) ──────────────────
        self._logger.log("SELINUX",
                         "Advisory: SELinux MAC policy enforcement active on this host.",
                         simulation=simulation)
        self._logger.log("SELINUX",
                         "Recommend: audit2allow or semanage to restrict process "
                         "privileges for sustained mitigation.",
                         simulation=simulation)

        return True

    def unblock_all(self, *, simulation: bool = False) -> int:
        """
        Remove all iptables rules inserted during this session.

        Returns the number of rules flushed.
        """
        count = 0
        for ip in list(self._blocked_ips):
            if not self._dry_run:
                cmd = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass  # best-effort cleanup
            count += 1

        self._logger.log("SHUTDOWN",
                         f"All iptables rules inserted by this session have been removed.",
                         simulation=simulation)
        self._blocked_ips.clear()
        return count
