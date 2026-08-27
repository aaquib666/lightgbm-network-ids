"""
logger.py — Dual Logger (Terminal + File)

Dissertation Reference:
    Chapter 5, Section 5.5 — User Interface (terminal-based reporting)

Implements simultaneous logging to:
    1. stdout with ANSI colour-coded output
    2. Plain-text log file with ISO timestamps
"""

import os
import sys
import logging
from datetime import datetime, timezone


# ── ANSI colour codes ────────────────────────────────────────────────────────
class _Colours:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


COLOURS = _Colours()

# Tag → colour mapping
_TAG_COLOURS = {
    "ATTACK":     COLOURS.RED,
    "BLOCKED":    COLOURS.YELLOW,
    "DRY-RUN":    COLOURS.YELLOW,
    "INFO":       COLOURS.BLUE,
    "MODEL":      COLOURS.CYAN,
    "STARTED":    COLOURS.GREEN,
    "SHUTDOWN":   COLOURS.BLUE,
    "SUMMARY":    COLOURS.BLUE,
    "SELINUX":    COLOURS.CYAN,
    "SIMULATION": COLOURS.CYAN,
    "SUSPICIOUS": COLOURS.YELLOW,
    "SAFE MODE":  COLOURS.GREEN,
    "WARNING":    COLOURS.YELLOW,
}


class IDSLogger:
    """Dual-output logger for the IDS prototype."""

    def __init__(self, log_path: str = "logs/ids_detections.log"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # File handler — plain text, append mode
        self._file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))

        self._file_logger = logging.getLogger("ids_file")
        self._file_logger.setLevel(logging.INFO)
        self._file_logger.addHandler(self._file_handler)
        self._file_logger.propagate = False

    # ── public helpers ────────────────────────────────────────────────────

    def log(self, tag: str, message: str, *, to_file: bool = True,
            simulation: bool = False) -> None:
        """Print a tagged, colour-coded line to stdout and optionally to the log file."""
        prefix = "[SIMULATION MODE] " if simulation else ""
        colour = _TAG_COLOURS.get(tag, COLOURS.RESET)

        # Terminal (coloured)
        terminal_line = f"{prefix}{colour}[{tag}]{COLOURS.RESET} {message}"
        print(terminal_line, flush=True)

        # File (plain, timestamped)
        if to_file:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            file_line = f"{ts} | {tag} | {message}"
            self._file_logger.info(file_line)

    def log_detection(self, src_ip: str, confidence: float,
                      packet_count: int, bytes_per_sec: float,
                      syn_count: int, ack_count: int,
                      latency_us: float, blocked: bool,
                      *, simulation: bool = False) -> None:
        """Log a structured attack detection event."""
        prefix = "[SIMULATION MODE] " if simulation else ""

        # Terminal — rich multi-line block
        block = (
            f"\n{prefix}{COLOURS.RED}[ATTACK]{COLOURS.RESET} DoS/DDoS attack detected!\n"
            f"         Source IP   : {src_ip}\n"
            f"         Confidence  : {confidence * 100:.2f}%\n"
            f"         Flow Stats  : {packet_count:,} packets | "
            f"{bytes_per_sec / 1e6:.1f} MB/s | SYN={syn_count}, ACK={ack_count}\n"
            f"         Latency     : {latency_us:.1f} µs/flow\n"
        )
        print(block, flush=True)

        # File — single structured line
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        status = "YES" if blocked else "NO"
        file_line = (
            f"{ts} | ATTACK | SRC: {src_ip} | CONFIDENCE: {confidence:.4f} | "
            f"PACKETS: {packet_count} | BLOCKED: {status}"
        )
        self._file_logger.info(file_line)

    def log_raw(self, text: str) -> None:
        """Print raw text to terminal only (banners, summaries)."""
        print(text, flush=True)
