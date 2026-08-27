"""
packet_capture.py — Live Packet Sniffer

Dissertation Reference:
    Chapter 3, Section 3.4.3.2 — Scapy for live packet sniffing

Uses Scapy's AsyncSniffer (non-blocking) to capture TCP and UDP packets
on the specified interface, forwarding each to the FlowTracker for
stateful per-flow aggregation.
"""

from scapy.all import AsyncSniffer

from flow_tracker import FlowTracker


class PacketCapture:
    """Non-blocking packet sniffer backed by Scapy's AsyncSniffer."""

    def __init__(self, interface: str, flow_tracker: FlowTracker):
        self._interface = interface
        self._flow_tracker = flow_tracker
        self._sniffer = None

    def _packet_callback(self, packet) -> None:
        """Called for every captured packet — delegates to FlowTracker."""
        self._flow_tracker.process_packet(packet)

    def start(self) -> None:
        """Begin non-blocking packet capture on the configured interface."""
        self._sniffer = AsyncSniffer(
            iface=self._interface,
            filter="tcp or udp or icmp",
            prn=self._packet_callback,
            store=False,
        )
        self._sniffer.start()

    def stop(self) -> None:
        """Halt the async sniffer gracefully."""
        if self._sniffer is not None:
            self._sniffer.stop()
            self._sniffer = None
