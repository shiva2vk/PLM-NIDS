"""
DPI-free PCAP tokenizer.

Parses raw PCAP files using Scapy, extracts ONLY L3/L4 header fields
(no payload inspection, no DPI), groups packets into flows by 5-tuple,
and emits per-packet token sequences.

Per-packet token layout (9 tokens per packet):
  [DIR] [LEN_BIN] [DT_BIN] [TTL_BIN] [PROTO] [SPORT_H] [DPORT_H] [FLAGS] [PKT_SEP]

Per-flow sequence:
  [FLOW_BOS] pkt1_tokens pkt2_tokens ... pktN_tokens [FLOW_EOS]

This fixes Gap-1 (truly DPI-free) and Gap-2 (sequences of 80-900 tokens)
simultaneously compared to the FlowMeter CSV approach.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Special token IDs (must match tokenizer.py) ───────────────────────────────
PAD_ID      = 0
BOS_ID      = 1   # <FLOW_BOS>
EOS_ID      = 2   # <FLOW_EOS>
PKT_SEP_ID  = 3   # <PKT_SEP>
UNK_ID      = 4
_N_SPECIAL  = 5

# Protocol token IDs (fixed, not binned)
PROTO_TCP   = 5
PROTO_UDP   = 6
PROTO_ICMP  = 7
PROTO_OTHER = 8

# Direction tokens
DIR_C2S = 9   # client → server
DIR_S2C = 10  # server → client

_N_FIXED = 11  # first dynamic token starts here

# Per-packet token positions (relative to packet start in sequence)
_TOK_DIR    = 0
_TOK_LEN    = 1
_TOK_DT     = 2
_TOK_TTL    = 3
_TOK_PROTO  = 4
_TOK_SPORT  = 5
_TOK_DPORT  = 6
_TOK_FLAGS  = 7
_TOK_SEP    = 8
TOKENS_PER_PKT = 9


class PcapFlowTokenizer:
    """
    Tokenizes raw PCAP files into per-flow token sequences.

    Parameters
    ----------
    n_len_bins  : bins for packet length (frame.len)
    n_dt_bins   : bins for inter-arrival time (Δt)
    n_ttl_bins  : bins for TTL / hop-limit
    n_port_buckets : hash buckets for src/dst ports
    max_pkts_per_flow : truncate flows longer than this
    flow_timeout : seconds of inactivity → new flow
    """

    def __init__(
        self,
        n_len_bins: int = 32,
        n_dt_bins: int = 32,
        n_ttl_bins: int = 16,
        n_port_buckets: int = 64,
        max_pkts_per_flow: int = 128,
        flow_timeout: float = 120.0,
    ) -> None:
        self.n_len_bins = n_len_bins
        self.n_dt_bins  = n_dt_bins
        self.n_ttl_bins = n_ttl_bins
        self.n_port_buckets = n_port_buckets
        self.max_pkts   = max_pkts_per_flow
        self.flow_timeout = flow_timeout

        # Bin edges — fitted on training PCAPs
        self._len_edges: Optional[np.ndarray] = None
        self._dt_edges:  Optional[np.ndarray] = None
        self._ttl_edges: Optional[np.ndarray] = None
        self._fitted = False

        # Token ID ranges
        self._len_offset   = _N_FIXED
        self._dt_offset    = _N_FIXED + n_len_bins
        self._ttl_offset   = _N_FIXED + n_len_bins + n_dt_bins
        self._sport_offset = _N_FIXED + n_len_bins + n_dt_bins + n_ttl_bins
        self._dport_offset = _N_FIXED + n_len_bins + n_dt_bins + n_ttl_bins + n_port_buckets
        self._flag_offset  = _N_FIXED + n_len_bins + n_dt_bins + n_ttl_bins + 2 * n_port_buckets

        # TCP flags: SYN, ACK, FIN, RST, PSH, URG, ECE, CWR → 8 flag tokens
        self._n_flag_tokens = 8
        self._vocab_size = (
            _N_FIXED
            + n_len_bins
            + n_dt_bins
            + n_ttl_bins
            + 2 * n_port_buckets   # sport + dport
            + self._n_flag_tokens
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def tokens_per_pkt(self) -> int:
        return TOKENS_PER_PKT

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_from_pcap(self, pcap_paths: List[str | Path], sample_n: int = 500_000) -> "PcapFlowTokenizer":
        """
        Fit bin edges by sampling packets from training PCAPs.
        Avoids loading entire PCAPs into memory.
        """
        try:
            from scapy.all import PcapReader, IP, TCP, UDP
        except ImportError:
            raise ImportError("Install scapy: pip install scapy")

        lengths, dts, ttls = [], [], []
        flow_last_ts: Dict[str, float] = {}
        count = 0

        for pcap_path in pcap_paths:
            logger.info("Sampling from %s …", pcap_path)
            try:
                with PcapReader(str(pcap_path)) as reader:
                    for pkt in reader:
                        if not pkt.haslayer(IP):
                            continue
                        length = len(pkt)
                        ttl    = pkt[IP].ttl
                        ts     = float(pkt.time)
                        key    = self._flow_key(pkt)
                        dt     = ts - flow_last_ts.get(key, ts)
                        flow_last_ts[key] = ts

                        lengths.append(length)
                        dts.append(dt)
                        ttls.append(ttl)
                        count += 1
                        if count >= sample_n:
                            break
            except Exception as e:
                logger.warning("Error reading %s: %s", pcap_path, e)

        if count == 0:
            raise RuntimeError("No IP packets found in provided PCAPs.")

        pct = np.linspace(0, 100, self.n_len_bins + 1)
        self._len_edges = np.unique(np.percentile(lengths, pct))
        self._dt_edges  = np.unique(np.percentile(np.log1p(dts), np.linspace(0, 100, self.n_dt_bins + 1)))
        self._ttl_edges = np.unique(np.percentile(ttls, np.linspace(0, 100, self.n_ttl_bins + 1)))
        self._fitted = True
        logger.info(
            "Fitted on %d packets: len_bins=%d dt_bins=%d ttl_bins=%d vocab=%d",
            count, len(self._len_edges) - 1, len(self._dt_edges) - 1,
            len(self._ttl_edges) - 1, self._vocab_size,
        )
        return self

    def fit_from_arrays(
        self,
        lengths: np.ndarray,
        dts: np.ndarray,
        ttls: np.ndarray,
    ) -> "PcapFlowTokenizer":
        """Fit from pre-collected arrays (useful for testing without PCAPs)."""
        pct = np.linspace(0, 100, self.n_len_bins + 1)
        self._len_edges = np.unique(np.percentile(lengths, pct))
        self._dt_edges  = np.unique(np.percentile(np.log1p(dts), np.linspace(0, 100, self.n_dt_bins + 1)))
        self._ttl_edges = np.unique(np.percentile(ttls, np.linspace(0, 100, self.n_ttl_bins + 1)))
        self._fitted = True
        return self

    # ── PCAP → flow sequences ─────────────────────────────────────────────────

    def pcap_to_flows(
        self,
        pcap_path: str | Path,
        label: int = 0,
        max_flows: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Parse one PCAP file → list of per-flow token arrays + labels.

        Args:
            pcap_path : path to .pcap / .pcapng file
            label     : 0=benign, 1=attack (applies to all flows in this file)
            max_flows : stop after this many flows (None = all)

        Returns:
            sequences : list of int32 arrays, each shape (seq_len,)
            labels    : list of ints (same label for all flows in file)
        """
        assert self._fitted, "Call fit_from_pcap() first."
        try:
            from scapy.all import PcapReader, IP, TCP, UDP, ICMP
        except ImportError:
            raise ImportError("pip install scapy")

        # flow_key → {'origin': str, 'last_ts': float, 'pkts': list[tokens]}
        active_flows: Dict[str, dict] = {}
        completed: List[np.ndarray] = []
        flow_labels: List[int] = []

        def _flush(key: str) -> None:
            flow = active_flows.pop(key, None)
            if flow and flow["pkts"]:
                seq = self._build_sequence(flow["pkts"])
                completed.append(seq)
                flow_labels.append(label)

        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                if not pkt.haslayer(IP):
                    continue
                if max_flows and len(completed) >= max_flows:
                    break

                key    = self._flow_key(pkt)
                ts     = float(pkt.time)
                tokens = self._pkt_to_tokens(pkt, key, active_flows, ts)

                if key not in active_flows:
                    active_flows[key] = {"origin": pkt[IP].src, "last_ts": ts, "pkts": []}

                # TTL eviction for inactive flows
                if ts - active_flows[key]["last_ts"] > self.flow_timeout:
                    _flush(key)
                    active_flows[key] = {"origin": pkt[IP].src, "last_ts": ts, "pkts": []}

                active_flows[key]["last_ts"] = ts
                if len(active_flows[key]["pkts"]) < self.max_pkts:
                    active_flows[key]["pkts"].append(tokens)

                # Flush on TCP FIN/RST
                if pkt.haslayer(TCP):
                    flags = pkt[TCP].flags
                    if flags & 0x01 or flags & 0x04:   # FIN or RST
                        _flush(key)

        # Flush remaining active flows
        for key in list(active_flows.keys()):
            _flush(key)

        logger.info(
            "Parsed %s → %d flows (label=%d)", Path(pcap_path).name, len(completed), label
        )
        return completed, flow_labels

    # ── Token helpers ─────────────────────────────────────────────────────────

    def _flow_key(self, pkt) -> str:
        """Normalised 5-tuple key so A→B and B→A share the same flow."""
        from scapy.all import IP, TCP, UDP
        if not pkt.haslayer(IP):
            return "unknown"
        src, dst = pkt[IP].src, pkt[IP].dst
        proto = pkt[IP].proto
        sport = dport = 0
        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        # Canonical direction: sort endpoints so A→B == B→A
        ep_a, ep_b = (src, sport), (dst, dport)
        if ep_a > ep_b:
            ep_a, ep_b = ep_b, ep_a
        return f"{ep_a[0]}:{ep_a[1]}-{ep_b[0]}:{ep_b[1]}-{proto}"

    def _pkt_to_tokens(
        self,
        pkt,
        key: str,
        active_flows: dict,
        ts: float,
    ) -> List[int]:
        """Convert one packet → list of 8 token IDs (PKT_SEP added separately)."""
        from scapy.all import IP, TCP, UDP, ICMP

        ip = pkt[IP]
        # Direction: C2S if src == canonical origin
        if key in active_flows:
            is_c2s = (ip.src == active_flows[key]["origin"])
        else:
            is_c2s = True   # first packet defines client
        dir_tok = DIR_C2S if is_c2s else DIR_S2C

        # Packet length
        length  = len(pkt)
        len_tok = self._bin(length, self._len_edges, self._len_offset, self.n_len_bins)

        # Inter-arrival time (Δt)
        last_ts = active_flows[key]["last_ts"] if key in active_flows else ts
        dt      = max(ts - last_ts, 0.0)
        dt_tok  = self._bin(np.log1p(dt), self._dt_edges, self._dt_offset, self.n_dt_bins)

        # TTL
        ttl     = ip.ttl
        ttl_tok = self._bin(ttl, self._ttl_edges, self._ttl_offset, self.n_ttl_bins)

        # Protocol
        proto = ip.proto
        if proto == 6:
            proto_tok = PROTO_TCP
        elif proto == 17:
            proto_tok = PROTO_UDP
        elif proto == 1:
            proto_tok = PROTO_ICMP
        else:
            proto_tok = PROTO_OTHER

        # Ports (hashed into buckets — no semantic DPI)
        sport = dport = 0
        tcp_flags = 0
        if pkt.haslayer(TCP):
            sport     = pkt[TCP].sport
            dport     = pkt[TCP].dport
            tcp_flags = int(pkt[TCP].flags)
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport

        sport_tok = self._sport_offset + (sport % self.n_port_buckets)
        dport_tok = self._dport_offset + (dport % self.n_port_buckets)

        # TCP flags bitmap → single token (0–7 based on dominant flag)
        flag_tok = self._flag_token(tcp_flags)

        return [dir_tok, len_tok, dt_tok, ttl_tok, proto_tok,
                sport_tok, dport_tok, flag_tok, PKT_SEP_ID]

    def _bin(self, value: float, edges: np.ndarray, offset: int, n_bins: int) -> int:
        idx = int(np.searchsorted(edges, value, side="right")) - 1
        return offset + max(0, min(idx, n_bins - 1))

    def _flag_token(self, flags: int) -> int:
        """Map TCP flags byte to one of 8 representative flag tokens."""
        # Priority: RST > FIN > SYN > PSH > ACK > URG > ECE > CWR
        priority = [(0x04, 0), (0x01, 1), (0x02, 2), (0x08, 3),
                    (0x10, 4), (0x20, 5), (0x40, 6), (0x80, 7)]
        for mask, idx in priority:
            if flags & mask:
                return self._flag_offset + idx
        return self._flag_offset   # no flags

    def _build_sequence(self, pkt_token_lists: List[List[int]]) -> np.ndarray:
        """Flatten per-packet token lists into one sequence with BOS/EOS."""
        flat = [BOS_ID]
        for toks in pkt_token_lists:
            flat.extend(toks)
        flat.append(EOS_ID)
        return np.array(flat, dtype=np.int32)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=4)
        logger.info("PcapFlowTokenizer saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PcapFlowTokenizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def token_name(self, tid: int) -> str:
        """Human-readable token name for debugging."""
        names = {
            PAD_ID: "<PAD>", BOS_ID: "<FLOW_BOS>", EOS_ID: "<FLOW_EOS>",
            PKT_SEP_ID: "<PKT_SEP>", UNK_ID: "<UNK>",
            PROTO_TCP: "PROTO_TCP", PROTO_UDP: "PROTO_UDP",
            PROTO_ICMP: "PROTO_ICMP", PROTO_OTHER: "PROTO_OTHER",
            DIR_C2S: "<DIR_C2S>", DIR_S2C: "<DIR_S2C>",
        }
        if tid in names:
            return names[tid]
        if self._len_offset <= tid < self._len_offset + self.n_len_bins:
            return f"LEN_BIN_{tid - self._len_offset}"
        if self._dt_offset <= tid < self._dt_offset + self.n_dt_bins:
            return f"DT_BIN_{tid - self._dt_offset}"
        if self._ttl_offset <= tid < self._ttl_offset + self.n_ttl_bins:
            return f"TTL_BIN_{tid - self._ttl_offset}"
        if self._sport_offset <= tid < self._sport_offset + self.n_port_buckets:
            return f"SPORT_H_{tid - self._sport_offset}"
        if self._dport_offset <= tid < self._dport_offset + self.n_port_buckets:
            return f"DPORT_H_{tid - self._dport_offset}"
        flag_names = ["FLAG_RST", "FLAG_FIN", "FLAG_SYN", "FLAG_PSH",
                      "FLAG_ACK", "FLAG_URG", "FLAG_ECE", "FLAG_CWR"]
        if self._flag_offset <= tid < self._flag_offset + 8:
            return flag_names[tid - self._flag_offset]
        return f"<UNK:{tid}>"
