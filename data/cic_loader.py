"""
CIC-IDS-2017 PCAP loader.

Responsibilities
----------------
1. Discover PCAP files by day name (Monday … Friday).
2. Parse each PCAP with Scapy, extract L3/L4 headers only (zero DPI).
3. Group packets into flows by normalised 5-tuple.
4. Assign binary labels: Monday=0 (benign), all other days=1 (attack).
5. Return train / val / test splits ready for the PLM datasets.

Label note
----------
Each attack day (Tue-Fri) also carries benign background traffic.
Without the companion FlowMeter CSV we cannot separate benign from
attack *within* an attack day at flow granularity.  We assign label=1
to all flows on attack days — this produces a ~20% false-label rate
on those days but is a published, reproducible approach for raw-PCAP
experiments.  If the companion CSV is later obtained, override via
`companion_csv` parameter.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.pcap_tokenizer import PcapFlowTokenizer

logger = logging.getLogger(__name__)

# ── Day → metadata ────────────────────────────────────────────────────────────
DAY_META: Dict[str, dict] = {
    "Monday-WorkingHours":    {"label": 0, "attack": "BENIGN"},
    "Tuesday-WorkingHours":   {"label": 1, "attack": "FTP-Patator_SSH-Patator"},
    "Wednesday-workingHours": {"label": 1, "attack": "DoS_Heartbleed"},
    "Thursday-WorkingHours":  {"label": 1, "attack": "WebAttacks_Infiltration"},
    "Friday-WorkingHours ":   {"label": 1, "attack": "Botnet_PortScan_DDoS"},
}


def _day_key(fname: str) -> Optional[str]:
    """Return the DAY_META key matching a filename, or None."""
    stem = Path(fname).stem
    for key in DAY_META:
        if key.lower() in stem.lower() or stem.lower() in key.lower():
            return key
    # Try prefix match
    for key in DAY_META:
        if stem.startswith(key.strip()):
            return key
    return None


def discover_cic_pcaps(pcap_dir: str | Path) -> Dict[str, Path]:
    """Return {day_key: path} for each CIC-IDS-2017 PCAP found."""
    root = Path(pcap_dir)
    found: Dict[str, Path] = {}
    for p in sorted(root.glob("*.pcap")) + sorted(root.glob("*.pcapng")):
        key = _day_key(p.name)
        if key:
            found[key] = p
            logger.info("Found %-40s → %s", p.name, key)
        else:
            logger.warning("Unrecognised PCAP (skipped): %s", p.name)
    if not found:
        raise FileNotFoundError(f"No CIC-IDS-2017 PCAPs found in {pcap_dir}")
    return found


# ── Core PCAP → flow tokeniser ────────────────────────────────────────────────

def _parse_pcap_to_flows(
    path: Path,
    label: int,
    attack_type: str,
    tokenizer: PcapFlowTokenizer,
    max_flows: Optional[int] = None,
    flow_timeout: float = 120.0,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Parse one PCAP file → (sequences, labels, attack_types).

    Uses only IP/TCP/UDP header fields — zero payload inspection.
    """
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
    except ImportError:
        raise ImportError("pip install scapy")

    # flow_key → {origin, last_ts, pkts}
    active: Dict[str, dict] = {}
    done_seqs:  List[np.ndarray] = []
    done_labels: List[int] = []
    done_atypes: List[str] = []

    def _flush(key: str) -> None:
        flow = active.pop(key, None)
        if flow and len(flow["pkts"]) >= 2:   # skip single-packet flows
            seq = tokenizer._build_sequence(flow["pkts"])
            done_seqs.append(seq)
            done_labels.append(label)
            done_atypes.append(attack_type)

    t0 = time.time()
    pkt_count = 0

    with PcapReader(str(path)) as reader:
        for pkt in reader:
            if max_flows and len(done_seqs) >= max_flows:
                break
            if IP not in pkt:
                continue

            ip    = pkt[IP]
            src, dst = ip.src, ip.dst
            proto = ip.proto
            ts    = float(pkt.time)
            sp = dp = 0
            tcp_flags = 0

            if TCP in pkt:
                sp, dp = pkt[TCP].sport, pkt[TCP].dport
                tcp_flags = int(pkt[TCP].flags)
            elif UDP in pkt:
                sp, dp = pkt[UDP].sport, pkt[UDP].dport

            # Normalised 5-tuple key (direction-agnostic)
            a, b = (src, sp), (dst, dp)
            if a > b:
                a, b = b, a
            key = f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"

            if key not in active:
                active[key] = {"origin": src, "last_ts": ts, "pkts": []}

            # TTL-evict stale flows
            if ts - active[key]["last_ts"] > flow_timeout:
                _flush(key)
                active[key] = {"origin": src, "last_ts": ts, "pkts": []}

            active[key]["last_ts"] = ts

            if len(active[key]["pkts"]) < tokenizer.max_pkts:
                toks = tokenizer._pkt_to_tokens(pkt, key, active, ts)
                active[key]["pkts"].append(toks)

            # Flush on TCP FIN / RST
            if tcp_flags & 0x01 or tcp_flags & 0x04:
                _flush(key)

            pkt_count += 1

    # Flush remaining
    for key in list(active.keys()):
        _flush(key)

    elapsed = time.time() - t0
    logger.info(
        "%-45s → %5d flows | %7d pkts | label=%d | %.1fs",
        path.name, len(done_seqs), pkt_count, label, elapsed,
    )
    return done_seqs, done_labels, done_atypes


# ── Public pipeline ───────────────────────────────────────────────────────────

def build_tokenizer_from_monday(
    monday_path: Path,
    cfg_tok: dict,
) -> PcapFlowTokenizer:
    """Fit PcapFlowTokenizer on Monday (benign) PCAP only."""
    tok = PcapFlowTokenizer(
        n_len_bins=cfg_tok.get("n_len_bins", 32),
        n_dt_bins=cfg_tok.get("n_dt_bins", 32),
        n_ttl_bins=cfg_tok.get("n_ttl_bins", 16),
        n_port_buckets=cfg_tok.get("n_port_buckets", 64),
        max_pkts_per_flow=cfg_tok.get("max_pkts_per_flow", 128),
        flow_timeout=cfg_tok.get("flow_timeout", 120.0),
    )
    tok.fit_from_pcap(
        [monday_path],
        sample_n=cfg_tok.get("fit_sample_n", 300_000),
    )
    tok.save(cfg_tok["save_path"])
    logger.info("Tokenizer: vocab=%d  seq_max=%d", tok.vocab_size,
                tok.max_pkts * tok.tokens_per_pkt + 2)
    return tok


def load_all_days(
    pcap_map: Dict[str, Path],
    tokenizer: PcapFlowTokenizer,
    cfg: dict,
    seed: int = 42,
) -> dict:
    """
    Parse all CIC-IDS-2017 PCAPs → train / val / test splits.

    Returns
    -------
    dict with keys:
      'monday_seqs'   : all Monday sequences (benign, for Phase-1)
      'monday_labels'
      'all_seqs'      : Monday + all attack-day sequences (for Phase-2)
      'all_labels'
      'all_atypes'    : attack type string per flow
      'splits'        : {'train','val','test'} each as (seqs, labels, atypes)
      'day_data'      : per-day {seqs, labels} for per-day evaluation
    """
    rng = random.Random(seed)
    flow_timeout = cfg["tokenizer"].get("flow_timeout", 120.0)

    monday_seqs, monday_labels = [], []
    all_seqs, all_labels, all_atypes = [], [], []
    day_data: Dict[str, dict] = {}

    for day_key, path in pcap_map.items():
        meta = DAY_META.get(day_key, {"label": 0, "attack": "UNKNOWN"})
        label      = meta["label"]
        attack_type = meta["attack"]

        seqs, labels, atypes = _parse_pcap_to_flows(
            path, label, attack_type, tokenizer,
            max_flows=None,
            flow_timeout=flow_timeout,
        )

        day_data[day_key] = {
            "seqs": seqs, "labels": labels,
            "atypes": atypes, "meta": meta,
        }

        if label == 0:
            monday_seqs.extend(seqs)
            monday_labels.extend(labels)

        all_seqs.extend(seqs)
        all_labels.extend(labels)
        all_atypes.extend(atypes)

    logger.info(
        "Total flows: %d | benign=%d | attack=%d",
        len(all_seqs),
        sum(l == 0 for l in all_labels),
        sum(l == 1 for l in all_labels),
    )

    # Shuffle + split
    def _split(seqs, labels, atypes, val_f=0.10, test_f=0.15):
        combined = list(zip(seqs, labels, atypes))
        rng.shuffle(combined)
        n = len(combined)
        n_test = int(n * test_f)
        n_val  = int(n * val_f)
        tr = combined[:n - n_test - n_val]
        va = combined[n - n_test - n_val: n - n_test]
        te = combined[n - n_test:]
        def unzip(lst):
            s, l, a = zip(*lst) if lst else ([], [], [])
            return list(s), list(l), list(a)
        return unzip(tr), unzip(va), unzip(te)

    m_tr, m_va, m_te = _split(monday_seqs, monday_labels,
                                ["BENIGN"] * len(monday_labels))
    a_tr, a_va, a_te = _split(all_seqs, all_labels, all_atypes)

    return {
        "monday_seqs":   monday_seqs,
        "monday_labels": monday_labels,
        "all_seqs":      all_seqs,
        "all_labels":    all_labels,
        "all_atypes":    all_atypes,
        "splits": {
            "monday_train": m_tr,
            "monday_val":   m_va,
            "monday_test":  m_te,
            "all_train":    a_tr,
            "all_val":      a_va,
            "all_test":     a_te,
        },
        "day_data": day_data,
    }
