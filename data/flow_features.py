"""
DPI-free per-flow feature extractor for baseline models.

Computes the same statistical features used in CICFlowMeter,
but derived purely from L3/L4 headers — zero payload inspection.
This ensures a fair comparison: baselines use IDENTICAL information
as the PLM tokenizer, just aggregated into a flat feature vector
instead of a token sequence.

Features extracted (25 total):
  Packet counts  : fwd_pkts, bwd_pkts, total_pkts
  Byte counts    : fwd_bytes, bwd_bytes
  Packet lengths : fwd_len_{min,max,mean,std}, bwd_len_{min,max,mean,std}
  Inter-arrival  : fwd_iat_{min,max,mean,std}, bwd_iat_{min,max,mean,std}
  TCP flags      : syn, ack, fin, rst, psh counts
  Flow meta      : duration, down_up_ratio, mean_ttl
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


FEATURE_NAMES = [
    "fwd_pkts", "bwd_pkts", "total_pkts",
    "fwd_bytes", "bwd_bytes",
    "fwd_len_min", "fwd_len_max", "fwd_len_mean", "fwd_len_std",
    "bwd_len_min", "bwd_len_max", "bwd_len_mean", "bwd_len_std",
    "fwd_iat_min", "fwd_iat_max", "fwd_iat_mean", "fwd_iat_std",
    "bwd_iat_min", "bwd_iat_max", "bwd_iat_mean", "bwd_iat_std",
    "flag_syn", "flag_ack", "flag_fin", "flag_rst",
    "duration", "down_up_ratio", "mean_ttl",
]
N_FEATURES = len(FEATURE_NAMES)   # 28


def extract_from_pcap(
    pcap_path: str,
    label: int,
    max_flows: int | None = None,
    flow_timeout: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse one PCAP → (X, y) where X is (N, 28) float32, y is (N,) int.
    Uses ONLY L3/L4 header fields — zero DPI.
    """
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
    except ImportError:
        raise ImportError("pip install scapy")

    # flow_key → accumulated stats
    flows: Dict[str, dict] = {}

    def _new_flow(src: str, ts: float) -> dict:
        return {
            "origin": src, "last_ts": ts,
            "fwd_lens": [], "bwd_lens": [],
            "fwd_ts": [],   "bwd_ts": [],
            "ttls": [],
            "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0,
            "start_ts": ts,
        }

    completed_X: List[np.ndarray] = []
    completed_y: List[int] = []

    def _flush(key: str) -> None:
        f = flows.pop(key, None)
        if f is None or (len(f["fwd_lens"]) + len(f["bwd_lens"])) < 2:
            return
        vec = _flow_to_vec(f)
        completed_X.append(vec)
        completed_y.append(label)

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if max_flows and len(completed_X) >= max_flows:
                break
            if IP not in pkt:
                continue

            ip    = pkt[IP]
            src, dst = ip.src, ip.dst
            proto = ip.proto
            ts    = float(pkt.time)
            length = len(pkt)
            ttl    = ip.ttl
            sp = dp = tcp_flags = 0

            if TCP in pkt:
                sp, dp = pkt[TCP].sport, pkt[TCP].dport
                tcp_flags = int(pkt[TCP].flags)
            elif UDP in pkt:
                sp, dp = pkt[UDP].sport, pkt[UDP].dport

            # Normalised key
            a, b = (src, sp), (dst, dp)
            if a > b:
                a, b = b, a
            key = f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"

            if key not in flows:
                flows[key] = _new_flow(src, ts)

            f = flows[key]
            # TTL-evict
            if ts - f["last_ts"] > flow_timeout:
                _flush(key)
                flows[key] = _new_flow(src, ts)
                f = flows[key]

            f["last_ts"] = ts
            f["ttls"].append(ttl)

            is_fwd = (ip.src == f["origin"])
            if is_fwd:
                f["fwd_lens"].append(length)
                f["fwd_ts"].append(ts)
            else:
                f["bwd_lens"].append(length)
                f["bwd_ts"].append(ts)

            if tcp_flags & 0x02: f["syn"] += 1
            if tcp_flags & 0x10: f["ack"] += 1
            if tcp_flags & 0x01: f["fin"] += 1
            if tcp_flags & 0x04: f["rst"] += 1
            if tcp_flags & 0x08: f["psh"] += 1

            if tcp_flags & 0x01 or tcp_flags & 0x04:
                _flush(key)

    for key in list(flows.keys()):
        _flush(key)

    if not completed_X:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return (
        np.vstack(completed_X).astype(np.float32),
        np.array(completed_y, dtype=np.int32),
    )


def _flow_to_vec(f: dict) -> np.ndarray:
    def _stats(lst):
        if not lst:
            return 0.0, 0.0, 0.0, 0.0
        a = np.array(lst, dtype=np.float64)
        return float(a.min()), float(a.max()), float(a.mean()), float(a.std())

    def _iats(ts_list):
        if len(ts_list) < 2:
            return 0.0, 0.0, 0.0, 0.0
        diffs = np.diff(sorted(ts_list))
        return _stats(diffs.tolist())

    fwd_l = f["fwd_lens"]; bwd_l = f["bwd_lens"]
    fl_min, fl_max, fl_mean, fl_std = _stats(fwd_l)
    bl_min, bl_max, bl_mean, bl_std = _stats(bwd_l)
    fi_min, fi_max, fi_mean, fi_std = _iats(f["fwd_ts"])
    bi_min, bi_max, bi_mean, bi_std = _iats(f["bwd_ts"])

    fwd_b = sum(fwd_l); bwd_b = sum(bwd_l)
    total = len(fwd_l) + len(bwd_l)
    dur   = f["last_ts"] - f["start_ts"]
    ratio = len(bwd_l) / max(len(fwd_l), 1)
    mttl  = float(np.mean(f["ttls"])) if f["ttls"] else 0.0

    return np.array([
        len(fwd_l), len(bwd_l), total,
        fwd_b, bwd_b,
        fl_min, fl_max, fl_mean, fl_std,
        bl_min, bl_max, bl_mean, bl_std,
        fi_min, fi_max, fi_mean, fi_std,
        bi_min, bi_max, bi_mean, bi_std,
        f["syn"], f["ack"], f["fin"], f["rst"],
        dur, ratio, mttl,
    ], dtype=np.float64)


def load_all_days_features(
    pcap_map: Dict,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse all PCAP days → (X_train, X_test, y_train, y_test).
    Same 85/15 split as the PLM pipeline for fair comparison.
    """
    import random
    from data.cic_loader import DAY_META

    rng = random.Random(seed)
    all_X, all_y = [], []

    for day_key, path in pcap_map.items():
        meta  = DAY_META.get(day_key, {"label": 0})
        label = meta["label"]
        import logging
        logging.getLogger(__name__).info("Extracting features from %s …", path.name)
        X, y = extract_from_pcap(str(path), label)
        if len(X):
            all_X.append(X)
            all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Shuffle + split
    idx = list(range(len(y_all)))
    rng.shuffle(idx)
    idx = np.array(idx)
    n_test = int(len(idx) * 0.15)
    train_idx = idx[n_test:]
    test_idx  = idx[:n_test]

    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]
