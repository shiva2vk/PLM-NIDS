"""
PCAP preprocessing pipeline — replaces preprocessing.py for raw-PCAP mode.

Expected directory structure:
  pcap_dir/
    benign/          ← .pcap files of clean traffic (Label=0)
    attack/
      Bruteforce/    ← .pcap files per attack category (Label=1)
      Probing/
      CryptoMiner/
      ...

For HIKARI-2021: download raw PCAPs from Zenodo DOI 10.5281/zenodo.6463389
For CIC-IDS-2017: download from https://www.unb.ca/cic/datasets/ids-2017.html
                  Monday/ = benign only (best for Phase-1 pre-training)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.pcap_tokenizer import PcapFlowTokenizer
from data.pcap_dataset import PcapLMDataset, PcapClassifierDataset, sequence_length_stats

logger = logging.getLogger(__name__)


def discover_pcaps(pcap_dir: str | Path) -> Dict[str, List[Path]]:
    """
    Walk pcap_dir and return dict:
      'benign'  → list of PCAP paths
      'attack'  → list of PCAP paths
    """
    root = Path(pcap_dir)
    benign_paths, attack_paths = [], []

    for p in sorted(root.rglob("*.pcap")) + sorted(root.rglob("*.pcapng")):
        parts = {part.lower() for part in p.parts}
        if any(k in parts for k in ("benign", "normal", "background", "monday")):
            benign_paths.append(p)
        else:
            attack_paths.append(p)

    logger.info(
        "Discovered %d benign PCAPs and %d attack PCAPs under %s",
        len(benign_paths), len(attack_paths), pcap_dir,
    )
    return {"benign": benign_paths, "attack": attack_paths}


def load_and_tokenise(
    pcap_paths: List[Path],
    label: int,
    tokenizer: PcapFlowTokenizer,
    max_flows_per_file: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load multiple PCAPs with one label, return all flow sequences."""
    all_seqs, all_labels = [], []
    for p in pcap_paths:
        try:
            seqs, labels = tokenizer.pcap_to_flows(
                p, label=label, max_flows=max_flows_per_file
            )
            all_seqs.extend(seqs)
            all_labels.extend(labels)
        except Exception as e:
            logger.warning("Skipping %s: %s", p.name, e)
    return all_seqs, all_labels


def prepare_pcap_pipeline(cfg: dict) -> dict:
    """
    Full PCAP preprocessing pipeline.

    Returns dict with:
      'tokenizer'     : fitted PcapFlowTokenizer
      'train_benign'  : (sequences, labels) for Phase-1
      'train_all'     : (sequences, labels) for Phase-2
      'val'           : (sequences, labels)
      'test'          : (sequences, labels)
      'length_stats'  : dict of sequence length statistics
    """
    pcap_cfg  = cfg["data"]["pcap"]
    tok_cfg   = cfg["tokenizer"]["pcap"]
    seed      = cfg["data"]["random_seed"]
    rng       = random.Random(seed)

    pcap_dir  = pcap_cfg["dir"]
    pcaps     = discover_pcaps(pcap_dir)

    # ── Fit tokenizer on benign training PCAPs ────────────────────────────────
    tok = PcapFlowTokenizer(
        n_len_bins=tok_cfg.get("n_len_bins", 32),
        n_dt_bins=tok_cfg.get("n_dt_bins", 32),
        n_ttl_bins=tok_cfg.get("n_ttl_bins", 16),
        n_port_buckets=tok_cfg.get("n_port_buckets", 64),
        max_pkts_per_flow=tok_cfg.get("max_pkts_per_flow", 128),
        flow_timeout=tok_cfg.get("flow_timeout", 120.0),
    )

    n_fit = int(len(pcaps["benign"]) * 0.7) + 1
    fit_paths = pcaps["benign"][:n_fit]
    logger.info("Fitting tokenizer on %d benign PCAPs …", len(fit_paths))
    tok.fit_from_pcap(fit_paths, sample_n=tok_cfg.get("fit_sample_n", 500_000))
    tok.save(tok_cfg["save_path"])
    logger.info("Vocab size: %d | tokens/packet: %d", tok.vocab_size, tok.tokens_per_pkt)

    # ── Load benign flows ─────────────────────────────────────────────────────
    logger.info("Loading benign flows …")
    benign_seqs, benign_labels = load_and_tokenise(
        pcaps["benign"], label=0, tokenizer=tok,
        max_flows_per_file=pcap_cfg.get("max_flows_per_file"),
    )

    # ── Load attack flows ─────────────────────────────────────────────────────
    logger.info("Loading attack flows …")
    attack_seqs, attack_labels = load_and_tokenise(
        pcaps["attack"], label=1, tokenizer=tok,
        max_flows_per_file=pcap_cfg.get("max_flows_per_file"),
    )

    # ── Log sequence length statistics (important for paper) ──────────────────
    all_seqs_for_stats = benign_seqs + attack_seqs
    stats = sequence_length_stats(all_seqs_for_stats)
    logger.info(
        "Sequence lengths → min=%d median=%.0f mean=%.0f max=%d p95=%.0f",
        stats["min"], stats["median"], stats["mean"], stats["max"], stats["p95"],
    )

    # ── Shuffle + split ───────────────────────────────────────────────────────
    def split(seqs, labels, test_frac=0.15, val_frac=0.10):
        combined = list(zip(seqs, labels))
        rng.shuffle(combined)
        n = len(combined)
        n_test = int(n * test_frac)
        n_val  = int(n * val_frac)
        n_train = n - n_test - n_val
        train = combined[:n_train]
        val   = combined[n_train:n_train + n_val]
        test  = combined[n_train + n_val:]
        return (
            ([x[0] for x in train], [x[1] for x in train]),
            ([x[0] for x in val],   [x[1] for x in val]),
            ([x[0] for x in test],  [x[1] for x in test]),
        )

    b_train, b_val, b_test = split(benign_seqs, benign_labels)
    a_train, a_val, a_test = split(attack_seqs, attack_labels)

    def merge(p, q):
        seqs   = p[0] + q[0]
        labels = p[1] + q[1]
        combined = list(zip(seqs, labels))
        rng.shuffle(combined)
        return [x[0] for x in combined], [x[1] for x in combined]

    train_benign = b_train
    train_all    = merge(b_train, a_train)
    val_data     = merge(b_val,   a_val)
    test_data    = merge(b_test,  a_test)

    logger.info(
        "Split → train_benign=%d | train_all=%d | val=%d | test=%d",
        len(train_benign[0]), len(train_all[0]), len(val_data[0]), len(test_data[0]),
    )

    return {
        "tokenizer":    tok,
        "train_benign": train_benign,
        "train_all":    train_all,
        "val":          val_data,
        "test":         test_data,
        "length_stats": stats,
    }
