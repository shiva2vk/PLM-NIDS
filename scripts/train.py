"""
Training script for PLM-NIDS on CIC-IDS-2017 raw PCAPs.

DEFAULT: trains ALL phases (Phase 1 → Phase 2) with no flags needed.

Usage
-----
# Train everything (default — runs Phase 1 then Phase 2):
  python scripts/train.py --config config.yaml

# Explicitly train all phases:
  python scripts/train.py --config config.yaml --phase all

# Phase 1 only (causal LM on Monday benign traffic):
  python scripts/train.py --config config.yaml --phase 1

# Phase 2 only (loads Phase-1 checkpoint automatically):
  python scripts/train.py --config config.yaml --phase 2

# Force re-parse PCAPs:
  python scripts/train.py --config config.yaml --no-cache
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cic_loader import discover_cic_pcaps, build_tokenizer_from_monday, load_all_days
from data.pcap_tokenizer import PcapFlowTokenizer
from data.pcap_dataset import (
    PcapLMDataset, PcapClassifierDataset,
    pcap_collate_lm, pcap_collate_cls,
)
from data.dataset import make_weighted_sampler
from models.plm import PLM
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(cfg_device)


def build_model(cfg: dict, vocab_size: int) -> PLM:
    m = cfg["model"]
    return PLM(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        dropout=m["dropout"],
        n_classes=m["n_classes"],
        tie_embeddings=m.get("tie_embeddings", True),
    )


def save_data_cache(data: dict, path: str) -> None:
    """Cache parsed flows to disk so re-runs skip the slow PCAP parsing step."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    logger.info("Flow cache saved → %s", path)


def load_data_cache(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLM-NIDS on CIC-IDS-2017")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--phase",   choices=["1", "2", "both", "all"],
                        default="all",
                        help="'all'/'both' = Phase1 then Phase2 (default). "
                             "'1' = Phase1 only. '2' = Phase2 only.")
    parser.add_argument("--resume",  default=None, help="Checkpoint to resume from")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-parse PCAPs even if flow cache exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["plot_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["training"].get("device", "auto"))
    seed   = cfg["training"].get("random_seed", 42)
    logger.info("Device: %s", device)

    # ── Step 1: Check cache first — PCAPs only needed if cache is missing ───
    cache_path = str(Path(cfg["paths"]["output_dir"]) / "flow_cache.pkl")
    tok_path   = cfg["tokenizer"]["save_path"]
    cache_exists = Path(cache_path).exists() and not args.no_cache
    tok_exists   = Path(tok_path).exists()   and not args.no_cache

    if cache_exists and tok_exists:
        # Fast path: load everything from disk — no PCAPs needed
        logger.info("Loading tokenizer from cache: %s", tok_path)
        tokenizer = PcapFlowTokenizer.load(tok_path)
        logger.info("Loading flow cache: %s", cache_path)
        data = load_data_cache(cache_path)
    else:
        # Need PCAPs — discover and parse
        pcap_map  = discover_cic_pcaps(cfg["paths"]["pcap_dir"])
        if tok_exists:
            tokenizer = PcapFlowTokenizer.load(tok_path)
        else:
            monday_key = next(k for k in pcap_map if "monday" in k.lower())
            tokenizer  = build_tokenizer_from_monday(pcap_map[monday_key],
                                                     cfg["tokenizer"])
        logger.info("Parsing all PCAP files (this takes ~10-30 min first time) …")
        data = load_all_days(pcap_map, tokenizer, cfg, seed=seed)
        save_data_cache(data, cache_path)

    logger.info("Vocab size: %d", tokenizer.vocab_size)

    splits = data["splits"]
    logger.info(
        "Flows → Monday-train=%d | All-train=%d | val=%d | test=%d",
        len(splits["monday_train"][0]),
        len(splits["all_train"][0]),
        len(splits["all_val"][0]),
        len(splits["all_test"][0]),
    )

    # ── Step 4: Build model ────────────────────────────────────────────────
    model = build_model(cfg, tokenizer.vocab_size)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Resumed from %s", args.resume)
    elif args.phase == "2":
        p1_ckpt = cfg["training"]["phase1"]["checkpoint"]
        if Path(p1_ckpt).exists():
            ckpt = torch.load(p1_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.info("Loaded Phase-1 checkpoint: %s", p1_ckpt)
        else:
            logger.warning("Phase-1 checkpoint not found; fine-tuning from scratch")

    trainer = Trainer(cfg, model)

    # Normalise: "all" is an alias for "both"
    if args.phase == "all":
        args.phase = "both"

    # max_seq_len: config override lets you reduce memory without re-parsing
    max_seq_len = cfg["training"].get(
        "max_seq_len",
        tokenizer.max_pkts * tokenizer.tokens_per_pkt + 2,
    )
    logger.info("max_seq_len=%d  (tokenizer max=%d)",
                max_seq_len, tokenizer.max_pkts * tokenizer.tokens_per_pkt + 2)

    # ── Phase 1: Causal LM pre-training on Monday (benign) ────────────────
    if args.phase in ("1", "both"):
        p1 = cfg["training"]["phase1"]
        m_train_seqs, m_train_lbl, _ = splits["monday_train"]
        m_val_seqs,   m_val_lbl,   _ = splits["monday_val"]

        train_ds = PcapLMDataset(m_train_seqs, m_train_lbl, max_len=max_seq_len)
        val_ds   = PcapLMDataset(m_val_seqs,   m_val_lbl,   max_len=max_seq_len)

        logger.info("Phase-1 → train=%d benign flows | val=%d", len(train_ds), len(val_ds))
        train_dl = DataLoader(train_ds, batch_size=p1["batch_size"],
                              shuffle=True, collate_fn=pcap_collate_lm, num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=p1["batch_size"] * 2,
                              shuffle=False, collate_fn=pcap_collate_lm, num_workers=0)
        trainer.train_phase1(train_dl, val_dl)

    # ── Phase 2: Supervised fine-tuning on all days ────────────────────────
    if args.phase in ("2", "both"):
        p2 = cfg["training"]["phase2"]
        a_train_seqs, a_train_lbl, _ = splits["all_train"]
        a_val_seqs,   a_val_lbl,   _ = splits["all_val"]

        import numpy as np
        train_ds = PcapClassifierDataset(a_train_seqs, a_train_lbl, max_len=max_seq_len)
        val_ds   = PcapClassifierDataset(a_val_seqs,   a_val_lbl,   max_len=max_seq_len)

        sampler = make_weighted_sampler(
            np.array(a_train_lbl),
            attack_weight=p2.get("attack_class_weight", 8.0),
        )
        train_dl = DataLoader(train_ds, batch_size=p2["batch_size"],
                              sampler=sampler, collate_fn=pcap_collate_cls, num_workers=0)
        val_dl   = DataLoader(val_ds, batch_size=p2["batch_size"] * 2,
                              shuffle=False, collate_fn=pcap_collate_cls, num_workers=0)

        logger.info(
            "Phase-2 → train=%d | val=%d | attack_rate=%.1f%%",
            len(train_ds), len(val_ds), 100 * np.mean(a_train_lbl),
        )
        trainer.train_phase2(train_dl, val_dl)

    logger.info("Training complete. Checkpoints in: %s",
                cfg["paths"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
