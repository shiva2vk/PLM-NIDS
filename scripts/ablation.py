"""
Ablation study — Table 3 of the paper.

Tests the contribution of each design decision by systematically
disabling or varying one component at a time.

Ablation dimensions
-------------------
  A. Model size          : d_model × n_layers  (3 configurations)
  B. Phase-1 pretraining : with vs without
  C. Sequence length     : max_pkts = 32 / 64 / 128
  D. Scoring mode        : perplexity vs supervised vs combined
  E. Token features      : full set vs drop flags vs drop IAT

Usage
-----
  python scripts/ablation.py --config config.yaml

Output
------
  outputs/ablation_results.json   — all ablation results
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cic_loader import discover_cic_pcaps
from data.pcap_tokenizer import PcapFlowTokenizer
from data.pcap_dataset import (PcapLMDataset, PcapClassifierDataset,
                                 pcap_collate_lm, pcap_collate_cls)
from data.dataset import make_weighted_sampler
from evaluation.metrics import compute_all_metrics
from inference.scorer import AnomalyScorer
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
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def _make_loaders(splits, cfg, p1=True):
    """Return (train_lm_dl, val_lm_dl, train_cls_dl, val_cls_dl)."""
    max_len = cfg["tokenizer"].get("max_pkts_per_flow", 128) * 9 + 2
    p1c = cfg["training"]["phase1"]
    p2c = cfg["training"]["phase2"]

    tr_b_seqs, tr_b_lbl, _ = splits["monday_train"]
    va_b_seqs, va_b_lbl, _ = splits["monday_val"]
    tr_a_seqs, tr_a_lbl, _ = splits["all_train"]
    va_a_seqs, va_a_lbl, _ = splits["all_val"]

    lm_train = DataLoader(
        PcapLMDataset(tr_b_seqs, tr_b_lbl, max_len=max_len),
        batch_size=p1c["batch_size"], shuffle=True,
        collate_fn=pcap_collate_lm, num_workers=0,
    )
    lm_val = DataLoader(
        PcapLMDataset(va_b_seqs, va_b_lbl, max_len=max_len),
        batch_size=p1c["batch_size"]*2, shuffle=False,
        collate_fn=pcap_collate_lm, num_workers=0,
    )
    sampler = make_weighted_sampler(np.array(tr_a_lbl),
                                     p2c.get("attack_class_weight", 8.0))
    cls_train = DataLoader(
        PcapClassifierDataset(tr_a_seqs, tr_a_lbl, max_len=max_len),
        batch_size=p2c["batch_size"], sampler=sampler,
        collate_fn=pcap_collate_cls, num_workers=0,
    )
    cls_val = DataLoader(
        PcapClassifierDataset(va_a_seqs, va_a_lbl, max_len=max_len),
        batch_size=p2c["batch_size"]*2, shuffle=False,
        collate_fn=pcap_collate_cls, num_workers=0,
    )
    return lm_train, lm_val, cls_train, cls_val


def run_experiment(
    name: str,
    cfg: dict,
    splits: dict,
    vocab_size: int,
    device: torch.device,
    d_model: int = 256,
    n_layers: int = 6,
    max_pkts: int = 128,
    skip_phase1: bool = False,
    score_mode: str = "perplexity",
    fast_epochs: int = 5,
) -> Dict:
    """Train one ablation configuration; return metrics dict."""
    logger.info("=== Ablation: %s ===", name)
    t0 = time.time()

    cfg_run = deepcopy(cfg)
    cfg_run["training"]["phase1"]["epochs"]     = fast_epochs
    cfg_run["training"]["phase2"]["epochs"]     = fast_epochs
    cfg_run["training"]["phase1"]["checkpoint"] = f"./checkpoints/abl_{name}_p1.pt"
    cfg_run["training"]["phase2"]["checkpoint"] = f"./checkpoints/abl_{name}_p2.pt"
    cfg_run["training"]["log_dir"]              = f"./logs/abl_{name}"
    cfg_run["training"]["save_dir"]             = f"./checkpoints/abl_{name}"
    # Use batch=256 for ablation — fast on A100, avoids OOM from fragmentation
    cfg_run["training"]["phase1"]["batch_size"] = 256
    cfg_run["training"]["phase2"]["batch_size"] = 256
    cfg_run["training"]["grad_accumulation_steps"] = 1

    model = PLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                dropout=0.1, n_classes=2)

    trainer = Trainer(cfg_run, model)
    lm_tr, lm_va, cls_tr, cls_va = _make_loaders(splits, cfg_run)

    if not skip_phase1:
        trainer.train_phase1(lm_tr, lm_va)
    trainer.train_phase2(cls_tr, cls_va)

    # Score test set
    scorer  = AnomalyScorer(model, device, score_mode=score_mode)
    te_seqs, te_lbl, _ = splits["all_test"]
    va_seqs, va_lbl, _ = splits["monday_val"]
    max_len = max_pkts * 9 + 2

    # Calibrate threshold
    benign_val = torch.from_numpy(
        np.vstack([s[:max_len] if len(s) >= max_len
                   else np.pad(s, (0, max_len - len(s))) for s in va_seqs])
    ).long()
    scorer.calibrate_threshold(benign_val[:500], percentile=95.0)

    # Test scores
    chunk, scores_all = 256, []
    for i in range(0, len(te_seqs), chunk):
        batch = te_seqs[i:i+chunk]
        padded = np.zeros((len(batch), max_len), dtype=np.int32)
        for j, s in enumerate(batch):
            t = min(len(s), max_len)
            padded[j, :t] = s[:t]
        sc = scorer.score_batch(torch.from_numpy(padded).long())
        scores_all.append(sc)

    scores = np.concatenate(scores_all)
    labels = np.array(te_lbl)
    m = compute_all_metrics(labels, scores, threshold=scorer.threshold)
    m["elapsed_s"] = round(time.time() - t0, 1)
    m["params"]    = model.count_parameters()
    logger.info("  PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  (%.0fs)",
                m["pr_auc"], m["roc_auc"], m["f1"], m["elapsed_s"])
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--fast-epochs",  type=int, default=5,
                        help="Epochs per ablation run (default 5 for speed)")
    parser.add_argument("--no-cache",     action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg["training"].get("device", "auto"))

    # Load flow cache (must exist — run train.py first)
    cache = output_dir / "flow_cache.pkl"
    if not cache.exists():
        logger.error("flow_cache.pkl not found. Run train.py first.")
        sys.exit(1)
    with open(cache, "rb") as f:
        data = pickle.load(f)
    splits   = data["splits"]
    tok_path = cfg["tokenizer"]["save_path"]
    tokenizer = PcapFlowTokenizer.load(tok_path)

    results: Dict = {}
    FE = args.fast_epochs

    # ── A. Full model (reference) ─────────────────────────────────────────
    results["Full_d256_L6"] = run_experiment(
        "Full_d256_L6", cfg, splits, tokenizer.vocab_size, device,
        d_model=256, n_layers=6, score_mode="combined",
        fast_epochs=FE,
    )

    # ── B. Model size ─────────────────────────────────────────────────────
    results["Small_d64_L2"] = run_experiment(
        "Small_d64_L2", cfg, splits, tokenizer.vocab_size, device,
        d_model=64, n_layers=2, score_mode="combined",
        fast_epochs=FE,
    )
    results["Medium_d128_L4"] = run_experiment(
        "Medium_d128_L4", cfg, splits, tokenizer.vocab_size, device,
        d_model=128, n_layers=4, score_mode="combined",
        fast_epochs=FE,
    )

    # ── C. Without Phase-1 pretraining ────────────────────────────────────
    results["No_Phase1_pretrain"] = run_experiment(
        "No_Phase1", cfg, splits, tokenizer.vocab_size, device,
        d_model=256, n_layers=6, score_mode="supervised",
        skip_phase1=True, fast_epochs=FE,
    )

    # ── D. Scoring mode ───────────────────────────────────────────────────
    results["Score_perplexity"] = run_experiment(
        "Score_PPL", cfg, splits, tokenizer.vocab_size, device,
        d_model=256, n_layers=6, score_mode="perplexity",
        fast_epochs=FE,
    )
    results["Score_supervised"] = run_experiment(
        "Score_CLS", cfg, splits, tokenizer.vocab_size, device,
        d_model=256, n_layers=6, score_mode="supervised",
        fast_epochs=FE,
    )

    # ── E. Sequence length (max_pkts) ─────────────────────────────────────
    for mp in [32, 64]:
        results[f"MaxPkts_{mp}"] = run_experiment(
            f"MaxPkts_{mp}", cfg, splits, tokenizer.vocab_size, device,
            d_model=256, n_layers=6, max_pkts=mp, score_mode="combined",
            fast_epochs=FE,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    out = output_dir / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Ablation results saved → %s", out)

    # Summary table
    print("\n" + "═" * 80)
    print(f"  {'Configuration':<28} {'PR-AUC':>8} {'ROC-AUC':>8} "
          f"{'F1':>8} {'FPR@95':>8} {'Params':>10}")
    print("─" * 80)
    for name, m in results.items():
        if not isinstance(m, dict) or "pr_auc" not in m:
            continue
        print(f"  {name:<28} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} "
              f"{m['f1']:>8.4f} {m['fpr_at_tpr95']:>8.4f} "
              f"{m.get('params', 0):>10,}")
    print("═" * 80)


if __name__ == "__main__":
    main()
