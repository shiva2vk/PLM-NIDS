"""
Complete evaluation script — all paper metrics.

Usage
-----
# Unsupervised (Phase-1 perplexity model):
  python scripts/evaluate.py --config config.yaml \
      --checkpoint checkpoints/phase1_best.pt --mode perplexity

# Supervised (Phase-2 classifier):
  python scripts/evaluate.py --config config.yaml \
      --checkpoint checkpoints/phase2_best.pt --mode supervised

# Combined score:
  python scripts/evaluate.py --config config.yaml \
      --checkpoint checkpoints/phase2_best.pt --mode combined

Outputs (saved to outputs/)
--------
  results_<mode>.json        — all metrics as JSON
  scores_<mode>.npz          — raw scores + labels for plotting
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cic_loader import discover_cic_pcaps, build_tokenizer_from_monday, load_all_days
from data.pcap_tokenizer import PcapFlowTokenizer
from data.pcap_dataset import PcapClassifierDataset, pcap_collate_cls
from evaluation.metrics import compute_all_metrics, print_report
from inference.scorer import AnomalyScorer
from models.plm import PLM
from torch.utils.data import DataLoader

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


def load_model(cfg: dict, checkpoint: str, vocab_size: int, device: torch.device) -> PLM:
    m = cfg["model"]
    model = PLM(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        dropout=0.0,
        n_classes=m["n_classes"],
        tie_embeddings=m.get("tie_embeddings", True),
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint from %s (epoch %d, val_loss=%.4f)",
                checkpoint, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))
    return model.to(device)


def score_split(
    seqs: List[np.ndarray],
    labels: List[int],
    scorer: AnomalyScorer,
    max_len: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a list of flow sequences; return (scores, labels) arrays."""
    all_scores = []
    n = len(seqs)
    for start in range(0, n, batch_size):
        chunk_seqs = seqs[start:start + batch_size]
        # Pad to max_len within batch
        max_t = min(max(len(s) for s in chunk_seqs), max_len)
        padded = np.zeros((len(chunk_seqs), max_t), dtype=np.int32)
        for i, s in enumerate(chunk_seqs):
            t = min(len(s), max_t)
            padded[i, :t] = s[:t]
        t = torch.from_numpy(padded).long()
        sc = scorer.score_batch(t)
        all_scores.append(sc)
        if (start // batch_size) % 20 == 0:
            logger.info("  Scored %d / %d flows", min(start + batch_size, n), n)
    return np.concatenate(all_scores), np.array(labels, dtype=int)


def evaluate_overall(
    scorer: AnomalyScorer,
    splits: dict,
    data: dict,
    max_len: int,
    output_dir: Path,
    mode: str,
    threshold_pct: float,
) -> dict:
    """Overall test-set metrics + per-attack-day breakdown."""
    results = {}

    # ── Calibrate threshold on Monday validation flows ─────────────────────
    logger.info("Calibrating threshold on Monday val flows …")
    m_val_seqs, m_val_lbl, _ = splits["monday_val"]
    val_scores, _ = score_split(m_val_seqs, m_val_lbl, scorer, max_len)
    benign_scores  = val_scores  # all monday flows are benign
    scorer.threshold = float(np.percentile(benign_scores, threshold_pct))
    logger.info("Threshold: %.4f (p%.0f of benign val)", scorer.threshold, threshold_pct)

    # ── Overall test evaluation ────────────────────────────────────────────
    logger.info("Scoring overall test set …")
    a_test_seqs, a_test_lbl, a_test_atypes = splits["all_test"]
    test_scores, test_labels = score_split(a_test_seqs, a_test_lbl, scorer, max_len)

    overall = compute_all_metrics(test_labels, test_scores, threshold=scorer.threshold)
    print_report(overall, f"OVERALL TEST — mode={mode}")
    results["overall"] = overall

    # Per-day breakdown skipped (attack days are single-class → NaN)
        # ── Save raw scores for plotting ───────────────────────────────────────
    np.savez(
        output_dir / f"scores_{mode}.npz",
        scores=test_scores,
        labels=test_labels,
        atypes=np.array(a_test_atypes),
        threshold=np.array([scorer.threshold]),
        benign_val_scores=benign_scores,
    )
    logger.info("Raw scores saved → %s", output_dir / f"scores_{mode}.npz")

    # Save per-day scores for detailed plotting
    per_day_scores = {}
    for day_key, day_info in data["day_data"].items():
        if day_info["seqs"]:
            sc, lb = score_split(day_info["seqs"], day_info["labels"], scorer, max_len)
            per_day_scores[day_key] = {"scores": sc, "labels": lb,
                                        "attack": day_info["meta"]["attack"]}
    with open(output_dir / f"per_day_scores_{mode}.pkl", "wb") as f:
        pickle.dump(per_day_scores, f)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--mode",          choices=["perplexity","supervised","combined"],
                        default="perplexity")
    parser.add_argument("--threshold_pct", type=float, default=95.0)
    parser.add_argument("--no-cache",      action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["training"].get("device", "auto"))
    seed   = cfg["training"].get("random_seed", 42)

    # ── Load tokenizer ────────────────────────────────────────────────────
    tok_path = cfg["tokenizer"]["save_path"]
    if not Path(tok_path).exists():
        pcap_map = discover_cic_pcaps(cfg["paths"]["pcap_dir"])
        monday_key = next(k for k in pcap_map if "monday" in k.lower())
        tokenizer  = build_tokenizer_from_monday(pcap_map[monday_key], cfg["tokenizer"])
    else:
        tokenizer = PcapFlowTokenizer.load(tok_path)

    max_len = tokenizer.max_pkts * tokenizer.tokens_per_pkt + 2

    # ── Load / parse flows ────────────────────────────────────────────────
    cache_path = output_dir / "flow_cache.pkl"
    if cache_path.exists() and not args.no_cache:
        logger.info("Loading flow cache …")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:
        pcap_map = discover_cic_pcaps(cfg["paths"]["pcap_dir"])
        data = load_all_days(pcap_map, tokenizer, cfg, seed=seed)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    splits = data["splits"]

    # ── Load model ────────────────────────────────────────────────────────
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model  = load_model(cfg, args.checkpoint, tokenizer.vocab_size, device)
    scorer = AnomalyScorer(model, device, score_mode=args.mode)

    # ── Run evaluation ────────────────────────────────────────────────────
    results = evaluate_overall(
        scorer, splits, data, max_len, output_dir, args.mode, args.threshold_pct
    )

    # ── Save JSON results ─────────────────────────────────────────────────
    out_json = output_dir / f"results_{args.mode}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved → %s", out_json)

    # Print summary table
    print("\n" + "═" * 60)
    print(f"  SUMMARY  (mode={args.mode}  threshold_pct={args.threshold_pct})")
    print("═" * 60)
    print(f"  {'Metric':<28} {'Value':>10}")
    print("─" * 60)
    ov = results.get("overall", {})
    for k in ["pr_auc","roc_auc","f1","precision","recall","fpr_at_tpr95"]:
        v = ov.get(k, float("nan"))
        print(f"  {k:<28} {v:>10.4f}")
    print("═" * 60)


if __name__ == "__main__":
    main()
