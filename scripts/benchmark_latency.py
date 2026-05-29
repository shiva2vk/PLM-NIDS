"""
Inference latency benchmark — Table for the paper's deployment section.

Measures actual throughput and latency at different batch sizes and
sequence lengths. Compares RWKV streaming vs Transformer (estimated).

Usage
-----
  python scripts/benchmark_latency.py --config config.yaml \
      --checkpoint checkpoints/phase1_best.pt

Output
------
  outputs/latency_results.json
  plots/fig_latency_benchmark.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pcap_tokenizer import PcapFlowTokenizer
from models.plm import PLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WARMUP_ITERS = 10
BENCH_ITERS  = 50


def _bench_batch(model: PLM, batch_size: int, seq_len: int,
                 device: torch.device, mode: str = "lm") -> Dict:
    """Measure latency and throughput for a fixed (B, T) batch."""
    model.eval()
    ids = torch.randint(1, model.vocab_size, (batch_size, seq_len),
                        device=device)
    tgt = torch.randint(1, model.vocab_size, (batch_size, seq_len),
                        device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(ids, targets=tgt, mode=mode)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            model(ids, targets=tgt, mode=mode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_batch  = elapsed / BENCH_ITERS * 1000
    flows_per_sec = batch_size / (elapsed / BENCH_ITERS)
    ms_per_flow   = ms_per_batch / batch_size
    tokens_per_sec = batch_size * seq_len / (elapsed / BENCH_ITERS)

    return {
        "batch_size":     batch_size,
        "seq_len":        seq_len,
        "ms_per_batch":   round(ms_per_batch, 3),
        "ms_per_flow":    round(ms_per_flow, 4),
        "flows_per_sec":  round(flows_per_sec, 1),
        "tokens_per_sec": round(tokens_per_sec, 0),
    }


def _bench_streaming(model: PLM, seq_len: int, device: torch.device) -> Dict:
    """Measure single-flow streaming (token-by-token) latency."""
    model.eval()
    states = model.init_states(1, device)
    tokens = torch.randint(1, model.vocab_size, (seq_len,), device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            s = model.init_states(1, device)
            for tok in tokens[:10]:
                _, s = model.step(tok.unsqueeze(0), s)

    # Benchmark one full flow
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            s = model.init_states(1, device)
            for tok in tokens:
                _, s = model.step(tok.unsqueeze(0), s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_flow   = elapsed / BENCH_ITERS * 1000
    ms_per_token  = ms_per_flow / seq_len
    flows_per_sec = BENCH_ITERS / elapsed

    return {
        "seq_len":        seq_len,
        "ms_per_flow":    round(ms_per_flow, 3),
        "ms_per_token":   round(ms_per_token, 4),
        "flows_per_sec":  round(flows_per_sec, 1),
        "mode":           "streaming (token-by-token)",
    }


def plot_latency(results: Dict, plot_dir: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Throughput vs batch size (seq=512)
    ax = axes[0]
    batch_results = [r for r in results["batch_sweep"] if r["seq_len"] == 512]
    if batch_results:
        bs  = [r["batch_size"]    for r in batch_results]
        fps = [r["flows_per_sec"] for r in batch_results]
        ax.bar(range(len(bs)), fps, color="#2196F3", alpha=0.85)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs])
        for i, v in enumerate(fps):
            ax.text(i, v + max(fps)*0.01, f"{v:,.0f}", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("Flows / second", fontsize=12)
    ax.set_title("Batch Inference Throughput\n(seq_len=512 tokens)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Latency vs sequence length (batch=64)
    ax2 = axes[1]
    seq_results = [r for r in results["batch_sweep"] if r["batch_size"] == 64]
    if seq_results:
        seqs = [r["seq_len"]    for r in seq_results]
        ms   = [r["ms_per_flow"] for r in seq_results]
        ax2.plot(seqs, ms, "b-o", linewidth=2.5, markersize=8, label="RWKV O(T)")
        # Estimated Transformer O(T²) scaling
        if seqs:
            ref_ms = ms[0]
            ref_t  = seqs[0]
            t_ms = [ref_ms * (s / ref_t) ** 2 for s in seqs]
            ax2.plot(seqs, t_ms, "r--s", linewidth=2, markersize=6,
                     alpha=0.7, label="Transformer O(T²) est.")
    ax2.set_xlabel("Sequence length (tokens)", fontsize=12)
    ax2.set_ylabel("Latency per flow (ms)", fontsize=12)
    ax2.set_title("Latency vs Sequence Length\n(batch=64)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Streaming vs batch comparison
    ax3 = axes[2]
    stream = results.get("streaming", {})
    batch64 = next((r for r in results.get("batch_sweep", [])
                    if r["batch_size"] == 64 and r["seq_len"] == 512), {})
    labels  = ["Batch\n(B=64, T=512)", "Streaming\n(B=1, T=512)"]
    fps_vals = [
        batch64.get("flows_per_sec", 0),
        stream.get("flows_per_sec", 0),
    ]
    colors = ["#2196F3", "#FF9800"]
    bars = ax3.bar(labels, fps_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, fps_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, v + max(fps_vals)*0.01,
                 f"{v:,.0f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold")
    ax3.set_ylabel("Flows / second", fontsize=12)
    ax3.set_title("Batch vs Streaming Inference\nThroughput Comparison",
                  fontsize=12, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"PLM-NIDS Inference Latency Benchmark\n"
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS'})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "fig_latency_benchmark.pdf"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    plot_dir   = Path(cfg["paths"]["plot_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Benchmarking on: %s", device)

    # Load model
    tok_path = cfg["tokenizer"]["save_path"]
    if Path(tok_path).exists():
        tokenizer = PcapFlowTokenizer.load(tok_path)
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = 227

    m = cfg["model"]
    model = PLM(vocab_size=vocab_size, d_model=m["d_model"],
                n_layers=m["n_layers"], dropout=0.0, n_classes=2)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    model = model.to(device).eval()
    logger.info("Model params: %s", f"{model.count_parameters():,}")

    results: Dict = {"device": str(device), "model_params": model.count_parameters()}

    # ── Batch sweep ───────────────────────────────────────────────────────
    batch_sweep = []
    logger.info("Benchmarking batch inference …")
    for seq_len in [128, 256, 512, 1154]:
        for batch_size in [1, 8, 32, 64, 128]:
            try:
                r = _bench_batch(model, batch_size, seq_len, device)
                batch_sweep.append(r)
                logger.info("  B=%3d T=%4d → %.1f flows/s  %.3f ms/flow",
                            batch_size, seq_len, r["flows_per_sec"], r["ms_per_flow"])
            except Exception as e:
                logger.warning("  B=%d T=%d OOM: %s", batch_size, seq_len, e)

    results["batch_sweep"] = batch_sweep

    # ── Streaming (token-by-token) ────────────────────────────────────────
    logger.info("Benchmarking streaming inference …")
    try:
        stream = _bench_streaming(model, seq_len=512, device=device)
        results["streaming"] = stream
        logger.info("  Streaming T=512 → %.1f flows/s  %.3f ms/flow  %.4f ms/token",
                    stream["flows_per_sec"], stream["ms_per_flow"], stream["ms_per_token"])
    except Exception as e:
        logger.warning("Streaming benchmark failed: %s", e)

    # ── Save ──────────────────────────────────────────────────────────────
    out_json = output_dir / "latency_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved → %s", out_json)

    plot_latency(results, plot_dir, cfg.get("evaluation", {}).get("dpi", 150))

    # Print summary table
    print("\n" + "═" * 65)
    print(f"  Latency Benchmark  ({device})")
    print("─" * 65)
    print(f"  {'B':>5}  {'T':>5}  {'ms/flow':>10}  {'flows/s':>10}  {'tok/s':>12}")
    print("─" * 65)
    for r in batch_sweep:
        if r["batch_size"] in [1, 64, 128] and r["seq_len"] in [128, 512, 1154]:
            print(f"  {r['batch_size']:>5}  {r['seq_len']:>5}  "
                  f"{r['ms_per_flow']:>10.3f}  {r['flows_per_sec']:>10.1f}  "
                  f"{r['tokens_per_sec']:>12.0f}")
    if "streaming" in results:
        s = results["streaming"]
        print(f"  {'1(stream)':>5}  {s['seq_len']:>5}  "
              f"{s['ms_per_flow']:>10.3f}  {s['flows_per_sec']:>10.1f}  {'—':>12}")
    print("═" * 65)


if __name__ == "__main__":
    main()
