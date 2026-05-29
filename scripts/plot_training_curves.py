"""
Extract training loss curves from TensorBoard logs → publication-quality PDF.

Run AFTER training completes (or during training for live curves).

Usage
-----
  python scripts/plot_training_curves.py --config config.yaml

Output
------
  plots/fig_training_curves.pdf   Phase-1 LM loss + Phase-2 cls loss + val loss
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_tb(log_dir: Path, tag: str):
    """Read a scalar tag from TensorBoard event files. Returns (steps, values)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("pip install tensorboard")

    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_training_curves(log_dir: Path, plot_dir: Path, dpi: int = 150) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Phase 1: LM loss ──────────────────────────────────────────────────
    ax = axes[0]
    tr_steps, tr_vals = _read_tb(log_dir, "phase1/train_loss")
    va_steps, va_vals = _read_tb(log_dir, "phase1/val_loss")

    if tr_vals:
        ax.plot(tr_steps, tr_vals, color="#2196F3", linewidth=1.5,
                alpha=0.7, label="Train loss")
        # Smooth with rolling mean for clarity
        if len(tr_vals) > 20:
            window = max(len(tr_vals) // 20, 5)
            smooth = np.convolve(tr_vals, np.ones(window)/window, mode="valid")
            ax.plot(tr_steps[window-1:], smooth, color="#1565C0",
                    linewidth=2.5, label=f"Train loss (smooth, w={window})")
    if va_vals:
        ax.plot(va_steps, va_vals, color="#F44336", linewidth=2.5,
                marker="o", markersize=5, label="Val loss")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Phase 1 — Causal LM Pre-training\n(Benign Grammar Learning)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if not tr_vals and not va_vals:
        ax.text(0.5, 0.5, "Training in progress…\nRe-run after Phase-1 completes.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ── Phase 2: Classification loss + accuracy ────────────────────────────
    ax2 = axes[1]
    tr2_steps, tr2_vals = _read_tb(log_dir, "phase2/train_loss")
    va2_steps, va2_vals = _read_tb(log_dir, "phase2/val_loss")
    ac_steps,  ac_vals  = _read_tb(log_dir, "phase2/val_acc")

    if tr2_vals:
        ax2.plot(tr2_steps, tr2_vals, color="#FF9800", linewidth=1.5,
                 alpha=0.7, label="Train loss")
        if len(tr2_vals) > 20:
            window = max(len(tr2_vals) // 20, 5)
            smooth = np.convolve(tr2_vals, np.ones(window)/window, mode="valid")
            ax2.plot(tr2_steps[window-1:], smooth, color="#E65100",
                     linewidth=2.5, label=f"Train loss (smooth)")
    if va2_vals:
        ax2.plot(va2_steps, va2_vals, color="#F44336", linewidth=2.5,
                 marker="o", markersize=5, label="Val loss")

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Phase 2 — Supervised Fine-tuning\n(Attack Classification)",
                  fontsize=12, fontweight="bold")

    # Overlay accuracy on secondary axis
    if ac_vals:
        ax2b = ax2.twinx()
        ax2b.plot(ac_steps, ac_vals, color="#4CAF50", linewidth=2.5,
                  linestyle="--", marker="s", markersize=5, label="Val accuracy")
        ax2b.set_ylabel("Validation Accuracy", fontsize=12, color="#4CAF50")
        ax2b.tick_params(axis="y", labelcolor="#4CAF50")
        ax2b.set_ylim(0, 1.05)
        lines2, labels2 = ax2b.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    if not tr2_vals and not va2_vals:
        ax2.text(0.5, 0.5, "Training in progress…\nRe-run after Phase-2 completes.",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="gray",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("PLM-NIDS Training Convergence — CIC-IDS-2017",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = plot_dir / "fig_training_curves.pdf"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir  = Path(cfg["paths"]["log_dir"])
    plot_dir = Path(cfg["paths"]["plot_dir"])
    dpi      = cfg.get("evaluation", {}).get("dpi", 150)

    plot_training_curves(log_dir, plot_dir, dpi)
    print(f"Saved → {plot_dir}/fig_training_curves.pdf")


if __name__ == "__main__":
    main()
