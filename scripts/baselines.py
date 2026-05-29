"""
Baseline models for paper comparison (Table 1).

All baselines use the SAME DPI-free L3/L4 features as the PLM —
ensuring a fair comparison where the only variable is the model architecture.

Baselines:
  1. Random           — trivial baseline (AUC=0.50 target)
  2. Isolation Forest — unsupervised anomaly detection
  3. Random Forest    — best classical ML baseline for NIDS
  4. MLP              — feedforward neural network on flat feature vector
  5. LSTM             — sequential model on token sequences (same input as PLM)

Usage
-----
  python scripts/baselines.py --config config.yaml

Output
------
  outputs/baseline_results.json    — all metrics
  outputs/baseline_scores.npz      — raw scores for plotting
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cic_loader import discover_cic_pcaps, DAY_META
from data.flow_features import load_all_days_features, FEATURE_NAMES, N_FEATURES
from evaluation.metrics import compute_all_metrics, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── LSTM baseline ─────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM over the same token sequences as the PLM.
    Fair comparison: identical input, different architecture.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden: int = 128, n_layers: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(embed_dim, hidden, n_layers,
                               batch_first=True, dropout=0.2)
        self.head   = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        return self.head(h[-1])


def train_lstm(
    train_seqs, train_labels, val_seqs, val_labels,
    vocab_size: int, device: torch.device,
    epochs: int = 10, batch_size: int = 128,
) -> LSTMClassifier:
    from data.pcap_dataset import PcapClassifierDataset, pcap_collate_cls
    from data.dataset import make_weighted_sampler

    max_len = max(len(s) for s in train_seqs + val_seqs)
    train_ds = PcapClassifierDataset(train_seqs, train_labels, max_len=max_len)
    val_ds   = PcapClassifierDataset(val_seqs,   val_labels,   max_len=max_len)

    sampler  = make_weighted_sampler(np.array(train_labels), attack_weight=8.0)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          collate_fn=pcap_collate_cls)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False,
                          collate_fn=pcap_collate_cls)

    model = LSTMClassifier(vocab_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 8.0], device=device)
    )
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_loss += crit(model(x), y).item()
        val_loss /= len(val_dl)
        logger.info("LSTM epoch %02d/%02d | val_loss=%.4f", epoch+1, epochs, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_lstm(model: LSTMClassifier, seqs, labels,
               vocab_size: int, device: torch.device) -> np.ndarray:
    from data.pcap_dataset import PcapClassifierDataset, pcap_collate_cls
    max_len = max(len(s) for s in seqs)
    ds = PcapClassifierDataset(seqs, labels, max_len=max_len)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=pcap_collate_cls)
    model.eval().to(device)
    probs = []
    for x, _ in dl:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["training"].get("random_seed", 42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Feature-based data ────────────────────────────────────────────────
    feat_cache = output_dir / "baseline_features.npz"
    if feat_cache.exists() and not args.no_cache:
        logger.info("Loading feature cache …")
        d = np.load(feat_cache)
        X_tr, X_te, y_tr, y_te = d["X_tr"], d["X_te"], d["y_tr"], d["y_te"]
    else:
        pcap_map = discover_cic_pcaps(cfg["paths"]["pcap_dir"])
        X_tr, X_te, y_tr, y_te = load_all_days_features(pcap_map, seed=seed)
        np.savez(feat_cache, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)
        logger.info("Features: train=%d test=%d features=%d", len(X_tr), len(X_te), X_tr.shape[1])

    # ── Sequence data for LSTM ────────────────────────────────────────────
    flow_cache = output_dir / "flow_cache.pkl"
    run_lstm = flow_cache.exists()
    if run_lstm:
        with open(flow_cache, "rb") as f:
            flow_data = pickle.load(f)
        tok_path = cfg["tokenizer"]["save_path"]
        from data.pcap_tokenizer import PcapFlowTokenizer
        tokenizer  = PcapFlowTokenizer.load(tok_path)
        splits     = flow_data["splits"]
        tr_seqs, tr_lbl, _ = splits["all_train"]
        te_seqs, te_lbl, _ = splits["all_test"]
        va_seqs, va_lbl, _ = splits["all_val"]
    else:
        logger.warning("flow_cache.pkl not found — skipping LSTM baseline. Run train.py first.")

    results: Dict = {}
    all_scores: Dict = {}

    # ── 1. Random baseline ────────────────────────────────────────────────
    logger.info("Baseline 1/5: Random …")
    np.random.seed(seed)
    rand_scores = np.random.rand(len(y_te)).astype(np.float32)
    m = compute_all_metrics(y_te, rand_scores)
    results["Random"] = m; all_scores["Random"] = rand_scores
    print_report(m, "Random Baseline")

    # ── 2. Isolation Forest (unsupervised) ────────────────────────────────
    logger.info("Baseline 2/5: Isolation Forest …")
    t0 = time.time()
    iso = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  IsolationForest(n_estimators=100, contamination=0.1,
                                    random_state=seed, n_jobs=-1)),
    ])
    iso.fit(X_tr[y_tr == 0])   # train on benign only (unsupervised)
    # IsolationForest: lower score = more anomalous → negate
    iso_scores = -iso.named_steps["model"].score_samples(
        iso.named_steps["scaler"].transform(X_te)
    )
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.ptp() + 1e-9)
    elapsed = time.time() - t0
    m = compute_all_metrics(y_te, iso_scores)
    m["latency_ms"] = round(elapsed / len(X_te) * 1000, 4)
    results["IsolationForest"] = m; all_scores["IsolationForest"] = iso_scores
    print_report(m, "Isolation Forest")

    # ── 3. Random Forest ─────────────────────────────────────────────────
    logger.info("Baseline 3/5: Random Forest …")
    t0 = time.time()
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1, max_depth=20,
        )),
    ])
    rf.fit(X_tr, y_tr)
    rf_scores = rf.predict_proba(X_te)[:, 1].astype(np.float32)
    elapsed = time.time() - t0
    m = compute_all_metrics(y_te, rf_scores)
    m["latency_ms"] = round(elapsed / len(X_te) * 1000, 4)
    results["RandomForest"] = m; all_scores["RandomForest"] = rf_scores
    print_report(m, "Random Forest")

    # ── 4. MLP ────────────────────────────────────────────────────────────
    logger.info("Baseline 4/5: MLP …")
    t0 = time.time()
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", max_iter=200,
            random_state=seed, early_stopping=True, validation_fraction=0.1,
        )),
    ])
    mlp.fit(X_tr, y_tr)
    mlp_scores = mlp.predict_proba(X_te)[:, 1].astype(np.float32)
    elapsed = time.time() - t0
    m = compute_all_metrics(y_te, mlp_scores)
    m["latency_ms"] = round(elapsed / len(X_te) * 1000, 4)
    results["MLP"] = m; all_scores["MLP"] = mlp_scores
    print_report(m, "MLP")

    # ── 5. LSTM (sequential, same tokens as PLM) ──────────────────────────
    if run_lstm:
        logger.info("Baseline 5/5: LSTM …")
        t0 = time.time()
        lstm_model = train_lstm(
            tr_seqs, tr_lbl, va_seqs, va_lbl,
            vocab_size=tokenizer.vocab_size, device=device,
        )
        lstm_scores = score_lstm(lstm_model, te_seqs, te_lbl,
                                  tokenizer.vocab_size, device)
        elapsed = time.time() - t0
        m = compute_all_metrics(np.array(te_lbl), lstm_scores)
        m["latency_ms"] = round(elapsed / max(len(te_seqs), 1) * 1000, 4)
        results["LSTM"] = m; all_scores["LSTM"] = lstm_scores
        print_report(m, "LSTM")
    else:
        results["LSTM"] = {"note": "skipped — run train.py first to generate flow cache"}

    # ── Save ──────────────────────────────────────────────────────────────
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(
        output_dir / "baseline_scores.npz",
        y_test=y_te,
        **{k: v for k, v in all_scores.items()},
    )

    logger.info("Saved → outputs/baseline_results.json + baseline_scores.npz")

    # Quick comparison table
    print("\n" + "═" * 72)
    print(f"  {'Method':<20} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'FPR@95':>8}")
    print("─" * 72)
    for name, m in results.items():
        if not isinstance(m, dict) or "pr_auc" not in m:
            continue
        print(f"  {name:<20} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} "
              f"{m['f1']:>8.4f} {m['fpr_at_tpr95']:>8.4f}")
    print("═" * 72)


if __name__ == "__main__":
    main()
