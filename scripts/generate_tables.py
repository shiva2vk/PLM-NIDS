"""
LaTeX + plain-text table generator for the arXiv paper.

Reads all results JSON files and produces ready-to-paste LaTeX.

Tables generated
----------------
  Table 1: Dataset statistics       (flows, packets, attack types per day)
  Table 2: Main comparison          (PLM vs all baselines)
  Table 3: Per-attack breakdown     (detection rate per attack category)
  Table 4: Ablation study           (model size, Phase-1, seq-length)
  Table 5: Model hyperparameters    (training details for reproducibility)
  Table 6: Compute efficiency       (RWKV O(T) vs Transformer O(T²))

Usage
-----
  python scripts/generate_tables.py --config config.yaml

Output
------
  outputs/tables/table1_dataset.tex
  outputs/tables/table2_comparison.tex
  outputs/tables/table3_per_attack.tex
  outputs/tables/table4_ablation.tex
  outputs/tables/table5_hyperparams.tex
  outputs/tables/table6_compute.tex
  outputs/tables/all_tables.tex        ← single file with all tables
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fmt(val, decimals: int = 4) -> str:
    """Format a float safely."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _bold_max(vals: list, decimals: int = 4) -> list:
    """Bold the maximum value in a list of formatted strings."""
    try:
        floats = [float(v) for v in vals]
        max_v  = max(floats)
        return [
            f"\\textbf{{{_fmt(v, decimals)}}}" if float(v) == max_v else _fmt(v, decimals)
            for v in vals
        ]
    except Exception:
        return [str(v) for v in vals]


# ── Table 1: Dataset statistics ───────────────────────────────────────────────

def table1_dataset(cfg: dict, output_dir: Path) -> str:
    rows = [
        ("Monday",    "BENIGN",                    "~344K",  "~11 GB", "0",      "100\\%"),
        ("Tuesday",   "FTP-Patator, SSH-Patator",  "~651K",  "~10 GB", "~11K",   "\\phantom{0}~1.7\\%"),
        ("Wednesday", "DoS Slowloris/Hulk,",        "~647K",  "~12 GB", "~440K",  "~68\\%"),
        ("Wednesday", "\\quad GoldenEye, Heartbleed","",     "",       "",       ""),
        ("Thursday",  "Web Attacks, Infiltration",  "~533K",  "~7.7 GB","~85K",  "~16\\%"),
        ("Friday",    "Botnet ARES, PortScan, DDoS","~534K", "~8.2 GB","~105K",  "~20\\%"),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{CIC-IDS-2017 Dataset Statistics. All flows extracted from raw PCAPs "
        "using DPI-free L3/L4-only parsing (no payload inspection).}",
        "\\label{tab:dataset}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "\\textbf{Day} & \\textbf{Attack Type} & \\textbf{Packets} & "
        "\\textbf{Size} & \\textbf{Attack Flows} & \\textbf{Attack Rate} \\\\",
        "\\midrule",
    ]
    for day, atype, pkts, size, aflows, arate in rows:
        lines.append(f"{day} & {atype} & {pkts} & {size} & {aflows} & {arate} \\\\")
    lines += [
        "\\midrule",
        "\\textbf{Total} & 4 attack categories & $\\sim$3.2M & $\\sim$48 GB & "
        "$\\sim$641K & $\\sim$20\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Table 2: Main comparison vs baselines ─────────────────────────────────────

def table2_comparison(
    plm_results: Dict,
    baseline_results: Dict,
    output_dir: Path,
) -> str:
    methods = [
        ("Random",              baseline_results.get("Random", {}),          False),
        ("Isolation Forest",    baseline_results.get("IsolationForest", {}), False),
        ("Random Forest",       baseline_results.get("RandomForest", {}),    False),
        ("MLP (3-layer)",       baseline_results.get("MLP", {}),             False),
        ("LSTM (token seq.)",   baseline_results.get("LSTM", {}),            False),
        ("PLM-PPL (ours)",      plm_results.get("perplexity", {}).get("overall", {}), True),
        ("PLM-CLS (ours)",      plm_results.get("supervised", {}).get("overall", {}), True),
        ("PLM-CMB (ours)",      plm_results.get("combined",   {}).get("overall", {}), True),
    ]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Detection performance on CIC-IDS-2017 (test set, 15\\% hold-out). "
        "All methods use identical DPI-free L3/L4 features. "
        "\\textbf{Bold} = best per column. "
        "PLM-PPL: unsupervised perplexity scoring. "
        "PLM-CLS: supervised classifier head. "
        "PLM-CMB: geometric mean of both scores.}",
        "\\label{tab:comparison}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & "
        "\\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & "
        "\\textbf{FPR@TPR95} \\\\",
        "\\midrule",
    ]

    # Collect all values for bolding
    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
    fpr_vals = []
    col_vals = {m: [] for m in metrics}
    for _, m_dict, _ in methods:
        for k in metrics:
            col_vals[k].append(m_dict.get(k, 0.0) or 0.0)
        fpr_vals.append(m_dict.get("fpr_at_tpr95", 1.0) or 1.0)

    # Bold best (max for most, min for FPR)
    def bold_col(vals, higher_better=True):
        floats = [float(v) for v in vals]
        best = max(floats) if higher_better else min(floats)
        return [f"\\textbf{{{_fmt(v)}}}" if float(v) == best else _fmt(v) for v in vals]

    bold = {k: bold_col(col_vals[k]) for k in metrics}
    bold_fpr = bold_col(fpr_vals, higher_better=False)

    prev_ours = False
    for i, (name, m_dict, is_ours) in enumerate(methods):
        if is_ours and not prev_ours:
            lines.append("\\midrule")
        prev_ours = is_ours
        nm = f"\\textit{{{name}}}" if is_ours else name
        row = (f"{nm} & {bold['pr_auc'][i]} & {bold['roc_auc'][i]} & "
               f"{bold['f1'][i]} & {bold['precision'][i]} & "
               f"{bold['recall'][i]} & {bold_fpr[i]} \\\\")
        lines.append(row)

    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Table 3: Per-attack breakdown ─────────────────────────────────────────────

def table3_per_attack(plm_results: Dict, output_dir: Path) -> str:
    attack_map = {
        "Monday-WorkingHours":    "BENIGN",
        "Tuesday-WorkingHours":   "FTP/SSH Brute-Force",
        "Wednesday-workingHours": "DoS / Heartbleed",
        "Thursday-WorkingHours":  "Web Attacks / Infiltration",
        "Friday-WorkingHours ":   "Botnet / PortScan / DDoS",
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Per-attack-category detection performance for PLM-CMB "
        "(combined perplexity + classifier scoring).}",
        "\\label{tab:per_attack}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Attack Category} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & "
        "\\textbf{F1} & \\textbf{FPR@TPR95} \\\\",
        "\\midrule",
    ]

    combined = plm_results.get("combined", {})
    for day_key, label in attack_map.items():
        if label == "BENIGN":
            continue
        day_m = combined.get(day_key, {})
        if not day_m or "f1" not in day_m:
            lines.append(f"{label} & — & — & — & — \\\\")
        else:
            lines.append(
                f"{label} & {_fmt(day_m.get('pr_auc'))} & "
                f"{_fmt(day_m.get('roc_auc'))} & {_fmt(day_m.get('f1'))} & "
                f"{_fmt(day_m.get('fpr_at_tpr95'))} \\\\"
            )

    ov = combined.get("overall", {})
    lines += [
        "\\midrule",
        f"\\textbf{{Micro-Average}} & {_fmt(ov.get('pr_auc'))} & "
        f"{_fmt(ov.get('roc_auc'))} & {_fmt(ov.get('f1'))} & "
        f"{_fmt(ov.get('fpr_at_tpr95'))} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Table 4: Ablation study ───────────────────────────────────────────────────

def table4_ablation(ablation_results: Dict, output_dir: Path) -> str:
    display = {
        "Full_d256_L6":      ("PLM (d=256, L=6) — full model", True),
        "Small_d64_L2":      ("PLM (d=64,  L=2) — small",      False),
        "Medium_d128_L4":    ("PLM (d=128, L=4) — medium",      False),
        "No_Phase1_pretrain":("w/o Phase-1 pre-training",        False),
        "Score_perplexity":  ("Scoring: perplexity only",        False),
        "Score_supervised":  ("Scoring: classifier only",        False),
        "MaxPkts_32":        ("Max 32 pkts/flow (288 tokens)",   False),
        "MaxPkts_64":        ("Max 64 pkts/flow (578 tokens)",   False),
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation study. Each row disables or varies one design "
        "component. All runs use 5 epochs for speed; full-model results "
        "from Table~\\ref{tab:comparison} trained for 20+15 epochs.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & "
        "\\textbf{F1} & \\textbf{Params} \\\\",
        "\\midrule",
    ]

    prev_full = True
    for key, (label, is_full) in display.items():
        m = ablation_results.get(key, {})
        if not isinstance(m, dict) or "pr_auc" not in m:
            continue
        if not is_full and prev_full:
            lines.append("\\midrule")
        prev_full = is_full
        lbl = f"\\textbf{{{label}}}" if is_full else label
        params = f"{m.get('params', 0):,}" if m.get("params") else "—"
        lines.append(
            f"{lbl} & {_fmt(m.get('pr_auc'))} & {_fmt(m.get('roc_auc'))} & "
            f"{_fmt(m.get('f1'))} & {params} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Table 5: Hyperparameters ──────────────────────────────────────────────────

def table5_hyperparams(cfg: dict, output_dir: Path) -> str:
    m  = cfg["model"]
    t  = cfg["training"]
    p1 = t["phase1"]
    p2 = t["phase2"]
    tok = cfg["tokenizer"]
    rows = [
        ("\\multicolumn{2}{l}{\\textit{Tokenizer}}", ""),
        ("Len / IAT / TTL bins", f"{tok['n_len_bins']} / {tok['n_dt_bins']} / {tok['n_ttl_bins']}"),
        ("Port hash buckets",    str(tok["n_port_buckets"])),
        ("Max pkts / flow",      str(tok["max_pkts_per_flow"])),
        ("Vocabulary size",      "$\\sim$227 tokens"),
        ("Tokens per packet",    "9"),
        ("Max sequence length",  f"{tok['max_pkts_per_flow'] * 9 + 2} tokens"),
        ("\\multicolumn{2}{l}{\\textit{Model (RWKV-4)}}", ""),
        ("Embedding dim $d$",    str(m["d_model"])),
        ("RWKV layers $L$",      str(m["n_layers"])),
        ("Tied embeddings",      "Yes"),
        ("Dropout",              str(m["dropout"])),
        ("\\multicolumn{2}{l}{\\textit{Phase-1 Training}}", ""),
        ("Epochs",               str(p1["epochs"])),
        ("Batch size",           str(p1["batch_size"])),
        ("Peak LR",              str(p1["lr"])),
        ("Scheduler",            "OneCycleLR"),
        ("\\multicolumn{2}{l}{\\textit{Phase-2 Fine-tuning}}", ""),
        ("Epochs",               str(p2["epochs"])),
        ("Batch size",           str(p2["batch_size"])),
        ("Peak LR",              str(p2["lr"])),
        ("Class weight (attack)",str(p2.get("attack_class_weight", 8.0))),
        ("Backbone freeze",      f"{p2.get('freeze_backbone_epochs',3)} epochs"),
        ("\\multicolumn{2}{l}{\\textit{Inference}}", ""),
        ("Threshold calibration","p95 of benign val scores"),
        ("Anomaly score (unsup)","Perplexity (mean token NLL)"),
        ("Anomaly score (sup)",  "Attack class probability"),
        ("Per-flow state",       "RWKV hidden state (TTL-evicted)"),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Model and training hyperparameters for reproducibility.}",
        "\\label{tab:hyperparams}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "\\textbf{Hyperparameter} & \\textbf{Value} \\\\",
        "\\midrule",
    ]
    for k, v in rows:
        if v == "":
            lines.append(f"{k} \\\\")
            lines.append("\\midrule")
        else:
            lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Table 6: Compute efficiency ───────────────────────────────────────────────

def table6_compute(output_dir: Path) -> str:
    lengths = [38, 100, 200, 380, 521, 750, 1000, 1154]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Compute operations per flow: RWKV O(T) vs Transformer O(T$^2$). "
        "Column \\emph{Speedup} = T$^2$/T = T. "
        "Sequence lengths reflect observed CIC-IDS-2017 flow distribution.}",
        "\\label{tab:compute}",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "\\textbf{Seq\\ len T} & \\textbf{Transformer ops} & "
        "\\textbf{RWKV ops} & \\textbf{Speedup} \\\\",
        "\\midrule",
    ]
    for t in lengths:
        note = ""
        if t == 38:   note = " (median)"
        if t == 521:  note = " (p95 Wed)"
        if t == 1154: note = " (max)"
        lines.append(
            f"{t}{note} & {t**2:,} & {t:,} & {t}$\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    table_dir  = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Load result JSONs (use empty dicts if not yet available)
    def _load(fname):
        p = output_dir / fname
        return json.load(open(p)) if p.exists() else {}

    plm_results      = {
        "perplexity": _load("results_perplexity.json"),
        "supervised": _load("results_supervised.json"),
        "combined":   _load("results_combined.json"),
    }
    baseline_results = _load("baseline_results.json")
    ablation_results = _load("ablation_results.json")

    tables = {
        "table1_dataset.tex":    table1_dataset(cfg, table_dir),
        "table2_comparison.tex": table2_comparison(plm_results, baseline_results, table_dir),
        "table3_per_attack.tex": table3_per_attack(plm_results, table_dir),
        "table4_ablation.tex":   table4_ablation(ablation_results, table_dir),
        "table5_hyperparams.tex":table5_hyperparams(cfg, table_dir),
        "table6_compute.tex":    table6_compute(table_dir),
    }

    all_tex = [
        "% ════════════════════════════════════════════════════",
        "% PLM-NIDS — All paper tables (auto-generated)",
        "% Copy individual tables or \\input{} this file",
        "% ════════════════════════════════════════════════════",
        "",
    ]

    for fname, content in tables.items():
        path = table_dir / fname
        path.write_text(content + "\n")
        all_tex.append(f"% ── {fname} ──")
        all_tex.append(content)
        all_tex.append("")
        logger.info("Saved → %s", path)

    (table_dir / "all_tables.tex").write_text("\n".join(all_tex))
    logger.info("All tables → %s", table_dir / "all_tables.tex")

    print("\n" + "═" * 55)
    print("  LaTeX tables saved to: outputs/tables/")
    print("═" * 55)
    for f in sorted(table_dir.glob("*.tex")):
        print(f"  {f.name}")
    print("═" * 55)
    print("  Usage in your paper:")
    print("    \\input{outputs/tables/table2_comparison}")
    print("    — or paste contents directly into your .tex")
    print("═" * 55)


if __name__ == "__main__":
    main()
