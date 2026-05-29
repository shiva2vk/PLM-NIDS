#!/usr/bin/env bash
# PLM-NIDS Complete Pipeline — GCS Bucket edition
# Checkpoints/outputs stay in bucket (survive VM shutdown)
# Logs go to /tmp (fast local writes)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$PROJECT_DIR/config.yaml"
NO_CACHE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-cache) NO_CACHE="--no-cache"; shift ;;
    --config)   CONFIG="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/run_pipeline.sh [--no-cache] [--config PATH]"
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

cd "$PROJECT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'
step() { echo -e "\n${BOLD}${BLUE}══ $* ${NC}"; }
ok()   { echo -e "${GREEN}✓  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }

echo -e "${BOLD}"
echo "════════════════════════════════════════════════════════"
echo "  PLM-NIDS  Complete Training + Evaluation Pipeline"
echo "  CIC-IDS-2017  |  RWKV-4 SSM  |  DPI-Free  |  A100"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
step "Pre-flight checks"

python3 -c "import torch, scapy, sklearn, yaml, tqdm, tensorboard, matplotlib, seaborn" \
  2>/dev/null || { echo "Missing packages — run: pip install -r requirements.txt"; exit 1; }
ok "Python packages present"

if [[ -f "outputs/flow_cache.pkl" ]]; then
  ok "flow_cache.pkl found ($(du -sh outputs/flow_cache.pkl | cut -f1)) — skipping PCAP parse"
else
  echo "ERROR: outputs/flow_cache.pkl not found"; exit 1
fi

mkdir -p checkpoints logs plots outputs /tmp/plm_logs
ok "Directories ready"

# ══════════════════════════════════════════════════════════════════════════════
step "PHASE 1 — Causal LM pre-training on Monday (benign only)"
echo "  Input  : flow_cache.pkl (Monday benign flows)"
echo "  Output : checkpoints/phase1_best.pt  [saved to GCS bucket]"
echo "  Epochs : $(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['phase1']['epochs'])")"

python3 scripts/train.py --config "$CONFIG" --phase 1 $NO_CACHE
ok "Phase 1 done → checkpoints/phase1_best.pt"

# ══════════════════════════════════════════════════════════════════════════════
step "PHASE 2 — Supervised fine-tuning on all 5 days"
echo "  Output : checkpoints/phase2_best.pt  [saved to GCS bucket]"
echo "  Epochs : $(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['phase2']['epochs'])")"

python3 scripts/train.py --config "$CONFIG" --phase 2 $NO_CACHE
ok "Phase 2 done → checkpoints/phase2_best.pt"

# ══════════════════════════════════════════════════════════════════════════════
step "EVALUATION — Mode 1/3: Perplexity (unsupervised)"
python3 scripts/evaluate.py \
  --config "$CONFIG" \
  --checkpoint checkpoints/phase1_best.pt \
  --mode perplexity --threshold_pct 95
ok "Perplexity evaluation done"

step "EVALUATION — Mode 2/3: Supervised (classifier)"
python3 scripts/evaluate.py \
  --config "$CONFIG" \
  --checkpoint checkpoints/phase2_best.pt \
  --mode supervised --threshold_pct 95
ok "Supervised evaluation done"

step "EVALUATION — Mode 3/3: Combined"
python3 scripts/evaluate.py \
  --config "$CONFIG" \
  --checkpoint checkpoints/phase2_best.pt \
  --mode combined --threshold_pct 95
ok "Combined evaluation done"

# ══════════════════════════════════════════════════════════════════════════════
step "BASELINES — RF / MLP / LSTM / IsolationForest"
python3 scripts/baselines.py --config "$CONFIG" $NO_CACHE
ok "Baselines done → outputs/baseline_results.json"

# ══════════════════════════════════════════════════════════════════════════════
step "ABLATION — 7 configurations × 2 epochs"
python3 scripts/ablation.py --config "$CONFIG" --fast-epochs 1
ok "Ablation done → outputs/ablation_results.json"

# ══════════════════════════════════════════════════════════════════════════════
step "LATEX TABLES — All 6 paper tables"
python3 scripts/generate_tables.py --config "$CONFIG"
ok "Tables done → outputs/tables/"

# ══════════════════════════════════════════════════════════════════════════════
step "TRAINING CURVES — Loss plots from TensorBoard logs"
python3 scripts/plot_training_curves.py --config "$CONFIG"
ok "Training curves → plots/fig_training_curves.pdf"

# ══════════════════════════════════════════════════════════════════════════════
step "LATENCY BENCHMARK"
python3 scripts/benchmark_latency.py \
  --config "$CONFIG" \
  --checkpoint checkpoints/phase1_best.pt
ok "Latency benchmark done"

# ══════════════════════════════════════════════════════════════════════════════
step "VISUALISATION — All paper figures"
for MODE in perplexity supervised combined; do
  python3 scripts/visualize.py --config "$CONFIG" --mode "$MODE"
  ok "Figures for mode=$MODE saved"
done

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-DATASET (HIKARI-2021) — optional
HIKARI_CSV="outputs/ALLFLOWMETER_HIKARI2021.csv"
if [[ -f "$HIKARI_CSV" ]]; then
  step "CROSS-DATASET — HIKARI-2021 TLS evaluation"
  python3 scripts/eval_cross_dataset.py \
    --config "$CONFIG" \
    --checkpoint checkpoints/phase1_best.pt \
    --hikari_csv "$HIKARI_CSV"
  ok "Cross-dataset eval done"
else
  warn "HIKARI CSV not found — skipping cross-dataset eval"
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
ok "PIPELINE COMPLETE — All results in GCS bucket"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*.pt 2>/dev/null | awk '{print "    "$NF, $5}' || echo "    (none)"
echo ""
echo "  Metrics:"
for F in outputs/results_*.json; do
  [[ -f "$F" ]] || continue
  RAUC=$(python3 -c "import json; d=json.load(open('$F')); print(f\"{d.get('overall',{}).get('roc_auc',0):.4f}\")" 2>/dev/null || echo "N/A")
  PRAUC=$(python3 -c "import json; d=json.load(open('$F')); print(f\"{d.get('overall',{}).get('pr_auc',0):.4f}\")" 2>/dev/null || echo "N/A")
  F1=$(python3 -c "import json; d=json.load(open('$F')); print(f\"{d.get('overall',{}).get('f1',0):.4f}\")" 2>/dev/null || echo "N/A")
  echo "    $(basename $F) → ROC-AUC=$RAUC  PR-AUC=$PRAUC  F1=$F1"
done
echo ""
echo "  Figures: $(ls plots/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDF files → plots/"
echo ""
echo "  TensorBoard: tensorboard --logdir /tmp/plm_logs/ --port 6006"
echo "════════════════════════════════════════════════════════"
