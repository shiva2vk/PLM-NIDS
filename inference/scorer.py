"""
Anomaly scoring and per-flow state cache for streaming inference.

AnomalyScorer       → batch or streaming perplexity/classifier scoring
FlowStateCache      → TTL-evicting dict mapping flow-key → RWKV hidden state
StreamingPipeline   → ties tokenizer + scorer + cache for live or offline use
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.tokenizer import FlowTokenizer
from models.plm import PLM
from models.rwkv import RWKVState

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    Wraps a trained PLM and computes anomaly scores for batches of flows.

    score_mode = "perplexity"  → unsupervised (Phase-1 model)
    score_mode = "supervised"  → uses classifier head probability (Phase-2 model)
    score_mode = "combined"    → geometric mean of perplexity and classifier score
    """

    def __init__(
        self,
        model: PLM,
        device: torch.device,
        score_mode: str = "perplexity",
        threshold: Optional[float] = None,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.score_mode = score_mode
        self.threshold = threshold

    @torch.no_grad()
    def score_batch(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        Args:
            token_ids : (B, T) long tensor
        Returns:
            scores    : (B,) numpy float32 — higher = more anomalous
        """
        token_ids = token_ids.to(self.device)

        if self.score_mode == "perplexity":
            ppl = self.model.flow_perplexity(token_ids)
            return ppl.cpu().numpy().astype(np.float32)

        if self.score_mode == "supervised":
            out = self.model(token_ids, mode="cls")
            probs = F.softmax(out["cls_logits"], dim=-1)
            return probs[:, 1].cpu().numpy().astype(np.float32)  # attack probability

        if self.score_mode == "combined":
            ppl = self.model.flow_perplexity(token_ids).cpu().numpy()
            out = self.model(token_ids, mode="cls")
            attack_prob = F.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
            # Normalise PPL to [0,1] via sigmoid(log(ppl/median))
            ppl_norm = 1 / (1 + np.exp(-(np.log(ppl + 1e-9) - 3.0)))
            return (ppl_norm * attack_prob) ** 0.5   # geometric mean

        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def calibrate_threshold(
        self, benign_token_ids: torch.Tensor, percentile: float = 95.0
    ) -> float:
        """Set and return threshold from benign validation scores."""
        scores = self.score_batch(benign_token_ids)
        self.threshold = float(np.percentile(scores, percentile))
        logger.info("Threshold calibrated to %.4f (p%.0f of benign scores)",
                    self.threshold, percentile)
        return self.threshold

    def predict(self, token_ids: torch.Tensor) -> np.ndarray:
        """Return binary predictions (1=attack) based on calibrated threshold."""
        assert self.threshold is not None, "Call calibrate_threshold() first."
        scores = self.score_batch(token_ids)
        return (scores >= self.threshold).astype(np.int32)


# ── Per-flow state cache ──────────────────────────────────────────────────────

class FlowStateCache:
    """
    Maps flow 5-tuples to their RWKV hidden states with TTL eviction.

    Keeps at most `max_flows` entries; evicts oldest-first (LRU-like).
    """

    def __init__(self, ttl_seconds: float = 300.0, max_flows: int = 50_000) -> None:
        self.ttl = ttl_seconds
        self.max_flows = max_flows
        self._cache: OrderedDict[str, Tuple[List[RWKVState], float]] = OrderedDict()

    def _make_key(self, src: str, dst: str, sport: int, dport: int, proto: int) -> str:
        # Normalise direction so A→B and B→A share a key
        a = (src, sport)
        b = (dst, dport)
        if a > b:
            a, b = b, a
        return f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"

    def get(
        self, src: str, dst: str, sport: int, dport: int, proto: int
    ) -> Optional[List[RWKVState]]:
        key = self._make_key(src, dst, sport, dport, proto)
        entry = self._cache.get(key)
        if entry is None:
            return None
        states, ts = entry
        if time.time() - ts > self.ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return states

    def put(
        self,
        src: str, dst: str, sport: int, dport: int, proto: int,
        states: List[RWKVState],
    ) -> None:
        key = self._make_key(src, dst, sport, dport, proto)
        self._cache[key] = (states, time.time())
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_flows:
            self._cache.popitem(last=False)   # evict oldest

    def evict_expired(self) -> int:
        """Evict TTL-expired entries; returns count removed."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self.ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._cache)


# ── Streaming pipeline (CSV/offline simulation) ───────────────────────────────

class StreamingPipeline:
    """
    Simulates streaming inference on a DataFrame of flows.

    For each flow row, it:
      1. Tokenises the row
      2. Runs token-by-token through the PLM using the flow's cached state
      3. Accumulates NLL → per-flow perplexity
      4. Alerts if score exceeds threshold
    """

    def __init__(
        self,
        model: PLM,
        tokenizer: FlowTokenizer,
        scorer: AnomalyScorer,
        flow_cache: FlowStateCache,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.flow_cache = flow_cache
        self.device = device

    @torch.no_grad()
    def process_flow(
        self,
        row: "pd.Series",
        src_col: str = "originh",
        dst_col: str = "responh",
        sport_col: str = "originp",
        dport_col: str = "responp",
    ) -> Dict:
        """
        Process one flow row.  Returns alert dict with score and decision.
        """
        src   = str(row.get(src_col, "0.0.0.0"))
        dst   = str(row.get(dst_col, "0.0.0.0"))
        sport = int(row.get(sport_col, 0))
        dport = int(row.get(dport_col, 0))
        proto = 6   # TCP by default (could be derived from column if available)

        tokens = self.tokenizer.encode_row(row)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        # Per-flow perplexity scoring
        ppl = self.model.flow_perplexity(token_tensor).item()

        is_alert = self.scorer.threshold is not None and ppl >= self.scorer.threshold
        return {
            "flow_key": f"{src}:{sport}-{dst}:{dport}",
            "perplexity": ppl,
            "alert": is_alert,
            "threshold": self.scorer.threshold,
        }

    def run_offline(
        self,
        df: "pd.DataFrame",
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Score all flows in *df* in batches.  Returns score array (N,).
        """
        import pandas as pd
        all_scores = []
        n = len(df)
        for start in range(0, n, batch_size):
            chunk = df.iloc[start : start + batch_size]
            token_ids = self.tokenizer.encode_dataframe_fast(chunk)
            t = torch.from_numpy(token_ids).long()
            scores = self.scorer.score_batch(t)
            all_scores.append(scores)
            if (start // batch_size) % 10 == 0:
                logger.info("Scored %d / %d flows …", min(start + batch_size, n), n)

        return np.concatenate(all_scores)
