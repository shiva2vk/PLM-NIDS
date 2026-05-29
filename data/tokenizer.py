"""
DPI-free flow tokenizer.

Converts FlowMeter CSV rows into discrete token-ID sequences that the PLM
can process.  Each continuous feature is quantised into `n_bins` buckets
(fitted on training data); each flag feature becomes a binary present/absent
token.  The resulting vocabulary is closed and fixed after fitting.

Sequence layout per flow:
  [BOS, feat0_bin, feat1_bin, ..., flag0, flag1, ..., EOS]
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Special token IDs ────────────────────────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
_N_SPECIAL = 4  # first user-token starts at 4


class FlowTokenizer:
    """
    Fits bin edges on training data and converts flow rows to token sequences.

    Parameters
    ----------
    continuous_features : list[str]
        Column names for continuously-valued features (packet sizes, IAT, …).
    flag_features : list[str]
        Column names for integer count features treated as binary flags.
    n_bins : int
        Number of quantisation bins per continuous feature.
    log_transform : bool
        Apply log1p before fitting / transforming continuous features.
        Recommended for heavy-tailed network statistics.
    """

    def __init__(
        self,
        continuous_features: List[str],
        flag_features: List[str],
        n_bins: int = 32,
        log_transform: bool = True,
    ) -> None:
        self.continuous_features = continuous_features
        self.flag_features = flag_features
        self.n_bins = n_bins
        self.log_transform = log_transform

        self._bin_edges: Dict[str, np.ndarray] = {}
        self._feat_offsets: Dict[str, int] = {}
        self._flag_offsets: Dict[str, Tuple[int, int]] = {}  # (absent_id, present_id)
        self._vocab_size: int = 0
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FlowTokenizer":
        """Fit bin edges on *df* (should be training split only)."""
        offset = _N_SPECIAL

        for col in self.continuous_features:
            if col not in df.columns:
                logger.warning("Column %s not found; skipping.", col)
                continue
            values = df[col].fillna(0).values.astype(np.float64)
            if self.log_transform:
                values = np.log1p(np.clip(values, 0, None))
            # Percentile-based edges so bins are roughly equal-density
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(values, percentiles))
            self._bin_edges[col] = edges
            self._feat_offsets[col] = offset
            offset += self.n_bins  # each bin maps to one token ID

        for col in self.flag_features:
            if col not in df.columns:
                logger.warning("Flag column %s not found; skipping.", col)
                continue
            self._flag_offsets[col] = (offset, offset + 1)  # 0=absent, 1=present
            offset += 2

        self._vocab_size = offset
        self._fitted = True
        logger.info(
            "Tokenizer fitted: %d continuous × %d bins + %d flags → vocab=%d",
            len(self._bin_edges),
            self.n_bins,
            len(self._flag_offsets),
            self._vocab_size,
        )
        return self

    def encode_row(self, row: pd.Series) -> List[int]:
        """Encode a single flow row as a list of token IDs (including BOS/EOS)."""
        assert self._fitted, "Call fit() before encode_row()."
        tokens: List[int] = [BOS_ID]

        for col, offset in self._feat_offsets.items():
            val = float(row.get(col, 0) or 0)
            if self.log_transform:
                val = np.log1p(max(val, 0))
            edges = self._bin_edges[col]
            bin_idx = int(np.searchsorted(edges, val, side="right")) - 1
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))
            tokens.append(offset + bin_idx)

        for col, (absent_id, present_id) in self._flag_offsets.items():
            count = int(row.get(col, 0) or 0)
            tokens.append(present_id if count > 0 else absent_id)

        tokens.append(EOS_ID)
        return tokens

    def encode_dataframe(self, df: pd.DataFrame) -> List[List[int]]:
        """Encode all rows; returns list of token-ID lists."""
        assert self._fitted
        return [self.encode_row(row) for _, row in df.iterrows()]

    def encode_dataframe_fast(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorised encoding: returns int32 array of shape (N, seq_len).
        All sequences have the same length (BOS + n_feats + n_flags + EOS).
        """
        assert self._fitted
        n = len(df)
        seq_len = 1 + len(self._feat_offsets) + len(self._flag_offsets) + 1
        out = np.zeros((n, seq_len), dtype=np.int32)
        out[:, 0] = BOS_ID

        col_idx = 1
        for col, offset in self._feat_offsets.items():
            vals = df[col].fillna(0).values.astype(np.float64)
            if self.log_transform:
                vals = np.log1p(np.clip(vals, 0, None))
            edges = self._bin_edges[col]
            bin_ids = np.searchsorted(edges, vals, side="right") - 1
            bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
            out[:, col_idx] = (offset + bin_ids).astype(np.int32)
            col_idx += 1

        for col, (absent_id, present_id) in self._flag_offsets.items():
            counts = df[col].fillna(0).values.astype(np.int32)
            out[:, col_idx] = np.where(counts > 0, present_id, absent_id)
            col_idx += 1

        out[:, col_idx] = EOS_ID
        return out

    @property
    def vocab_size(self) -> int:
        assert self._fitted, "Call fit() first."
        return self._vocab_size

    @property
    def seq_len(self) -> int:
        """Fixed sequence length (BOS + features + flags + EOS)."""
        assert self._fitted
        return 1 + len(self._feat_offsets) + len(self._flag_offsets) + 1

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=4)
        logger.info("Tokenizer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FlowTokenizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls.__new__(cls)
        tok.__dict__.update(state)
        return tok

    def token_name(self, token_id: int) -> str:
        """Human-readable name for a token ID (for debugging)."""
        if token_id == PAD_ID:
            return "<PAD>"
        if token_id == BOS_ID:
            return "<BOS>"
        if token_id == EOS_ID:
            return "<EOS>"
        if token_id == UNK_ID:
            return "<UNK>"
        for col, offset in self._feat_offsets.items():
            if offset <= token_id < offset + self.n_bins:
                return f"{col}_BIN_{token_id - offset}"
        for col, (absent_id, present_id) in self._flag_offsets.items():
            if token_id == absent_id:
                return f"{col}_ABSENT"
            if token_id == present_id:
                return f"{col}_PRESENT"
        return f"<UNK:{token_id}>"
