"""
Data loading, splitting, and tokenisation for the HIKARI-2021 FlowMeter CSV.

Label schema
------------
  Label == 0  →  benign  (Benign + Background categories)
  Label == 1  →  attack  (Probing, Bruteforce, Bruteforce-XML, XMRIGCC CryptoMiner)

Phase-1 pre-training uses *benign_only* subset.
Phase-2 fine-tuning uses the full dataset with labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.tokenizer import FlowTokenizer

logger = logging.getLogger(__name__)

_META_COLS = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh",
              "traffic_category", "Label", "originp", "responp"]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    logger.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop meta columns; replace inf / NaN with 0."""
    drop_cols = [c for c in _META_COLS if c in df.columns and c not in ("traffic_category", "Label")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
    exclude_background_from_pretrain: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict with keys:
      'train_all', 'train_benign', 'val', 'test'

    train_benign  →  Phase-1 causal-LM pre-training (benign flows only)
    train_all     →  Phase-2 supervised fine-tuning (all labelled flows)
    val / test    →  evaluation (stratified by Label)
    """
    label = df["Label"]

    # Stratified train / (val+test) split
    train_idx, valtest_idx = train_test_split(
        df.index, test_size=test_size + val_size, stratify=label, random_state=seed
    )
    val_frac_of_remaining = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=1 - val_frac_of_remaining,
        stratify=label[valtest_idx],
        random_state=seed,
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    benign_mask = train_df["Label"] == 0
    if exclude_background_from_pretrain and "traffic_category" in train_df.columns:
        benign_mask = benign_mask & (train_df["traffic_category"] != "Background")

    train_benign = train_df[benign_mask].reset_index(drop=True)

    logger.info(
        "Split → train_all=%d | train_benign=%d | val=%d | test=%d",
        len(train_df), len(train_benign), len(val_df), len(test_df),
    )
    logger.info(
        "Attack ratios → train=%.2f%% | val=%.2f%% | test=%.2f%%",
        100 * train_df["Label"].mean(),
        100 * val_df["Label"].mean(),
        100 * test_df["Label"].mean(),
    )

    return {
        "train_all": train_df,
        "train_benign": train_benign,
        "val": val_df,
        "test": test_df,
    }


def build_tokenizer(
    cfg_tok: dict,
    train_df: pd.DataFrame,
) -> FlowTokenizer:
    """Fit tokenizer on training data; save to disk."""
    tok = FlowTokenizer(
        continuous_features=cfg_tok["continuous_features"],
        flag_features=cfg_tok["flag_features"],
        n_bins=cfg_tok["n_bins"],
        log_transform=cfg_tok.get("log_transform", True),
    )
    tok.fit(train_df)
    tok.save(cfg_tok["save_path"])
    return tok


def tokenise_split(
    split_df: pd.DataFrame,
    tokenizer: FlowTokenizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        token_ids : int32 array (N, seq_len)
        labels    : int32 array (N,)
    """
    logger.info("Tokenising %d flows …", len(split_df))
    token_ids = tokenizer.encode_dataframe_fast(split_df)
    labels = split_df["Label"].values.astype(np.int32)
    return token_ids, labels


def prepare_all(cfg: dict) -> Dict:
    """
    Full preprocessing pipeline.  Returns a dict with:
      'tokenizer', 'splits' (dict of split name → (token_ids, labels))
    """
    csv_path = cfg["data"]["csv_path"]
    seed = cfg["data"]["random_seed"]
    exclude_bg = cfg["data"].get("exclude_background_from_pretrain", True)

    raw = load_raw(csv_path)
    df = clean(raw)

    splits = split_dataset(
        df,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=seed,
        exclude_background_from_pretrain=exclude_bg,
    )

    # Fit tokenizer on full training set (not just benign) for better bin coverage
    tokenizer = build_tokenizer(cfg["tokenizer"], splits["train_all"])

    tokenised: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, split_df in splits.items():
        tokenised[name] = tokenise_split(split_df, tokenizer)

    return {"tokenizer": tokenizer, "splits": tokenised, "raw_splits": splits}
