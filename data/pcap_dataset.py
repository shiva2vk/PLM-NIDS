"""
Variable-length dataset for PCAP-derived flow sequences.

Key difference from dataset.py (FlowMeter mode):
  - Sequences have DIFFERENT lengths per flow (10–1000+ tokens)
  - Requires padding + a collate function for batching
  - This is where RWKV/Mamba's O(T) advantage over O(T²) Transformers matters
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.tokenizer import PAD_ID


class PcapLMDataset(Dataset):
    """
    Causal LM dataset from PCAP-derived variable-length sequences.

    Each item: (input_ids[:-1], target_ids[1:]) — both variable length.
    The collate_fn pads to the longest sequence in the batch.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_len: int = 1024,
        benign_only: bool = False,
    ) -> None:
        if benign_only:
            pairs = [(s, l) for s, l in zip(sequences, labels) if l == 0]
            sequences, labels = zip(*pairs) if pairs else ([], [])

        # Truncate long sequences
        self.sequences = [
            torch.from_numpy(s[:max_len]).long() for s in sequences
        ]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]   # input, target


class PcapClassifierDataset(Dataset):
    """Supervised dataset: full sequence → label."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_len: int = 1024,
    ) -> None:
        self.sequences = [
            torch.from_numpy(s[:max_len]).long() for s in sequences
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def pcap_collate_lm(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads variable-length LM batches.
    Returns (input_ids, target_ids) both (B, T_max) with PAD_ID filling.
    """
    inputs, targets = zip(*batch)
    inputs_padded  = pad_sequence(inputs,  batch_first=True, padding_value=PAD_ID)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_ID)
    return inputs_padded, targets_padded


def pcap_collate_cls(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pads variable-length classifier batches."""
    seqs, labels = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    labels_t    = torch.stack(labels)
    return seqs_padded, labels_t


def sequence_length_stats(sequences: List[np.ndarray]) -> dict:
    """Print useful statistics about sequence lengths — key for paper's Table 1."""
    lengths = np.array([len(s) for s in sequences])
    return {
        "min":    int(lengths.min()),
        "max":    int(lengths.max()),
        "mean":   float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95":    float(np.percentile(lengths, 95)),
        "total_tokens": int(lengths.sum()),
    }
