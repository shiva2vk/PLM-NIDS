"""
PyTorch Datasets for the PLM.

LMDataset        → causal language-model training (next-token prediction)
ClassifierDataset → supervised classification (returns full sequence + label)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """
    Returns (input_ids, target_ids) pairs for causal LM training.

    input_ids  = token_ids[:, :-1]   (all tokens except last)
    target_ids = token_ids[:, 1:]    (shifted right by 1)
    """

    def __init__(
        self,
        token_ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
        benign_only: bool = False,
    ) -> None:
        if benign_only and labels is not None:
            mask = labels == 0
            token_ids = token_ids[mask]
            if labels is not None:
                labels = labels[mask]

        self.token_ids = torch.from_numpy(token_ids).long()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.token_ids[idx]
        return seq[:-1], seq[1:]  # input, target


class ClassifierDataset(Dataset):
    """
    Returns (token_ids, label) for supervised classification.
    """

    def __init__(self, token_ids: np.ndarray, labels: np.ndarray) -> None:
        self.token_ids = torch.from_numpy(token_ids).long()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


def make_weighted_sampler(labels: np.ndarray, attack_weight: float = 10.0):
    """
    Returns a WeightedRandomSampler that upsamples attack flows.
    Helps with the ~14:1 class imbalance in HIKARI-2021.
    """
    from torch.utils.data import WeightedRandomSampler

    weights = np.where(labels == 1, attack_weight, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler
