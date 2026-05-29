"""
Protocol-Language Model (PLM) — full model combining:
  Embedding → RWKVBackbone → LM head (causal language modelling)
                           → Classifier head (optional supervised detection)

Training modes
--------------
  mode="lm"         : causal LM, returns per-token NLL / loss
  mode="cls"        : supervised classifier, returns logits + cross-entropy loss
  mode="lm+cls"     : joint training (Phase-2 fine-tuning)

Streaming inference
-------------------
  Call model.step(token_id, states) per token to get logits + updated state.
  Accumulate NLL → perplexity for anomaly scoring.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv import RWKVBackbone, RWKVState


class PLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_classes: int = 2,
        tie_embeddings: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.backbone = RWKVBackbone(d_model, n_layers, dropout)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Classifier head: mean-pool hidden states → label
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not self.embedding.weight:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T)
        targets: Optional[torch.Tensor] = None,  # (B, T) for LM loss
        labels: Optional[torch.Tensor] = None,   # (B,) for cls loss
        mode: str = "lm",                 # "lm" | "cls" | "lm+cls"
        past_states: Optional[List] = None,
        lm_weight: float = 1.0,
        cls_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that always contains 'loss' plus mode-specific keys.
        """
        x = self.embedding(input_ids)     # (B, T, d_model)
        hidden, new_states = self.backbone(x, past_states)

        out: Dict[str, torch.Tensor] = {"states": new_states}
        total_loss = torch.tensor(0.0, device=x.device)

        # ── LM branch ────────────────────────────────────────────────────────
        if mode in ("lm", "lm+cls"):
            logits = self.lm_head(hidden)          # (B, T, vocab)
            out["lm_logits"] = logits
            if targets is not None:
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=self.pad_id,
                )
                out["lm_loss"] = lm_loss
                total_loss = total_loss + lm_weight * lm_loss

        # ── Classifier branch ─────────────────────────────────────────────────
        if mode in ("cls", "lm+cls"):
            # Mean-pool over non-pad positions
            mask = (input_ids != self.pad_id).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            cls_logits = self.cls_head(pooled)    # (B, n_classes)
            out["cls_logits"] = cls_logits
            if labels is not None:
                cls_loss = F.cross_entropy(cls_logits, labels)
                out["cls_loss"] = cls_loss
                total_loss = total_loss + cls_weight * cls_loss

        out["loss"] = total_loss
        return out

    # ── Streaming step ────────────────────────────────────────────────────────

    def step(
        self,
        token_id: torch.Tensor,        # (B,) or scalar
        states: List[RWKVState],
    ) -> Tuple[torch.Tensor, List[RWKVState]]:
        """
        Process one token and return logits for next-token prediction.

        Returns:
            logits    : (B, vocab_size)
            new_states: updated per-layer states
        """
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        x = self.embedding(token_id)       # (B, d_model)
        hidden, new_states = self.backbone.step(x, states)
        logits = self.lm_head(hidden)      # (B, vocab_size)
        return logits, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List[RWKVState]:
        return self.backbone.init_states(batch_size, device)

    # ── Perplexity scoring ────────────────────────────────────────────────────

    @torch.no_grad()
    def flow_perplexity(
        self,
        token_ids: torch.Tensor,   # (B, T) — full sequence including BOS/EOS
        reduction: str = "mean",   # "mean" → scalar PPL per flow
    ) -> torch.Tensor:
        """
        Compute per-flow perplexity via streaming step-by-step.
        High PPL → anomalous.
        """
        B, T = token_ids.shape
        device = token_ids.device
        states = self.init_states(B, device)

        nll_sum = torch.zeros(B, device=device)
        nll_count = torch.zeros(B, device=device)

        for t in range(T - 1):
            inp = token_ids[:, t]
            tgt = token_ids[:, t + 1]

            logits, states = self.step(inp, states)
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather NLL for each target token
            nll = -log_probs[torch.arange(B, device=device), tgt]
            # Mask pad tokens in target
            pad_mask = (tgt != self.pad_id).float()
            nll_sum += nll * pad_mask
            nll_count += pad_mask

        avg_nll = nll_sum / nll_count.clamp(min=1)
        ppl = torch.exp(avg_nll)
        return ppl

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        for p in self.embedding.parameters():
            p.requires_grad = not freeze
