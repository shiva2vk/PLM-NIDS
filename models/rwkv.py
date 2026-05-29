"""
RWKV-4 implementation in pure PyTorch.

Supports both:
  * Parallel / teacher-forced training  (forward pass over full sequence)
  * Streaming / recurrent inference     (step() with per-flow hidden state)

Architecture per block
----------------------
  LayerNorm → TimeMix (WKV attention) → residual
  LayerNorm → ChannelMix (gated FFN)  → residual

WKV (Weighted Key-Value) recurrence
------------------------------------
  Given time-decay w (per-channel, learned) and bonus u (current-token):

    wkv_t = ( exp(u+k_t)*v_t  +  Σ_{i<t} exp(-(t-1-i)*w + k_i)*v_i )
            ─────────────────────────────────────────────────────────
            ( exp(u+k_t)       +  Σ_{i<t} exp(-(t-1-i)*w + k_i)     )

  Recurrent form (numerically stable via log-space):
    log_a_t = logaddexp(log_w + log_a_{t-1}, k_t + log_v_t_abs)
    log_b_t = logaddexp(log_w + log_b_{t-1}, k_t)
    wkv_t   = a_t / b_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RWKVState:
    """Per-layer recurrent state for streaming inference."""
    # TimeMix state: (shift_x, wkv_num, wkv_den)
    tm_x_prev: torch.Tensor      # (d_model,)
    wkv_num: torch.Tensor        # (d_model,)  numerator accumulator
    wkv_den: torch.Tensor        # (d_model,)  denominator accumulator
    # ChannelMix state: shift_x
    cm_x_prev: torch.Tensor      # (d_model,)


class TimeMix(nn.Module):
    """RWKV-4 Time-Mixing block."""

    def __init__(self, d_model: int, layer_id: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Time-shift mix ratios (learnable per-channel)
        ratio_0 = 1.0 - layer_id / max(n_layers - 1, 1)
        ratio_1 = 1.0 - (layer_id + 0.5) / n_layers
        decay_speed = torch.tensor(
            [-(5 + 8 * (i / max(d_model - 1, 1)) ** 0.7) for i in range(d_model)]
        )

        self.time_decay = nn.Parameter(decay_speed)            # w (log-domain)
        self.time_first = nn.Parameter(torch.full((d_model,), math.log(0.3)))  # u

        self.time_mix_r = nn.Parameter(torch.full((1, 1, d_model), ratio_0))
        self.time_mix_k = nn.Parameter(torch.full((1, 1, d_model), ratio_1))
        self.time_mix_v = nn.Parameter(torch.full((1, 1, d_model), ratio_1))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key        = nn.Linear(d_model, d_model, bias=False)
        self.value      = nn.Linear(d_model, d_model, bias=False)
        self.output     = nn.Linear(d_model, d_model, bias=False)

        # Small init for output projection (stability)
        nn.init.zeros_(self.output.weight)

    def _wkv_parallel(
        self, w: torch.Tensor, u: torch.Tensor,
        k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel WKV scan for training (B, T, d_model).
        We compute it step-by-step (T-loop) for correctness;
        for long sequences this is the bottleneck — acceptable for T≈20.
        """
        B, T, C = k.shape
        # log-space decay per step
        log_w = -torch.exp(w)          # (C,) negative → decay
        log_u = u                      # (C,) bonus for current token

        num = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        den = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        out = torch.zeros(B, T, C, device=k.device, dtype=k.dtype)

        EPS = 1e-9
        for t in range(T):
            kt = k[:, t, :]   # (B, C)
            vt = v[:, t, :]   # (B, C)
            # Current-token contribution (with bonus u)
            e_uk = torch.exp(torch.clamp(log_u + kt, max=30))
            # Past contribution
            e_prev = torch.exp(torch.clamp(log_w, max=0))
            num = e_prev * num + e_uk * vt
            den = e_prev * den + e_uk
            out[:, t, :] = num / (den + EPS)

        return out

    def forward(
        self,
        x: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x      : (B, T, d_model)
            x_prev : (B, 1, d_model)  last token from previous call (for shift)
                     If None, uses zeros (first call in a sequence).
        Returns:
            out    : (B, T, d_model)
            x_last : (B, 1, d_model)  to be passed on next call
        """
        B, T, C = x.shape

        if x_prev is None:
            x_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)

        # Shift: x_{t-1} for each position
        x_shifted = torch.cat([x_prev, x[:, :-1, :]], dim=1)   # (B, T, C)

        # Interpolate current vs previous
        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)

        w = self.time_decay   # (C,)
        u = self.time_first   # (C,)
        wkv = self._wkv_parallel(w, u, k, v)

        out = self.output(r * wkv)
        return out, x[:, -1:, :]

    def step(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Single-token streaming step.

        Args:
            x     : (B, d_model)
            state : (x_prev, wkv_num, wkv_den)  each (B, d_model)
        Returns:
            out   : (B, d_model)
            state : updated state tuple
        """
        x_prev, num, den = state
        C = x.shape[-1]

        xr = x * self.time_mix_r.squeeze() + x_prev * (1 - self.time_mix_r.squeeze())
        xk = x * self.time_mix_k.squeeze() + x_prev * (1 - self.time_mix_k.squeeze())
        xv = x * self.time_mix_v.squeeze() + x_prev * (1 - self.time_mix_v.squeeze())

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)

        log_w = -torch.exp(self.time_decay)
        log_u = self.time_first

        e_uk  = torch.exp(torch.clamp(log_u + k, max=30))
        e_prev = torch.exp(torch.clamp(log_w, max=0))

        new_num = e_prev * num + e_uk * v
        new_den = e_prev * den + e_uk

        wkv = new_num / (new_den + 1e-9)
        out = self.output(r * wkv)

        return out, (x, new_num, new_den)


class ChannelMix(nn.Module):
    """RWKV-4 Channel-Mixing block (gated FFN)."""

    def __init__(self, d_model: int, layer_id: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        ratio = 1.0 - layer_id / max(n_layers - 1, 1)

        self.time_mix_r = nn.Parameter(torch.full((1, 1, d_model), ratio))
        self.time_mix_k = nn.Parameter(torch.full((1, 1, d_model), ratio))

        d_ffn = int(d_model * 3.5)  # RWKV-4 uses ~3.5× expansion
        self.key        = nn.Linear(d_model, d_ffn, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value      = nn.Linear(d_ffn, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        if x_prev is None:
            x_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)

        x_shifted = torch.cat([x_prev, x[:, :-1, :]], dim=1)

        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)

        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk)))   # squared-ReLU
        out = r * self.value(k)
        return out, x[:, -1:, :]

    def step(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xr = x * self.time_mix_r.squeeze() + x_prev * (1 - self.time_mix_r.squeeze())
        xk = x * self.time_mix_k.squeeze() + x_prev * (1 - self.time_mix_k.squeeze())

        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk)))
        out = r * self.value(k)
        return out, x


class RWKVBlock(nn.Module):
    """One RWKV-4 block: LayerNorm → TimeMix → LN → ChannelMix."""

    def __init__(self, d_model: int, layer_id: int, n_layers: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix = TimeMix(d_model, layer_id, n_layers)
        self.chan_mix = ChannelMix(d_model, layer_id, n_layers)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        tm_prev: Optional[torch.Tensor] = None,
        cm_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x         : (B, T, d_model)
            tm_x_last : last token from TimeMix (for continuity across calls)
            cm_x_last : last token from ChannelMix
        """
        tm_out, tm_last = self.time_mix(self.ln1(x), tm_prev)
        x = x + self.drop(tm_out)

        cm_out, cm_last = self.chan_mix(self.ln2(x), cm_prev)
        x = x + self.drop(cm_out)

        return x, tm_last, cm_last

    def step(
        self,
        x: torch.Tensor,
        state: RWKVState,
    ) -> Tuple[torch.Tensor, RWKVState]:
        """Single-token step for streaming inference."""
        # TimeMix
        ln1_x = F.layer_norm(x, (x.shape[-1],),
                              self.ln1.weight, self.ln1.bias, self.ln1.eps)
        tm_out, (new_tm_x, new_num, new_den) = self.time_mix.step(
            ln1_x, (state.tm_x_prev, state.wkv_num, state.wkv_den)
        )
        x = x + tm_out

        # ChannelMix
        ln2_x = F.layer_norm(x, (x.shape[-1],),
                              self.ln2.weight, self.ln2.bias, self.ln2.eps)
        cm_out, new_cm_x = self.chan_mix.step(ln2_x, state.cm_x_prev)
        x = x + cm_out

        new_state = RWKVState(
            tm_x_prev=new_tm_x,
            wkv_num=new_num,
            wkv_den=new_den,
            cm_x_prev=new_cm_x,
        )
        return x, new_state


class RWKVBackbone(nn.Module):
    """Stack of RWKVBlocks."""

    def __init__(self, d_model: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [RWKVBlock(d_model, i, n_layers, dropout) for i in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.n_layers = n_layers
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        past_states: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Args:
            x           : (B, T, d_model)
            past_states : list of (tm_prev, cm_prev) per layer; None = fresh
        Returns:
            x       : (B, T, d_model) after all blocks + final LayerNorm
            states  : list of (tm_last, cm_last) per layer  (for streaming)
        """
        new_states = []
        for i, block in enumerate(self.blocks):
            tm_prev = past_states[i][0] if past_states else None
            cm_prev = past_states[i][1] if past_states else None
            x, tm_last, cm_last = block(x, tm_prev, cm_prev)
            new_states.append((tm_last, cm_last))

        x = self.ln_out(x)
        return x, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List[RWKVState]:
        """Create zero-initialised per-layer states for streaming."""
        states = []
        for _ in range(self.n_layers):
            states.append(RWKVState(
                tm_x_prev=torch.zeros(batch_size, self.d_model, device=device),
                wkv_num=torch.zeros(batch_size, self.d_model, device=device),
                wkv_den=torch.zeros(batch_size, self.d_model, device=device),
                cm_x_prev=torch.zeros(batch_size, self.d_model, device=device),
            ))
        return states

    def step(
        self,
        x: torch.Tensor,
        states: List[RWKVState],
    ) -> Tuple[torch.Tensor, List[RWKVState]]:
        """Single-token streaming step through all layers."""
        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block.step(x, state)
            new_states.append(new_state)
        x = F.layer_norm(x, (x.shape[-1],),
                         self.ln_out.weight, self.ln_out.bias, self.ln_out.eps)
        return x, new_states
