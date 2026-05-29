from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import torch
from torch import nn


def _rank0() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def flatten_selected_world_tokens(
    world_tokens: Dict[str, torch.Tensor],
    use_scene: bool = True,
    use_agent: bool = True,
    use_goal: bool = True,
) -> Optional[torch.Tensor]:
    selected: List[torch.Tensor] = []
    if use_scene and world_tokens.get("scene") is not None:
        selected.append(world_tokens["scene"])
    if use_agent and world_tokens.get("agent") is not None:
        selected.append(world_tokens["agent"])
    if use_goal and world_tokens.get("goal") is not None:
        selected.append(world_tokens["goal"])
    if not selected:
        return None
    return torch.cat(selected, dim=1)


def pool_world_tokens(world_memory: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    if pooling == "none":
        return world_memory
    if pooling == "cls":
        return world_memory[:, :1, :]
    return world_memory.mean(dim=1, keepdim=True)


class WorldTokenProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class WorldDenoiseCrossAttention(nn.Module):
    def __init__(self, denoise_dim: int, num_heads: int = 4, dropout: float = 0.0, gate_init: float = -2.0, residual_scale: float = 1.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(denoise_dim)
        self.norm_kv = nn.LayerNorm(denoise_dim)
        self.attn = nn.MultiheadAttention(denoise_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.residual_scale = residual_scale

    def forward(self, denoise_hidden: torch.Tensor, world_memory: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.norm_q(denoise_hidden)
        kv = self.norm_kv(world_memory)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        gate = torch.sigmoid(self.gate)
        updated = denoise_hidden + gate * self.residual_scale * out
        return updated, {"gate_mean": gate.detach(), "gate_min": gate.detach(), "gate_max": gate.detach()}


class WorldDenoiseFiLM(nn.Module):
    def __init__(self, denoise_dim: int, cond_dim: int, gate_init: float = -2.0, residual_scale: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(denoise_dim)
        self.mlp = nn.Sequential(nn.Linear(cond_dim, denoise_dim * 3), nn.GELU(), nn.Linear(denoise_dim * 3, denoise_dim * 3))
        self.gate_bias = nn.Parameter(torch.tensor(gate_init))
        self.residual_scale = residual_scale

    def forward(self, denoise_hidden: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = self.mlp(context.squeeze(1))
        gamma, beta, gate = torch.chunk(params, 3, dim=-1)
        h_mod = self.norm(denoise_hidden) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        g = torch.sigmoid(gate.unsqueeze(1) + self.gate_bias)
        updated = denoise_hidden + g * self.residual_scale * h_mod
        return updated, {"gate_mean": g.mean().detach(), "gate_min": g.min().detach(), "gate_max": g.max().detach()}


class WorldDenoiseModulator(nn.Module):
    def __init__(self, denoise_dim: int, world_dim: int, influence_type: str = "cross_attn", num_heads: int = 4, dropout: float = 0.0, pooling: str = "none", residual_scale: float = 1.0, gate_init: float = -2.0, use_cross_attn: bool = False, use_film: bool = False):
        super().__init__()
        self.influence_type = influence_type
        self.pooling = pooling
        self.world_proj = WorldTokenProjector(world_dim, denoise_dim)
        self.cross_attn = WorldDenoiseCrossAttention(denoise_dim, num_heads, dropout, gate_init, residual_scale)
        self.film = WorldDenoiseFiLM(denoise_dim, denoise_dim, gate_init, residual_scale)
        self.use_cross_attn = use_cross_attn
        self.use_film = use_film
        self._printed = False

    def forward(self, denoise_hidden: torch.Tensor, world_tokens: Dict[str, torch.Tensor], use_scene=True, use_agent=True, use_goal=True, step_idx: Optional[int]=None, total_steps: Optional[int]=None, apply_steps: str="all", custom_steps: Optional[List[int]]=None, debug: bool=False):
        wm = flatten_selected_world_tokens(world_tokens, use_scene, use_agent, use_goal)
        if wm is None:
            return denoise_hidden, {}
        wm = self.world_proj(wm)
        context = pool_world_tokens(wm, pooling=self.pooling)

        use_ca = self.use_cross_attn or self.influence_type in ["cross_attn", "cross_attn_plus_film"]
        use_fm = self.use_film or self.influence_type in ["film", "adaln", "cross_attn_plus_film"]

        stats = {}
        h = denoise_hidden
        if use_ca:
            h, st = self.cross_attn(h, wm)
            stats.update({f"cross_{k}": v for k, v in st.items()})
        if use_fm:
            h, st = self.film(h, context)
            stats.update({f"film_{k}": v for k, v in st.items()})

        if debug and (not self._printed) and _rank0():
            print(f"[WorldDenoiseModulator] influence={self.influence_type} denoise_before={tuple(denoise_hidden.shape)} world_memory={tuple(wm.shape)} denoise_after={tuple(h.shape)}")
            self._printed = True
        return h, stats
