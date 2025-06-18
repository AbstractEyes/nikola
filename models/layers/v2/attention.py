import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonantMultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        collapse_threshold: float = 0.27,
        sharpness: float = 15.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.inner_dim = self.head_dim * heads

        self.collapse_threshold = collapse_threshold
        self.sharpness = sharpness

        # Q, K, V projections (renamed for clarity)
        self.rotor_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.rotor_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.carrier_v = nn.Linear(dim, self.inner_dim, bias=False)

        # Collapse gating (1D projection per token)
        self.gate_q = nn.Linear(self.head_dim, 1, bias=False)
        self.gate_k = nn.Linear(self.head_dim, 1, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.norm(x)

        # Linear projections and reshape
        q = self.rotor_q(x_norm).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.rotor_k(x_norm).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.carrier_v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Normalize for phase alignment
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        align = torch.matmul(q, k.transpose(-1, -2))  # [B, H, T, T]

        # Phase-bandpass modulation (no suppression)
        resonance = 0.5 + 0.5 * torch.tanh((align - self.collapse_threshold) * self.sharpness)

        # Soft conductance scaling with minimum floor
        gate_q = 0.25 + 0.75 * torch.sigmoid(self.gate_q(q)).squeeze(-1)
        gate_k = 0.25 + 0.75 * torch.sigmoid(self.gate_k(k)).squeeze(-1)
        conductance = gate_q.unsqueeze(-1) * gate_k.unsqueeze(-2)

        modulation = resonance * conductance
        fused = torch.matmul(modulation, v)

        fused = fused.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.out_proj(fused)
