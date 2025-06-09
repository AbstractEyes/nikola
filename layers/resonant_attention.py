import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonantMultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        collapse_threshold: float = 0.29514,
        sharpness: float = 15.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.inner_dim = self.head_dim * heads

        self.collapse_threshold = collapse_threshold
        self.sharpness = sharpness

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)

        # Collapse gating (1D projection per token)
        self.gate_q = nn.Linear(self.head_dim, 1, bias=False)
        self.gate_k = nn.Linear(self.head_dim, 1, bias=False)

        # Output projection back to input dim
        self.out_proj = nn.Linear(self.inner_dim, dim)

        # Input normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resonant attention mechanism.
        Collapse only permitted where phase alignment and conductance permit it.
        Args:
            x: Input tensor [B, T, D]
        Returns:
            Tensor of shape [B, T, D] with phase-permissioned modulation applied.
        """
        B, T, D = x.shape
        x_norm = self.norm(x)

        # Linear projections: [B, T, H * D_head] → [B, H, T, D_head]
        q = self.q_proj(x_norm).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Normalize Q, K for phase alignment
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Alignment score: [B, H, T, T]
        align = torch.matmul(q, k.transpose(-1, -2))

        # Resonance gate: permission to collapse [B, H, T, T]
        resonance = torch.sigmoid((align - self.collapse_threshold) * self.sharpness)

        # Per-token conductance gating
        gate_q = self.gate_q(q).squeeze(-1)  # [B, H, T]
        gate_k = self.gate_k(k).squeeze(-1)  # [B, H, T]

        conductance = gate_q.unsqueeze(-1) * gate_k.unsqueeze(-2)  # [B, H, T, T]

        # Final modulation matrix
        modulation = resonance * conductance  # [B, H, T, T]

        # Collapse output: [B, H, T, D_head]
        collapse = torch.matmul(modulation, v)

        # Reshape back to [B, T, D]
        collapse = collapse.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.out_proj(collapse)

        return out
