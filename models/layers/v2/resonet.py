import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.v2.modulators import PhaseModulator  # Now corrected name
from models.layers.v2.attention import ResonantMultiheadAttention  # Now corrected name

class ResoNet(nn.Module):
    """
    Nikola-aligned symbolic residual unit.
    Replaces Conv+ReLU+Skip with phase-aligned modulation + guided residual merge.
    Dropout is removed for symbolic field coherence.
    """
    def __init__(self, dim, use_attention=True, phase_mode='sin'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.phase = PhaseModulator(mode=phase_mode)

        self.transform = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim)
        )

        self.use_attention = use_attention
        if use_attention:
            self.attn = ResonantMultiheadAttention(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.phase(x)
        x = self.transform(x)

        if self.use_attention:
            x = self.attn(x)

        return residual + x