"""
    This is another of Nikola Tesla's inventions, a phase-regulated conductance gate.
    This variation is fairly untested, however the concept is to control resonant output based on field alignment,
    damping (τ), and conductance ceiling (χ). All gating is multiplicative — no division used.

    The conductance gate is designed to modulate the output of a resonant system based on the alignment of input signals,
    damping effects, and a soft limit on conductance. It uses a phase gate to control resonance and applies a permission
    gate to allow or restrict the output based on the alignment of the input signals.

    Conceptually and structurally, this is similar to Tesla's original work on resonant circuits and phase alignment -
    which are still fundamental to many modern electrical systems and often used in devices like radio transmitters and receivers,
    albeit in very different forms.

"""

import torch
import torch.nn as nn
from ..modulation.phase_gate import ResonantPhaseGate  # Assuming phase_gate.py is in the same directory


class ConductanceGate(nn.Module):
    """
    A phase-regulated gate that controls resonant output based on field alignment,
    damping (τ), and conductance ceiling (χ). All gating is multiplicative — no division used.
    """
    def __init__(self, input_dim, tau=0.25, chi_limit=0.29514, use_context=True):
        super().__init__()
        self.tau = tau
        self.chi_limit = chi_limit
        self.use_context = use_context

        # Field preparation
        self.norm = nn.LayerNorm(input_dim)
        self.resonance_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            ResonantPhaseGate(mode='tau', tau=tau),  # Using the phase gate for resonance
            nn.Linear(input_dim, input_dim)
        )

        # Collapse permission gate (sigmoid-based)
        self.permission_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        # Conductance soft limit (modulated but clamped later)
        self.conductance_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1),
            nn.Tanh(),
            nn.Sigmoid()  # final range [0, 1]
        )

    def forward(self, x, context=None):
        """
        x: [B, T, D] - primary signal
        context: [B, T, D] or None - optional resonance guide vector
        """
        x_norm = self.norm(x)

        # Optional contextual alignment (dot product, no normalization, no division)
        if self.use_context and context is not None:
            raw_alignment = torch.sum(x * context, dim=-1, keepdim=True)  # [B, T, 1]
            safe_alignment = torch.sigmoid(raw_alignment - self.chi_limit)  # gate opens near χ
        else:
            safe_alignment = torch.ones_like(x[..., :1])  # default to open

        # Damping based on τ (entropy-based resistance)
        damping = 1.0 - torch.exp(-self.tau * x.abs())
        resonance = self.resonance_proj(x_norm) * damping

        # Field-limited output signal
        conductance = self.conductance_gate(resonance)  # [B, T, 1]
        output = resonance * conductance.clamp(max=self.chi_limit)

        # Collapse permission is gated alignment
        permission = self.permission_gate(resonance) * safe_alignment  # [B, T, 1]

        return output, permission
