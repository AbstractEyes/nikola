import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.v2.modulators import PhaseModulator  # Now corrected name

class ResonantIgnitionLayer(nn.Module):
    """
    Refactored ResonantIgnitionLayer
    Computes symbolic ignition potential — the readiness of a field to express energy —
    based on alignment to intent and internal expression freedom.

    No suppression, no decay — only modulation and phase-based permission.
    """
    def __init__(self, dim, ignition_threshold=0.27, sharpness=25.0, phase_mode='sin'):
        super().__init__()
        self.dim = dim
        self.ignition_threshold = ignition_threshold
        self.sharpness = sharpness

        self.intent_proj = nn.Linear(dim, dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.intent_norm = nn.LayerNorm(dim)

        self.modulator = PhaseModulator(mode=phase_mode)

        self.energy_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor, intent: torch.Tensor):
        """
        latent: [B, T, D] - modulated symbolic field
        intent: [B, D]    - current symbolic directive or global vector
        """
        B, T, D = latent.shape

        latent = self.latent_norm(latent)             # [B, T, D]
        intent = self.intent_norm(intent)             # [B, D]
        intent_proj = self.intent_proj(intent).unsqueeze(1).expand(-1, T, -1)

        # Phase-aligned symbolic activation (dot-product, phase-compatible)
        alignment = torch.sum(latent * intent_proj, dim=-1, keepdim=True)  # [B, T, 1]

        # Modulate activation curve
        modulated_alignment = self.modulator(alignment)  # [B, T, 1]

        # Sharp sigmoid gating based on threshold
        pressure = torch.sigmoid((modulated_alignment - self.ignition_threshold) * self.sharpness)

        # Project field openness
        conductance = self.energy_projector(latent)  # [B, T, 1]

        # Final ignition = modulated pressure scaled by field expressiveness
        ignition = pressure * conductance
        return ignition
