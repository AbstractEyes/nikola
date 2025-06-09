import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonantIgnitionLayer(nn.Module):
    def __init__(self, dim, collapse_threshold=0.29514, sharpness=25.0):
        super().__init__()
        self.dim = dim
        self.collapse_threshold = collapse_threshold
        self.sharpness = sharpness

        self.intent_proj = nn.Linear(dim, dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.intent_norm = nn.LayerNorm(dim)

        self.pressure_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor, intent: torch.Tensor):
        """
        latent: [B, T, D]
        intent: [B, D]  — symbolic or learned purpose vector
        """
        B, T, D = latent.shape

        # Normalize
        latent = self.latent_norm(latent)           # [B, T, D]
        intent = self.intent_norm(intent)           # [B, D]
        intent = self.intent_proj(intent).unsqueeze(1).expand(-1, T, -1)  # [B, T, D]

        # Alignment: dot product between purpose and token latent
        alignment = torch.sum(latent * intent, dim=-1, keepdim=True)  # [B, T, 1]

        # Gate pressure from alignment
        pressure = torch.sigmoid((alignment - self.collapse_threshold) * self.sharpness)  # [B, T, 1]

        # Conductance gate (how much purpose energy is tolerated)
        conductance = self.pressure_gate(latent)  # [B, T, 1]

        # Final ignition potential
        ignition = pressure * conductance  # [B, T, 1]

        return ignition  # To be used as modulation gate or collapse trigger
