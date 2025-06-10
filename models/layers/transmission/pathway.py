import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.modulation.attention import ResonantMultiheadAttention
from models.layers.modulation.ignition import ResonantIgnitionLayer


# this is a legacy artifact BottleneckResBlock to ensure the first setups conform to need.
# This will be rewritten soon and will become obsoleted once rewritten into resonant form.
class BottleneckResBlock(nn.Module):
    def __init__(self, dim, kernel=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.proj(x)


class PathwayCoil(nn.Module):
    """
    Full PathwayCoil reconstruction using MinimalShunt principles.
    Includes bidirectional resonance, pocket storage, and proper gating.
    """

    def __init__(self, dim, heads=8, collapse_threshold=0.29514, sharpness=25.0, tau_init=0.01, dropout=0.0):
        super().__init__()

        # Bidirectional resonant coupling (not attention!)
        self.cross_latent2intent = ResonantMultiheadAttention(
            dim=dim,
            heads=heads,
            collapse_threshold=collapse_threshold,
            sharpness=2 * torch.pi  # Gentler for cross-coupling
        )
        self.cross_intent2latent = ResonantMultiheadAttention(
            dim=dim,
            heads=heads,
            collapse_threshold=collapse_threshold,
            sharpness=2 * torch.pi
        )

        # Pocket blocks for discharge pattern storage
        self.pocket_l2i = BottleneckResBlock(dim, dropout=dropout)
        self.pocket_i2l = BottleneckResBlock(dim, dropout=dropout)

        # Full ignition layer
        self.ignition_layer = ResonantIgnitionLayer(
            dim=dim,
            collapse_threshold=collapse_threshold,
            sharpness=sharpness
        )

        # Dual projections like MinimalShunt
        self.delta_proj = nn.Linear(dim, dim)
        self.anchor_proj = nn.Linear(dim, dim)

        # Gate projection matching MinimalShunt design
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Tanh(),
            nn.Sigmoid()
        )

        # CRE basis and attractor
        self.cre_basis = nn.Parameter(torch.randn(6, dim))
        self.attractor = nn.Parameter(torch.randn(1, 1, dim))
        self.tau = nn.Parameter(torch.full((heads, 1, 1), tau_init))

        # Initialize gate to 0.29514
        with torch.no_grad():
            self.gate_proj[-3].bias.data.fill_(-1.527)

    def forward(self, latent: torch.Tensor, intent: torch.Tensor) -> dict:
        """
        latent: [B, T, D] - latent signal tensor
        intent: [B, D] - symbolic or learned purpose vector
        Returns: dict with delta, anchor, gate, and resonance info
        """
        B, T, D = latent.shape

        # Handle intent dimensions
        if intent.dim() == 2:
            intent_expanded = intent.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        else:
            intent_expanded = intent  # Already [B, T, D]

        # Bidirectional resonant coupling
        # For l2i: latent resonates with intent phase reference
        l2i_combined = latent + intent_expanded * 0.1  # Gentle phase coupling
        l2i = self.cross_latent2intent(l2i_combined)

        # For i2l: intent resonates with latent phase reference
        i2l_combined = intent_expanded + latent * 0.1
        i2l = self.cross_intent2latent(i2l_combined)

        # Store discharge patterns
        l2i = self.pocket_l2i(l2i)
        i2l = self.pocket_i2l(i2l)

        # Combine bidirectional flows
        core = (l2i.mean(1) + i2l.mean(1)) / 2  # [B, D]

        # Compute ignition using the combined representation
        combined = l2i + i2l  # [B, T, D]
        # Pass intent in its original form (B, D) to ignition layer
        intent_for_ignition = intent if intent.dim() == 2 else intent.mean(1)
        ignition = self.ignition_layer(combined, intent_for_ignition)  # [B, T, 1]

        # Gate computation
        gate = self.gate_proj(core)  # [B, 1]

        # Dual projections
        delta = self.delta_proj(core) * gate  # [B, D]
        anchor = self.anchor_proj(core)  # [B, D]

        # CRE energy calculation
        delta_norm = F.normalize(delta, dim=-1)
        cre_energy = torch.stack([
            torch.sum(delta_norm * basis, dim=-1)
            for basis in self.cre_basis
        ], dim=-1)  # [B, 6]

        # CVP alignment
        attractor_norm = F.normalize(self.attractor.squeeze(0), dim=-1)
        cvp_alignment = torch.sum(delta_norm * attractor_norm, dim=-1, keepdim=True)  # [B, 1]

        # Modulated output
        modulated_output = combined * ignition  # [B, T, D]

        return {
            "output": modulated_output,
            "delta": delta,
            "anchor": anchor,
            "gate": gate,
            "ignition": ignition.mean(dim=1),  # [B, 1]
            "cre_energy": cre_energy,
            "cvp_alignment": cvp_alignment,
            "tau": self.tau,
        }