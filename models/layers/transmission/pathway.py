import torch
from models.layers.modulation.attention import ResonantMultiheadAttention
from models.layers.modulation.ignition import ResonantIgnitionLayer


class PathwayCoil:
    """
        A very simple and very sensitive pathway coil that modulates its output based on the alignment of latent signals
        and an ignition intent vector. It uses resonant attention to determine phase alignment
        and a gating mechanism to control the output conductance.
        This coil is meant to overfit and learn very specific patterns in the latent space that cannot be unlearned.
    """
    def __init__(self, dim, heads=8, collapse_threshold=0.29514, sharpness=15.0):
        self.attention = ResonantMultiheadAttention(dim, heads, collapse_threshold, sharpness)
        self.ignition_layer = ResonantIgnitionLayer(dim, collapse_threshold, sharpness)

    def forward(self, latent: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        latent: [B, T, D] - latent signal tensor
        intent: [B, D] - symbolic or learned purpose vector
        """
        # Apply resonant attention to the latent signal
        attention_output = self.attention(latent)

        # Compute ignition potential based on intent
        ignition = self.ignition_layer(attention_output, intent)

        return ignition * attention_output  # Modulate output by ignition potential