import torch
from models.layers.modulation.attention import ResonantMultiheadAttention
from models.layers.modulation.ignition import ResonantIgnitionLayer

class EmbedCoil:
    """
        Learns the purpose of the input data and attempts to provide an approximate representation that can be accessed.
        Meant specifically for the intermingling of traditional linear embeddings into resonant attention mechanisms.
        This experimental artifact is to ensure the first setups will properly be tested and conform to need.
        Depending on the results, this will be rewritten into a more resonant form or deprecated entirely.

        Tuned to target the coils responsible for intentionally diverse information representations through hardcoded gating offsets.
    """
    def __init__(self, gating_offset: float = 0.29514, sharpness: float = 15.0):
        self.gating_offset = gating_offset
        self.sharpness = sharpness

        # Initialize the resonant ignition layer
        self.ignition_layer = ResonantIgnitionLayer(dim=32, collapse_threshold=gating_offset, sharpness=sharpness)
        # Initialize the resonant attention layer
        self.attention_layer = ResonantMultiheadAttention(dim=32, collapse_threshold=gating_offset, sharpness=sharpness)

    def forward(self, latent: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embed coil.
        :param latent:
        :param intent:
        :return:
        """
        # Apply the ignition layer to compute the modulation gate
        ignition = self.ignition_layer(latent, intent)

        # Apply the resonant attention mechanism
        attended_output = self.attention_layer(latent)

        # Combine the ignition modulation with the attended output
        output = attended_output * ignition

        # Should give an approximate targeted representation offset as an answer for the collective request.
        return output
