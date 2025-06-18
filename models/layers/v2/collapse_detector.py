import torch
import torch.nn as nn


class CollapseDetector(nn.Module):
    """
    Nikola-compliant symbolic collapse detector.
    Detects collapse via entropy excess, alignment loss, and conductance decay.
    Outputs a soft collapse score (not hard gate).
    """

    def __init__(self, entropy_threshold=0.85, alignment_threshold=0.15, conductance_floor=0.25):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.alignment_threshold = alignment_threshold
        self.conductance_floor = conductance_floor

    def forward(self, log_sigma, alignment, conductance):
        """
        Args:
            log_sigma: [B, D] — latent entropy signature
            alignment: [B, 1] — symbolic alignment score (0..1)
            conductance: [B, 1] — symbolic conductance flow

        Returns:
            collapse_score: [B, 1] — soft signal (0 = stable, 1 = collapsed)
        """

        # Compute entropy activation
        entropy_energy = torch.exp(log_sigma.abs()).mean(dim=-1, keepdim=True)  # [B, 1]
        entropy_trigger = (entropy_energy > self.entropy_threshold).float()

        # Alignment failure
        alignment_trigger = (alignment < self.alignment_threshold).float()

        # Conductance too weak
        conductance_trigger = (conductance < self.conductance_floor).float()

        # Combine as weighted indicators
        collapse_score = (
                0.4 * entropy_trigger +
                0.3 * alignment_trigger +
                0.3 * conductance_trigger
        ).clamp(0, 1)

        return collapse_score  # not a gate — this is an indicator
