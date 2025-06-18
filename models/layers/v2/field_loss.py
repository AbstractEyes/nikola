import torch
import torch.nn as nn
import torch.nn.functional as F

class NikolaFieldLoss(nn.Module):
    """
    Nikola-compliant symbolic field loss.
    Measures symbolic deviation without enforcing collapse.
    Phase divergence is softly penalized. No component is zeroed or clipped.
    """
    def __init__(self, reduction='mean', use_cosine=True, cosine_weight=0.5):
        super().__init__()
        self.reduction = reduction
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight

    def forward(self, pred, target, permission, alignment, log_sigma=None):
        """
        pred:      [B, D] predicted symbolic field vector
        target:    [B, D] target field vector (grounded symbolically)
        permission: [B, 1] symbolic modulation signal
        alignment:  [B, 1] phase alignment coefficient (0..1)
        log_sigma: [B, D] optional symbolic entropy field (per dimension)
        """

        # Symbolic deviation (not "error")
        base_diff = pred - target
        magnitude_loss = (base_diff ** 2).sum(dim=-1, keepdim=True)  # [B, 1]

        # Optional phase alignment penalty (cosine divergence)
        if self.use_cosine:
            cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-8).unsqueeze(-1)  # [B, 1]
            phase_divergence = 1.0 - cos_sim
            # Phase divergence does not suppress magnitude — only scales it
            loss = magnitude_loss + self.cosine_weight * phase_divergence * magnitude_loss.detach()
        else:
            loss = magnitude_loss

        # Entropy mask suppresses unstable feedback, not symbolic form
        if log_sigma is not None:
            entropy_mask = torch.exp(-log_sigma.abs())  # [B, D] or broadcastable
        else:
            entropy_mask = 1.0

        # Final resonance-aligned symbolic loss (no zeroing)
        field_loss = loss * permission * alignment * entropy_mask  # [B, 1] or [B, D]

        if self.reduction == 'mean':
            return field_loss.mean()
        elif self.reduction == 'sum':
            return field_loss.sum()
        return field_loss  # unaggregated
