import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldLoss(nn.Module):
    """
    Field-aware collapse loss with resonance-controlled gating.
    Includes:
    - Entropy masking (lightweight)
    - Cosine-phase alignment penalty
    - Differentiable alignment-triggered collapse (soft threshold)
    """
    def __init__(self,
                 reduction='mean',
                 use_cosine=True,
                 cosine_weight=0.5,
                 use_alignment_gate=True,
                 align_gate_thresh=0.25,
                 align_gate_sharpness=10.0):
        super().__init__()
        self.reduction = reduction
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight
        self.use_alignment_gate = use_alignment_gate
        self.align_gate_thresh = align_gate_thresh
        self.align_gate_sharpness = align_gate_sharpness

    def forward(self, pred, target, permission, alignment, log_sigma=None):
        """
        pred:       [B, D] - predicted vector
        target:     [B, D] - target vector
        permission: [B, 1] - scalar gate (0–1)
        alignment:  [B, 1] - phase agreement (e.g. dot product or cosine)
        log_sigma:  [B, 1] or None - optional entropy penalty
        """
        # Base loss (MSE)
        base_mse = F.mse_loss(pred, target, reduction='none').sum(dim=-1, keepdim=True)

        # Cosine alignment penalty (directional)
        if self.use_cosine:
            cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-8).unsqueeze(-1)
            angle_loss = 1.0 - cos_sim  # [B, 1]
            loss = (1.0 - self.cosine_weight) * base_mse + self.cosine_weight * angle_loss
        else:
            loss = base_mse

        # Differentiable alignment-triggered gating
        if self.use_alignment_gate:
            align_gate = torch.sigmoid((alignment - self.align_gate_thresh) * self.align_gate_sharpness)
        else:
            align_gate = alignment.clamp(min=0.0)  # fallback for hard gating

        # Entropy masking
        if log_sigma is not None:
            entropy_mask = 1.0 - torch.sigmoid(log_sigma)  # high entropy → low weight
        else:
            entropy_mask = 1.0

        # Final modulator
        modulator = permission * align_gate * entropy_mask
        gated_loss = loss * modulator

        # Reduction
        if self.reduction == 'mean':
            return gated_loss.mean()
        elif self.reduction == 'sum':
            return gated_loss.sum()
        else:
            return gated_loss  # [B, 1]
