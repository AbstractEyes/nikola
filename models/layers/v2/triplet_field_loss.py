import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNikolaFieldLoss(nn.Module):
    """
    Field-aware triplet loss for symbolic resonance systems.
    Computes:
      - Positive alignment (anchor ↔ positive)
      - Repulsion penalty (anchor ↔ negative)
    Collapse is not enforced. All divergence is phase-scaled and entropy-aware.
    """
    def __init__(self, reduction='mean', use_cosine=True, cosine_weight=0.5, margin=0.1):
        super().__init__()
        self.reduction = reduction
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight
        self.margin = margin  # margin in alignment space (field units)

    def forward(
        self,
        anchor,         # [B, D]
        positive,       # [B, D]
        negative,       # [B, D]
        permission,     # [B, 1]
        alignment_pos,  # [B, 1] (optional per anchor-positive)
        alignment_neg,  # [B, 1] (optional per anchor-negative)
        log_sigma=None  # [B, D] (optional entropy field)
    ):
        # Base magnitude loss (symbolic drift)
        pos_delta = anchor - positive
        neg_delta = anchor - negative

        pos_mag = (pos_delta ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
        neg_mag = (neg_delta ** 2).sum(dim=-1, keepdim=True)  # [B, 1]

        if self.use_cosine:
            pos_cos = F.cosine_similarity(anchor, positive, dim=-1, eps=1e-8).unsqueeze(-1)
            neg_cos = F.cosine_similarity(anchor, negative, dim=-1, eps=1e-8).unsqueeze(-1)

            pos_phase = 1.0 - pos_cos
            neg_phase = 1.0 - neg_cos

            pos_loss = pos_mag + self.cosine_weight * pos_phase * pos_mag.detach()
            neg_loss = neg_mag + self.cosine_weight * neg_phase * neg_mag.detach()
        else:
            pos_loss = pos_mag
            neg_loss = neg_mag

        # Optional entropy modulation
        if log_sigma is not None:
            entropy_mask = torch.exp(-log_sigma.abs())
        else:
            entropy_mask = 1.0

        # Phase-scaled resonance guidance
        align_pos = alignment_pos if alignment_pos is not None else 1.0
        align_neg = alignment_neg if alignment_neg is not None else 1.0

        # Final modulated losses
        pos_field = pos_loss * permission * align_pos * entropy_mask
        neg_field = neg_loss * permission * align_neg * entropy_mask

        # Soft-margin field triplet loss
        field_triplet = F.relu(pos_field - neg_field + self.margin)

        if self.reduction == 'mean':
            return field_triplet.mean()
        elif self.reduction == 'sum':
            return field_triplet.sum()
        return field_triplet
