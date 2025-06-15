import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldLoss(nn.Module):
    """
    Unclamped field-aware collapse loss.
    All collapse is now driven purely by resonance math.
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
        base_mse = F.mse_loss(pred, target, reduction='none').sum(dim=-1, keepdim=True)

        if self.use_cosine:
            cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-8).unsqueeze(-1)
            angle_loss = 1.0 - cos_sim
            loss = base_mse + self.cosine_weight * angle_loss * base_mse.detach()
        else:
            loss = base_mse

        if self.use_alignment_gate:
            gate_input = (alignment - self.align_gate_thresh) * self.align_gate_sharpness
            align_gate = torch.sigmoid(gate_input)
        else:
            align_gate = alignment  # no clamp

        if log_sigma is not None:
            entropy_mask = torch.exp(-log_sigma.abs())  # no clamp
        else:
            entropy_mask = 1.0

        # No clamps — permission flows as-is
        modulator = permission * align_gate * entropy_mask
        gated_loss = loss * modulator

        if self.reduction == 'mean':
            return gated_loss.mean()
        elif self.reduction == 'sum':
            return gated_loss.sum()
        return gated_loss


class FieldLossRigid(nn.Module):
    """
    Rigid field collapse loss.
    Collapse is triggered only when alignment surpasses a hard threshold.
    Loss modulation is unforgiving; feedback is scaled by structural precision.
    """
    def __init__(self,
                 reduction='mean',
                 align_gate_thresh=0.25,
                 repulsion_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.align_gate_thresh = align_gate_thresh
        self.repulsion_weight = repulsion_weight

    def forward(self, pred, target, permission, alignment, log_sigma=None):
        # Base structural error
        base_mse = F.mse_loss(pred, target, reduction='none').sum(dim=-1, keepdim=True)

        # Hard collapse: only triggers when resonance is sufficiently aligned
        collapse_trigger = (alignment >= self.align_gate_thresh).float()

        # Repulsion if alignment exceeds unity
        repel_force = (alignment - 1.0).clamp(max=0.0).abs()

        # Entropy suppression
        if log_sigma is not None:
            entropy_mask = torch.exp(-log_sigma.abs()).clamp(min=1e-2)
        else:
            entropy_mask = 1.0

        # Final modulator (vacuum consequence field)
        modulator = permission * collapse_trigger * entropy_mask

        # Apply rigid collapse and opposing structure force
        loss = base_mse * modulator + self.repulsion_weight * repel_force

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
