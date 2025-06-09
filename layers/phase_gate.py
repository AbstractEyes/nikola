import torch
import torch.nn as nn
import math


class ResonantPhaseGate(nn.Module):
    """
    Division-free, resonance-aware activation function.
    Implements phase-permissive gating in three selectable modes:

    Modes:
        - 'sine': Uses sin(sigmoid(x) * π/2) for smooth flow control
        - 'entropy': Applies entropy-weighted suppression using sigmoid and tanh
        - 'tau': Uses exponential decay to represent inertial damping (preferred)
    """

    def __init__(self, mode='tau', tau=0.25, beta=1.5):
        super().__init__()
        self.mode = mode
        self.tau = tau
        self.beta = beta

    def forward(self, x):
        if self.mode == 'sine':
            gated = torch.sigmoid(x) * (math.pi * 0.5)  # no division
            return x * torch.sin(gated)

        elif self.mode == 'entropy':
            return x * torch.sigmoid(self.beta * x) * torch.tanh(x)

        elif self.mode == 'tau':
            damping = 1.0 - torch.exp(-self.tau * x.abs())
            return x * damping

        else:
            raise ValueError(f"[ResonantPhaseGate] Unknown mode: {self.mode}")
