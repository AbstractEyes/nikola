import torch
import torch.nn as nn
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseModulator(nn.Module):
    """
    Formerly ResonantPhaseGate.
    Modulates symbolic expression based on field phase translation.
    Replaces the idea of 'activation' with 'form release'.

    Modes:
        - 'sin': Soft oscillatory modulation (default)
        - 'translation': Linear-preserving phase shift
        - 'wrap': Hard phase loop mapping for symbolic phase rotation
    """
    def __init__(self, mode='sin', phase_scalar=1.0):
        super().__init__()
        self.mode = mode
        self.phase_scalar = phase_scalar

    def forward(self, x):
        if self.mode == 'sin':
            # Soft harmonic release
            phase = torch.sigmoid(x * self.phase_scalar) * (torch.pi / 2)
            return x * torch.sin(phase)

        elif self.mode == 'translation':
            # Direct modulation without suppression
            shift = torch.tanh(self.phase_scalar * x)
            return x + shift

        elif self.mode == 'wrap':
            # Force signal into bounded rotational field space
            wrapped = torch.fmod(self.phase_scalar * x, 2 * torch.pi)
            return torch.sin(wrapped)

        else:
            raise ValueError(f"Unknown phase modulation mode: {self.mode}")