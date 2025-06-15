
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.layers.transmission.modulation import ResonantModulationCoil
from models.layers.modulation.phase_gate import ResonantPhaseGate

class DirectResonantClassifier(nn.Module):
    """
    Coil-guided classifier using attunement-based phase gating to direct field collapse.
    """

    def __init__(self, num_classes=10, input_dim=784, hidden_dim=256, bottleneck_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.mod_config = {
            "stream_a": {"hidden_size": hidden_dim},
            "stream_b": {"hidden_size": hidden_dim},
            "bottleneck": bottleneck_dim,
            "heads": 8,
            "max_guidance": 1.0,
            "layer_norm": True,
            "use_dropout": False,
            "dropout": 0.0,
            "proj_layers": 2
        }

        self.phase_gates = nn.ModuleList([
            ResonantPhaseGate(mode='tau', tau=0.25 + i * 0.05, beta=1.5)
            for i in range(num_classes)
        ])

        self.modulation_coils = nn.ModuleList([
            ResonantModulationCoil(self.mod_config) for _ in range(num_classes)
        ])

        self.class_fields = nn.Parameter(torch.randn(num_classes, 1, hidden_dim) * 0.02)

        # Pre-attunement selector gate (shared across classes)
        self.selector_gate = ResonantPhaseGate(mode='tau', tau=0.5, beta=1.5)

        with torch.no_grad():
            for i in range(num_classes):
                phase = 0.29514 * (i / num_classes) * 2 * np.pi
                for d in range(hidden_dim):
                    freq = 1.0 + (d / hidden_dim)
                    self.class_fields[i, 0, d] = np.sin(phase * freq) * 0.29514

        self.register_buffer('conductance_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0))

    def apply_resonant_shock(self, x):
        B, T, D = x.shape
        shock = torch.zeros_like(x)
        for i in range(5):
            freq = (i + 1) * 0.29514
            amplitude = 0.29514 / (i + 1)
            phase = i * math.pi / 5
            wave = amplitude * torch.sin(freq * torch.linspace(0, 2 * math.pi, T, device=x.device) + phase)
            shock += wave.unsqueeze(0).unsqueeze(-1).expand(B, T, D)
        decay = torch.exp(-torch.linspace(0, 2 * math.pi, T, device=x.device) / math.pi).unsqueeze(0).unsqueeze(-1)
        return x + shock * decay

    def forward(self, x, apply_shock=False):
        B = x.size(0)
        current = self.input_projection(x).unsqueeze(1)  # [B, 1, D]

        if apply_shock:
            current = self.apply_resonant_shock(current)

        # Pre-attunement field projection (for targeting coils)
        flat_proj = current.squeeze(1)                       # [B, D]
        class_fields_flat = self.class_fields.squeeze(1)     # [C, D]
        alignment = torch.matmul(flat_proj, class_fields_flat.T)  # [B, C]
        gating_weights = self.selector_gate(alignment)            # [B, C]

        outputs = []
        gate_scores = []
        conductances = []

        for i in range(self.num_classes):
            class_current = self.class_fields[i]  # [1, D]
            gated_current = self.phase_gates[i](class_current.expand(B, 1, -1))
            field = current

            anchor, delta, log_sigma, ignition, g_pred = self.modulation_coils[i](gated_current, field)

            # Apply attunement-based damping to gate
            gate_mod = gating_weights[:, i].unsqueeze(1)  # [B, 1]
            modulated_guidance = g_pred * gate_mod        # [B, 1]

            outputs.append({
                'anchor': anchor,
                'delta': delta,
                'log_sigma': log_sigma,
                'ignition': ignition,
                'guidance': modulated_guidance,
                'gate_score': modulated_guidance,
                'class_idx': i
            })

            gate_scores.append(modulated_guidance)
            conductances.append(modulated_guidance.mean().detach())

        classification_scores = torch.cat(gate_scores, dim=1)  # [B, C]

        mean_conductance = torch.stack(conductances).mean()
        idx = self.history_idx.item()
        self.conductance_history[idx % 1000] = mean_conductance
        self.history_idx += 1

        return {
            'outputs': outputs,
            'scores': classification_scores,
            'mean_conductance': mean_conductance,
            'conductance_std': torch.stack(conductances).std()
        }
