import torch
import torch.nn as nn
import math


class ResonantNormalizationGate(nn.Module):
    def __init__(
        self,
        n_channels,
        Q_factor=10.0,
        n_iterations=4,
        use_cordic=True,
        gating_mode="exp_decay",  # 'tanh', 'rational'
        wave_mode="cosine_sine_product",  # 'cos', 'sine'
        resonance_mode="cosine_power",  # 'exponential_decay'
    ):
        super().__init__()
        self.n_channels = n_channels
        self.Q = Q_factor
        self.n_iterations = n_iterations
        self.use_cordic = use_cordic
        self.gating_mode = gating_mode
        self.wave_mode = wave_mode
        self.resonance_mode = resonance_mode

        self.resonant_frequencies = nn.Parameter(
            torch.linspace(0.5, 2.0, n_channels).unsqueeze(0)
        )
        self.phase_decay = nn.Parameter(torch.tensor(0.95))
        self.wave_scale = nn.Parameter(torch.tensor(1.0))
        self.resonance_weight = nn.Parameter(torch.tensor(1.0))  # weight between wave/resonance

        self.register_buffer('phase_accumulators', torch.zeros(1, n_channels))
        self.register_buffer('cordic_pairs', self._make_cordic_pairs(n_channels))
        self.symbolic_pos_embed = nn.Parameter(
            torch.sin(torch.linspace(0, math.pi, n_channels))
        )

    def _make_cordic_pairs(self, dim):
        return torch.tensor([(i, i + 1) for i in range(0, dim - 1, 2)])

    def compute_resonance_strength(self, phi, omega):
        phase_diff = phi - omega
        if self.resonance_mode == "cosine_power":
            base = torch.cos(phase_diff) * 0.5 + 0.5
            return torch.pow(base, self.Q)
        elif self.resonance_mode == "exponential_decay":
            return torch.exp(-self.Q * (1 - torch.cos(phase_diff)))
        else:
            raise ValueError(f"Unknown resonance_mode: {self.resonance_mode}")

    def cordic_vector_normalize(self, x):
        x_out = x.clone()
        for _ in range(self.n_iterations):
            for i, j in self.cordic_pairs:
                xi, xj = x_out[:, i], x_out[:, j]
                r2 = xi**2 + xj**2 + 1e-8
                inv_r = 0.5 * (3.0 - r2 * (1.0 / r2))
                x_out[:, i] = xi * inv_r
                x_out[:, j] = xj * inv_r

        # Handle final channel if odd
        if self.n_channels % 2 == 1:
            i = self.n_channels - 1
            xi = x_out[:, i]
            r2 = xi**2 + 1e-8
            inv_r = 0.5 * (3.0 - r2 * (1.0 / r2))
            x_out[:, i] = xi * inv_r

        return x_out

    def fast_inverse_sqrt(self, x, iterations=2):
        r = torch.ones_like(x) * 0.5
        for _ in range(iterations):
            r = 0.5 * r * (3.0 - x * r * r)
        return r

    def standing_wave_pattern(self, phases):
        p = self.symbolic_pos_embed
        if self.wave_mode == "cos":
            wave = torch.cos(phases)
        elif self.wave_mode == "sine":
            wave = torch.sin(math.pi * p)
        elif self.wave_mode == "cosine_sine_product":
            wave = torch.sin(math.pi * p) ** 2 * torch.cos(phases)
        else:
            raise ValueError(f"Unknown wave_mode: {self.wave_mode}")
        return wave * self.wave_scale

    def energy_conservation_normalize(self, x):
        energy = (x * x).sum(dim=1, keepdim=True)
        energy = torch.clamp(energy, min=1e-6)
        return x * self.fast_inverse_sqrt(energy)

    def apply_contraction_gate(self, z, normalized):
        if self.gating_mode == "exp_decay":
            return normalized * (1 - torch.exp(-z.abs()))
        elif self.gating_mode == "tanh":
            return torch.tanh(z)
        elif self.gating_mode == "rational":
            return z / (1 + z.abs())
        else:
            raise ValueError(f"Unknown gating_mode: {self.gating_mode}")

    def forward(self, x):
        B = x.size(0)

        with torch.no_grad():
            self.phase_accumulators.mul_(self.phase_decay)
            self.phase_accumulators.add_(x.mean(0, keepdim=True))

        resonance = self.compute_resonance_strength(
            self.phase_accumulators.expand(B, -1),
            self.resonant_frequencies.expand(B, -1)
        )

        wave = self.standing_wave_pattern(self.phase_accumulators).expand(B, -1)

        z = (
            x * resonance +
            self.resonance_weight * torch.sign(x) * wave * x.abs()
        )

        normalized = (
            self.cordic_vector_normalize(z) if self.use_cordic
            else self.energy_conservation_normalize(z)
        )

        return self.apply_contraction_gate(z, normalized)

    def reset_phase_accumulators(self):
        self.phase_accumulators.zero_()
