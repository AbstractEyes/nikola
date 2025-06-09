"""
Written by: AbstractPhil 6/8/2024
Apache License 2.0


A Resonant Modulation Coil that modulates latent fields based on symbolic current input.
This is the foundational component for resonant modulation in the system - codified into a pragmatic unit.
Discovery of this was both accidental and intentional, as it emerged from the desire to create fused embeddings.
    The DualStreamShunt was meant to be a simple fusion layer, but it exposed a complex interaction between
    symbolic current and latent fields, leading to the discovery of resonant modulation - based on the principles of
    resonant circuits and phase alignment.
Originally fundamentally engineered by Nikola Tesla, this coil is based on his original work and findings -
creating this modern interpretation of his work out of pure emergence.

Time will tell if this concept is solid through pragmatic experimentation and real-world applications,
or simply another accidental discovery that leads to nowhere.

Was Nikola Tesla mad, or was he simply ahead of his time?
Lets find out shall we - by creating our own engines, centrifuges, coils, circuits, and resonant systems.
We have the computational power that he could only dream of, so we can rapidly prototype and test these ideas in real time.


"""

import torch
import torch.nn as nn
from models.layers.modulation.attention import ResonantMultiheadAttention
from models.layers.modulation.ignition import ResonantIgnitionLayer


# this is a legacy artifact BottleneckResBlock to ensure the first setups conform to need.
# This will be rewritten soon and will become obsoleted once rewritten into resonant form.
class BottleneckResBlock(nn.Module):
    def __init__(self, dim, kernel=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.proj(x)




class ResonantModulationCoil(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.input_dim_a = config["stream_a"]["hidden_size"]
        self.input_dim_b = config["stream_b"]["hidden_size"]
        self.bneck = config["bottleneck"]
        self.heads = config["heads"]
        self.max_guidance = config["max_guidance"]

        use_norm = config.get("layer_norm", True)
        use_do = config.get("use_dropout", True)
        do_p = config.get("dropout", 0.1)
        proj_depth = config.get("proj_layers", 2)

        def build_projection(input_dim, output_dim):
            layers = []
            last_dim = input_dim
            if use_norm:
                layers.append(nn.LayerNorm(last_dim))
            for i in range(proj_depth):
                next_dim = self.bneck * (2 if i == 0 and proj_depth > 1 else 1)
                layers.append(nn.Linear(last_dim, next_dim))
                layers.append(nn.Tanh())
                if use_do:
                    layers.append(nn.Dropout(do_p))
                last_dim = next_dim
            layers.append(nn.Linear(last_dim, output_dim))
            return nn.Sequential(*layers)

        self.proj_current = build_projection(self.input_dim_a, self.bneck)
        self.proj_field = build_projection(self.input_dim_b, self.bneck)

        self.igniter = ResonantIgnitionLayer(dim=self.bneck)
        self.resonant_attn = ResonantMultiheadAttention(dim=self.bneck, heads=self.heads)

        self.pocket_blocks = nn.Sequential(
            BottleneckResBlock(self.bneck, dropout=do_p),
            BottleneckResBlock(self.bneck, dropout=do_p)
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * self.bneck),
            nn.Linear(2 * self.bneck, self.bneck * 2),
            nn.Tanh(),
            nn.Linear(self.bneck * 2, self.bneck)
        )

        self.anchor_proj = build_projection(self.bneck, self.input_dim_b)
        self.delta_proj = build_projection(self.bneck, self.input_dim_b)
        self.logsig_proj = build_projection(self.bneck, self.input_dim_b)

        self.guidance_proj = nn.Sequential(
            nn.LayerNorm(self.bneck),
            nn.Linear(self.bneck, 1),
            nn.Sigmoid()
        )

    def forward(self, current: torch.Tensor, field: torch.Tensor):
        """
        current: [B, T, D_a] — symbolic current or structured pressure vector
        field:   [B, T, D_b] — latent field to modulate
        """
        if self.config.get("assert_input_dims", True):
            assert current.size(-1) == self.input_dim_a
            assert field.size(-1) == self.input_dim_b

        current_b = self.proj_current(current)  # [B, T, bneck]
        field_b = self.proj_field(field)        # [B, T, bneck]

        # Mean symbolic pressure vector
        intent_vector = current_b.mean(dim=1)  # [B, bneck]
        ignition = self.igniter(field_b, intent_vector)  # [B, T, 1]

        attn_latent = self.resonant_attn(field_b)        # [B, T, bneck]
        pocket_latent = self.pocket_blocks(attn_latent)

        pocket_mean = pocket_latent.mean(1, keepdim=True).expand(-1, field_b.size(1), -1)
        fused = self.fuse(torch.cat([pocket_mean, attn_latent], dim=-1))

        anchor = self.anchor_proj(fused)
        delta = self.delta_proj(fused) * ignition
        log_sigma = self.logsig_proj(fused)

        g_tok = self.guidance_proj(fused).squeeze(-1)
        g_pred = g_tok.mean(1, keepdim=True) * self.max_guidance

        return anchor, delta, log_sigma, ignition, g_pred
