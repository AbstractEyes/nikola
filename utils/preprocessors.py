# nikola/utils/preprocessors.py

import torch

def default_preprocess(x: torch.Tensor) -> torch.Tensor:
    """
    Default preprocessing:
    - Flatten if image-like
    - Normalize to [-1, 1]
    - Convert to float32
    """
    if x.ndim > 2:
        x = x.view(x.size(0), -1)
    x = x.to(torch.float32)
    return x * 2 - 1  # Assuming x ∈ [0,1]

def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)

def whiten(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)
