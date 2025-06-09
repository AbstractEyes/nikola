import torch
import torch.nn.functional as F
from typing import Callable, Union

class DataConfiguration:
    """
    Holds configuration for data shaping and preprocessing.
    """
    def __init__(self,
                 target_shape: Union[tuple, torch.Size],
                 strict: bool = False,
                 eager: bool = True,
                 preprocess: Callable = None):
        self.target_shape = target_shape
        self.strict = strict
        self.eager = eager
        self.preprocess = preprocess  # Optional callable


class SimpleDataShaper:
    """
    Utility to preprocess and reshape data into target dimensions
    for resonance conditioning and training input normalization.
    """
    def __init__(self, config: DataConfiguration, data: torch.Tensor):
        self.data = data
        self.config = config
        self.target_shape = config.target_shape
        self.strict = config.strict
        self.eager = config.eager
        self.preprocess_fn = config.preprocess

    def preprocess(self, data: torch.Tensor = None, target_shape=None):
        data = data if data is not None else self.data
        target_shape = target_shape if target_shape is not None else self.target_shape

        input_shape = data.shape

        if self.preprocess_fn:
            data = self.preprocess_fn(data)

        if input_shape == torch.Size(target_shape):
            return data

        # Attempt padding or trimming to match target shape
        current_dim = data.shape[-1]
        target_dim = target_shape[-1]

        if current_dim < target_dim:
            pad_size = target_dim - current_dim
            padded = F.pad(data, (0, pad_size), value=0.0)
            reshaped = padded
        else:
            reshaped = data[..., :target_dim]

        if self.strict and reshaped.shape[-1] != target_dim:
            raise ValueError(f"Cannot match target shape {target_shape}, got {reshaped.shape}")

        return reshaped
