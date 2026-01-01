"""
Normalization layer for models - enables attacks in pixel space
"""

import torch
import torch.nn as nn


class NormalizeLayer(nn.Module):
    """Normalization layer that can be inserted into model forward pass"""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        if x.size(1) != self.mean.size(1):
            raise ValueError(f"NormalizeLayer expected {self.mean.size(1)} channels, got {x.size(1)}")

        # Cast to input dtype for AMP/mixed precision compatibility
        mean = self.mean.to(dtype=x.dtype)
        std = self.std.to(dtype=x.dtype)
        return (x - mean) / std


def create_imagenet_normalizer():
    """Create ImageNet normalization layer"""
    return NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
