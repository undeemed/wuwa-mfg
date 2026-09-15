# SPDX-License-Identifier: Apache-2.0
"""Global token exchange for the private compact-student experiment.

The caller supplies learned features at 1/32 of each image dimension. Original
pixels and the existing detail skips remain in the surrounding network. This
is a new architecture to train, not an equivalent replacement for native code.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GlobalAttention(nn.Module):
    def __init__(self, width, head_dim=32):
        super().__init__()
        if width < head_dim or width % head_dim:
            raise ValueError('Feature width must be a positive multiple of head_dim.')
        self.width = width
        self.head_dim = head_dim
        self.heads = width // head_dim
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.project = nn.Linear(width, width)
        # Preserve the starting backbone function while learning this branch.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features):
        batch, channels, height, width = features.shape
        if channels != self.width:
            raise ValueError('Feature channel count differs from the configured width.')
        tokens = features.flatten(2).transpose(1, 2)
        qkv = self.qkv(self.norm(tokens)).reshape(
            batch, height * width, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
        residual = self.project(attended).transpose(1, 2).reshape_as(features)
        return features + residual * 0.1
