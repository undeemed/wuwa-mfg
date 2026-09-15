# SPDX-License-Identifier: Apache-2.0
"""Original region-routed context experiment on frozen student features.

Inspired by MoBA's block selection and BiFormer's region routing. This is not
either paper's model or implementation. All spatial queries are retained; only
the context keys/values are selected. Uses existing PyTorch operators only.
"""
import torch
from torch import nn
from torch.nn import functional as F

from shared_feature_student import SharedFeatureRefinement


class RegionContext(nn.Module):
    def __init__(self, channels=96, inner=48, heads=3, region_size=4,
                 selected_regions=4, mode='routed'):
        super().__init__()
        if (mode not in ('dense', 'routed') or min(channels, inner, heads, region_size, selected_regions) < 1
                or inner % heads):
            raise ValueError('Invalid region-attention configuration.')
        self.channels, self.inner, self.heads = channels, inner, heads
        self.region_size, self.selected_regions, self.mode = region_size, selected_regions, mode
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3*inner)
        self.output = nn.Linear(inner, channels)
        self.scale = nn.Parameter(torch.full((channels,), .1))

    def prepare(self, source):
        if source.ndim != 4 or source.shape[1] != self.channels or min(source.shape[-2:]) < 1:
            raise ValueError('Expected nonempty NCHW feature tensor with the configured channels.')
        batch, _, height, width = source.shape
        side = self.region_size
        rows, cols = (height+side-1)//side, (width+side-1)//side
        padded = F.pad(source, (0, cols*side-width, 0, rows*side-height))
        tokens = padded.reshape(batch, self.channels, rows, side, cols, side)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(batch, rows*cols, side*side, self.channels)
        # Padding is excluded from region means AND token-level key attention.
        valid = torch.ones((1, 1, height, width), device=source.device, dtype=torch.bool)
        valid = F.pad(valid, (0, cols*side-width, 0, rows*side-height), value=False)
        valid = valid.reshape(1, 1, rows, side, cols, side).permute(0, 2, 4, 3, 5, 1)
        valid = valid.reshape(1, rows*cols, side*side).expand(batch, -1, -1)
        q, k, v = self.qkv(self.norm(tokens)).chunk(3, dim=-1)
        return q, k, v, valid, (height, width, rows, cols)

    def route(self, q, k, valid):
        count = q.shape[1]
        selected = min(count, self.selected_regions)
        own = torch.arange(count, device=q.device).view(1, count, 1).expand(q.shape[0], -1, -1)
        if selected == 1:
            return own
        # Discrete top-k has no gradient. Shared q/k projections learn through
        # token attention; FP32 region scores reduce low-precision routing noise.
        with torch.no_grad():
            weight = valid.float().unsqueeze(-1)
            denom = weight.sum(dim=2).clamp_min(1)
            mean_q = (q.detach().float()*weight).sum(dim=2)/denom
            mean_k = (k.detach().float()*weight).sum(dim=2)/denom
            scores = mean_q @ mean_k.transpose(-1, -2) / self.inner**.5
            diagonal = torch.eye(count, device=q.device, dtype=torch.bool).unsqueeze(0)
            other = scores.masked_fill(diagonal, -torch.inf).topk(selected-1, dim=-1).indices
        return torch.cat((own, other), dim=-1)

    def attend(self, q, k, v, valid, routes=None):
        batch, regions, tokens, _ = q.shape
        head_size = self.inner//self.heads
        if routes is None:
            queries = q.reshape(batch, regions*tokens, self.heads, head_size).transpose(1, 2)
            keys = k.reshape(batch, regions*tokens, self.heads, head_size).transpose(1, 2)
            values = v.reshape(batch, regions*tokens, self.heads, head_size).transpose(1, 2)
            mask = valid.reshape(batch, 1, 1, regions*tokens)
            result = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, dropout_p=0.)
            return result.transpose(1, 2).reshape(batch, regions, tokens, self.inner)
        chosen = routes.shape[-1]
        batch_ids = torch.arange(batch, device=q.device)[:, None, None]
        # Actual gather reduces the attention key dimension. This is not a
        # full dense attention matrix with a sparse mask applied afterwards.
        keys = k[batch_ids, routes].reshape(batch*regions, chosen*tokens, self.heads, head_size).transpose(1, 2)
        values = v[batch_ids, routes].reshape(batch*regions, chosen*tokens, self.heads, head_size).transpose(1, 2)
        mask = valid[batch_ids, routes].reshape(batch*regions, 1, 1, chosen*tokens)
        queries = q.reshape(batch*regions, tokens, self.heads, head_size).transpose(1, 2)
        result = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, dropout_p=0.)
        return result.transpose(1, 2).reshape(batch, regions, tokens, self.inner)

    def forward(self, source):
        q, k, v, valid, (height, width, rows, cols) = self.prepare(source)
        routes = self.route(q, k, valid) if self.mode == 'routed' else None
        update = self.output(self.attend(q, k, v, valid, routes))*self.scale
        side = self.region_size
        update = update.reshape(source.shape[0], rows, cols, side, side, self.channels)
        update = update.permute(0, 5, 1, 3, 2, 4).reshape(source.shape[0], self.channels, rows*side, cols*side)
        return source + update[:, :, :height, :width]


class RegionFeatureRefinement(SharedFeatureRefinement):
    def __init__(self, first, mode):
        super().__init__(first)
        # All old decoder parameters are initialized before the added branch.
        # Resetting the same seed gives the dense/routed pair identical weights.
        self.refinement.deep = nn.Sequential(self.refinement.deep, RegionContext(mode=mode))


def architecture(mode):
    return {'type': 'shared-feature decoder with region context', 'mode': mode,
            'input_channels': 96, 'inner_channels': 48, 'heads': 3,
            'region_size': [4, 4], 'selected_regions': 4 if mode == 'routed' else 'all',
            'own_region_always_included': True, 'routing_scores': 'FP32 pooled q/k dot products',
            'routing_differentiable': False, 'query_pixels_dropped': False,
            'padding_excluded_from_pooling_and_keys': True, 'new_custom_kernels': False}
