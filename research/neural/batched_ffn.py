# SPDX-License-Identifier: Apache-2.0
"""Batch independent FFN branches while preserving sequential head additions.

Adapted from the pinned MLX-DLSS mathematical reference. No vendor code or weights.
This is a research candidate; GEMM implementation changes require output checks.
"""
import torch


def branched_feed_forward(value, *, expansion_weight, branch_projection_weight,
                          output_projection_weight):
    from mlxdlss.model import e4m3_round_trip, quadratic_gate_activation
    channels = value.shape[-1]
    groups = channels // 32
    if channels < 64 or channels % 32:
        raise ValueError('Expected 32-aligned channels >=64.')
    if expansion_weight.shape != (groups, 4, groups, 32, 32):
        raise ValueError('Unexpected expansion layout.')
    if branch_projection_weight.shape != (groups, 4, 32, 32):
        raise ValueError('Unexpected branch projection layout.')
    if output_projection_weight.shape != (channels, channels):
        raise ValueError('Unexpected output projection layout.')
    flat = value.reshape(-1, channels)
    weights = expansion_weight.reshape(groups * 4, groups, 32, 32)
    expanded = None
    for head in range(groups):
        current = torch.bmm(flat[:, head*32:(head+1)*32].unsqueeze(0).expand(groups*4,-1,-1),
                            weights[:,head])
        # Same order as Python sum in the reference; no single wider reduction.
        expanded = current + 0 if expanded is None else expanded + current
    activated = e4m3_round_trip(quadratic_gate_activation(expanded))
    branches = torch.bmm(activated, branch_projection_weight.reshape(groups*4,32,32))
    branches = branches.reshape(groups,4,flat.shape[0],32)
    projected = branches[:,0] + 0
    for branch in range(1,4):
        projected = projected + branches[:,branch]
    published = e4m3_round_trip(projected)
    merged = published.permute(1,0,2).reshape(flat.shape[0],channels)
    return (merged @ output_projection_weight).reshape(value.shape)


def split_group_feed_forward(value, *, first_projection_weight, expand_weight, project_weight):
    from mlxdlss.model import e4m3_round_trip, quadratic_gate_activation
    channels = value.shape[-1]
    groups = channels // 64
    if channels % 64 or expand_weight.shape != (groups,64,256) or project_weight.shape != (groups,256,64):
        raise ValueError('Unexpected split FFN layout.')
    hidden = e4m3_round_trip(value @ first_projection_weight)
    grouped = hidden.reshape(-1,groups,64).permute(1,0,2)
    expanded = torch.bmm(grouped,expand_weight)
    projected = torch.bmm(quadratic_gate_activation(expanded),project_weight)
    merged = projected.permute(1,0,2).reshape(value.shape)
    return e4m3_round_trip(merged)
