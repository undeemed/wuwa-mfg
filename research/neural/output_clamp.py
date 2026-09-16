# SPDX-License-Identifier: Apache-2.0
"""Experimental straight-through gradient; the forward clamp stays exact."""
import torch


class StraightThroughClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value):
        return value.clamp(0,1)

    @staticmethod
    def backward(ctx,gradient):
        return gradient
