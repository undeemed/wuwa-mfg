# SPDX-License-Identifier: Apache-2.0
"""Original implementation of a detached, per-image/channel spectral weight.

Inspired by Jiang et al., Focal Frequency Loss (ICCV 2021), equations 7-10.
This auxiliary training loss adds no inference operations or learned parameters.
"""
import torch


def frequency_loss(predicted,target):
    if predicted.shape!=target.shape or predicted.ndim!=4 or predicted.dtype not in (torch.float32,torch.float64):
        raise ValueError('Expected matching NCHW FP32/FP64 images.')
    if predicted.dtype!=target.dtype or predicted.device!=target.device:
        raise ValueError('Expected matching image dtype and device.')
    difference=torch.fft.fft2(predicted-target,norm='ortho')
    squared=difference.real.square()+difference.imag.square()
    amplitude=squared.detach().sqrt()
    weight=amplitude/amplitude.amax(dim=(-2,-1),keepdim=True).clamp(min=1e-12)
    return (weight*squared).mean()
