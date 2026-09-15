# SPDX-License-Identifier: Apache-2.0
"""Apply a coarse learned affine color field and a full-resolution residual.

The field predicts a transform, not a reduced-resolution replacement image.
Every original RGB pixel and every detail-residual pixel participates in output.
"""
import torch
from torch.nn import functional as F


def compose_affine(source,detail,coefficients):
    if source.ndim!=4 or source.shape[1]!=3 or detail.shape!=source.shape:
        raise ValueError('Expected matching NCHW RGB source and detail tensors.')
    batch,_,height,width=source.shape
    if coefficients.shape!=(batch,12,(height+31)//32,(width+31)//32):
        raise ValueError('Expected twelve coefficients per padded 32x32 cell.')
    dense=F.interpolate(coefficients,scale_factor=32,mode='bilinear',align_corners=False)
    dense=dense[:,:,:height,:width].reshape(batch,3,4,height,width)
    color=dense[:,:,0]*source[:,0:1]
    color=color+dense[:,:,1]*source[:,1:2]
    color=color+dense[:,:,2]*source[:,2:3]
    color=color+dense[:,:,3]
    return (source+.25*(detail+color)).clamp(0,1)
