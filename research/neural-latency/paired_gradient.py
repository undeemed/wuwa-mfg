# SPDX-License-Identifier: Apache-2.0
"""Two-domain shared-parameter PCGrad, or its matched mean-gradient control.

Inspired by Yu et al., Gradient Surgery for Multi-Task Learning (2020),
Algorithm 1. Both projections use the original opposite gradient. The final
average fixes the overall scale across the projected and ordinary controls.
"""
import torch


def combine_pair(a,b,*,project):
    if a.ndim!=1 or b.shape!=a.shape or a.device!=b.device or a.dtype!=b.dtype or a.numel()==0:
        raise ValueError('Expected matching nonempty gradient vectors.')
    if a.dtype not in (torch.float32,torch.float64) or a.requires_grad or b.requires_grad:
        raise ValueError('Expected detached FP32/FP64 gradients.')
    dot=torch.dot(a,b)
    if project:
        negative=dot.clamp(max=0)
        a_scale=negative/torch.dot(b,b).clamp(min=1e-30)
        b_scale=negative/torch.dot(a,a).clamp(min=1e-30)
        result=.5*((a-a_scale*b)+(b-b_scale*a))
    else:
        result=.5*(a+b)
    return result,dot<0


def paired_gradients(model,training_views,rng,mode):
    from student_training_pairs import rgb_loss
    assert len(training_views)==46 and mode in ('mean','pcgrad')
    indices=[int(rng.integers(30)),30+int(rng.integers(16))]
    parameters=list(model.parameters())
    vectors,losses,pixels=[],[],[]
    for index in indices:
        source,target,_=training_views[index]
        loss,pixel=rgb_loss(model(source.contiguous(memory_format=torch.channels_last)),target)
        gradients=torch.autograd.grad(loss,parameters)
        vectors.append(torch.cat([g.flatten() for g in gradients]).detach())
        losses.append(loss.detach());pixels.append(pixel.detach())
    combined,conflict=combine_pair(*vectors,project=mode=='pcgrad')
    offset=0
    for parameter in parameters:
        parameter.grad=combined[offset:offset+parameter.numel()].view_as(parameter)
        offset+=parameter.numel()
    assert offset==combined.numel()
    return .5*(losses[0]+losses[1]),.5*(pixels[0]+pixels[1]),conflict
