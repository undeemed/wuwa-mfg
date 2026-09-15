# SPDX-License-Identifier: Apache-2.0
"""Check decoder skip/conditioning fusion without changing FP16 rounding."""
import numpy as np
import torch
from torch.nn import functional as F


def operator_tests(kernel):
    tests=[]
    def check(label,x,k,s,b):
        expected=(F.interpolate(x,size=k.shape[-2:],mode='nearest')+k)*(1+s)+b
        actual=kernel.decoder_conditioned_add(x,k,s,b)
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),label
        tests.append({'label':label,'shape':list(k.shape),'input_shape':list(x.shape),
                      'strides':[list(t.stride()) for t in (x,k,s,b)],'bit_equal':True})
    with torch.inference_mode():
        for channels in (3,16,32,64,128):
            for ratio in (1,2):
                k=torch.randn((2,channels,18,30),device='cuda',dtype=torch.float16)
                x=torch.randn((2,channels,18//ratio,30//ratio),device='cuda',dtype=torch.float16)
                coeff=torch.randn((2,channels*4,1,1),device='cuda',dtype=torch.float16)
                s,b=coeff[:,:channels],coeff[:,channels:2*channels]
                for layout in ('planar','channels-last','strided'):
                    tensors=(x,k,s,b)
                    if layout=='channels-last':tensors=tuple(t.contiguous(memory_format=torch.channels_last) for t in (x,k))+(s,b)
                    elif layout=='strided':tensors=tuple(torch.stack((t,t),dim=-1)[...,1] for t in tensors)
                    check(f'{layout}-c{channels}-r{ratio}',*tensors)
        for channels,height,width in ((16,272,480),(32,272,480),(128,68,120)):
            k=torch.randn((1,channels,height,width),device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
            x=k[:,:,::2,::2].contiguous(memory_format=torch.channels_last)
            s=torch.randn((1,channels,1,1),device='cuda',dtype=torch.float16)
            check('decoder-extent',x,k,s,-s)
        bits=np.arange(65536,dtype=np.uint16).view(np.float16)
        values=torch.from_numpy(bits[np.isfinite(bits)].copy()).cuda()
        for channels in (16,32,64,128):
            count=((values.numel()+2*channels-1)//(2*channels))*(2*channels)
            packed=values.repeat(2)[:count]
            k=packed.reshape(1,1,-1,channels).permute(0,3,1,2)
            x=k.roll(37,dims=-1).contiguous(memory_format=torch.channels_last)
            s=torch.tensor([.1,-1.,-.9995,65504.,-65504.,0.,-0.,.00000011920928955078125],device='cuda',dtype=torch.float16).repeat(channels//8).reshape(1,channels,1,1)
            b=s.roll(3,dims=1)
            check('dense-finite-half-patterns',x,k,s,b)
        s=values.reshape(1,-1,1,1);k=torch.ones((1,values.numel(),1,2),device='cuda',dtype=torch.float16)
        check('all-finite-coefficients',k,-k,s,s.roll(5,dims=1))
        x=torch.randn((1,16,6,10),device='cuda',dtype=torch.float16)
        s=torch.randn((1,16,1,1),device='cuda',dtype=torch.float16)
        check('broadcast-strides',x[:,:1].expand_as(x),x,s[:,:1].expand_as(s),s)
        check('batch-broadcast-coefficients',x.expand(2,-1,-1,-1),x.expand(2,-1,-1,-1),s.expand(2,-1,-1,-1),s.expand(2,-1,-1,-1))
        raw=torch.randn((16*6*10+1,),device='cuda',dtype=torch.float16)
        odd=raw[1:].reshape(1,6,10,16).permute(0,3,1,2)
        check('misaligned-packed-input',odd,odd,s,s)
        check('odd-spatial-same-size',x[:,:,:5,:9],x[:,:,:5,:9],s,s)
    bad=[('dtype',x.float(),x,s,s),('rank',x[0],x,s,s),('extent',x[:,:,:,:3],x,s,s),
         ('scale-shape',x,x,s[:,:1],s),('shift-shape',x,x,s,s[:,:1]),
         ('cpu',x.cpu(),x.cpu(),s.cpu(),s.cpu()),('empty',x[:0],x[:0],s[:0],s[:0]),
         ('autograd',x,x,torch.ones_like(s,requires_grad=True),s)]
    rejected=[]
    for label,*tensors in bad:
        try:kernel.decoder_conditioned_add(*tensors)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label+' accepted')
    return {'tests':tests,'rejected_inputs':rejected,
            'scope':'Finite half patterns in selected operand combinations, not every possible combination. Explicit round-to-nearest per operation.'}
