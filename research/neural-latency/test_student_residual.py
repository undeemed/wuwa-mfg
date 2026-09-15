# SPDX-License-Identifier: Apache-2.0
"""Finite-pattern and layout checks for the inference-only residual fusion."""
import numpy as np
import torch


def operator_tests(kernel):
    tests=[]
    def check(label,x,r,s):
        expected=x+r*s
        actual=kernel.residual_scale_add(x,r,s)
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),label
        tests.append({'label':label,'shape':list(x.shape),'input_stride':list(x.stride()),
                      'residual_stride':list(r.stride()),'scale_stride':list(s.stride()),'bit_equal':True})
    with torch.inference_mode():
        for shape in [(2,3,17,29),(1,16,272,480),(1,96,34,60)]:
            x=torch.randn(shape,device='cuda',dtype=torch.float16)
            r=torch.randn_like(x)*4
            s=torch.randn((1,shape[1],1,1),device='cuda',dtype=torch.float16)*.3
            for layout in ['planar','channels-last','strided']:
                a,b,c=x,r,s
                if layout=='channels-last':a,b=(t.contiguous(memory_format=torch.channels_last) for t in (x,r))
                elif layout=='strided':a,b,c=(torch.stack((t,t),dim=-1)[...,1] for t in (x,r,s))
                check(layout,a,b,c)
        bits=np.arange(65536,dtype=np.uint16).view(np.float16)
        values=torch.from_numpy(bits[np.isfinite(bits)].copy()).cuda()
        x=torch.stack((values,values.roll(5),values.roll(97))).reshape(1,3,1,-1)
        r=x.roll(113,dims=-1)
        for scales in [[.1,-.25,1.],[0.,-0.,65504.],[-65504.,.00006103515625,-1.]]:
            s=torch.tensor(scales,device='cuda',dtype=torch.float16).reshape(1,3,1,1)
            check('finite-half-patterns',x,r,s)
        check('broadcast',x[:,:1].expand_as(x),r[:,:1].expand_as(r),s[:,:1].expand_as(s))
        for channels in (16,32,64,96,128,192):
            count=((values.numel()+channels-1)//channels)*channels
            packed=values.repeat(2)[:count]
            x=packed.reshape(1,1,-1,channels).permute(0,3,1,2)
            r=packed.roll(113).reshape(1,1,-1,channels).permute(0,3,1,2)
            scales=torch.tensor([.1,-.25,1.,0.,-0.,65504.,-65504.,.00006103515625],device='cuda',dtype=torch.float16)
            s=scales.repeat(channels//8).reshape(1,channels,1,1)
            check('dense-finite-half-patterns',x,r,s)
        # An odd half-element offset must use the generic scalar fallback.
        shape=(1,16,4,8)
        storage=torch.randn((16*4*8+1,),device='cuda',dtype=torch.float16)
        x=storage[1:].reshape(1,4,8,16).permute(0,3,1,2)
        check('misaligned-dense-view',x,x,s[:,:16])
        x=torch.zeros((1,3,4,8),device='cuda',dtype=torch.float16)
        s=torch.ones((1,3,1,1),device='cuda',dtype=torch.float16)
    bad=[('dtype',x.float(),x,s),('rank',x[0],x,s),('extent',x,x[:,:,:,:2],s),
         ('scale-shape',x,x,s[:,:1]),('cpu',x.cpu(),x.cpu(),s.cpu()),
         ('empty',x[:0],x[:0],s),('autograd',x,x,torch.ones_like(s,requires_grad=True))]
    rejected=[]
    for label,a,b,c in bad:
        try:kernel.residual_scale_add(a,b,c)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label+' was accepted')
    return {'tests':tests,'rejected_inputs':rejected,
            'scope':'Finite FP16 patterns in selected operand combinations, not every possible triple.'}
