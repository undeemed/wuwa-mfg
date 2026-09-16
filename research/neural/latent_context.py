# SPDX-License-Identifier: Apache-2.0
"""Bounded spatial context exchange inspired by Perceiver IO's latent interface.

This is an original small CNN branch, not a replication of its full models.
Source pixels and local skips remain in the surrounding network.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LatentContext(nn.Module):
    def __init__(self,width,latents=32,head_dim=32):
        super().__init__()
        if width not in (96,192) or head_dim!=32 or latents!=32:
            raise ValueError('This experiment supports widths 96/192, 32 latents and head dimension 32.')
        self.width=width;self.heads=width//head_dim;self.head_dim=head_dim
        self.latents=nn.Parameter(torch.randn(1,latents,width)*.02)
        self.input_norm=nn.LayerNorm(width)
        self.position=nn.Linear(2,width,bias=False)
        self.read_norm=nn.LayerNorm(width)
        self.read_q=nn.Linear(width,width)
        self.read_kv=nn.Linear(width,2*width)
        self.read_project=nn.Linear(width,width)
        self.process_norm=nn.LayerNorm(width)
        self.process_qkv=nn.Linear(width,3*width)
        self.process_project=nn.Linear(width,width)
        self.mlp_norm=nn.LayerNorm(width)
        self.mlp=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.write_norm=nn.LayerNorm(width)
        self.write_q=nn.Linear(width,width)
        self.write_kv=nn.Linear(width,2*width)
        self.project=nn.Linear(width,width)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
        self._coordinate_key=None;self._coordinates=None

    def attention(self,q,k,v):
        def heads(t):return t.reshape(t.shape[0],t.shape[1],self.heads,self.head_dim).transpose(1,2)
        result=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),dropout_p=0.0)
        return result.transpose(1,2).reshape(q.shape[0],q.shape[1],self.width)

    def forward(self,features):
        if features.ndim!=4 or features.shape[1]!=self.width:
            raise ValueError('Expected NCHW features with the configured channel width.')
        batch,channels,height,width=features.shape
        if batch<1 or height<1 or width<1 or height*width>8192:
            raise ValueError('Feature extent exceeds the bounded experiment.')
        key=(height,width,features.device,features.dtype)
        if key!=self._coordinate_key:
            # Keep reusable constants ordinary even when first called in inference mode.
            with torch.inference_mode(False):
                y,x=torch.meshgrid(torch.linspace(-1,1,height,device=features.device,dtype=features.dtype),
                                   torch.linspace(-1,1,width,device=features.device,dtype=features.dtype),indexing='ij')
                self._coordinates=torch.stack((x,y),dim=-1).reshape(1,height*width,2)
            self._coordinate_key=key
        tokens=features.flatten(2).transpose(1,2)
        encoded=self.input_norm(tokens)+self.position(self._coordinates)
        latent=self.latents.expand(batch,-1,-1)
        k,v=self.read_kv(encoded).chunk(2,dim=-1)
        latent=latent+self.read_project(self.attention(self.read_q(self.read_norm(latent)),k,v))
        q,k,v=self.process_qkv(self.process_norm(latent)).chunk(3,dim=-1)
        latent=latent+self.process_project(self.attention(q,k,v))
        latent=latent+self.mlp(self.mlp_norm(latent))
        k,v=self.write_kv(self.write_norm(latent)).chunk(2,dim=-1)
        correction=self.project(self.attention(self.write_q(encoded),k,v))
        correction=correction.transpose(1,2).reshape_as(features)
        return features+correction*.1
