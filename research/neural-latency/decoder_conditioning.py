# SPDX-License-Identifier: Apache-2.0
"""Image-conditioned decoder features with an initially neutral affine branch.

FiLM supplies the feature-wise conditioning idea; zero initialization preserves
the starting backbone. This is not a diffusion or language-conditioned model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DecoderConditioning(nn.Module):
    def __init__(self,context_width,decoder_widths):
        super().__init__()
        self.widths=tuple(decoder_widths)
        self.norm=nn.LayerNorm(context_width)
        self.hidden=nn.Linear(context_width,context_width)
        self.project=nn.Linear(context_width,2*sum(self.widths))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self,pooled):
        features=pooled.flatten(1)
        coefficients=self.project(F.gelu(self.hidden(self.norm(features))))
        return coefficients[:,:,None,None]

    def stage(self,coefficients,index):
        start=2*sum(self.widths[:index]);end=start+2*self.widths[index]
        return coefficients[:,start:end].chunk(2,dim=1)
