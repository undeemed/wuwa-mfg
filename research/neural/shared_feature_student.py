# SPDX-License-Identifier: Apache-2.0
"""Frozen student features feeding an original correction decoder.

Inspired by cross-stage feature exchange, not an implementation of MPRNet.
No native teacher feature, previous-frame cache or second encoder is required.
"""
import torch
from torch import nn
from torch.nn import functional as F

from decoder_conditioning import DecoderConditioning
from output_grade import grade_torch
from student_probe import ResidualBlock
from test_student_output import configure


class FeatureCorrectionDecoder(nn.Module):
    def __init__(self, grade_parameters):
        super().__init__()
        widths = (16, 32, 64, 96)
        self.deep = nn.Conv2d(96, 96, 1)
        self.skip = nn.ModuleList(nn.Conv2d(2*w, w, 1) for w in widths[:3])
        self.up = nn.ModuleList(nn.Conv2d(widths[i+1], widths[i], 1) for i in range(3))
        self.blocks = nn.ModuleList(nn.Sequential(ResidualBlock(w), ResidualBlock(w)) for w in widths[:3])
        self.local = nn.Conv2d(48, 16, 1)
        self.conditioning = DecoderConditioning(96, widths[:3])
        self.head = nn.Conv2d(16, 48, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.grade_parameters = tuple(grade_parameters)
        self.fused_backend = None

    def forward(self, source, base, features):
        if source.ndim != 4 or source.shape[1] != 3 or source.shape != base.shape:
            raise ValueError('Expected matching NCHW RGB input and ungraded base prediction.')
        if len(features) != 7 or any(t.dtype != source.dtype or t.device != source.device for t in (base, *features)):
            raise ValueError('Expected seven first-stage features with matching device and precision.')
        height, width = source.shape[-2:]
        expected_hw = ((height+31)//32*8, (width+31)//32*8)
        for index, (channels, scale) in enumerate(((16,1),(32,2),(64,4),(96,8),(64,4),(32,2),(16,1))):
            if features[index].shape != (source.shape[0], channels, expected_hw[0]//scale, expected_hw[1]//scale):
                raise ValueError('Feature shape does not match the width-16 hierarchy.')
        encoded, decoded = features[:4], (features[6], features[5], features[4])
        modulation = self.conditioning(encoded[3].mean(dim=(2,3), keepdim=True))
        value = self.deep(encoded[3])
        for index in (2, 1, 0):
            skip = self.skip[index](torch.cat((encoded[index], decoded[index]), dim=1))
            value = F.interpolate(self.up[index](value), size=skip.shape[-2:], mode='nearest') + skip
            if index == 0:
                padded = F.pad(source, (0,(-width)%32,0,(-height)%32), mode='reflect')
                value = value + self.local(F.pixel_unshuffle(padded, 4))
            scale, shift = self.conditioning.stage(modulation, index)
            value = self.blocks[index](value*(1+scale)+shift)
        head = self.head(value)
        if self.fused_backend is not None:
            return self.fused_backend.student_output(base, head, self.grade_parameters)
        residual = F.pixel_shuffle(head, 4)[:, :, :height, :width]
        return grade_torch((base + .25*residual).clamp(0,1), self.grade_parameters)


class SharedFeatureRefinement(nn.Module):
    def __init__(self, first):
        super().__init__()
        if first.network.head.in_channels != 16 or first.network.decoder_conditioning is None:
            raise ValueError('Expected the width-16 conditioned first student.')
        self.first = first.requires_grad_(False).eval()
        self.refinement = FeatureCorrectionDecoder(first.grade_parameters)

    def train(self, mode=True):
        super().train(mode)
        self.first.eval()
        return self

    def extract(self, source):
        features = []
        with torch.no_grad():
            base = self.first.network(source, feature_sink=features)
        return base, tuple(features)

    def forward(self, source):
        base, features = self.extract(source)
        return self.refinement(source, base, features)


def configure_shared(model, mode, backend):
    if mode not in ('baseline', 'all-conditioning'):
        raise ValueError('Unsupported shared-feature inference configuration.')
    model.first.fused_backend = backend
    configure(model.first, mode, backend)
    model.refinement.fused_backend = backend if mode == 'all-conditioning' else None
    for layer in model.refinement.modules():
        if isinstance(layer, ResidualBlock):
            layer.fused_residual_backend = backend if mode == 'all-conditioning' else None
