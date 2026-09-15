# SPDX-License-Identifier: Apache-2.0
"""Differentiable observed output grade and an optional fused inference path.

This preserves our algebraic grading approximation, not native instruction-level
rounding. Controls are obtained from the validated, pinned output contract.
"""
import numpy as np
import torch
from torch import nn

from compare_output_grade import read_contract


def contract_parameters(trial):
    fields=read_contract(trial)['float_by_offset']
    return (float(np.exp2(np.float32(fields['324']))),
            float(np.float32(fields['332'])),
            float(np.float32(1)+np.float32(fields['336'])))


def grade_torch(image, parameters):
    if image.ndim!=4 or image.shape[1]!=3:
        raise ValueError('Expected NCHW RGB.')
    exposure,contrast,saturation=parameters
    value=image.float().clamp(0,1)
    value=(value*exposure).clamp(0,1)
    delta=value*value*(3-2*value)-value
    value=(value+contrast*delta).clamp(0,1)
    lightness=(value.amax(dim=1,keepdim=True)+value.amin(dim=1,keepdim=True))*.5
    return (lightness+saturation*(value-lightness)).clamp(0,1).to(image.dtype)


class GradedStudent(nn.Module):
    def __init__(self,network,parameters):
        super().__init__()
        self.network=network
        # Python numbers remain FP32-valued when the network is converted to half.
        self.grade_parameters=tuple(parameters)
        self.fused_backend=None
        self.fused_student_output=False

    def forward(self,image):
        if self.fused_student_output:
            if self.fused_backend is None:
                raise ValueError('Fused student output requires an inference backend.')
            return self.network(image,fused_output_backend=self.fused_backend,grade_parameters=self.grade_parameters)
        value=self.network(image)
        if self.fused_backend is not None:
            return self.fused_backend.output_grade(value,self.grade_parameters)
        return grade_torch(value,self.grade_parameters)
