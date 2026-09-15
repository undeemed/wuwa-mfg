# SPDX-License-Identifier: Apache-2.0
"""Original frozen-base residual refinement; not an implementation of MPRNet."""
import torch
from torch import nn
from student_probe import HierarchicalStudent
from output_grade import GradedStudent


class FrozenStudentRefinement(nn.Module):
    def __init__(self,first):
        super().__init__()
        if first.network.head.in_channels!=16 or first.network.decoder_conditioning is None:
            raise ValueError('Expected the width-16 conditioned base student.')
        self.first=first.requires_grad_(False).eval()
        network=HierarchicalStudent(16,2,conditioned=True)
        # Two full-resolution RGB images are reversibly rearranged into channels.
        network.stem=nn.Conv2d(96,16,1)
        self.refinement=GradedStudent(network,first.grade_parameters)

    def train(self,mode=True):
        super().train(mode);self.first.eval();return self

    def first_ungraded(self,source):
        with torch.no_grad():return self.first.network(source)

    def refine_cached(self,first_ungraded,source):
        if source.ndim!=4 or source.shape[1]!=3 or first_ungraded.shape!=source.shape:
            raise ValueError('Expected matching NCHW RGB source and ungraded first output.')
        if first_ungraded.dtype!=source.dtype or first_ungraded.device!=source.device:
            raise ValueError('Both stages must use the same dtype and device.')
        combined=torch.cat((first_ungraded,source),dim=1).contiguous(memory_format=torch.channels_last)
        return self.refinement(combined)

    def forward(self,source):return self.refine_cached(self.first_ungraded(source),source)


def configure_progressive(model,mode,kernel):
    from test_student_output import configure
    for stage in (model.first,model.refinement):
        stage.fused_backend=kernel
        configure(stage,mode,kernel)
    # The first stage is deliberately called before grading. Its unfused final
    # shuffle/clamp remains included; the second stage applies the grade once.


def load_first(directory):
    from student_training_pairs import read
    record=read(directory/'result.json');a=record['architecture']
    if a['variant']!='hierarchical-film' or a['width']!=16 or a['blocks']!=2:
        raise ValueError('Unsupported base architecture.')
    first=GradedStudent(HierarchicalStudent(16,2,conditioned=True),a['explicit_output_grading'])
    checkpoint=torch.load(directory/'student-private.pt',map_location='cpu',weights_only=True)
    assert checkpoint['architecture']==a
    first.load_state_dict(checkpoint['state_dict'],strict=True)
    return first,record
