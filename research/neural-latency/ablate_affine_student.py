# SPDX-License-Identifier: Apache-2.0
"""Disable learned branches after training to diagnose an affine student.

No retraining or speed measurement. Disabled branches are still computed and
discarded; these comparisons cannot demonstrate an optimized renderer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from compare_output_grade import read_capture
from fused_norm import FusedNorm
from output_grade import GradedStudent
from student_probe import HierarchicalStudent


class BranchProbe:
    def __init__(self,backend):
        self.backend=backend
        self.mode='both'

    def affine_compose(self,source,detail,field):
        if self.mode in ('affine-only','neither'):detail=torch.zeros_like(detail)
        if self.mode in ('detail-only','neither'):field=torch.zeros_like(field)
        return self.backend.affine_compose(source,detail,field)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student',type=Path,required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,metavar=('LABEL','CAPTURE','ROLE'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    trained=json.loads((args.student/'result.json').read_text())
    checkpoint_path=args.student/'student-private.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    architecture=checkpoint['architecture']
    if architecture!=trained['architecture'] or architecture['variant']!='hierarchical-affine' or architecture['noise_channels']!=0:
        raise ValueError('Expected the matched, noise-free affine student checkpoint.')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    network=HierarchicalStudent(architecture['width'],architecture['blocks'],False,True)
    model=GradedStudent(network,architecture['explicit_output_grading'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model=model.cuda().half().eval().to(memory_format=torch.channels_last)
    backend=FusedNorm();probe=BranchProbe(backend)
    model.fused_backend=backend;model.network.fused_affine_backend=probe
    report={'schema':1,'target_achieved':False,'quality_gate_passed':False,
            'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'cases':[],'scope':'Post-training branch ablation only; no retraining or latency claim.'}
    with torch.inference_mode():
        for label,capture,role in args.case:
            if role=='primary':hashes=trained['capture_hashes'];filename='student-output.npy'
            elif role=='validation':hashes=trained['validation_capture_hashes'];filename='validation-output.npy'
            elif role=='extra-0':hashes=trained['extra_validation'][0]['capture_hashes'];filename='extra-validation-0.npy'
            else:raise ValueError('Supported roles are primary, validation and extra-0.')
            controls,images,actual_hashes=read_capture(Path(capture))
            if controls!=trained['controls'] or actual_hashes!=hashes:
                raise ValueError('Capture does not match the trained experiment.')
            source=torch.from_numpy(images['color']).permute(2,0,1)[None].cuda().half().contiguous(memory_format=torch.channels_last)
            target=torch.from_numpy(images['output']).permute(2,0,1)[None].cuda()
            metrics={}
            for mode in ('both','affine-only','detail-only','neither'):
                probe.mode=mode;output=model(source)
                if mode=='both':
                    saved=np.load(args.student/filename,allow_pickle=False)
                    actual=output[0].permute(1,2,0).float().cpu().numpy()
                    if not np.array_equal(saved,actual):raise ValueError('Reloaded student differs from saved output.')
                error=(output.float()-target).abs()
                metrics[mode]={'mae':float(error.mean()),'rmse':float(error.square().mean().sqrt()),
                               'p99':float(torch.quantile(error.flatten(),.99)),'max_abs':float(error.max())}
            report['cases'].append({'label':label,'role':role,'capture_hashes':hashes,
                'saved_output_reproduced_exactly':True,'metrics':metrics})
            print(label,{k:v['mae'] for k,v in metrics.items()},flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
