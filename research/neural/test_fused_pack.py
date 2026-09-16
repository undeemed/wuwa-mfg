# SPDX-License-Identifier: Apache-2.0
"""Verify direct FP8 packing and fused half-rounded activation byte for byte."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from fused_norm import FusedNorm
from probe_block_sensitivity import SOURCE_COMMIT
from test_output_grade import graph_measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output path.')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Wrong reference source.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Modified reference source.')
    sys.path.insert(0,str(args.source/'python'))
    from mlxdlss.model import quadratic_gate_activation
    kernel=FusedNorm();torch.manual_seed(97323)
    records=[]
    def expected(x,activate):
        value=quadratic_gate_activation(x) if activate else x
        return value.clamp(-448,448).to(torch.float8_e4m3fn)
    def check(label,x):
        for activate in (False,True):
            original=expected(x,activate);actual=kernel.pack(x,activate=activate)
            unequal=int((original.contiguous().view(torch.uint8)!=actual.view(torch.uint8)).sum())
            records.append({'label':label,'shape':list(x.shape),'stride':list(x.stride()),
                'activate':activate,'elements':x.numel(),'unequal_bytes':unequal})
            if unequal:raise AssertionError((label,activate,unequal))
    with torch.inference_mode():
        all_half=np.arange(65536,dtype=np.uint16).view(np.float16)
        check('all finite FP16 bit patterns',torch.from_numpy(all_half[np.isfinite(all_half)].copy()).cuda())
        check('scalar negative zero',torch.tensor(-0.,dtype=torch.float16,device='cuda'))
        check('empty',torch.empty((0,32),device='cuda',dtype=torch.float16))
        for count in (1,2,3,511,512,513,1048579):
            check('partial pair or block '+str(count),torch.randn(count,device='cuda',dtype=torch.float16)*17)
        value=torch.randn((5,37,19),device='cuda',dtype=torch.float16)*5
        check('transposed and sliced',value.transpose(1,2)[::2,::2,1::2])
        check('broadcast',value[:1,:1,:1].expand(3,23,31))
        check('unaligned pair origin',value.flatten()[1:])
        check('eight dimensions',value[:1,:1,:1].reshape(1,1,1,1,1,1,1,1).expand(2,2,1,2,1,2,2,3))
        timings=[]
        for shape,activate in [((262144,32),False),((262144,128),True)]:
            value=torch.randn(shape,device='cuda',dtype=torch.float16)*3
            check('representative token chunk',value)
            prior=lambda x:((kernel.gate(x) if activate else x).clamp(-448,448).to(torch.float8_e4m3fn))
            fused=lambda x:kernel.pack(x,activate=activate)
            old_time,old_output=graph_measure(prior,value)
            new_time,new_output=graph_measure(fused,value)
            exact=torch.equal(old_output.view(torch.uint8),new_output.view(torch.uint8))
            if not exact:raise AssertionError('Graph changed packed operand bytes.')
            timings.append({'shape':list(shape),'activate':activate,'prior_graph':old_time,
                'fused_graph':new_time,'graph_output_matches_exactly':exact})
    report={'schema':1,'passed':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'source_commit':SOURCE_COMMIT,'tests':records,'total_values':sum(x['elements'] for x in records),
        'graph_timings':timings,
        'scope':'All finite FP16 scalar values and representative layouts; includes signed-zero bytes and saturated gate overflow. No native model speed or quality claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Passed',len(records),'cases,',report['total_values'],'values.')
    for item in timings:print(item['shape'],item['activate'],item['prior_graph']['median_ms'],item['fused_graph']['median_ms'])


if __name__=='__main__':main()
