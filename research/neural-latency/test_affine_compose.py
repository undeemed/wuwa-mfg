# SPDX-License-Identifier: Apache-2.0
"""Check the fused FP16 affine/detail composition against its Torch reference."""
import argparse
import json
from pathlib import Path

import torch

from affine_student import compose_affine
from fused_norm import FusedNorm
from test_output_grade import graph_measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.manual_seed(72315);fused=FusedNorm();records=[]
    def check(label,source,detail,field):
        expected=compose_affine(source,detail,field)
        actual=fused.affine_compose(source,detail,field)
        delta=(expected.float()-actual.float()).abs()
        records.append({'label':label,'shape':list(source.shape),'field_shape':list(field.shape),
                        'strides':[list(x.stride()) for x in (source,detail,field)],
                        'elements':source.numel(),'unequal':int((expected!=actual).sum()),
                        'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
                        'finite':bool(torch.isfinite(actual).all())})
    with torch.inference_mode():
        for n,h,w in [(1,1,1),(1,19,37),(2,65,97),(1,128,192),(1,1080,1920)]:
            source=torch.rand((n,3,h,w),device='cuda',dtype=torch.float16)
            detail=torch.randn_like(source)*.2
            field=torch.randn((n,12,(h+31)//32,(w+31)//32),device='cuda',dtype=torch.float16)*.3
            check('planar',source,detail,field)
            check('interleaved',*(x.contiguous(memory_format=torch.channels_last) for x in (source,detail,field)))
            check('zero field',source,detail,torch.zeros_like(field))
        source=torch.rand((2,3,130,194),device='cuda',dtype=torch.float16)[:,:,::2,::2]
        detail=torch.randn((2,3,97,65),device='cuda',dtype=torch.float16).transpose(2,3)*.1
        field=torch.randn((2,12,4,3),device='cuda',dtype=torch.float16).transpose(2,3)*.1
        check('strided source detail and field',source,detail,field)
        check('broadcast batch and coefficients',source[:1].expand(2,-1,-1,-1),detail,
              field[:1,:1].expand(2,12,-1,-1))
        source=torch.randn((1,3,63,95),device='cuda',dtype=torch.float16)*.6+.5
        detail=torch.randn_like(source)*.5
        field=torch.randn((1,12,2,3),device='cuda',dtype=torch.float16)*.5
        check('input undershoot overshoot and output clipping',source,detail,field)
        source=torch.rand((1,3,1080,1920),device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        detail=torch.randn_like(source)*.2
        field=torch.randn((1,12,34,60),device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)*.3
        inputs=(source,detail,field);timings={};outputs={}
        for label,function in [('torch',lambda xs:compose_affine(*xs)),('fused',lambda xs:fused.affine_compose(*xs))]:
            timings[label],outputs[label]=graph_measure(function,inputs)
        graph_equal=bool(torch.equal(outputs['torch'],outputs['fused']))
    report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'all_equal':all(r['unequal']==0 and r['finite'] for r in records),
            'graph_equal':graph_equal,'tests':records,'graph_timings':timings,
            'quality_gate_passed':False,'target_achieved':False,
            'scope':'Finite FP16 affine/detail composition only; not a native renderer benchmark or model quality test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','graph_timings')},indent=2))
    print({k:v['median_ms'] for k,v in timings.items()},flush=True)
    if not report['all_equal'] or not graph_equal:
        print([r for r in records if r['unequal']],flush=True)
        raise SystemExit('Affine composition mismatch.')


if __name__=='__main__':main()
