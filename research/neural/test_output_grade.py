# SPDX-License-Identifier: Apache-2.0
"""Check fused grading against Torch and compare full-1080p graph intervals.

Finite-input arithmetic test of our approximation, not native output parity.
"""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from fused_norm import FusedNorm
from output_grade import contract_parameters,grade_torch


def graph_measure(function,image):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):function(image)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=function(image)
    for _ in range(5):graph.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(30):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();graph.replay();end.record();end.synchronize()
        samples.append(begin.elapsed_time(end))
    return {'median_ms':statistics.median(samples),'p95_ms':float(np.percentile(samples,95)),
            'samples_ms':samples},output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    parameters=contract_parameters(args.contract_trial)
    fused=FusedNorm();torch.manual_seed(37021)
    records=[]
    def check(label,image,controls=parameters):
        expected=grade_torch(image,controls);actual=fused.output_grade(image,controls)
        delta=(actual.float()-expected.float()).abs()
        records.append({'label':label,'shape':list(image.shape),'stride':list(image.stride()),
                        'dtype':str(image.dtype),'parameters':list(controls),
                        'unequal':int((actual!=expected).sum()),'max_abs':float(delta.max()),
                        'elements':image.numel()})
    with torch.inference_mode():
        for dtype in (torch.float16,torch.float32):
            image=torch.randn((2,3,19,37),device='cuda',dtype=dtype)*.7+.5
            check('partial block planar',image)
            check('partial block interleaved',image.contiguous(memory_format=torch.channels_last))
            check('sliced and transposed',image[:,:,::2,1::2].transpose(2,3))
            check('broadcast channel and batch',image[:1,:1].expand(3,3,19,37))
            for controls in ((1.,0.,1.),(2.,1.,0.),(.25,-1.,.5)):
                check('parameter boundaries',image,controls)
        values=np.arange(65536,dtype=np.uint16).view(np.float16)
        values=torch.from_numpy(values[np.isfinite(values)].copy()).cuda()
        colors=torch.stack((values,values.roll(1),values.roll(177))).reshape(1,3,1,-1)
        check('all finite half patterns in mixed RGB',colors)
        check('finite half patterns promoted to float',colors.float())
        image=torch.randn((1,3,1080,1920),device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)*.4+.5
        check('full 1080p',image)
        timings={}
        for name,function in [('torch',lambda x:grade_torch(x,parameters)),
                              ('fused',lambda x:fused.output_grade(x,parameters))]:
            timings[name],output=graph_measure(function,image)
        graph_exact=torch.equal(output,grade_torch(image,parameters))
    report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'parameters':list(parameters),'tests':records,
            'all_equal':all(r['unequal']==0 for r in records),'graph_matches_torch':graph_exact,
            'timing_shape':list(image.shape),'graph_timings':timings,
            'scope':'Finite-input grading approximation only; not native bit parity or a complete renderer benchmark.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2),flush=True)
    if not report['all_equal'] or not graph_exact:raise SystemExit('Fused grade mismatch.')


if __name__=='__main__':main()
