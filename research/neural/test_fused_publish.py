# SPDX-License-Identifier: Apache-2.0
"""Check fused strided normalization, head scaling and E4M3 publication."""
import argparse,json,statistics,sys
from pathlib import Path
import torch
from fused_norm import FusedNorm

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Use a fresh output path.')
sys.path.insert(0,str(a.source/'python'))
from mlxdlss.model import vendor_cosine_publish
kernel=FusedNorm();torch.manual_seed(84277)
records=[]
def check(label,x,scale):
    expected=vendor_cosine_publish(x,scale);actual=kernel.publish(x,scale)
    equal=(expected==actual)|(torch.isnan(expected)&torch.isnan(actual))
    finite=torch.isfinite(expected)&torch.isfinite(actual)
    records.append({'label':label,'dtype':str(x.dtype),'shape':list(x.shape),'stride':list(x.stride()),
                    'scaled':scale is not None,'elements':x.numel(),'unequal':int((~equal).sum()),
                    'max_abs_finite':float((expected[finite]-actual[finite]).abs().max()) if finite.any() else 0.})
with torch.inference_mode():
    for dtype in (torch.float16,torch.float32):
        for heads in (1,2,4,8,16):
            raw=torch.randn(3,67,heads*32*3,device='cuda',dtype=dtype)
            x=raw[...,:heads*32].reshape(3,67,heads,32).permute(0,2,1,3)
            scale=torch.linspace(-2,16,heads,device='cuda',dtype=dtype)
            check('strided QKV view',x,scale);check('unscaled key',x,None)
        for magnitude in (0.,.0001,.1,16.,100.):
            x=torch.randn(1,4,19,64,device='cuda',dtype=dtype)[...,::2]*magnitude
            check('magnitude '+str(magnitude),x,torch.tensor([0.,.125,448.,65504.],device='cuda',dtype=dtype))
        backing=torch.randn(3,8,17,64,device='cuda',dtype=dtype)
        check('strided channels',backing[...,::2],torch.arange(16,device='cuda',dtype=dtype)[::2])
    raw=torch.randn(256,64,3*256,device='cuda',dtype=torch.float16)
    x=raw[...,:256].reshape(256,64,8,32).permute(0,2,1,3)
    scale=torch.arange(1,9,device='cuda',dtype=torch.float16)
    def previous(value):
        normalized=kernel(value).half()
        normalized=(normalized*scale.reshape(1,8,1,1)).half()
        return normalized.clamp(-448,448).to(torch.float8_e4m3fn).to(value.dtype)
    timings={}
    for name,fn in [('previous_fused_norm_plus_separate_scale_fp8',previous),('fused_publish',lambda v:kernel.publish(v,scale))]:
        warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):fn(x)
        torch.cuda.current_stream().wait_stream(warm)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(x)
        for _ in range(5):graph.replay()
        torch.cuda.synchronize();samples=[]
        for _ in range(30):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end))
        timings[name]={'median_ms':statistics.median(samples),'samples_ms':samples}
    report={'schema':1,'tests':records,'all_equal':all(r['unequal']==0 for r in records),
            'total_elements':sum(r['elements'] for r in records),'timing_shape':list(x.shape),'timings':timings,
            'scope':'CUDA Graph intervals in reconstruction only, not vendor or game latency.'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))
if not report['all_equal']:raise SystemExit('Rejected: numerical mismatch.')
