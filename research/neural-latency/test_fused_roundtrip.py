# SPDX-License-Identifier: Apache-2.0
"""Test direct strided FP8 roundtrip, saturation and signed zero behavior."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
import torch
from fused_norm import FusedNorm
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a fresh output path.')
kernel=FusedNorm();torch.manual_seed(18982);records=[]
def previous(x):return x.clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype)
def check(label,x):
    y=previous(x);z=kernel.roundtrip(x)
    equal=(y==z)|(torch.isnan(y)&torch.isnan(z))
    bitdtype=torch.int16 if x.dtype==torch.float16 else torch.int32
    records.append({'label':label,'dtype':str(x.dtype),'shape':list(x.shape),'elements':x.numel(),
        'unequal':int((~equal).sum()),'non_nan_bit_mismatches':int(((y.view(bitdtype)!=z.view(bitdtype))&~torch.isnan(y)).sum())})
with torch.inference_mode():
    all_half=np.arange(65536,dtype=np.uint16).view(np.float16)
    finite=torch.from_numpy(all_half[np.isfinite(all_half)].copy()).cuda()
    check('all finite half patterns',finite);check('half values as float',finite.float())
    for dtype in (torch.float16,torch.float32):
        for factor in (.0001,.1,1.,16.,1000.):check('random '+str(factor),torch.randn(1048591,device='cuda',dtype=dtype)*factor)
        backing=torch.randn(3,8,67,64,device='cuda',dtype=dtype)
        check('slice offset',backing[:,1::2,::3,1::2]);check('permuted',backing.permute(2,1,0,3))
        check('zero strides',backing[:1,:1,:1,:].expand(3,8,67,64))
        check('specials',torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),448.,449.,464.,480.],device='cuda',dtype=dtype))
        check('scalar',torch.tensor(-0.,device='cuda',dtype=dtype));check('empty',torch.empty(0,32,device='cuda',dtype=dtype))
    timings={};x=torch.randn(1,1088,1920,32,device='cuda',dtype=torch.float16)
    for name,fn in [('separate_clamp_casts',previous),('fused_roundtrip',kernel.roundtrip)]:
        warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):fn(x)
        torch.cuda.current_stream().wait_stream(warm);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(x)
        for _ in range(5):graph.replay()
        torch.cuda.synchronize();samples=[]
        for _ in range(20):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end))
        timings[name]={'median_ms':statistics.median(samples),'samples_ms':samples}
report={'schema':1,'all_equal':all(not r['unequal'] for r in records),
        'all_non_nan_bits_equal':all(not r['non_nan_bit_mismatches'] for r in records),
        'total_elements':sum(r['elements'] for r in records),'tests':records,'timing_shape':list(x.shape),'timings':timings,
        'scope':'Isolated CUDA Graph operation, not full model or NVIDIA runtime latency.'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))
if not report['all_equal'] or not report['all_non_nan_bits_equal']:raise SystemExit('Rejected: numerical mismatch.')
