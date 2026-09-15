# SPDX-License-Identifier: Apache-2.0
"""Check half-rounded quadratic activation, including every finite FP16 input."""
import argparse,json,statistics,sys
from pathlib import Path
import numpy as np
import torch
from fused_norm import FusedNorm

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a fresh output file.')
sys.path.insert(0,str(a.source/'python'))
from mlxdlss.model import quadratic_gate_activation
kernel=FusedNorm()
torch.manual_seed(12809)
records=[]
def check(label,x):
    expected=quadratic_gate_activation(x);actual=kernel.gate(x)
    equal=(expected==actual)|(torch.isnan(expected)&torch.isnan(actual))
    finite=torch.isfinite(expected)&torch.isfinite(actual)
    records.append({'label':label,'dtype':str(x.dtype),'elements':x.numel(),
        'unequal':int(torch.count_nonzero(~equal)),
        'max_finite_abs':float((expected[finite]-actual[finite]).abs().max()) if finite.any() else 0.,
        'matching_nonfinite_outputs':int(torch.count_nonzero(equal&~finite))})
with torch.inference_mode():
    all_half=np.arange(65536,dtype=np.uint16).view(np.float16)
    half=torch.from_numpy(all_half[np.isfinite(all_half)].copy()).cuda()
    check('all finite FP16 bit patterns',half)
    check('finite FP16 values as FP32',half.float())
    for dtype in [torch.float16,torch.float32]:
        for scale in [.001,1,4,100]:
            check('random scale '+str(scale),torch.randn((1048579,),device='cuda',dtype=dtype)*scale)
        check('noncontiguous',torch.randn((29,127,17),device='cuda',dtype=dtype).transpose(1,2))
    x=torch.randn((1,256,256,128),device='cuda',dtype=torch.float16)
    timings={}
    for name,fn in [('reference',quadratic_gate_activation),('fused',kernel.gate)]:
        for _ in range(3):fn(x)
        torch.cuda.synchronize()
        samples=[]
        for _ in range(12):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();fn(x);end.record();end.synchronize();samples.append(start.elapsed_time(end))
        timings[name]={'median_ms':statistics.median(samples),'samples_ms':samples}
report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'tests':records,'all_equal':all(r['unequal']==0 for r in records),
        'total_elements':sum(r['elements'] for r in records),'timing_shape':list(x.shape),'timings':timings,
        'scope':'Recovered PyTorch operation only; no vendor runtime speedup claim.'}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
if not report['all_equal']:raise SystemExit('Rejected: numerical mismatch.')
