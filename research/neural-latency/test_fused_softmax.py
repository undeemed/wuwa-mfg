# SPDX-License-Identifier: Apache-2.0
"""Compare the custom bit-affine row kernel to the pinned reference."""
import argparse,json,statistics,sys
from pathlib import Path
import torch
from fused_norm import FusedNorm
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a new output path.')
sys.path.insert(0,str(a.source/'python'))
from mlxdlss.model import vendor_approximate_softmax
torch.manual_seed(123456)
kernel=FusedNorm()
records=[]
with torch.inference_mode():
    for dtype in [torch.float16,torch.float32]:
        for columns in [2,16,64,128,256,512,640,1024,2048]:
            for scale in [0,0.1,1,8,32]:
                x=torch.randn((259,columns),device='cuda',dtype=dtype)*scale
                y,z=vendor_approximate_softmax(x),kernel.softmax(x)
                mismatch=(y!=z)&~(torch.isnan(y)&torch.isnan(z))
                records.append({'dtype':str(dtype),'columns':columns,'scale':scale,
                    'unequal':int(torch.count_nonzero(mismatch)), 'elements':x.numel(),
                    'max_abs':float(torch.nan_to_num((y-z).abs()).max())})
        x=torch.randn((7,64,39),device='cuda',dtype=dtype).transpose(1,2)
        y,z=vendor_approximate_softmax(x),kernel.softmax(x)
        records.append({'dtype':str(dtype),'noncontiguous':True,'unequal':int(torch.count_nonzero(y!=z)),
                        'elements':x.numel(),'max_abs':float((y-z).abs().max())})
    timings={}
    x=torch.randn((32768,64),device='cuda',dtype=torch.float16)
    for name,fn in [('reference',vendor_approximate_softmax),('fused',kernel.softmax)]:
        for _ in range(3):fn(x)
        torch.cuda.synchronize()
        ms=[]
        for _ in range(12):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();fn(x);end.record();end.synchronize();ms.append(start.elapsed_time(end))
        timings[name]={'median_ms':statistics.median(ms),'samples_ms':ms}
report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'tests':records,
        'total_elements':sum(t['elements'] for t in records),'all_equal':all(t['unequal']==0 for t in records),
        'timing_shape':list(x.shape),'timings':timings,
        'scope':'PyTorch reconstruction only; not a NVIDIA DLL or game speedup.'}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))
bad=[r for r in records if r['unequal']]
if bad:
    print(json.dumps(bad,indent=2));raise SystemExit('Rejected: numerical differences.')
