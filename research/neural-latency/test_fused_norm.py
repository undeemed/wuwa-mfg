# SPDX-License-Identifier: Apache-2.0
"""Numerical and timing checks for an isolated 32-channel CUDA kernel."""
import argparse, json, statistics, sys
from pathlib import Path
import torch
from fused_norm import FusedNorm

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
sys.path.insert(0,str(a.source/'python'))
from mlxdlss.model import vendor_cosine_normalize
if a.output.exists(): raise SystemExit('Use a fresh output path.')
torch.manual_seed(5678)
kernel = FusedNorm()
results = []
with torch.inference_mode():
    for dtype in [torch.float16,torch.float32]:
        for scale in [0.0, 0.0001, 0.1, 1.0, 16.0, 100.0]:
            x = torch.randn((4099,32),device='cuda',dtype=dtype)*scale
            y = vendor_cosine_normalize(x)
            z = kernel(x)
            torch.cuda.synchronize()
            finite = torch.isfinite(y) & torch.isfinite(z)
            equal = (y == z) | (torch.isnan(y) & torch.isnan(z))
            results.append({'dtype':str(dtype),'scale':scale,'rows':4099,
                'unequal':int(torch.count_nonzero(~equal)),
                'max_abs_finite':float((y[finite]-z[finite]).abs().max()),
                'finite_class_mismatches':int(torch.count_nonzero(torch.isfinite(y) != torch.isfinite(z)))})
        x = torch.randn((7,32,19),device='cuda',dtype=dtype).transpose(1,2)
        y, z = vendor_cosine_normalize(x), kernel(x)
        results.append({'dtype':str(dtype),'layout':'noncontiguous [7,19,32]',
            'unequal':int(torch.count_nonzero(y != z)), 'max_abs_finite':float((y-z).abs().max())})
    timings = {}
    x = torch.randn((262144,32),device='cuda',dtype=torch.float16)
    for name, fn in [('reference',vendor_cosine_normalize),('fused',kernel)]:
        for _ in range(3): fn(x)
        torch.cuda.synchronize()
        values=[]
        for _ in range(12):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); fn(x); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        timings[name]={'median_ms':statistics.median(values),'samples_ms':values}
report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'tests':results,'timing_shape':list(x.shape),'timings':timings,
        'matches_tested_inputs':all(row['unequal']==0 for row in results),
        'scope':'Isolated kernel in PyTorch reconstruction; not a vendor DLL or game speedup.'}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not report['matches_tested_inputs']:
    raise SystemExit('Numerical checks failed; reject the kernel.')
