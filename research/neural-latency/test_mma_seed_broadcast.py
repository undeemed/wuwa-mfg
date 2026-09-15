# SPDX-License-Identifier: Apache-2.0
"""Check periodic trailing-batch MMA seeds against expanded seeds and exact math."""
import argparse
import json
from pathlib import Path

import torch
from fused_norm import FusedNorm
from test_output_grade import graph_measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private result path.')
    kernel=FusedNorm();torch.manual_seed(9144);records=[];timings=[]
    with torch.inference_mode():
        for dtype in (torch.float8_e4m3fn,torch.float16):
            for batch_shape in ((2,3),(2,3,4)):
                for m,n,k in ((1,1,32),(17,9,32),(33,65,64)):
                    a=torch.randint(-2,3,(*batch_shape,m,k),device='cuda').to(dtype)
                    b=torch.randint(-2,3,(*batch_shape,k,n),device='cuda').to(dtype)
                    for seed_dims in range(len(batch_shape)+1):
                        leading=batch_shape[-seed_dims:] if seed_dims else ()
                        seed=torch.randint(-4,5,(*leading,n,m),device='cuda').half().transpose(-1,-2)*.5
                        compact=kernel.mma(a,b,seed)
                        expanded=kernel.mma(a,b,seed.expand(*batch_shape,m,n).contiguous())
                        exact=a.float()@b.float()+seed.float()
                        if not torch.equal(compact,expanded) or not torch.equal(compact.float(),exact):
                            raise AssertionError('Compact seed differs from expansion or exact arithmetic.')
                        records.append({'dtype':str(dtype),'batch_shape':list(batch_shape),'matrix_shape':[m,n,k],
                            'seed_shape':list(seed.shape),'strided_seed':not seed.is_contiguous(),
                            'elements':compact.numel(),'expanded_equal':True,'exact_arithmetic_equal':True})
        rejections=[]
        a=torch.ones((2,3,17,32),device='cuda',dtype=torch.float8_e4m3fn)
        b=torch.ones((32,9),device='cuda',dtype=torch.float8_e4m3fn)
        for shape in ((9,),(2,17,9),(1,3,17,9),(3,16,9),(2,3,4,17,9)):
            seed=torch.zeros(shape,device='cuda',dtype=torch.float16)
            try:kernel.mma(a,b,seed)
            except ValueError:rejections.append(list(shape))
            else:raise AssertionError('Invalid seed shape accepted.')
        for windows,heads in ((2160,2),(135,8)):
            shape=(windows,heads,64,32)
            a=(torch.randn(shape,device='cuda',dtype=torch.float16)*.25).to(torch.float8_e4m3fn)
            b=(torch.randn((windows,heads,32,64),device='cuda',dtype=torch.float16)*.25).to(torch.float8_e4m3fn)
            bias=torch.randn((heads,64,64),device='cuda',dtype=torch.float16)
            original=lambda x:kernel.mma(x,b,bias.unsqueeze(0).expand(windows,-1,-1,-1))
            compact=lambda x:kernel.mma(x,b,bias)
            old,old_output=graph_measure(original,a)
            new,new_output=graph_measure(compact,a)
            if not torch.equal(old_output,new_output):raise AssertionError('Graph bias broadcast changes output.')
            timings.append({'shape':list(shape),'expanded_graph':old,'compact_graph':new,
                'graph_outputs_equal':True,'bias_bytes':bias.numel()*2,
                'expanded_bias_bytes':windows*bias.numel()*2})
    report={'schema':1,'passed':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'tests':records,'invalid_seed_shapes_rejected':rejections,'timings':timings,
        'scope':'MMA seed mapping and isolated attention-score graphs only; no native renderer speed or quality claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Passed',len(records),'mapping cases and',len(rejections),'shape guards.')
    for item in timings:print(item['shape'],item['expanded_graph']['median_ms'],item['compact_graph']['median_ms'])


if __name__=='__main__':main()
