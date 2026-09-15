# SPDX-License-Identifier: Apache-2.0
"""Check FP8 MMA fragment layout, matrix tails, batching and initial C ordering."""
import argparse
import json
from pathlib import Path
import torch
from fused_norm import FusedNorm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.manual_seed(821)
    kernel=FusedNorm();records=[]
    with torch.inference_mode():
        for batch,m,n,k in [(1,1,1,32),(1,17,9,32),(3,33,31,64),(5,63,65,128),(2,128,32,128)]:
            a=torch.randint(-2,3,(batch,m,k),device='cuda').to(torch.float8_e4m3fn)
            for shared in (False,True):
                b=torch.randint(-2,3,(k,n) if shared else (batch,k,n),device='cuda').to(torch.float8_e4m3fn)
                for seeded in (False,True):
                    seed=torch.randint(-4,5,(m,n) if shared else (batch,m,n),device='cuda').half()*.5 if seeded else None
                    output=kernel.mma(a,b,seed)
                    expected=a.float()@b.float()
                    if seed is not None:expected+=seed.float()
                    unequal=int((output.float()!=expected).sum())
                    records.append({'shape':[batch,m,n,k],'shared_B_C':shared,'seeded':seeded,
                                    'elements':output.numel(),'unequal':unequal})
                    if unequal:raise ValueError(records[-1])
        a=torch.ones((16,64),device='cuda').to(torch.float8_e4m3fn)
        b=torch.cat([torch.full((32,8),-256.,device='cuda'),torch.full((32,8),.046875,device='cuda')]).to(torch.float8_e4m3fn)
        seed=torch.full((16,8),8192.,device='cuda',dtype=torch.float16)
        initial=kernel.mma(a,b,seed)
        separate=kernel.mma(a,b)+seed
        if not (torch.all(initial==1.5) and torch.all(separate==0)):
            raise ValueError('Initial-accumulator ordering test failed.')
        for batch,m,n,k in [(1,1,1,16),(3,17,33,16),(2,31,9,32),(4,65,31,64)]:
            a=torch.randint(-2,3,(batch,m,k),device='cuda').half()
            b=torch.randint(-2,3,(k,n),device='cuda').half()
            seed=torch.randint(-4,5,(batch,m,n),device='cuda').half()*.5
            output=kernel.mma(a,b,seed)
            expected=a.float()@b.float()+seed.float()
            unequal=int((output.float()!=expected).sum())
            records.append({'dtype':'FP16','shape':[batch,m,n,k],'shared_B_C':False,'seeded':True,
                            'elements':output.numel(),'unequal':unequal})
            if unequal:raise ValueError(records[-1])
        for dtype in (torch.float8_e4m3fn,torch.float16):
            # A contiguous view can still start at an address misaligned for a
            # packed four-byte load. Also cover noncontiguous batched B and C.
            batch,m,n,k=2,17,9,32
            backing=torch.randint(-2,3,(batch*m*k+1,),device='cuda').to(dtype)
            a=backing[1:].reshape(batch,m,k)
            b=torch.randint(-2,3,(batch,n,k),device='cuda').to(dtype).transpose(-1,-2)
            seed=torch.randint(-4,5,(batch,n,m),device='cuda').half().transpose(-1,-2)*.5
            output=kernel.mma(a,b,seed)
            expected=a.float()@b.float()+seed.float()
            unequal=int((output.float()!=expected).sum())
            records.append({'dtype':str(dtype),'shape':[batch,m,n,k],'misaligned_A':True,
                            'strided_B_C':True,'elements':output.numel(),'unequal':unequal})
            if unequal:raise ValueError(records[-1])
        report={'schema':1,'tests':records,'all_exact_layout_tests_passed':True,
                'initial_C_test':{'elements':initial.numel(),'initial_C_result':1.5,'separate_add_result':0.},
                'quality_gate_passed':False,'scope':'Synthetic exact arithmetic and accumulator-order test only; not model parity or performance.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(records),'all_exact_layout_tests_passed':True,'initial_C_test':report['initial_C_test']}))


if __name__=='__main__':main()
