# SPDX-License-Identifier: Apache-2.0
"""Check branched adapters against unchanged math, chunking and compact bias."""
import argparse
import json
from pathlib import Path
import subprocess

from probe_block_sensitivity import setup_model,sha,SOURCE_COMMIT,WEIGHTS_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private output.')
    if sha(args.weights)!=WEIGHTS_SHA or subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Requires pinned source and weights.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():raise ValueError('Modified reference.')
    import torch
    from fused_norm import FusedNorm
    from mma_branched_block import MmaBranchedBlock,BRANCHED
    pipeline=setup_model(args.source,args.weights);kernel=FusedNorm()
    from mlxdlss import model as ref
    torch.manual_seed(97512);checks=[];chunking=[]
    original_chunk=ref.CHUNK_TOKENS
    try:
        with torch.inference_mode():
            for index in sorted(BRANCHED):
                adapter=MmaBranchedBlock(ref,pipeline.model,kernel,block_index=index)
                value=torch.randn((1,17,23,adapter.channels),device='cuda',dtype=torch.float16)*.5
                ref.CHUNK_TOKENS=256
                expected=pipeline.model._window(value,index,head_count=adapter.heads,publish=False)
                actual=adapter(value,ffn_mma=False,attention_mma=False)
                if not torch.equal(expected,actual):raise AssertionError('Unchanged reference path differs.')
                checks.append({'block':index,'heads':adapter.heads,'shape':list(value.shape),'reference_exact':True})
            for index in (5,6,9,10,15,16):
                expanded=MmaBranchedBlock(ref,pipeline.model,kernel,block_index=index)
                compact=MmaBranchedBlock(ref,pipeline.model,kernel,block_index=index,compact_bias=True)
                value=torch.randn((2,17,23,expanded.channels),device='cuda',dtype=torch.float16)*.5
                for ffn,attention in ((True,False),(False,True),(True,True)):
                    ref.CHUNK_TOKENS=0
                    whole=expanded(value,ffn_mma=ffn,attention_mma=attention)
                    ref.CHUNK_TOKENS=256
                    chunks=expanded(value,ffn_mma=ffn,attention_mma=attention)
                    small=compact(value,ffn_mma=ffn,attention_mma=attention)
                    if not torch.equal(whole,chunks) or not torch.equal(chunks,small):raise AssertionError('Chunking or compact bias changes the adapter.')
                    chunking.append({'block':index,'shape':list(value.shape),'ffn_mma':ffn,'attention_mma':attention,
                        'chunking_exact':True,'compact_bias_exact':True})
    finally:ref.CHUNK_TOKENS=original_chunk
    report={'schema':1,'passed':True,'source_commit':SOURCE_COMMIT,'weights_sha256':WEIGHTS_SHA,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'reference_checks':checks,'chunking_checks':chunking,
        'scope':'Reference wiring and numerical invariance only; does not establish native intermediate parity.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Passed',len(checks),'reference checks and',len(chunking),'chunk/compact comparisons.')


if __name__=='__main__':main()
