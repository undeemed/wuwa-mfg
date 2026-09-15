# SPDX-License-Identifier: Apache-2.0
"""Check the boundary-block extension against the prior block-0 diagnostic.

Requires repository history at the pinned prior commit, private reference
weights, and CUDA. Compares actual GPU outputs; does not launch an application.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from probe_block_sensitivity import setup_model,sha,SOURCE_COMMIT,WEIGHTS_SHA

PREVIOUS_COMMIT='3beec244d0aea1a35b744f9210818830801d1a2d'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.resolve().is_relative_to(repository):
        raise ValueError('Use a fresh private result path.')
    if sha(args.weights)!=WEIGHTS_SHA or subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Requires the pinned reference and weights.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Modified reference source.')
    previous=subprocess.check_output(['git','show',PREVIOUS_COMMIT+':research/neural-latency/mma_first_block.py'],cwd=repository)
    namespace={}
    exec(compile(previous,'previous_mma_first_block.py','exec'),namespace)
    import torch
    from fused_norm import FusedNorm
    from mma_first_block import MmaFirstBlock
    pipeline=setup_model(args.source,args.weights)
    from mlxdlss import model as reference
    kernel=FusedNorm()
    old=namespace['MmaFirstBlock'](reference,pipeline.model,kernel)
    first=MmaFirstBlock(reference,pipeline.model,kernel)
    last=MmaFirstBlock(reference,pipeline.model,kernel,block_index=70)
    torch.manual_seed(41982)
    options=[dict(zip(('attention_mma','seed_residual','seed_logits'),values))
             for values in itertools.product((False,True),repeat=3)]
    report={'schema':1,'passed':False,'source_commit':SOURCE_COMMIT,'weights_sha256':WEIGHTS_SHA,
            'previous_commit':PREVIOUS_COMMIT,'previous_source_sha256':hashlib.sha256(previous).hexdigest(),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cases':[],'packing_cases':[],
            'scope':'Block-0 regression, block-70 chunk invariance and single-head packing fusion on finite synthetic inputs; not native parity or a speed test.'}
    original_chunk=reference.CHUNK_TOKENS
    try:
        with torch.inference_mode():
            for shape in ((1,8,8,32),(2,16,24,32),(1,24,40,32)):
                value=torch.randn(shape,device='cuda',dtype=torch.float16)*.5
                for option in options:
                    reference.CHUNK_TOKENS=256
                    expected,actual=old(value,**option),first(value,**option)
                    if not torch.equal(expected,actual):raise AssertionError('Block-0 regression.')
                    reference.CHUNK_TOKENS=0
                    unchunked=last(value,**option)
                    reference.CHUNK_TOKENS=256
                    chunked=last(value,**option)
                    if not torch.equal(unchunked,chunked):raise AssertionError('Shifted block chunk mismatch.')
                    if not torch.isfinite(actual).all() or not torch.isfinite(chunked).all():
                        raise AssertionError('Nonfinite output.')
                    report['cases'].append({'shape':list(shape),'options':option,
                        'block0_matches_previous_exactly':True,'block70_chunking_matches_exactly':True})
            for index in (*range(5),*range(66,71)):
                variants={mode:MmaFirstBlock(reference,pipeline.model,kernel,block_index=index,
                    fused_packing=mode!='none',fused_input_packing=mode=='all') for mode in ('none','gate','all')}
                for shape in ((1,8,8,32),(2,16,24,32),(1,24,40,32)):
                    value=torch.randn(shape,device='cuda',dtype=torch.float16)*.5
                    outputs={mode:adapter(value,attention_mma=True,seed_residual=True,seed_logits=True)
                             for mode,adapter in variants.items()}
                    if not all(torch.equal(outputs['none'],item) for item in outputs.values()):
                        raise AssertionError('Packing fusion changes the single-head output.')
                    report['packing_cases'].append({'block':index,'shape':list(shape),
                        'gate_only_matches_exactly':True,'all_packing_matches_exactly':True})
    finally:
        reference.CHUNK_TOKENS=original_chunk
    report['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Passed',len(report['cases']),'block-0 and shifted block-70 cases.')
    print('Passed',len(report['packing_cases']),'single-head packing comparisons.')


if __name__=='__main__':main()
