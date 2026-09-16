# SPDX-License-Identifier: Apache-2.0
"""Validate the native student against its unchanged trained PyTorch model.

Use a private cases JSON containing label/capture rows. Only aggregate numerical
results may be published; case files and checkpoint/export artifacts stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from progressive_student import load_first
from region_context_student import RegionFeatureRefinement
from compare_output_grade import read_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('first','candidate','model','probe','cases','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Validation pixels must stay private.')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.model/'model.json').read_text())
    checkpoint=a.candidate/'student-private.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    first,_=load_first(a.first)
    model=RegionFeatureRefinement(first,'routed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict'],strict=True)
    model=model.eval().cuda().half().to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    with torch.inference_mode():
        for row in json.loads(a.cases.read_text()):
            controls,arrays,hashes=read_capture(Path(row['capture']))
            source=torch.from_numpy(arrays['color']).permute(2,0,1).unsqueeze(0).cuda().half()
            expected=model(source).squeeze(0).permute(1,2,0).float().cpu().numpy()
            rgba=np.ones((1080,1920,4),np.float16);rgba[:,:,:3]=arrays['color']
            rgba.tofile(a.output/'input.bin')
            run=subprocess.run([str(a.probe.resolve()),str(a.model.resolve()),str((a.output/'input.bin').resolve()),str((a.output/'output.bin').resolve())],
                capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
            if run.returncode: raise RuntimeError(run.stdout+'\n'+run.stderr)
            timing=json.loads(run.stdout.strip().splitlines()[-1])
            actual=np.fromfile(a.output/'output.bin',np.float16).reshape(1080,1920,4)[:,:,:3].astype(np.float32)
            delta=np.abs(actual-expected)
            measured={'label':row['label'],'finite':bool(np.isfinite(actual).all()),
                'mae':float(delta.mean()),'max':float(delta.max()),'p99':float(np.quantile(delta,.99)),
                'median_ms':timing['median_ms']}
            measured['parity_passed']=measured['finite'] and measured['mae']<.0005 and measured['p99']<.003
            rows.append(measured);print(json.dumps({'completed':len(rows),**measured}),flush=True)
    result={'complete':True,'checkpoint_sha256':manifest['checkpoint_sha256'],
        'probe_sha256':hashlib.sha256(a.probe.read_bytes()).hexdigest(),
        'width':1920,'height':1080,'case_count':len(rows),'all_passed':all(r['parity_passed'] for r in rows),
        'mean_mae':float(np.mean([r['mae'] for r in rows])),
        'max_p99':max(r['p99'] for r in rows),'cases':rows,
        'scope':'DirectML versus the same student weights in CUDA. This is port parity, not NVIDIA-quality or temporal/gameplay validation.'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    if not result['all_passed']: raise SystemExit('Native student parity gate failed.')


if __name__=='__main__':main()
