# SPDX-License-Identifier: Apache-2.0
"""Training-only removal of the learned context branch; no weight fitting."""
import argparse,json,statistics
from pathlib import Path
import torch
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,training_pairs,load_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','photos','images','model','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private report.')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    record=read(a.model/'result.json');arch=record['architecture'];pairs=training_pairs(a.base,a.photos,a.images,record)
    assert arch['variant']=='hierarchical-latent' and arch['width']==16 and arch['blocks']==2
    state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True);assert state['architecture']==arch
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True,latent=True),arch['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval().to(memory_format=torch.channels_last)
    branch=model.network.latent_context;capture={}
    def hook(module,inputs,output):
        x=inputs[0];capture['relative_feature_correction_l1']=float((output-x).abs().mean()/x.abs().mean().clamp_min(1e-8))
    handle=branch.register_forward_hook(hook);rows=[]
    try:
        with torch.inference_mode():
            for pair in pairs:
                source,target=load_pair(pair,record['controls']);prediction=model(source)
                try:
                    model.network.latent_context=None;without=model(source)
                finally:model.network.latent_context=branch
                rows.append({k:v for k,v in pair.items() if k!='path'}|{
                    'rgb_mae':float((prediction-target).abs().mean()),'without_latent_mae':float((without-target).abs().mean()),
                    'output_change_mae':float((prediction-without).abs().mean()),**capture})
    finally:handle.remove();model.network.latent_context=branch
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    groups={g:{key:statistics.mean(r[key] for r in rows if r['group']==g) for key in
        ['rgb_mae','without_latent_mae','output_change_mae','relative_feature_correction_l1']} for g in ['scene','photos']}
    report={'complete':True,'training_only':True,'optimizer_steps':0,'weights_unchanged':True,
        'target_achieved':False,'quality_gate_passed':False,'checkpoint_sha256':sha(a.model/'student-private.pt'),
        'result_sha256':sha(a.model/'result.json'),'rows':rows,'groups':groups,
        'limitations':['Removing a learned branch changes the distribution seen by downstream layers; this is not a separately trained no-branch control.',
            'Only 46 training images are evaluated, without fitting any weights. FP32 results are distinct from FP16 validation.',
            'Feature and output changes are scalar diagnostics, not a native-quality or generalization pass.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
