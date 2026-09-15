# SPDX-License-Identifier: Apache-2.0
"""Training-only oracle decomposition of spatially varying RGB errors.

Tile corrections use native targets. They are not available at inference and
do not establish which architecture can predict them or preserve visual quality.
"""
import argparse,json,statistics
from pathlib import Path
import torch
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,training_pairs,load_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','photos','images','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',nargs=2,action='append',required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private report.')
    assert 1<=len(a.model)<=2 and len({n for n,_ in a.model})==len(a.model)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    baseline=read(Path(a.model[0][1])/'result.json');pairs=training_pairs(a.base,a.photos,a.images,baseline);models={}
    for name,directory in a.model:
        directory=Path(directory);record=read(directory/'result.json');arch=record['architecture']
        assert record['training_capture_hashes']==baseline['training_capture_hashes'] and record['controls']==baseline['controls']
        assert arch['variant']=='hierarchical-film' and arch['width'] in (16,32) and arch['blocks']==2
        state=torch.load(directory/'student-private.pt',map_location='cpu',weights_only=True)
        assert state['architecture']==arch
        model=GradedStudent(HierarchicalStudent(arch['width'],2,conditioned=True),arch['explicit_output_grading'])
        model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval().to(memory_format=torch.channels_last)
        rows=[]
        with torch.inference_mode():
            for pair in pairs:
                source,target=load_pair(pair,record['controls']);prediction=model(source);delta=target-prediction
                assert delta.shape==(1,3,1080,1920)
                mae=float(delta.abs().mean());oracles={};previous=mae
                for gh,gw in [(1,1),(2,4),(4,8),(8,16)]:
                    hh,ww=1080//gh,1920//gw
                    offsets=delta.reshape(1,3,gh,hh,gw,ww).permute(0,1,2,4,3,5).reshape(1,3,gh,gw,-1).median(dim=-1).values
                    corrected=(prediction+offsets.repeat_interleave(hh,2).repeat_interleave(ww,3)).clamp(0,1)
                    error=float((corrected-target).abs().mean());assert error<=previous+1e-6;previous=error
                    oracles[f'{gh}x{gw}']={'mae':error,'reduction_fraction':1-error/mae}
                rows.append({k:v for k,v in pair.items() if k!='path'}|{'rgb_mae':mae,'oracle_tiles':oracles})
        assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
        summary={g:{'rgb_mae':statistics.mean(r['rgb_mae'] for r in rows if r['group']==g),
            'mean_oracle_reduction':{grid:statistics.mean(r['oracle_tiles'][grid]['reduction_fraction'] for r in rows if r['group']==g) for grid in ['1x1','2x4','4x8','8x16']}} for g in ['scene','photos']}
        models[name]={'checkpoint_sha256':sha(directory/'student-private.pt'),'result_sha256':sha(directory/'result.json'),
            'width':arch['width'],'weights_unchanged':True,'rows':rows,'summary':summary}
        del model,state
    report={'complete':True,'training_only':True,'optimizer_steps':0,'target_achieved':False,'quality_gate_passed':False,
        'precision':'FP32 student, gradients disabled','models':models,
        'limitations':['Corrections are fitted to native training targets independently per image/tile and are unavailable at inference.',
            'Only scalar errors are retained. This is a spatial error decomposition, not a learned correction or a causal architecture diagnosis.',
            'All sixteen validation captures are excluded. No game or sample is launched.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:v['summary'] for n,v in models.items()},indent=2))


if __name__=='__main__':main()
