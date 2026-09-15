# SPDX-License-Identifier: Apache-2.0
"""Training-only context saturation and global RGB-error diagnostics.

The RGB offset uses the native target and is an oracle diagnostic, not an
available inference correction. No weights are fitted or changed.
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
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private report.')
    assert 1<=len(a.model)<=3 and len({n for n,_ in a.model})==len(a.model)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    baseline=read(Path(a.model[0][1])/'result.json')
    pairs=training_pairs(a.base,a.photos,a.images,baseline)
    models={}
    for name,directory in a.model:
        directory=Path(directory);record=read(directory/'result.json');arch=record['architecture']
        assert record['training_capture_hashes']==baseline['training_capture_hashes'] and record['controls']==baseline['controls']
        assert arch['variant']=='hierarchical' and arch['width']==16 and arch['blocks']==2 and arch['noise_channels']==0
        state=torch.load(directory/'student-private.pt',map_location='cpu',weights_only=True)
        model=GradedStudent(HierarchicalStudent(16,2),arch['explicit_output_grading'])
        model.load_state_dict(state['state_dict'],strict=True)
        model=model.cuda().float().eval().to(memory_format=torch.channels_last)
        captured={}
        def capture(module,inputs,output):captured['preactivation']=output.detach()
        hook=model.network.context.register_forward_hook(capture)
        rows,gates=[],[]
        with torch.inference_mode():
            for pair in pairs:
                source,target=load_pair(pair,record['controls'])
                prediction=model(source)
                gate=torch.tanh(captured['preactivation']).flatten()
                gates.append(gate.clone())
                delta=target-prediction
                offset=delta.flatten(2).median(dim=-1).values[:,:,None,None]
                corrected=(prediction+offset).clamp(0,1)
                mae=float(delta.abs().mean());corrected_mae=float((corrected-target).abs().mean())
                assert corrected_mae<=mae+1e-6
                rows.append({k:v for k,v in pair.items() if k!='path'}|{
                    'rgb_mae':mae,'oracle_global_offset_mae':corrected_mae,
                    'oracle_offset_reduction_fraction':1-corrected_mae/mae,
                    'mean_absolute_oracle_offset':float(offset.abs().mean()),
                    'gate_abs_above_095_fraction':float((gate.abs()>.95).float().mean()),
                    'gate_abs_above_099_fraction':float((gate.abs()>.99).float().mean()),
                    'mean_absolute_gate':float(gate.abs().mean()),
                    'mean_gate_derivative':float((.1*(1-gate.square())).mean())})
        hook.remove()
        assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
        stacked=torch.stack(gates)
        summary={g:{k:statistics.mean(r[k] for r in rows if r['group']==g) for k in
                    ['rgb_mae','oracle_global_offset_mae','oracle_offset_reduction_fraction',
                     'gate_abs_above_095_fraction','gate_abs_above_099_fraction','mean_gate_derivative']}
                 for g in ['scene','photos']}
        models[name]={'checkpoint_sha256':sha(directory/'student-private.pt'),'result_sha256':sha(directory/'result.json'),
            'rows':rows,'summary':summary,'weights_unchanged':True,
            'mean_context_gain_std_across_training_images':float((.1*stacked).std(dim=0).mean()),
            'context_channels_always_saturated_099':int((stacked.abs()>.99).all(dim=0).sum()),
            'context_channel_count':int(stacked.shape[1])}
        del model,state,gates,stacked
    report={'schema':1,'complete':True,'training_only':True,'optimizer_steps':0,'quality_gate_passed':False,
        'target_achieved':False,'precision':'FP32 student and gradients disabled','models':models,
        'limitations':['Context saturation is measured only at these checkpoints; it is not a causal diagnosis.',
            'Per-image RGB offsets use native targets and cannot be used at inference. This is an error decomposition, not a replacement model.',
            'Only the 46 training frames enter these diagnostics. No validation cases are fitted or evaluated.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:{k:v for k,v in r.items() if k not in ['rows','checkpoint_sha256','result_sha256']} for n,r in models.items()},indent=2))


if __name__=='__main__':main()
