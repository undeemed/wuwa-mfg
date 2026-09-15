# SPDX-License-Identifier: Apache-2.0
"""Check neutral insertion, gradient flow and strict reload on private inputs."""
import argparse,io,json
from pathlib import Path

import torch

from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,load_pair,rgb_loss
from compare_output_grade import read_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private report.')
    assert len(a.capture)==2
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    record=read(a.model/'result.json');arch=record['architecture']
    assert arch['variant']=='hierarchical' and arch['width']==16 and arch['blocks']==2
    state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    def create(conditioned,dtype):
        model=GradedStudent(HierarchicalStudent(16,2,conditioned=conditioned),arch['explicit_output_grading'])
        if conditioned:
            missing,unexpected=model.load_state_dict(state['state_dict'],strict=False)
            assert len(missing)==6 and all(n.startswith('network.decoder_conditioning.') for n in missing) and not unexpected
        else:model.load_state_dict(state['state_dict'],strict=True)
        return model.cuda().to(dtype=dtype,memory_format=torch.channels_last).eval()
    pairs=[]
    train={r['color']:r for r in record['training_capture_hashes']}
    for path in a.capture:
        controls,arrays,hashes=read_capture(path)
        assert controls==record['controls'] and hashes==train[hashes['color']]
        pairs.append({'path':path,'capture_hashes':hashes})
    rows=[]
    for dtype in [torch.float32,torch.float16]:
        base,model=create(False,dtype),create(True,dtype)
        with torch.inference_mode():
            for pair in pairs:
                source,_=load_pair(pair,record['controls']);source=source.to(dtype)
                expected,actual=base(source),model(source)
                assert torch.equal(expected,actual)
                if dtype==torch.float16:assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                rows.append({'dtype':str(dtype),'capture_hashes':pair['capture_hashes'],'neutral_output_exact':True})
        del base,model
    model=create(True,torch.float32)
    condition=model.network.decoder_conditioning
    assert condition.norm.eps==1e-5
    source,target=load_pair(pairs[0],record['controls'])
    loss,_=rgb_loss(model(source),target);loss.backward()
    project_gradient=float(condition.project.weight.grad.abs().max())
    hidden_before=float(condition.hidden.weight.grad.abs().max())
    assert project_gradient>0 and hidden_before==0
    with torch.no_grad():
        condition.project.weight.add_(condition.project.weight.grad,alpha=-.01)
        condition.project.bias.add_(condition.project.bias.grad,alpha=-.01)
    model.zero_grad(set_to_none=True)
    loss,_=rgb_loss(model(source),target);loss.backward()
    hidden_after=float(condition.hidden.weight.grad.abs().max())
    assert hidden_after>0
    with torch.inference_mode():expected=model(source)
    memory=io.BytesIO();torch.save(model.state_dict(),memory);memory.seek(0)
    reloaded=create(True,torch.float32)
    reloaded.load_state_dict(torch.load(memory,map_location='cuda',weights_only=True),strict=True)
    with torch.inference_mode():assert torch.equal(reloaded(source),expected)
    rejected=[]
    for option in ['noise','affine','attention']:
        try:HierarchicalStudent(16,2,conditioned=True,**{option:True})
        except ValueError:rejected.append(option)
        else:raise AssertionError(option)
    result={'complete':True,'target_achieved':False,'quality_gate_passed':False,
        'source_checkpoint_sha256':sha(a.model/'student-private.pt'),'neutral_tests':rows,
        'added_parameters':sum(p.numel() for p in condition.parameters()),'total_parameters':sum(p.numel() for p in model.parameters()),
        'initial_projection_gradient_max':project_gradient,'hidden_gradient_before_projection_step':hidden_before,
        'hidden_gradient_after_projection_step':hidden_after,'strict_reload_exact':True,'rejected_combinations':rejected,
        'scope':'Mechanical checks only. One private projection-only test update; no candidate quality or performance claim.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['neutral_tests','source_checkpoint_sha256']},indent=2))


if __name__=='__main__':main()
