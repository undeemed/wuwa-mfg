# SPDX-License-Identifier: Apache-2.0
"""Check neutral insertion, learning path, coordinates and private strict reload."""
import argparse,io,json
from pathlib import Path
import torch
from latent_context import LatentContext
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,load_pair,rgb_loss
from compare_output_grade import read_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private report.')
    assert len(a.capture)==2
    torch.manual_seed(49321)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    record=read(a.model/'result.json');arch=record['architecture'];state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    assert state['architecture']==arch and arch['variant']=='hierarchical-film' and arch['width']==16 and arch['blocks']==2
    def create(latent,dtype):
        model=GradedStudent(HierarchicalStudent(16,2,conditioned=True,latent=latent),arch['explicit_output_grading'])
        if latent:
            missing,unexpected=model.load_state_dict(state['state_dict'],strict=False)
            assert len(missing)>0 and all(n.startswith('network.latent_context.') for n in missing) and not unexpected
        else:model.load_state_dict(state['state_dict'],strict=True)
        return model.cuda().to(dtype=dtype,memory_format=torch.channels_last).eval()
    train={r['color']:r for r in record['training_capture_hashes']};pairs=[]
    for path in a.capture:
        controls,_,hashes=read_capture(path);assert controls==record['controls'] and hashes==train[hashes['color']]
        pairs.append({'path':path,'capture_hashes':hashes})
    rows=[]
    for dtype in (torch.float32,torch.float16):
        base,model=create(False,dtype),create(True,dtype)
        with torch.inference_mode():
            for pair in pairs:
                source,_=load_pair(pair,record['controls']);source=source.to(dtype)
                expected,actual=base(source),model(source);assert torch.equal(expected,actual)
                if dtype==torch.float16:assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                rows.append({'dtype':str(dtype),'capture_hashes':pair['capture_hashes'],'neutral_output_exact':True})
        del base,model
    model=create(True,torch.float32);branch=model.network.latent_context
    source,target=load_pair(pairs[0],record['controls']);loss,_=rgb_loss(model(source),target);loss.backward()
    projection=float(branch.project.weight.grad.abs().max());before=float(branch.read_kv.weight.grad.abs().max())
    assert projection>0 and before==0
    with torch.no_grad():
        branch.project.weight.add_(branch.project.weight.grad,alpha=-.01);branch.project.bias.add_(branch.project.bias.grad,alpha=-.01)
    model.zero_grad(set_to_none=True);loss,_=rgb_loss(model(source),target);loss.backward()
    after=float(branch.read_kv.weight.grad.abs().max());assert after>0
    assert branch.position.weight.grad.abs().max()>0 and branch.latents.grad.abs().max()>0
    with torch.inference_mode():expected=model(source)
    memory=io.BytesIO();torch.save(model.state_dict(),memory);memory.seek(0);reloaded=create(True,torch.float32)
    reloaded.load_state_dict(torch.load(memory,map_location='cuda',weights_only=True),strict=True)
    with torch.inference_mode():assert torch.equal(reloaded(source),expected)
    coordinates=[]
    test=LatentContext(96).cuda()
    for shape,dtype in [((2,96,7,11),torch.float32),((1,96,9,13),torch.float16),((1,96,7,11),torch.float32)]:
        test=test.to(dtype)
        with torch.inference_mode():
            for _ in range(2):
                x=torch.randn(shape,device='cuda',dtype=dtype);assert torch.equal(test(x),x)
            xy=test._coordinates;assert xy.shape==(1,shape[2]*shape[3],2) and xy.dtype==dtype
            assert xy[0,0].tolist()==[-1.,-1.] and xy[0,-1].tolist()==[1.,1.]
            coordinates.append({'shape':list(shape),'dtype':str(dtype),'neutral_exact':True})
    test(torch.randn((1,96,7,11),device='cuda',requires_grad=True)).sum().backward()
    assert test.project.weight.grad.abs().max()>0
    rejected=[]
    for label,fn in [('width',lambda:LatentContext(64)),('latent-count',lambda:LatentContext(96,latents=16)),
        ('unconditioned',lambda:HierarchicalStudent(16,2,latent=True)),
        ('channels',lambda:branch(torch.zeros((1,3,2,2),device='cuda'))),
        ('empty',lambda:branch(torch.zeros((0,96,2,2),device='cuda'))),
        ('extent',lambda:branch(torch.zeros((1,96,1,8193),device='cuda')))]:
        try:fn()
        except ValueError:rejected.append(label)
        else:raise AssertionError(label+' accepted')
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'source_checkpoint_sha256':sha(a.model/'student-private.pt'),
        'neutral_tests':rows,'coordinate_tests':coordinates,'initial_projection_gradient_max':projection,
        'read_gradient_before_projection_step':before,'read_gradient_after_projection_step':after,
        'position_and_latents_receive_gradients':True,'strict_reload_exact':True,'rejected_inputs':rejected,
        'cached_inference_to_training_valid':True,
        'branch_parameters':sum(p.numel() for p in branch.parameters()),'total_parameters':sum(p.numel() for p in model.parameters()),
        'scope':'Mechanical tests and one private scratch projection update only, not a trained quality candidate.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['neutral_tests','source_checkpoint_sha256']},indent=2))


if __name__=='__main__':main()
