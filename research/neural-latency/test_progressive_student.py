# SPDX-License-Identifier: Apache-2.0
"""Check neutral initialization, frozen gradients and full-image refinement math."""
import argparse,copy,io,json
from pathlib import Path
import torch
from progressive_student import FrozenStudentRefinement,load_first,configure_progressive
from student_training_pairs import read,sha,load_pair,rgb_loss
from fused_norm import FusedNorm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('first','capture','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(a.first)
    from compare_output_grade import read_capture
    controls,_,hashes=read_capture(a.capture);assert controls==record['controls'] and hashes in record['training_capture_hashes']
    torch.manual_seed(28411);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    source,target=load_pair({'path':a.capture,'capture_hashes':hashes},controls)
    model=FrozenStudentRefinement(first).cuda().to(memory_format=torch.channels_last)
    saved={k:v.clone() for k,v in model.first.state_dict().items()};rows=[]
    for dtype in (torch.float32,torch.float16):
        m=copy.deepcopy(model).to(dtype=dtype).eval();x=source.to(dtype=dtype)
        with torch.no_grad():
            value=m(x);reference=m.first(x);cached=m.refine_cached(m.first_ungraded(x),x)
            assert torch.equal(value,reference) and torch.equal(value,cached)
        rows.append({'dtype':str(dtype),'neutral_output_exact':True,'cached_output_exact':True})
    model.train();assert not model.first.training and model.refinement.training
    loss,_=rgb_loss(model(source),target);loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.first.parameters())
    head=model.refinement.network.head.weight;norm=float(head.grad.norm());assert norm>0
    # A disposable local step tests gradient plumbing, never creates a candidate.
    with torch.no_grad():head.add_(head.grad,alpha=-.01)
    assert all(torch.equal(v,saved[k]) for k,v in model.first.state_dict().items())
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=copy.deepcopy(model);restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    with torch.no_grad():assert torch.equal(model(source),restored(source))
    m=copy.deepcopy(model).half().eval();x=source.half();kernel=FusedNorm()
    with torch.inference_mode():
        baseline=m(x);configure_progressive(m,'all-conditioning',kernel);fused=m(x)
        assert torch.equal(baseline.view(torch.int16),fused.view(torch.int16))
    rejected=[]
    for name,bad in [('channels',source[:,:2]),('shape',source[:,:,:-1]),('dtype',source.half())]:
        try:model.refine_cached(source,bad)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'training_capture_hashes':hashes,
        'first_checkpoint_sha256':sha(a.first/'student-private.pt'),'cases':rows,'first_stage_unchanged':True,
        'first_stage_has_gradients':False,'head_gradient_norm':norm,'strict_roundtrip_exact':True,'all_fusion_output_exact':True,
        'rejected_inputs':rejected,'scope':'One training image, two precisions and one disposable local step. No validation image or saved candidate used.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
