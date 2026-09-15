# SPDX-License-Identifier: Apache-2.0
"""Audit training-only spectral errors and fix one auxiliary-loss coefficient."""
import argparse,json,statistics
from pathlib import Path
import torch
from collect_training_brightness import audited_brightness
from student_training_pairs import read,sha,training_pairs,load_pair,rgb_loss
from student_probe import HierarchicalStudent
from output_grade import GradedStudent
from frequency_loss import frequency_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','photos','images','brightness','baseline','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    record=read(a.model/'result.json');pairs=training_pairs(a.base,a.photos,a.images,read(a.baseline/'result.json'))
    for r in audited_brightness(a.brightness,a.base,record['controls'],a.photos,a.images):
        pairs.append({'label':r['label'],'group':'bright-photos','path':a.base/(r['label']+'-state')/'capture','capture_hashes':r['capture_hashes']})
    assert len(pairs)==62 and [r['capture_hashes'] for r in pairs]==record['training_capture_hashes']
    val={record['validation_capture_hashes']['color']}|{r['capture_hashes']['color'] for r in record['extra_validation']}
    assert not val & {r['capture_hashes']['color'] for r in pairs}
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True),record['architecture']['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval().to(memory_format=torch.channels_last)
    fy=torch.fft.fftfreq(1080,device='cuda')[:,None];fx=torch.fft.fftfreq(1920,device='cuda')[None,:]
    radius=(fy.square()+fx.square()).sqrt()
    bands={'dc':radius==0,'below_1_per_128':(radius>0)&(radius<1/128),
        '1_per_128_to_1_per_32':(radius>=1/128)&(radius<1/32),
        '1_per_32_to_1_per_8':(radius>=1/32)&(radius<1/8),'at_least_1_per_8':radius>=1/8}
    assert torch.stack(list(bands.values())).sum(0).eq(1).all()
    rows=[]
    for pair in pairs:
        source,target=load_pair(pair,record['controls'])
        with torch.no_grad():prediction=model(source)
        prediction=prediction.detach().requires_grad_(True)
        loss,pixel=rgb_loss(prediction,target);ff=frequency_loss(prediction,target)
        g=torch.autograd.grad(loss,prediction)[0];gf=torch.autograd.grad(ff,prediction)[0]
        norm=torch.linalg.vector_norm(g);fnorm=torch.linalg.vector_norm(gf)
        with torch.no_grad():
            delta=prediction-target;power=torch.fft.fft2(delta,norm='ortho').abs().square()
            mse=delta.square().mean();parseval=power.mean()
            assert torch.allclose(mse,parseval,rtol=1e-5,atol=1e-10)
            total=power.sum();fractions={n:float(power[...,mask].sum()/total) for n,mask in bands.items()}
        rows.append({k:v for k,v in pair.items() if k!='path'}|{'mae':float(pixel),'mse':float(mse),
            'frequency_loss':float(ff),'spectral_energy_fractions':fractions,'ordinary_output_gradient_l2':float(norm),
            'frequency_output_gradient_l2':float(fnorm),'gradient_cosine':float((g*gf).sum()/(norm*fnorm).clamp(min=1e-30)),
            'coefficient_for_quarter_gradient_norm':float(.25*norm/fnorm)})
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    coefficient=statistics.median(r['coefficient_for_quarter_gradient_norm'] for r in rows)
    assert 0<coefficient<=100
    groups={g:{'mae':statistics.mean(r['mae'] for r in rows if r['group']==g),
        'spectral_energy_fractions':{b:statistics.mean(r['spectral_energy_fractions'][b] for r in rows if r['group']==g) for b in bands},
        'gradient_cosine':statistics.mean(r['gradient_cosine'] for r in rows if r['group']==g)} for g in sorted({r['group'] for r in rows})}
    report={'complete':True,'training_only':True,'optimizer_steps':0,'weights_unchanged':True,'parameter_grad_fields_unchanged':True,
        'target_achieved':False,'quality_gate_passed':False,'checkpoint_sha256':sha(a.model/'student-private.pt'),
        'rows':rows,'groups':groups,'frequency_coefficient':coefficient,'coefficient_rule':'Median of 0.25 * ordinary/frequency output-gradient L2 norms on all 62 training frames; fixed before training and validation.',
        'limitations':['Fourier energy describes squared RGB error, not a perceptual or causal diagnosis.',
            'The coefficient balances output-space gradients at the trained control; parameter-gradient balance during training is not guaranteed.',
            'Only training first-reset images are inspected. No optimizer steps, sample or game launches.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'groups':groups,'frequency_coefficient':coefficient},indent=2))


if __name__=='__main__':main()
