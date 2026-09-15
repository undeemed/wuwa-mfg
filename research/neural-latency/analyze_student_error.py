# SPDX-License-Identifier: MIT
"""Separate broad tone/structure errors from fine-detail errors in a private capture."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--prediction',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.capture/'frame-0.json').read_text())
assert manifest['complete'] and manifest['gpu_completed'] and manifest['evaluate_result']==1
images={}
for role in ('color','output'):
    info=manifest['resources'][role]
    assert info['format']==10 and info['row_bytes']==info['width']*8
    assert Path(info['file']).name==info['file']
    raw=(a.capture/info['file']).read_bytes()
    assert len(raw)==info['height']*info['row_bytes']
    images[role]=np.frombuffer(raw,dtype='<f2').reshape(info['height'],info['width'],4)[...,:3].astype(np.float32)
predicted=np.load(a.prediction,allow_pickle=False)
assert predicted.shape==images['output'].shape and all(np.isfinite(x).all() for x in [predicted,*images.values()])
h,w=predicted.shape[:2]
frequency=np.sqrt(np.fft.fftfreq(h)[:,None]**2+np.fft.rfftfreq(w)[None,:]**2)
edges=[0,1/256,1/64,1/16,1/4,np.inf]
labels=['wavelength >=256px','64..256px','16..64px','4..16px','<4px']
weight=np.full((1,w//2+1),2.);weight[:,0]=1
if w%2==0:weight[:,-1]=1
def spectrum(delta):
    power=np.zeros(frequency.shape,np.float64)
    for channel in range(3):
        transformed=np.fft.rfft2(delta[...,channel])
        power+=(transformed.real**2+transformed.imag**2)*weight/(h*w)**2/3
    total=float(power.sum())
    direct_mse=float(np.mean(delta.astype(np.float64)**2))
    assert abs(total-direct_mse)<=max(1e-12,direct_mse*1e-6), 'Parseval consistency failed'
    return {'mse':direct_mse,'mae':float(np.abs(delta).mean()),'dc_fraction':float(power[0,0]/total) if total else 0.,
            'bands':[{'spatial_wavelength':label,'mse':float(power[(frequency>=lo)&(frequency<hi)].sum()),
                       'fraction':float(power[(frequency>=lo)&(frequency<hi)].sum()/total) if total else 0.}
                      for label,lo,hi in zip(labels,edges,edges[1:])]}
report={'schema':1,'shape':list(predicted.shape),'quality_gate_passed':False,
        'teacher_effect':spectrum(images['output']-images['color'].clip(0,1)),
        'student_error':spectrum(predicted-images['output']),
        'limitation':'One first-reset camera view. FFT bands describe error scale, not perceptual or temporal quality. Image borders are periodic in this diagnostic.'}
with a.output.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2);handle.write('\n')
print(json.dumps(report,indent=2))
