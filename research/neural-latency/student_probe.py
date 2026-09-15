# SPDX-License-Identifier: Apache-2.0
"""Bounded first student experiment against private matched NVIDIA textures.

One scene with a spatial holdout tests training mechanics, NOT generalization.
Every input pixel is retained by pixel-unshuffle; no input image downscaling.
Never installs anything in the game or changes the vendor runtime.
"""
import argparse,copy,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.depthwise=nn.Conv2d(width,width,3,padding=1,groups=width)
        self.expand=nn.Conv2d(width,width*2,1)
        self.project=nn.Conv2d(width*2,width,1)
        self.scale=nn.Parameter(torch.full((1,width,1,1),0.1))

    def forward(self,x):
        y=self.project(F.gelu(self.expand(self.depthwise(x))))
        return x+y*self.scale


class PixelStudent(nn.Module):
    def __init__(self,width=32,blocks=4):
        super().__init__()
        self.stem=nn.Conv2d(48,width,1)
        self.blocks=nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head=nn.Conv2d(width,48,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,x):
        # A reversible rearrangement: 4x4 pixels become 48 channels.
        features=self.stem(F.pixel_unshuffle(x,4))
        residual=F.pixel_shuffle(self.head(self.blocks(features)),4)
        return (x+0.25*residual).clamp(0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--extra-train-capture',type=Path,action='append',default=[])
    p.add_argument('--validation-capture',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--max-seconds',type=int,default=120)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--loss-border',type=int,choices=[0,16,32],default=16)
    a=p.parse_args()
    if not 1<=a.steps<=10000 or not 1<=a.max_seconds<=600:
        raise SystemExit('Use a bounded training run.')
    if a.width not in (16,32,48,64) or not 1<=a.blocks<=8:
        raise SystemExit('Architecture exceeds the prototype bounds.')
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.capture/'frame-0.json').read_text())
    assert meta['complete'] and meta['gpu_completed'] and meta['evaluate_result']==1
    assert meta['controls']['DLSSNR.Reset']==1
    torch.manual_seed(28411)
    rng=np.random.default_rng(28411)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    def load_capture(directory):
        manifest=json.loads((directory/'frame-0.json').read_text())
        assert manifest['complete'] and manifest['gpu_completed'] and manifest['evaluate_result']==1
        assert manifest['controls']==meta['controls'], 'Keep all runtime controls matched.'
        images={};hashes={}
        for role in ['color','output']:
            info=manifest['resources'][role]
            assert info['format']==10 and info['row_bytes']==info['width']*8
            raw=(directory/info['file']).read_bytes()
            assert len(raw)==info['height']*info['row_bytes']
            image=np.frombuffer(raw,dtype='<f2').reshape(info['height'],info['width'],4)[...,:3].astype(np.float32)
            assert np.isfinite(image).all()
            images[role]=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda()
            hashes[role]=hashlib.sha256(raw).hexdigest()
        return images,hashes
    images,hashes=load_capture(a.capture)
    source,target=images['color'],images['output']
    height,width=source.shape[2:]
    assert source.shape==target.shape and height%4==0 and width%4==0
    train_end=width if a.validation_capture else (width*3//5)//4*4
    holdout_start=(width*4//5)//4*4
    patch=128;batch=8
    assert train_end>=patch and height>=patch and width-holdout_start>=patch
    training_views=[(source,target,train_end)]
    training_hashes=[hashes]
    for directory in a.extra_train_capture:
        pair,pair_hashes=load_capture(directory)
        assert pair['color'].shape==source.shape
        training_views.append((pair['color'],pair['output'],width))
        training_hashes.append(pair_hashes)
    validation=None
    validation_hashes=None
    if a.validation_capture:
        validation,validation_hashes=load_capture(a.validation_capture)
        assert validation['color'].shape==source.shape
        assert all(validation_hashes['color']!=h['color'] for h in training_hashes), 'Validation input overlaps training.'
    model=PixelStudent(a.width,a.blocks).cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0001)
    report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'capture_hashes':hashes,'training_capture_hashes':training_hashes,'validation_capture_hashes':validation_hashes,
        'controls':meta['controls'],'native_shape':[height,width],
        'architecture':{'type':'pixel-unshuffle residual depthwise CNN','width':a.width,'blocks':a.blocks,
                        'parameters':sum(p.numel() for p in model.parameters()),'shuffle_factor':4},
        'data_split':{'scene_count':1,'training_view_count':len(training_views),
                      'train_columns':[0,train_end],'holdout_columns':None if validation else [holdout_start,width],
                      'separate_validation_view':validation is not None,
                      'patch':patch,'batch':batch,'no_resize':True,'loss_border':a.loss_border},
        'quality_gate_passed':False,
        'limitations':['One scene and first-reset frames only; a spatial or camera holdout is not cross-scene generalization.',
                       'No temporal training or quality validation.',
                       'Torch timing excludes D3D12 integration and is not a hidden-demo benchmark.'],
        'training':[]}
    with torch.no_grad():
        error=(validation['color'].clamp(0,1)-validation['output']).abs() if validation else (source.clamp(0,1)-target).abs()[:,:,:,holdout_start:]
        report['identity_holdout_mae']=float(error.mean())
    torch.cuda.synchronize()
    started=time.perf_counter()
    for step in range(a.steps):
        crops=[];labels=[]
        for _ in range(batch):
            train_source,train_target,right=training_views[int(rng.integers(len(training_views)))]
            y=int(rng.integers((height-patch)//4+1))*4
            x=int(rng.integers((right-patch)//4+1))*4
            crops.append(train_source[:,:,y:y+patch,x:x+patch])
            labels.append(train_target[:,:,y:y+patch,x:x+patch])
        x=torch.cat(crops).contiguous(memory_format=torch.channels_last)
        y=torch.cat(labels).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        predicted=model(x)
        if a.loss_border:
            border=a.loss_border
            predicted=predicted[:,:,border:-border,border:-border]
            y=y[:,:,border:-border,border:-border]
        pixel=(predicted-y).abs().mean()
        detail=((predicted[:,:,:,1:]-predicted[:,:,:,:-1])-(y[:,:,:,1:]-y[:,:,:,:-1])).abs().mean()
        detail=detail+((predicted[:,:,1:,:]-predicted[:,:,:-1,:])-(y[:,:,1:,:]-y[:,:,:-1,:])).abs().mean()
        loss=pixel+0.25*detail
        loss.backward();optimizer.step()
        if step%100==0 or step+1==a.steps:
            sample={'step':step+1,'loss':float(loss),'pixel_mae':float(pixel),'seconds':time.perf_counter()-started}
            report['training'].append(sample);print(json.dumps(sample),flush=True)
        if time.perf_counter()-started>=a.max_seconds:
            break
    report['completed_steps']=step+1
    torch.cuda.synchronize()
    report['training_seconds']=time.perf_counter()-started
    # Private derivative checkpoint; publication includes source and metrics only.
    torch.save({'architecture':report['architecture'],'state_dict':model.cpu().state_dict()},a.output/'student-private.pt')
    inference=copy.deepcopy(model).cuda().half().eval().to(memory_format=torch.channels_last)
    x=source.half().contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        predicted=inference(x)
        error=(predicted.float()-target).abs()
        def metrics(region):
            err=error[:,:,:,region]
            mse=float(err.square().mean())
            return {'mae':float(err.mean()),'rmse':mse**.5,'p99':float(torch.quantile(err.flatten(),.99)),
                    'max_abs':float(err.max()),'psnr_db':-10*np.log10(mse) if mse else None}
        report['quality']={'train_region':metrics(slice(0,train_end)), 'whole_frame':metrics(slice(None))}
        if not validation:
            report['quality']['spatial_holdout']=metrics(slice(holdout_start,width))
        np.save(a.output/'student-output.npy',predicted[0].permute(1,2,0).float().cpu().numpy())
        if validation:
            validation_output=inference(validation['color'].half().contiguous(memory_format=torch.channels_last))
            error=(validation_output.float()-validation['output']).abs()
            report['quality']['heldout_camera']=metrics(slice(None))
            np.save(a.output/'validation-output.npy',validation_output[0].permute(1,2,0).float().cpu().numpy())
        warmup=torch.cuda.Stream();warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):inference(x)
        torch.cuda.current_stream().wait_stream(warmup)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):graph_output=inference(x)
        for _ in range(5):graph.replay()
        torch.cuda.synchronize()
        times=[]
        for _ in range(30):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graph.replay();end.record();end.synchronize()
            times.append(start.elapsed_time(end))
        report['torch_graph_timing']={'median_ms':statistics.median(times),'p95_ms':float(np.percentile(times,95)),
                                      'samples_ms':times,'output_matches_eager':bool(torch.equal(graph_output,predicted))}
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('training','controls')},indent=2),flush=True)


if __name__=='__main__':main()
