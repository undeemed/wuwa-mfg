# SPDX-License-Identifier: Apache-2.0
"""Bounded first student experiment against private matched NVIDIA textures.

One scene with a spatial holdout tests training mechanics, NOT generalization.
Every input pixel is retained by pixel-unshuffle; no input image downscaling.
Never installs anything in the game or changes the vendor runtime.
"""
import argparse,copy,hashlib,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__()
        self.depthwise=nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,groups=width)
        self.expand=nn.Conv2d(width,width*2,1)
        self.project=nn.Conv2d(width*2,width,1)
        self.scale=nn.Parameter(torch.full((1,width,1,1),0.1))
        self.fused_residual_backend=None

    def forward(self,x):
        y=self.project(F.gelu(self.expand(self.depthwise(x))))
        if self.fused_residual_backend is not None:
            return self.fused_residual_backend.residual_scale_add(x,y,self.scale.detach())
        return x+y*self.scale


class PixelStudent(nn.Module):
    def __init__(self,width=32,blocks=4,noise=False,dilations=None):
        super().__init__()
        self.stem=nn.Conv2d(96 if noise else 48,width,1)
        self.blocks=nn.Sequential(*(ResidualBlock(width,d) for d in (dilations or [1]*blocks)))
        self.head=nn.Conv2d(width,48,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,x):
        # A reversible rearrangement: 4x4 pixels become 48 channels.
        features=self.stem(F.pixel_unshuffle(x,4))
        residual=F.pixel_shuffle(self.head(self.blocks(features)),4)
        return (x[:,:3]+0.25*residual).clamp(0,1)


class HierarchicalStudent(nn.Module):
    """Learned multiscale features with full-resolution pixel rearrangement/skips."""
    def __init__(self,width=16,blocks=2,noise=False,affine=False,attention=False,conditioned=False,latent=False):
        super().__init__()
        if latent and not conditioned:raise ValueError('Latent context requires the conditioned RGB hierarchy.')
        if conditioned and (noise or affine or attention):
            raise ValueError('Decoder conditioning is tested only with the plain RGB hierarchy.')
        widths=[width,width*2,width*4,width*6]
        self.stem=nn.Conv2d(96 if noise else 48,width,1)
        self.encoder=nn.ModuleList(nn.Sequential(*(ResidualBlock(w) for _ in range(blocks))) for w in widths)
        self.down=nn.ModuleList(nn.Conv2d(widths[i],widths[i+1],2,stride=2) for i in range(3))
        self.up=nn.ModuleList(nn.Conv2d(widths[i+1],widths[i],1) for i in range(3))
        self.decoder=nn.ModuleList(nn.Sequential(*(ResidualBlock(w) for _ in range(blocks))) for w in widths[:3])
        self.context=nn.Conv2d(widths[-1],widths[-1],1)
        self.head=nn.Conv2d(width,48,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.affine_head=nn.Conv2d(widths[-1],12,1) if affine else None
        if self.affine_head is not None:
            nn.init.zeros_(self.affine_head.weight);nn.init.zeros_(self.affine_head.bias)
        self.fused_affine_backend=None
        self.reorder_decoder=False
        self.decoder_reorder_stages=(0,1,2)
        self.fused_decoder_backend=None
        self.fused_conditioning_backend=None
        self.global_attention=None
        if attention:
            from attention_student import GlobalAttention
            self.global_attention=GlobalAttention(widths[-1])
        self.decoder_conditioning=None
        if conditioned:
            from decoder_conditioning import DecoderConditioning
            self.decoder_conditioning=DecoderConditioning(widths[-1],widths[:3])
        self.latent_context=None
        if latent:
            from latent_context import LatentContext
            self.latent_context=LatentContext(widths[-1])

    def forward(self,x,*,fused_output_backend=None,grade_parameters=None):
        height,width=x.shape[-2:]
        # Only padding and reversible pixel-unshuffle touch the source image.
        padded=F.pad(x,(0,(-width)%32,0,(-height)%32),mode='reflect')
        value=self.stem(F.pixel_unshuffle(padded,4))
        skips=[]
        for i,stage in enumerate(self.encoder):
            value=stage(value)
            if i<3:skips.append(value);value=self.down[i](value)
        if self.latent_context is not None:value=self.latent_context(value)
        pooled=value.mean(dim=(2,3),keepdim=True)
        modulation=self.decoder_conditioning(pooled) if self.decoder_conditioning is not None else None
        value=value*(1+0.1*torch.tanh(self.context(pooled)))
        if self.global_attention is not None:value=self.global_attention(value)
        coefficients=self.affine_head(value) if self.affine_head is not None else None
        for i in (2,1,0):
            fused_conditioning=modulation is not None and self.fused_conditioning_backend is not None
            if modulation is not None:
                scale,shift=self.decoder_conditioning.stage(modulation,i)
            if self.reorder_decoder and i in self.decoder_reorder_stages:
                value=self.up[i](value)
                value=(self.fused_conditioning_backend.decoder_conditioned_add(value,skips[i],scale,shift)
                       if fused_conditioning else
                       self.fused_decoder_backend.decoder_upscale_add(value,skips[i])
                       if self.fused_decoder_backend is not None
                       else F.interpolate(value,size=skips[i].shape[-2:],mode='nearest')+skips[i])
            else:
                value=self.up[i](F.interpolate(value,size=skips[i].shape[-2:],mode='nearest'))
                value=(self.fused_conditioning_backend.decoder_conditioned_add(value,skips[i],scale,shift)
                       if fused_conditioning else value+skips[i])
            if modulation is not None and not fused_conditioning:
                value=value*(1+scale)+shift
            value=self.decoder[i](value)
        head=self.head(value)
        if fused_output_backend is not None:
            if coefficients is not None:
                raise ValueError('Fused student output does not support the affine branch.')
            return fused_output_backend.student_output(x[:,:3],head,grade_parameters)
        residual=F.pixel_shuffle(head,4)[:,:,:height,:width]
        if coefficients is not None:
            if self.fused_affine_backend is not None:
                return self.fused_affine_backend.affine_compose(x[:,:3],residual,coefficients)
            from affine_student import compose_affine
            return compose_affine(x[:,:3],residual,coefficients)
        return (x[:,:3]+0.25*residual).clamp(0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--extra-train-capture',type=Path,action='append',default=[])
    p.add_argument('--validation-capture',type=Path)
    p.add_argument('--extra-validation-capture',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--max-seconds',type=int,default=120)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--loss-border',type=int,choices=[0,16,32,64,96],default=16)
    p.add_argument('--noise-source',type=Path,help='Pinned MLX-DLSS source for deterministic first-frame noise channels.')
    p.add_argument('--dilations',help='One comma-separated dilation per residual block, e.g. 1,2,4,8,4,2.')
    p.add_argument('--patch-size',type=int,choices=[128,256,384],default=128)
    p.add_argument('--batch',type=int,choices=range(1,17),default=8)
    p.add_argument('--cosine-lr',action='store_true',help='Decay learning rate from .002 to .00002 over the bounded steps.')
    p.add_argument('--architecture',choices=['local','hierarchical','hierarchical-affine','hierarchical-attention','hierarchical-film','hierarchical-latent'],default='local')
    p.add_argument('--whole-frame',action='store_true',help='Train complete views, batch 1; requires a separate validation view.')
    p.add_argument('--output-grade-contract',type=Path,help='Learn a pre-grading residual, then apply the observed output grade. Requires a separate validation view.')
    p.add_argument('--evaluate-all-training',action='store_true',help='Report every full training view after fitting; requires separate validation views.')
    p.add_argument('--data-description',help='Describe a mixed-content dataset; scene count is then left unspecified rather than assumed to be one.')
    p.add_argument('--initial-lr',type=float,default=.002)
    p.add_argument('--final-lr',type=float,default=.00002)
    p.add_argument('--initialize-from',type=Path,help='Private matching student run; load weights only and start a fresh optimizer.')
    p.add_argument('--feature-targets',type=Path,help='Private native decoder hints used only during training.')
    p.add_argument('--feature-weight',type=float,default=.01)
    p.add_argument('--paired-gradient',choices=['mean','pcgrad'],help='Two training domains: first 30 scene views, then 16 photos. One example from each per optimizer step.')
    a=p.parse_args()
    if not 1<=a.steps<=10000 or not 1<=a.max_seconds<=600:
        raise SystemExit('Use a bounded training run.')
    if not all(math.isfinite(v) for v in (a.initial_lr,a.final_lr)) or not 0<a.final_lr<=a.initial_lr<=.002:
        raise SystemExit('Require finite learning rates with 0 < final <= initial <= .002.')
    if a.feature_targets and (a.architecture!='hierarchical' or not a.whole_frame or not a.output_grade_contract
                             or not math.isfinite(a.feature_weight) or not 0<a.feature_weight<=1):
        raise SystemExit('Feature hints require the graded whole-frame hierarchical model and a finite weight in (0,1].')
    if a.width not in (16,32,48,64) or not 1<=a.blocks<=8:
        raise SystemExit('Architecture exceeds the prototype bounds.')
    if a.paired_gradient and (a.architecture!='hierarchical' or not a.whole_frame or not a.output_grade_contract
                             or a.feature_targets or a.noise_source):
        raise SystemExit('Paired gradients require the plain graded whole-frame hierarchy without feature hints or noise.')
    dilations=list(map(int,a.dilations.split(','))) if a.dilations else [1]*a.blocks
    if len(dilations)!=a.blocks or any(d not in [1,2,4,8] for d in dilations):
        raise SystemExit('Expected one supported dilation per block.')
    if a.patch_size<=2*a.loss_border:raise SystemExit('Loss border removes the complete crop.')
    if a.whole_frame and (not a.validation_capture or a.batch!=1):
        raise SystemExit('Whole-frame training requires batch1 and a separate validation view.')
    if a.extra_validation_capture and not a.validation_capture:
        raise SystemExit('Extra validation views require a primary validation capture.')
    if a.architecture.startswith('hierarchical') and a.dilations:
        raise SystemExit('The hierarchical model uses its fixed multiscale topology, not a dilation list.')
    if a.output_grade_contract and not a.validation_capture:
        raise SystemExit('Output-grading experiments require a separate validation view.')
    if a.evaluate_all_training and not a.validation_capture:
        raise SystemExit('Full training-view evaluation requires separate validation views.')
    if a.architecture=='hierarchical-affine' and (not a.whole_frame or not a.output_grade_contract):
        raise SystemExit('The affine experiment requires whole-frame training and the observed output grade.')
    if a.architecture=='hierarchical-attention' and (not a.whole_frame or not a.output_grade_contract):
        raise SystemExit('The global-attention experiment requires whole-frame training and the observed output grade.')
    if a.architecture in ('hierarchical-film','hierarchical-latent') and (not a.whole_frame or not a.output_grade_contract or a.noise_source):
        raise SystemExit('Decoder conditioning requires graded whole-frame RGB training.')
    noise=None
    if a.noise_source:
        sys.path.insert(0,str(a.noise_source/'python'))
        from mlxdlss.features import deterministic_noise
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.capture/'frame-0.json').read_text())
    assert meta['complete'] and meta['gpu_completed'] and meta['evaluate_result']==1
    assert meta['controls']['DLSSNR.Reset']==1
    grade_parameters=None
    if a.output_grade_contract:
        from compare_output_grade import read_capture
        from output_grade import GradedStudent,contract_parameters,grade_torch
        grade_controls,_,grade_capture_hashes=read_capture(a.output_grade_contract/'capture')
        assert grade_controls==meta['controls'], 'Observed grade controls must match the training data.'
        grade_parameters=contract_parameters(a.output_grade_contract)
    torch.manual_seed(28411)
    rng=np.random.default_rng(28411)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    def load_capture(directory):
        nonlocal noise
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
        if a.noise_source:
            h,w=images['color'].shape[2:]
            if noise is None:
                noise=torch.from_numpy(deterministic_noise(h,w,0)).permute(2,0,1).unsqueeze(0).cuda()
            assert noise.shape==images['color'].shape
            images['input']=torch.cat([images['color'],noise],dim=1)
        else:images['input']=images['color']
        return images,hashes
    images,hashes=load_capture(a.capture)
    source,target=images['color'],images['output']
    height,width=source.shape[2:]
    assert source.shape==target.shape and height%4==0 and width%4==0
    train_end=width if a.validation_capture else (width*3//5)//4*4
    holdout_start=(width*4//5)//4*4
    patch=a.patch_size;batch=a.batch
    assert train_end>=patch and height>=patch and width-holdout_start>=patch
    training_views=[(images['input'],target,train_end)]
    training_hashes=[hashes]
    for directory in a.extra_train_capture:
        pair,pair_hashes=load_capture(directory)
        assert pair['color'].shape==source.shape
        training_views.append((pair['input'],pair['output'],width))
        training_hashes.append(pair_hashes)
    validation=None
    validation_hashes=None
    extra_validation=[]
    if a.validation_capture:
        validation,validation_hashes=load_capture(a.validation_capture)
        assert validation['color'].shape==source.shape
        assert all(validation_hashes['color']!=h['color'] for h in training_hashes), 'Validation input overlaps training.'
    validation_inputs={validation_hashes['color']} if validation_hashes else set()
    for directory in a.extra_validation_capture:
        pair,pair_hashes=load_capture(directory)
        assert pair['color'].shape==source.shape
        assert all(pair_hashes['color']!=h['color'] for h in training_hashes), 'Validation input overlaps training.'
        assert pair_hashes['color'] not in validation_inputs, 'Duplicate validation input.'
        validation_inputs.add(pair_hashes['color'])
        extra_validation.append((pair,pair_hashes))
    model=(HierarchicalStudent(a.width,a.blocks,bool(a.noise_source),a.architecture=='hierarchical-affine',a.architecture=='hierarchical-attention',a.architecture in ('hierarchical-film','hierarchical-latent'),a.architecture=='hierarchical-latent') if a.architecture.startswith('hierarchical')
           else PixelStudent(a.width,a.blocks,bool(a.noise_source),dilations)).cuda().to(memory_format=torch.channels_last)
    if grade_parameters is not None:model=GradedStudent(model,grade_parameters)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=0.0001)
    report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'capture_hashes':hashes,'training_capture_hashes':training_hashes,'validation_capture_hashes':validation_hashes,
        'controls':meta['controls'],'native_shape':[height,width],
        'architecture':{'type':'pixel-unshuffle residual depthwise CNN','width':a.width,'blocks':a.blocks,
                        'parameters':sum(p.numel() for p in model.parameters()),'shuffle_factor':4,
                        'dilations':dilations,'noise_channels':3 if noise is not None else 0,
                        'input_receptive_field_pixels':4*(1+2*sum(dilations))},
        'data_split':{'scene_count':None if a.data_description else 1,'training_view_count':len(training_views),
                      'train_columns':[0,train_end],'holdout_columns':None if validation else [holdout_start,width],
                      'separate_validation_view':validation is not None,
                      'patch':patch,'batch':batch,'no_resize':True,'loss_border':a.loss_border},
        'quality_gate_passed':False,
        'limitations':[a.data_description or 'One scene and first-reset frames only; a spatial or camera holdout is not cross-scene generalization.',
                       'No temporal training or quality validation.',
                       'Torch timing excludes D3D12 integration and is not a hidden-demo benchmark.'],
        'training':[]}
    report['optimization']={'initial_learning_rate':a.initial_lr,'cosine_decay':a.cosine_lr,
                            'final_learning_rate':a.final_lr if a.cosine_lr else a.initial_lr}
    report['architecture']['variant']=a.architecture
    if grade_parameters is not None:
        report['architecture']['explicit_output_grading']=list(grade_parameters)
        report['output_grading']={'parameters':list(grade_parameters),'observed_capture_hashes':grade_capture_hashes,
            'scope':'Unfitted algebraic approximation from the same runtime controls; other camera views reuse the parameters by inference.',
            'training':'Differentiable FP32 operations after the clamped learned residual.',
            'inference':'FP16 network, FP32 grading, FP16 final storage; all included in graph timing.'}
    report['data_split']['whole_frame_training']=a.whole_frame
    report['data_split']['validation_view_count']=len(validation_inputs)
    if a.whole_frame:report['data_split']['patch']=[height,width]
    if a.architecture.startswith('hierarchical'):
        report['architecture'].update(type='four-level hierarchical residual CNN with global context gate',
                                      widths=[a.width,a.width*2,a.width*4,a.width*6],
                                      blocks_per_stage=a.blocks,input_receptive_field_pixels='whole image via global context gate',
                                      internal_downsampling='learned features only; original pixels retained through input rearrangement and residual path')
        report['architecture'].pop('dilations')
        if a.architecture in ('hierarchical-film','hierarchical-latent'):
            report['architecture']['decoder_conditioning']={
                'source':'spatial mean of deepest encoder features, before the original context gate',
                'normalization':'LayerNorm across pooled channels, epsilon 1e-5',
                'hidden_width':a.width*6,'coefficient_channels':2*(a.width+a.width*2+a.width*4),
                'application':'per-channel value*(1+scale)+shift after upsample/skip combination at all three decoder stages',
                'projection_initialization':'zero; initially neutral',
                'conditioning_parameters':sum(v.numel() for v in model.network.decoder_conditioning.parameters())}
        if a.architecture=='hierarchical-latent':
            report['architecture']['latent_context']={'tokens':32,'width':a.width*6,'head_dim':32,
                'input_token_shape':[(height+31)//32,(width+31)//32],
                'stages':'input-to-latent cross-attention, one latent self-attention/MLP, latent-to-spatial cross-attention',
                'position_encoding':'normalized xy coordinates projected into features, shared by read keys and write queries',
                'application':'deepest encoder features before pooling, original context gate and decoder conditioning',
                'output_projection_initialization':'zero','residual_scale':.1,
                'parameters':sum(v.numel() for v in model.network.latent_context.parameters()),
                'operator':'PyTorch scaled_dot_product_attention, noncausal, dropout 0',
                'precision':'FP32 training, FP16 inference; all stages included in complete graph timings'}
        if a.architecture=='hierarchical-affine':
            report['architecture']['affine_field']={'channels':12,'cell_size':32,
                'interpolation':'bilinear, align_corners=False, over the padded image',
                'application':'source + .25*(full-resolution detail + learned 3x4 affine color correction)',
                'head_initialization':'zero'}
        if a.architecture=='hierarchical-attention':
            report['architecture']['type']='four-level hierarchical residual CNN with global context gate and spatial self-attention'
            report['architecture']['global_attention']={
                'cell_size':32,'token_shape':[(height+31)//32,(width+31)//32],
                'heads':a.width*6//32,'head_dim':32,'layers':1,
                'normalization':'per-token LayerNorm before learned Q/K/V',
                'operator':'PyTorch scaled_dot_product_attention, noncausal, dropout 0',
                'position_encoding':'none; attention receives the existing spatial CNN features',
                'residual_scale':.1,'output_projection_initialization':'zero',
                'precision':'FP32 training; FP16 inference',
                'timing_scope':'included in complete network and output-grading graph'}
    if noise is not None:
        report['limitations'].append('Noise is from the pinned reconstruction at counter 0; timing excludes its precomputation and input concatenation. No vendor intermediate feature parity is claimed.')
    if a.initialize_from:
        if a.initialize_from.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('Initialization checkpoints must remain private.')
        origin=json.loads((a.initialize_from/'result.json').read_text())
        checkpoint=torch.load(a.initialize_from/'student-private.pt',map_location='cpu',weights_only=True)
        assert origin['architecture']==checkpoint['architecture']==report['architecture']
        assert origin['controls']==report['controls'] and origin['native_shape']==report['native_shape']
        origin_hashes={h['color'] for h in origin['training_capture_hashes']}
        assert origin_hashes <= {h['color'] for h in training_hashes}
        assert not origin_hashes & validation_inputs
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        report['initialization']={'origin_result_sha256':hashlib.sha256((a.initialize_from/'result.json').read_bytes()).hexdigest(),
            'origin_checkpoint_sha256':hashlib.sha256((a.initialize_from/'student-private.pt').read_bytes()).hexdigest(),
            'origin_completed_steps':origin['completed_steps'],'origin_training_view_count':len(origin_hashes),
            'fresh_optimizer':True,'origin_validation_overlap':False}
    hint=None
    if a.feature_targets:
        from feature_hint import FeatureHints
        hint=FeatureHints(a.feature_targets,model.network,meta['controls'],training_hashes,
                         [validation_hashes]+[hashes for _,hashes in extra_validation])
        optimizer.add_param_group({'params':hint.projector.parameters()})
        report['feature_supervision']={**hint.info,'weight':a.feature_weight}
    with torch.no_grad():
        error=(validation['color'].clamp(0,1)-validation['output']).abs() if validation else (source.clamp(0,1)-target).abs()[:,:,:,holdout_start:]
        report['identity_holdout_mae']=float(error.mean())
        if grade_parameters is not None:
            pair=validation if validation else images
            graded_error=(grade_torch(pair['color'],grade_parameters)-pair['output']).abs()
            report['grade_only_holdout_mae']=float(graded_error.mean())
    torch.cuda.synchronize()
    started=time.perf_counter()
    paired_conflicts=torch.zeros((),device='cuda',dtype=torch.int32)
    if a.paired_gradient:
        assert len(training_views)==46 and batch==1
        from paired_gradient import paired_gradients
        report['paired_training']={'method':a.paired_gradient,'scene_frames':30,'photo_frames':16,
            'examples_per_step':2,'domain_weights':[.5,.5],
            'parameter_projection':'shared, symmetric against original gradients' if a.paired_gradient=='pcgrad' else 'none'}
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        if a.cosine_lr:
            lr=a.final_lr+.5*(a.initial_lr-a.final_lr)*(1+math.cos(math.pi*step/max(1,a.steps-1)))
            for group in optimizer.param_groups:group['lr']=lr
        if a.paired_gradient:
            loss,pixel,conflict=paired_gradients(model,training_views,rng,a.paired_gradient)
            paired_conflicts+=conflict
            feature_loss=None
        else:
            crops=[];labels=[]
            for _ in range(batch):
                selected_index=int(rng.integers(len(training_views)))
                train_source,train_target,right=training_views[selected_index]
                if a.whole_frame:
                    crops.append(train_source);labels.append(train_target)
                else:
                    y=int(rng.integers((height-patch)//4+1))*4
                    x=int(rng.integers((right-patch)//4+1))*4
                    crops.append(train_source[:,:,y:y+patch,x:x+patch])
                    labels.append(train_target[:,:,y:y+patch,x:x+patch])
            x=torch.cat(crops).contiguous(memory_format=torch.channels_last)
            y=torch.cat(labels).contiguous(memory_format=torch.channels_last)
            predicted=model(x)
            if a.loss_border:
                border=a.loss_border
                predicted=predicted[:,:,border:-border,border:-border]
                y=y[:,:,border:-border,border:-border]
            pixel=(predicted-y).abs().mean()
            detail=((predicted[:,:,:,1:]-predicted[:,:,:,:-1])-(y[:,:,:,1:]-y[:,:,:,:-1])).abs().mean()
            detail=detail+((predicted[:,:,1:,:]-predicted[:,:,:-1,:])-(y[:,:,1:,:]-y[:,:,:-1,:])).abs().mean()
            loss=pixel+0.25*detail
            feature_loss=hint.loss(training_hashes[selected_index]['color']) if hint is not None else None
            if feature_loss is not None:loss=loss+a.feature_weight*feature_loss
            loss.backward()
        optimizer.step()
        if step%100==0 or step+1==a.steps:
            sample={'step':step+1,'loss':float(loss),'pixel_mae':float(pixel),'seconds':time.perf_counter()-started}
            if hint is not None:sample['feature_mse']=float(feature_loss) if feature_loss is not None else None
            report['training'].append(sample);print(json.dumps(sample),flush=True)
        if time.perf_counter()-started>=a.max_seconds:
            break
    report['completed_steps']=step+1
    if a.paired_gradient:
        report['paired_training'].update(conflicting_steps=int(paired_conflicts),examples_seen=2*(step+1))
    torch.cuda.synchronize()
    report['training_seconds']=time.perf_counter()-started
    if hint is not None:
        pairs=[(view[0],hashes) for view,hashes in zip(training_views,training_hashes)]
        pairs += [(validation['input'],validation_hashes)]+[(pair['input'],hashes) for pair,hashes in extra_validation]
        report['feature_supervision']['final_feature_metrics']=hint.evaluate(model,pairs)
        hint.close(a.output)
    # Private derivative checkpoint; publication includes source and metrics only.
    torch.save({'architecture':report['architecture'],'state_dict':model.cpu().state_dict()},a.output/'student-private.pt')
    inference=copy.deepcopy(model).cuda().half().eval().to(memory_format=torch.channels_last)
    x=images['input'].half().contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        predicted=inference(x)
        if grade_parameters is not None:
            from fused_norm import FusedNorm
            from test_output_grade import graph_measure
            report['unfused_grade_graph_timing'],_=graph_measure(inference,x)
            inference.fused_backend=FusedNorm()
            if a.architecture=='hierarchical-affine':
                inference.network.fused_affine_backend=inference.fused_backend
            fused_predicted=inference(x)
            report['output_grading']['fused_matches_torch']=bool(torch.equal(fused_predicted,predicted))
            report['output_grading']['fused_max_abs']=float((fused_predicted-predicted).abs().max())
            if not report['output_grading']['fused_matches_torch']:
                raise RuntimeError('Fused grading changed this complete student output.')
            predicted=fused_predicted
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
            validation_output=inference(validation['input'].half().contiguous(memory_format=torch.channels_last))
            error=(validation_output.float()-validation['output']).abs()
            report['quality']['heldout_camera']=metrics(slice(None))
            np.save(a.output/'validation-output.npy',validation_output[0].permute(1,2,0).float().cpu().numpy())
        if extra_validation:
            report['extra_validation']=[]
            for index,(pair,pair_hashes) in enumerate(extra_validation):
                extra_output=inference(pair['input'].half().contiguous(memory_format=torch.channels_last))
                error=(extra_output.float()-pair['output']).abs()
                report['extra_validation'].append({'capture_hashes':pair_hashes,'quality':metrics(slice(None)),
                    'identity_mae':float((pair['color'].clamp(0,1)-pair['output']).abs().mean())})
                if grade_parameters is not None:
                    report['extra_validation'][-1]['grade_only_mae']=float((grade_torch(pair['color'],grade_parameters)-pair['output']).abs().mean())
                np.save(a.output/f'extra-validation-{index}.npy',extra_output[0].permute(1,2,0).float().cpu().numpy())
        if a.evaluate_all_training:
            report['training_view_evaluation']=[]
            for (train_input,train_target,_),train_hashes in zip(training_views,training_hashes):
                train_output=inference(train_input.half().contiguous(memory_format=torch.channels_last))
                error=(train_output.float()-train_target).abs()
                report['training_view_evaluation'].append({'capture_hashes':train_hashes,
                    'mae':float(error.mean()),'rmse':float(error.square().mean().sqrt())})
            report['mean_training_view_mae']=statistics.mean(item['mae'] for item in report['training_view_evaluation'])
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
