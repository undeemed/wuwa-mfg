# SPDX-License-Identifier: Apache-2.0
"""Compare first-reset vendor RGB against the recovered model on identical input."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--inspect-only',action='store_true')
p.add_argument('--precision',choices=['fast','reference'],default='fast')
p.add_argument('--network-height',type=int)
p.add_argument('--batched-ffn',action='store_true',help='Experimental FP16 branch batching; FP32 keeps the reference.')
p.add_argument('--fused-gate',action='store_true')
p.add_argument('--graph',action='store_true',help='Time fixed-shape CUDA Graph replay on the complete captured input.')
p.add_argument('--noise-frame',type=int,choices=range(4),default=0,
               help='Diagnostic noise counter; the vendor counter is not yet captured.')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.source/'python'))
metadata=json.loads((a.capture/'frame-0.json').read_text())
assert metadata['complete'] and metadata['gpu_completed'] and metadata['evaluate_result']==1
controls=metadata['controls']
assert controls['DLSSNR.Reset']==1, 'This script is a no-history, first-reset comparison.'
assert controls['DLSSNR.Hint.Render.Preset']==0, 'Other weight slots are not recovered.'
images={}
hashes={}
for role in ['color','output']:
    info=metadata['resources'][role]
    assert info['format']==10 and info['row_bytes']==info['width']*8 and info['rows']==info['height']
    file=a.capture/info['file']
    assert file.stat().st_size==info['height']*info['row_bytes']
    values=np.fromfile(file,dtype='<f2').reshape(info['height'],info['width'],4)[...,:3].astype(np.float32)
    assert np.isfinite(values).all(), f'{role} is nonfinite'
    images[role]=values
    hashes[role]=hashlib.sha256(file.read_bytes()).hexdigest()
source=images['color']; vendor=images['output']
assert source.shape==vendor.shape
height,width=source.shape[:2]
assert [width,height]==[controls['DLSSNR.Width'],controls['DLSSNR.Height']]
from PIL import Image
def preview(name,values):
    Image.fromarray(np.rint(np.clip(values,0,1)*255).astype(np.uint8)).save(a.output/(name+'.png'))
preview('model-input',source);preview('vendor-output',vendor)
stats={'schema':1,'capture_hashes':hashes,'source_shape':list(source.shape),'controls':controls,
       'input_mean':float(source.mean()),'input_range':[float(source.min()),float(source.max())],
       'vendor_effect_mae':float(np.abs(vendor-source).mean()),
       'vendor_range':[float(vendor.min()),float(vendor.max())], 'quality_gate_passed':False}
(a.output/'capture-summary.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2),flush=True)
if a.inspect_only:raise SystemExit()
import torch
torch.backends.cuda.matmul.allow_tf32=False
from mlxdlss import model as reference
from mlxdlss.pipeline import NeuralRenderingPipeline
from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
from fused_norm import FusedNorm
print('Loading recovered model; preserving complete captured resolution.',flush=True)
pipeline=NeuralRenderingPipeline.from_safetensors(a.weights,device='cuda',precision=a.precision)
fused=FusedNorm()
original_norm=reference.vendor_cosine_normalize
original_softmax=reference.vendor_approximate_softmax
original_round=reference.e4m3_round_trip
reference.vendor_cosine_normalize=lambda x: fused(x) if x.device.type=='cuda' and x.shape[-1]==32 else original_norm(x)
reference.vendor_approximate_softmax=lambda x: fused.softmax(x) if x.device.type=='cuda' and 2<=x.shape[-1]<=2048 and x.shape[-1]%2==0 else original_softmax(x)
reference.e4m3_round_trip=lambda x: x.clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype) if x.device.type=='cuda' else original_round(x)
bias_layout=reference.recover_attention_bias_layout
bias_cache={}
def cached_bias(bias):
    key=(bias.data_ptr(),tuple(bias.shape),bias.dtype,bias.device)
    if key not in bias_cache:bias_cache[key]=bias_layout(bias)
    return bias_cache[key]
reference.recover_attention_bias_layout=cached_bias
if a.fused_gate:
    reference.quadratic_gate_activation=fused.gate
if a.batched_ffn:
    import batched_ffn
    original_branched=reference.branched_feed_forward
    original_split=reference.split_group_feed_forward
    reference.branched_feed_forward=lambda x,**kw: batched_ffn.branched_feed_forward(x,**kw) if x.dtype==torch.float16 else original_branched(x,**kw)
    reference.split_group_feed_forward=lambda x,**kw: batched_ffn.split_group_feed_forward(x,**kw) if x.dtype==torch.float16 else original_split(x,**kw)
automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
prepared=pipeline.prepare(source,frame_index=a.noise_frame,normalized_style=controls['DLSSNR.Style']/128,
    local_tone_strength=controls['DLSSNR.LocalToneStrength'],
    local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
if a.network_height:
    assert a.network_height>=height and a.network_height%64==0
    prepared.geometry=NetworkGeometry(width,height,prepared.geometry.network_width,a.network_height)
    prepared.features=make_features(source,geometry=prepared.geometry,frame_index=a.noise_frame,
        normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
        local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
    stats['custom_network_height_diagnostic']=a.network_height
print(f'Prepared actual model features {prepared.features.shape}; executing first-frame inference.',flush=True)
start=time.perf_counter()
if a.graph:
    with torch.inference_mode():
        tensor=torch.from_numpy(prepared.features[None]).to('cuda',pipeline.dtype)
        warmup=torch.cuda.Stream();warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):pipeline.model(tensor)
        torch.cuda.current_stream().wait_stream(warmup)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):graph_output=pipeline.model(tensor)
        for _ in range(5):graph.replay()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(20):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();graph.replay();end.record();end.synchronize();samples.append(begin.elapsed_time(end))
        head=graph_output.float().cpu().numpy()[0]
        stats['cuda_graph']={'median_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
                             'samples_ms':samples,'trial_wall_seconds':time.perf_counter()-start,
                             'scope':'Fixed full-resolution input in PyTorch; no D3D12 integration or vendor speedup.'}
        network_seconds=float(np.median(samples))/1000
    stats['network_timing_kind']='CUDA Graph median GPU interval'
else:
    head=pipeline.run_features(prepared.features)
    network_seconds=time.perf_counter()-start
    stats['network_timing_kind']='Eager wall time, including output transfer'
print(f'Model timing {network_seconds:.4f}s ({stats["network_timing_kind"]}); comparing pixels.',flush=True)
result=pipeline.finish(prepared,head,intensity=controls['DLSSNR.Intensity'],network_seconds=network_seconds)
predicted=result.image
np.save(a.output/'reconstruction.npy',predicted)
preview('reconstruction',predicted)
preview('difference-amplified-8x',np.abs(predicted-vendor)*8)
error=np.abs(predicted-vendor)
mse=float(np.mean(error**2))
def corr(x,y):
    x=x.reshape(-1).astype(np.float64);y=y.reshape(-1).astype(np.float64)
    return float(np.corrcoef(x,y)[0,1])
def hp(x):
    return x[1:-1,1:-1]-(x[:-2,1:-1]+x[2:,1:-1]+x[1:-1,:-2]+x[1:-1,2:])*0.25
stats.update(precision=a.precision,network_extent=list(result.network_extent),network_seconds=network_seconds,
             noise_frame=a.noise_frame,
             batched_ffn=a.batched_ffn and a.precision=='fast',
             fused_gate=a.fused_gate,
             weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
             comparison={'rgb_mae':float(error.mean()),'rgb_rmse':mse**0.5,
             'rgb_max_abs':float(error.max()),'rgb_p99_abs':float(np.percentile(error,99)),
             'psnr_db':float(-10*np.log10(mse)) if mse else None,
             'rgb_correlation':corr(predicted,vendor),
             'effect_correlation':corr(predicted-source,vendor-source),
             'highpass_correlation':corr(hp(predicted),hp(vendor)),
             'reconstruction_effect_mae':float(np.abs(predicted-source).mean())},
             limitation='One first-reset scene; noise counter, controls and reconstruction remain unproven against vendor. Reconstruction timing is not a vendor-runtime benchmark. No speed/quality claim.',
             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
(a.output/'comparison.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2),flush=True)
