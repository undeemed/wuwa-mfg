# SPDX-License-Identifier: Apache-2.0
"""Measure a branch batching candidate against the recovered reference."""
import argparse,json,statistics,sys,hashlib
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',action='store_true')
p.add_argument('--fused-gate',action='store_true')
a=p.parse_args()
if a.output.exists():raise SystemExit('Use a fresh output file.')
sys.path.insert(0,str(a.source/'python'))
from mlxdlss import model as reference
from mlxdlss.pipeline import NeuralRenderingPipeline
from fused_norm import FusedNorm
import batched_ffn
torch.manual_seed(54271)
torch.backends.cuda.matmul.allow_tf32=False
pipeline=NeuralRenderingPipeline.from_safetensors(a.weights,device='cuda',precision='fast')
fused=FusedNorm()
reference.e4m3_round_trip=lambda x:x.clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype)
original_norm=reference.vendor_cosine_normalize
reference.vendor_cosine_normalize=lambda x:fused(x) if x.shape[-1]==32 else original_norm(x)
reference.vendor_approximate_softmax=fused.softmax
original_bias=reference.recover_attention_bias_layout
bias_cache={}
def bias_cached(x):
    key=(x.data_ptr(),tuple(x.shape),x.dtype,x.device)
    if key not in bias_cache:bias_cache[key]=original_bias(x)
    return bias_cache[key]
reference.recover_attention_bias_layout=bias_cached
report={'schema':1,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        'scope':'PyTorch reference research; not vendor parity or a game speedup.', 'cases':[]}

def compare(x,y):
    delta=(x.float()-y.float()).abs()
    return {'elements':x.numel(),'unequal':int(torch.count_nonzero(x!=y)),
            'mae':float(delta.mean()),'max_abs':float(delta.max())}

def graph_time(fn):
    warmup=torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(warmup)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=fn()
    for _ in range(3):graph.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(12):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        samples.append(start.elapsed_time(end))
    return output.clone(),{'median_ms':statistics.median(samples),'samples_ms':samples}

with torch.inference_mode():
    for index,rows in [(5,513),(9,127),(15,73)]:
        prefix=f'block{index}.layer0.'
        params={name:pipeline.model.weight(prefix+suffix) for name,suffix in
            [('expansion_weight','ffn_expand_weight'),('branch_projection_weight','ffn_branch_projection_weight'),
             ('output_projection_weight','ffn_output_projection_weight')]}
        channels=params['output_projection_weight'].shape[0]
        for dtype in [torch.float16,torch.float32]:
            weights={k:v.to(dtype) for k,v in params.items()}
            for scale in [0.1,1,4]:
                x=torch.randn((1,rows,3,channels),device='cuda',dtype=dtype)*scale
                old=reference.branched_feed_forward(x,**weights)
                new=batched_ffn.branched_feed_forward(x,**weights)
                report['cases'].append({'block':index,'dtype':str(dtype),'scale':scale,**compare(old,new)})
    prefix='block23.layer0.'
    weights={name:pipeline.model.weight(prefix+suffix) for name,suffix in
        [('first_projection_weight','first_projection_weight'),('expand_weight','group_expand_weight'),
         ('project_weight','group_project_weight')]}
    for dtype in [torch.float16,torch.float32]:
        typed={k:v.to(dtype) for k,v in weights.items()}
        x=torch.randn((1,127,3,512),device='cuda',dtype=dtype)*.1
        old=reference.split_group_feed_forward(x,**typed)
        new=batched_ffn.split_group_feed_forward(x,**typed)
        report['cases'].append({'block':23,'dtype':str(dtype),**compare(old,new)})
    print(json.dumps({'cases':report['cases']},indent=2),flush=True)
    yy,xx=np.indices((320,320),dtype=np.float32)
    image=np.stack((xx/319,yy/319,.5+.3*np.sin(xx*.21)*np.cos(yy*.13)),axis=-1)
    tensor=torch.from_numpy(pipeline.prepare(image).features[None]).cuda().half()
    old,old_times=graph_time(lambda:pipeline.model(tensor))
    if a.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            pipeline.model(tensor)
            torch.cuda.synchronize()
        report['profile_top_ops']=[{'name':event.key,'self_cuda_us':event.self_device_time_total,'calls':event.count}
            for event in sorted(profiler.key_averages(),key=lambda x:x.self_device_time_total,reverse=True)[:20]]
    reference.branched_feed_forward=batched_ffn.branched_feed_forward
    reference.split_group_feed_forward=batched_ffn.split_group_feed_forward
    new,new_times=graph_time(lambda:pipeline.model(tensor))
    report['full_model']={'shape':list(tensor.shape),'baseline':old_times,'candidate':new_times,**compare(old,new)}
    if a.fused_gate:
        reference.quadratic_gate_activation=fused.gate
        gated,gated_times=graph_time(lambda:pipeline.model(tensor))
        report['fused_gate_model']={'timing':gated_times,**compare(new,gated)}
    report['all_tested_values_equal']=all(case['unequal']==0 for case in report['cases']) and report['full_model']['unequal']==0
    report['fp16_tested_values_equal']=all(case['unequal']==0 for case in report['cases'] if case['dtype']=='torch.float16') and report['full_model']['unequal']==0 and report.get('fused_gate_model',{'unequal':0})['unequal']==0
    report['fp32_replacement_rejected']=any(case['unequal'] for case in report['cases'] if case['dtype']=='torch.float32')
    report['quality_gate_passed']=False
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
if not report['fp16_tested_values_equal']:raise SystemExit('Rejected: FP16 outputs differ.')
