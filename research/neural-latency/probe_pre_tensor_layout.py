# SPDX-License-Identifier: Apache-2.0
"""Compare a private native allocation prefix with reconstructed stem layouts.

This is a bounded layout diagnostic, not a numerical-parity or speed claim.
Uses the pinned MLX-DLSS first block and external local weights. No game access.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-channels',action='store_true',help='Also test channel axes split as 2x16 and 4x8.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((args.trial/'pre-tensor/metadata.json').read_text())
    assert meta['complete'] and meta['gpu_completed'] and meta['noise_counter']==0
    assert [meta['width'],meta['height']]==[1920,1080] and meta['bytes']==32*1024*1024
    raw=(args.trial/'pre-tensor'/meta['file']).read_bytes()
    assert len(raw)==meta['bytes']
    native=np.frombuffer(raw,np.uint8)
    capture=args.trial/'capture'
    frame=json.loads((capture/'frame-0.json').read_text())
    assert frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1
    controls=frame['controls'];assert controls['DLSSNR.Reset']==1
    color=frame['resources']['color'];assert color['format']==10 and color['row_bytes']==1920*8
    color_raw=(capture/color['file']).read_bytes()
    rgb=np.frombuffer(color_raw,'<f2').reshape(1080,1920,4)[...,:3].astype(np.float32)
    sys.path.insert(0,str(args.source/'python'))
    import torch
    from mlxdlss import model as reference
    from mlxdlss.pipeline import NeuralRenderingPipeline
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    torch.backends.cuda.matmul.allow_tf32=False
    pipeline=NeuralRenderingPipeline.from_safetensors(args.weights,device='cuda',precision='fast')
    # Same native E4M3 round trip used by the existing full-reference comparisons.
    reference.e4m3_round_trip=lambda x:x.clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype)
    automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
    # Native preprocessor grid is 240x144 tiles, each covering 8x8 input pixels.
    # This is an explicit layout hypothesis, not a proven logical tensor extent.
    features=make_features(rgb,geometry=NetworkGeometry(1920,1080,1920,1152),frame_index=0,
        normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
        local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
    with torch.inference_mode():
        x=torch.from_numpy(features[None]).cuda().half()
        adapter=x@pipeline.model.weight('block0.layer0.input_adapter_weight')
        block=pipeline.model._window(adapter,0,head_count=1)
        pool=reference.average_pool2(block)
        tensors={}
        for name,value in [('adapter',adapter),('block0',block),('pool',pool)]:
            quantized=reference.e4m3_round_trip(value).to(torch.float8_e4m3fn)
            tensors[name]=quantized[0].view(torch.uint8).cpu().numpy()
            np.save(args.output/(name+'-private.npy'),tensors[name])
        lut=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    del pipeline,x,adapter,block,pool
    torch.cuda.empty_cache()
    rng=np.random.default_rng(719)
    candidates=[]
    def score(tensor,tile,order,indices,split=None):
        height,width,channels=tensor.shape
        dims=(height//tile,width//tile,tile,tile,*split) if split else (height//tile,width//tile,tile,tile,channels)
        remainder=indices.copy();coords=[None]*len(dims)
        for axis in reversed(order):
            coords[axis]=remainder%dims[axis];remainder=remainder//dims[axis]
        channel=coords[4]*split[1]+coords[5] if split else coords[4]
        expected=tensor[coords[0]*tile+coords[2],coords[1]*tile+coords[3],channel]
        observed=native[indices]
        a,b=lut[expected].astype(np.float64),lut[observed].astype(np.float64)
        valid=np.isfinite(a)&np.isfinite(b)
        if valid.sum()<len(a)*.99:return {'correlation':-1,'finite_fraction':float(valid.mean())}
        a=a[valid];b=b[valid]
        return {'correlation':float(np.corrcoef(a,b)[0,1]),'mae':float(np.abs(a-b).mean()),
                'rmse':float(np.sqrt(np.mean((a-b)**2))),'exact_byte_fraction':float(np.mean(expected[valid]==observed[valid])),
                'finite_fraction':float(valid.mean())}
    for name,tensor in tensors.items():
        extent=min(native.size,tensor.size)
        indices=rng.choice(extent,size=8192,replace=False)
        for tile in [4,8,16]:
            for split in ([None,(2,16),(4,8)] if args.split_channels else [None]):
                for order in itertools.permutations(range(6 if split else 5)):
                    metrics=score(tensor,tile,order,indices,split)
                    candidates.append({'stage':name,'tile':tile,'axis_order':list(order),
                                       'channel_split':split,'search_sample':metrics})
        print(json.dumps({'searched_stage':name,'candidates_so_far':len(candidates)}),flush=True)
    candidates.sort(key=lambda item:item['search_sample']['correlation'],reverse=True)
    for candidate in candidates[:10]:
        tensor=tensors[candidate['stage']]
        indices=rng.choice(min(native.size,tensor.size),size=65536,replace=False)
        candidate['fresh_sample']=score(tensor,candidate['tile'],candidate['axis_order'],indices,candidate['channel_split'])
    report={'schema':1,'native_prefix_sha256':hashlib.sha256(raw).hexdigest(),
        'input_sha256':hashlib.sha256(color_raw).hexdigest(),'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'network_extent_hypothesis':[1152,1920],'prefix_bytes':len(raw),'candidate_count':len(candidates),
        'axis_names':['tile_y','tile_x','inner_y','inner_x','channel_or_channel_high','channel_low_when_split'],'top_candidates':candidates[:10],
        'quality_gate_passed':False,'layout_verified':False,
        'limitations':['Allocation prefix may span multiple tensors or unpublished scratch data.',
            'Only declared tiled axis permutations searched; arbitrary lane/channel maps and tensor offsets are not covered.',
            'Reference first block is not numerically proven against NVIDIA.',
            'Fresh samples can overlap the search sample; no data fitting or quality acceptance is performed.',
            'This is not a model-performance benchmark.']}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
