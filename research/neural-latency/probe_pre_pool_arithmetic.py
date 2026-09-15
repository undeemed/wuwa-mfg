# SPDX-License-Identifier: Apache-2.0
"""Compare pooled arithmetic using the best isolated direct-MMA first block.

Loads private captured inputs and weights. Never launches or modifies a game.
Pooling uses unpublished FP16 block values; no student training or speed claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from decode_pre_pool import compare, load_pooled
from fused_norm import FusedNorm
from mma_first_block import MmaFirstBlock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--trial', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'],
                                            text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
    if source_commit != '0ca2deab092fe6f3e331bf4f616271dbc64521d0':
        raise ValueError('Requires the pinned independent reference source.')
    weight_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weight_hash != 'fa6ebc71bc6f91347d51b3368b0ef6e952b6157b6d9a164e7db82cffb88d3143':
        raise ValueError('Requires the previously validated local weights.')
    sys.path.insert(0, str(args.source / 'python'))
    from mlxdlss import model as reference
    from mlxdlss.features import AutomaticMask, NetworkGeometry, make_features
    from mlxdlss.pipeline import NeuralRenderingPipeline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device='cuda', precision='fast')
    reference.e4m3_round_trip = lambda x: x.clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype)
    kernel = FusedNorm()
    mma = MmaFirstBlock(reference, pipeline.model, kernel)
    records = []
    with torch.inference_mode():
        noise = kernel.noise(1152, 1920, 0)
        for trial in args.trial:
            native, native_hash = load_pooled(trial)
            capture = trial / 'capture'
            frame = json.loads((capture / 'frame-0.json').read_text())
            controls = frame['controls']
            if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
                    and controls['DLSSNR.Reset'] == 1):
                raise ValueError('Requires a successful fenced first-reset frame.')
            color = frame['resources']['color']
            if (color['format'] != 10 or color['row_bytes'] != 1920*8
                    or [color['width'], color['height']] != [1920,1080]):
                raise ValueError('Requires tightly packed 1080p RGBA16F.')
            filename = Path(color['file'])
            if filename.name != str(filename):
                raise ValueError('Input texture filename must be a basename.')
            raw = (capture / filename).read_bytes()
            rgb = np.frombuffer(raw, '<f2').reshape(1080,1920,4)[...,:3].astype(np.float32)
            automatic = AutomaticMask(controls['DLSSNR.SkinStructureStrength'],
                controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
            features = make_features(rgb, geometry=NetworkGeometry(1920,1080,1920,1152), frame_index=0,
                normalized_style=controls['DLSSNR.Style']/128,
                local_tone_strength=controls['DLSSNR.LocalToneStrength'],
                local_structure_strength=controls['DLSSNR.LocalStructureStrength'], automatic_mask=automatic)
            value = torch.from_numpy(features[None]).cuda().half()
            value[0,...,:3] = noise
            adapter = kernel.mma(value.reshape(-1,16),
                pipeline.model.weight('block0.layer0.input_adapter_weight')).reshape(1,1152,1920,32)
            block = mma(adapter, attention_mma=True, seed_residual=True, seed_logits=True)
            a,b,c,d = block[:,0::2,0::2],block[:,1::2,0::2],block[:,0::2,1::2],block[:,1::2,1::2]
            variants = {
                'sequential_vertical_first': lambda: ((a+b)+c)+d,
                'paired_vertical': lambda: (a+b)+(c+d),
                'paired_horizontal': lambda: (a+c)+(b+d),
                'paired_diagonal': lambda: (a+d)+(b+c),
            }
            record = {'native_pooled_sha256': native_hash, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                      'variants': {}}
            for name, reduce in variants.items():
                pooled = (reduce()*0.25).clamp(-448,448).to(torch.float8_e4m3fn)
                expected = pooled[0].view(torch.uint8).cpu().numpy()
                result = compare(native, expected)
                record['variants'][name] = result
                print(json.dumps({'input_sha256': record['input_sha256'], 'variant': name,
                    **{k:result['all'][k] for k in ('mae','rmse','exact_byte_fraction')}}), flush=True)
                del pooled, expected
            records.append(record)
            del a,b,c,d,block,adapter,value
    report = {'schema': 1, 'weights_sha256': weight_hash, 'source_commit': source_commit,
              'first_block': 'GPU noise, direct FP16 adapter and FP8 MMA with initial residuals/bias',
              'records': records, 'quality_gate_passed': False,
              'limitations': ['Two first-reset views from one scene; no temporal or cross-scene quality pass.',
                  'Native pooled layout is fixed before the second view; no mapping refit.',
                  'This is an arithmetic diagnostic, not a native latency improvement.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
