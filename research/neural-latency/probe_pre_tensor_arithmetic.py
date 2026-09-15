# SPDX-License-Identifier: Apache-2.0
"""Test first-block arithmetic hypotheses against two fenced native prefixes.

Local captured data and weights only. No game access, binary patch or new demo
launch. Noise and branch-input rounding changes are isolated; none is a quality
acceptance or a claim of native model speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from decode_pre_tensor import compare, load_prefix
from fused_norm import FusedNorm
from first_block_rounding import block_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--trial', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source / 'python'))
    from mlxdlss import model as reference
    from mlxdlss.features import AutomaticMask, NetworkGeometry, make_features
    from mlxdlss.pipeline import NeuralRenderingPipeline
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device='cuda', precision='fast')
    reference.e4m3_round_trip = lambda x: x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)
    kernel = FusedNorm()
    records = []
    with torch.inference_mode():
        gpu_noise = kernel.noise(1152, 1920, 0)
        for trial in args.trial:
            native, digest = load_prefix(trial)
            capture = trial / 'capture'
            frame = json.loads((capture / 'frame-0.json').read_text())
            controls = frame['controls']
            if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
                    and controls['DLSSNR.Reset'] == 1):
                raise ValueError('Requires a complete successful first-reset frame.')
            color = frame['resources']['color']
            if color['format'] != 10 or color['row_bytes'] != 1920 * 8:
                raise ValueError('Expected tightly packed 1080p RGBA16F input.')
            raw = (capture / color['file']).read_bytes()
            rgb = np.frombuffer(raw, '<f2').reshape(1080, 1920, 4)[..., :3].astype(np.float32)
            automatic = AutomaticMask(controls['DLSSNR.SkinStructureStrength'],
                                      controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
            features = make_features(
                rgb, geometry=NetworkGeometry(1920, 1080, 1920, 1152), frame_index=0,
                normalized_style=controls['DLSSNR.Style']/128,
                local_tone_strength=controls['DLSSNR.LocalToneStrength'],
                local_structure_strength=controls['DLSSNR.LocalStructureStrength'], automatic_mask=automatic)
            noise_delta = np.abs(gpu_noise.cpu().numpy() - features[..., :3])
            record = {'native_prefix_sha256': digest, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                      'noise_cuda_vs_numpy': {'per_channel_mae': noise_delta.mean(axis=(0, 1)).tolist(),
                                             'per_channel_max': noise_delta.max(axis=(0, 1)).tolist()},
                      'variants': {}}
            original = torch.from_numpy(features[None]).cuda().half()
            for name, use_gpu, quantize_ffn, quantize_qkv in [
                ('baseline', False, False, False), ('gpu_noise', True, False, False),
                ('gpu_noise_fp8_ffn_input', True, True, False),
                ('gpu_noise_fp8_qkv_input', True, False, True),
                ('gpu_noise_fp8_both_branch_inputs', True, True, True),
            ]:
                x = original.clone()
                if use_gpu:
                    x[0, ..., :3] = gpu_noise
                adapter = x @ pipeline.model.weight('block0.layer0.input_adapter_weight')
                block = block_zero(reference, pipeline.model, adapter,
                                   round_ffn=quantize_ffn, round_qkv=quantize_qkv)
                quantized = reference.e4m3_round_trip(block).to(torch.float8_e4m3fn)
                array = quantized[0].view(torch.uint8).cpu().numpy()
                metrics = compare(native, array)
                record['variants'][name] = metrics
                print(json.dumps({'input_sha256': record['input_sha256'], 'variant': name,
                                  **{key: metrics[key] for key in ('mae', 'rmse', 'exact_byte_fraction')}}), flush=True)
                del x, adapter, block, quantized, array
            records.append(record)
    report = {'schema': 1, 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              'source_commit': '0ca2deab092fe6f3e331bf4f616271dbc64521d0', 'records': records,
              'quality_gate_passed': False, 'limitations': [
                  'Only the captured prefix of the first-block skip at reset zero is compared.',
                  'Branch-input E4M3 rounding preserves separate unquantized residual operands.',
                  'Noise agreement with NumPy alone cannot prove native noise equality.',
                  'Full-model parity, final image quality, temporal behavior and latency remain unverified.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
