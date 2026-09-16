# SPDX-License-Identifier: Apache-2.0
"""Audit native feature targets by replaying only the reconstructed final block.

Captured native activations are oracle inputs here, never an independent model.
No training, application launch or native-runtime modification occurs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from compare_output_grade import observed_grade
from decode_post_inputs import load_prefixes, decode_decoder, decode_skip
from decode_pre_tensor import e4m3_lut
from probe_block_sensitivity import setup_model, sha, image_metrics, SOURCE_COMMIT, WEIGHTS_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--case', nargs=2, action='append', required=True, metavar=('LABEL', 'TRIAL'))
    parser.add_argument('--historical-results', type=Path, help='Optional previous oracle outputs for matching labels.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    labels = [label for label, _ in args.case]
    if len(labels) != len(set(labels)) or any(not re.fullmatch('[a-z0-9-]+', label) for label in labels):
        raise ValueError('Use distinct simple labels.')
    if sha(args.weights) != WEIGHTS_SHA or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip() != SOURCE_COMMIT:
        raise ValueError('Requires the pinned reference and private weights.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.source, text=True).strip():
        raise ValueError('The reference source changed.')
    import torch
    from fused_norm import FusedNorm
    from mma_first_block import MmaFirstBlock
    pipeline = setup_model(args.source, args.weights)
    kernel = FusedNorm()
    from mlxdlss import model as reference
    from mlxdlss.features import NetworkGeometry
    from mlxdlss.pipeline import PreparedFrame
    post = MmaFirstBlock(reference, pipeline.model, kernel, block_index=70,
                         fused_packing=True, fused_input_packing=True)
    head_weight = torch.cat((pipeline.model.weight('block70.layer0.out_gain'),
                             pipeline.model.weight('block70.layer0.out_conv_weight')), dim=0).contiguous()
    merge_sin = pipeline.model.weight('block70.layer0.inp_merge_sin')
    merge_cos = pipeline.model.weight('block70.layer0.inp_merge_cos')
    lut = e4m3_lut().astype(np.float16)
    args.output.mkdir(parents=True)
    report = {'schema': 1, 'complete': False, 'target_achieved': False, 'quality_gate_passed': False,
              'source_commit': SOURCE_COMMIT, 'weights_sha256': WEIGHTS_SHA,
              'native_runtime_modified': False, 'uses_captured_native_activations': True,
              'output_extent': [1920, 1080], 'network_extent': [1920, 1152], 'cases': [],
              'limitations': ['Oracle native features are required; this is not an independently runnable replacement.',
                  'The decoder mapping and reconstructed final arithmetic remain approximations, not proven native parity.',
                  'First-reset static images only; no temporal or perceptual acceptance, and no native speedup claim.']}
    with torch.inference_mode():
        for label, trial_name in args.case:
            trial = Path(trial_name)
            decoder_raw, skip_raw, metadata, images = load_prefixes(trial)
            decoder = torch.from_numpy(lut[decode_decoder(decoder_raw)][None]).cuda()
            skip = torch.from_numpy(lut[decode_skip(skip_raw)][None]).cuda()
            assert torch.isfinite(decoder).all() and torch.isfinite(skip).all()
            up = reference.nearest_upsample2_crop(decoder, height=1152, width=1920)
            merged = reference._rows(lambda a, b: a*merge_sin+b*merge_cos, up, skip)
            features = post(merged, attention_mma=True, seed_residual=True, seed_logits=True)
            head = kernel.mma(features.reshape(-1, 32), head_weight).reshape(*features.shape[:-1], 4)
            # finish consumes source/processing/geometry, not prepared input features.
            prepared = PreparedFrame(images['color'], images['color'], np.empty(0, np.float32),
                                     NetworkGeometry(1920, 1080, 1920, 1152), None, 0.)
            image = observed_grade(pipeline.finish(prepared, head.float().cpu().numpy()[0], intensity=1).image,
                                   metadata['output_fields'])
            assert image.shape == images['color'].shape and np.isfinite(image).all()
            row = {'label': label, 'capture_hashes': metadata['capture_hashes'],
                   'feature_hashes': metadata['input_sha256'], 'controls': metadata['controls'],
                   'feature_capture_complete_and_fenced': True,
                   'decoder_shape': list(decoder.shape), 'skip_shape': list(skip.shape),
                   'oracle_vs_native': image_metrics(image, images['output']), 'feature_statistics': {}}
            for name, tensor in [('decoder', decoder), ('skip', skip)]:
                x = tensor.float()
                row['feature_statistics'][name] = {'minimum': float(x.min()), 'maximum': float(x.max()),
                    'mean': float(x.mean()), 'std': float(x.std()),
                    'per_channel_rms': x.square().mean(dim=(0, 1, 2)).sqrt().cpu().tolist()}
            historical = args.historical_results / (label+'-native-inputs-mma-all.npy') if args.historical_results else None
            if historical is not None and historical.exists():
                old_report = json.loads((args.historical_results/'result.json').read_text())
                old = next(r for r in old_report['cases'] if r['label'] == label)
                assert old['capture_hashes'] == metadata['capture_hashes'] and old['controls'] == metadata['controls']
                row['historical_output_reproduced_exactly'] = bool(np.array_equal(image, np.load(historical, allow_pickle=False)))
                assert row['historical_output_reproduced_exactly']
            np.save(args.output/(label+'-oracle.npy'), image)
            row['oracle_file_sha256'] = sha(args.output/(label+'-oracle.npy'))
            report['cases'].append(row)
            (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'label': label, 'oracle_vs_native': row['oracle_vs_native']}), flush=True)
            del decoder, skip, up, merged, features, head, prepared, image
    report['complete'] = True
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
