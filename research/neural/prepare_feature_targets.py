# SPDX-License-Identifier: Apache-2.0
"""Prepare private native decoder hints, fitting normalization on training only.

The captured half-resolution decoder is averaged 2x2 into quarter-resolution
training targets over valid image rows. This changes auxiliary supervision,
never the student's full-resolution RGB input or inference resolution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from decode_post_inputs import load_prefixes, decode_decoder
from decode_pre_tensor import e4m3_lut


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', nargs=3, action='append', required=True, metavar=('NAME', 'SPLIT', 'TRIAL'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    if not 2 <= len(args.case) <= 64 or {s for _, s, _ in args.case} != {'train', 'validation'}:
        raise ValueError('Require a bounded collection with training and validation splits.')
    names = [n for n, _, _ in args.case]
    if len(set(names)) != len(names) or any(not re.fullmatch('[a-z0-9-]+', n) for n in names):
        raise ValueError('Use distinct simple labels.')
    arrays, records, inputs = [], [], set()
    sum_values, sum_squares, count = np.zeros(32, np.float64), np.zeros(32, np.float64), 0
    controls_expected = None
    for name, split, trial_name in args.case:
        decoder_raw, _, data, _ = load_prefixes(Path(trial_name))
        if controls_expected is None:
            controls_expected = data['controls']
        assert data['controls'] == controls_expected and data['capture_hashes']['color'] not in inputs
        inputs.add(data['capture_hashes']['color'])
        decoded = e4m3_lut()[decode_decoder(decoder_raw)[:540]].astype(np.float32)
        pooled = decoded.reshape(270, 2, 480, 2, 32).mean(axis=(1, 3))
        assert pooled.shape == (270, 480, 32) and np.isfinite(pooled).all()
        if split == 'train':
            flat = pooled.reshape(-1, 32).astype(np.float64)
            sum_values += flat.sum(axis=0)
            sum_squares += np.square(flat).sum(axis=0)
            count += flat.shape[0]
        arrays.append(pooled)
        records.append({'name': name, 'split': split, 'capture_hashes': data['capture_hashes'],
                        'feature_hashes': data['input_sha256']})
    mean = sum_values/count
    standard_deviation = np.sqrt(np.maximum(0, sum_squares/count-np.square(mean)))
    scale = np.maximum(standard_deviation, .001)
    args.output.mkdir(parents=True)
    for record, pooled in zip(records, arrays):
        target = ((pooled-mean.astype(np.float32))/scale.astype(np.float32)).astype(np.float32)
        assert np.isfinite(target).all()
        filename = record['name']+'-target.npy'
        np.save(args.output/filename, target)
        record.update(target_file=filename, target_sha256=sha(args.output/filename),
                      normalized_mean=float(target.mean()), normalized_std=float(target.std()))
    report = {'schema': 1, 'complete': True, 'controls': controls_expected,
              'target_shape_hwc': [270, 480, 32], 'decoder_layout': 'two-plane-permuted',
              'valid_native_decoder_crop_hwc': [540, 960, 32], 'pool': '2x2 arithmetic mean, stride 2',
              'normalization': {'fit_split': 'train', 'pixels_per_channel': count,
                                'mean': mean.tolist(), 'std': standard_deviation.tolist(), 'scale': scale.tolist()},
              'cases': records, 'quality_gate_passed': False,
              'limitations': ['Native decoder layout is an evidence-supported hypothesis, not complete native parity.',
                  'Only auxiliary targets are pooled. Full-resolution RGB inputs and final output supervision are unchanged.',
                  'Targets are first-reset features; no temporal supervision or inference-time native feature substitution.']}
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'prepared': len(records), 'training_cases': sum(r['split']=='train' for r in records),
                      'normalization_fit_on_training_only': True}), flush=True)


if __name__ == '__main__':
    main()
