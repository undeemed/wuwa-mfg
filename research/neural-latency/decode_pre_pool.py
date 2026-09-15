# SPDX-License-Identifier: Apache-2.0
"""Decode the separately addressed first-block pooled tensor, not its skip tiles.

The observed 1080p SM89 layout is [2,576,960,16], with a fixed channel bit
permutation inside each plane. Layout agreement is not a model quality pass.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from decode_pre_tensor import e4m3_lut


HEIGHT, WIDTH, CHANNELS = 576, 960, 32
PHYSICAL_BITS = (0, 2, 3, 1)


def channel_indices():
    physical = np.arange(16)
    logical = sum(((physical >> bit) & 1) << i for i, bit in enumerate(PHYSICAL_BITS))
    if not np.array_equal(np.sort(logical), physical):
        raise ValueError('Channel map is not a bijection.')
    return np.argsort(logical)


def load_pooled(trial):
    meta = json.loads((trial / 'pre-pool/metadata.json').read_text())
    if not (meta['complete'] and meta['gpu_completed'] and meta['frame'] == 1
            and meta['noise_counter'] == 0 and meta['tensor_kind'] == 'pooled'
            and meta['pointer_argument_offset'] == 248
            and [meta['width'], meta['height']] == [1920, 1080]
            and [meta['pool_width'], meta['pool_height']] == [WIDTH, HEIGHT]
            and meta['bytes'] == HEIGHT*WIDTH*CHANNELS and meta['file'] == 'pre-pool.raw'):
        raise ValueError('Requires the complete fenced 1080p first-reset pooled capture.')
    raw = (trial / 'pre-pool/pre-pool.raw').read_bytes()
    if len(raw) != meta['bytes']:
        raise ValueError('Incomplete pooled tensor.')
    planes = np.frombuffer(raw, np.uint8).reshape(2, HEIGHT, WIDTH, 16)
    logical = planes[..., channel_indices()].transpose(1, 2, 0, 3).reshape(HEIGHT, WIDTH, CHANNELS)
    return logical, hashlib.sha256(raw).hexdigest()


def metrics(observed, expected):
    if observed.dtype != np.uint8 or expected.dtype != np.uint8 or observed.shape != expected.shape:
        raise ValueError('Requires equal-shape uint8 E4M3 arrays.')
    lut = e4m3_lut()
    a, b = lut[observed], lut[expected]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite values; refusing finite-only selection.')
    delta = a-b
    same = observed == expected
    def rank(code):
        code = code.astype(np.int16)
        return np.where(code & 128, 128-(code & 127), 128+code)
    hist = np.bincount(np.abs(rank(observed)-rank(expected)).ravel(), minlength=256)
    count = a.size
    sx, sy = a.sum(), b.sum()
    denominator = np.sqrt(((a*a).sum()-sx*sx/count)*((b*b).sum()-sy*sy/count))
    return {'elements': count, 'mae': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt((delta*delta).mean())),
            'maximum_absolute_error': float(np.abs(delta).max()),
            'exact_byte_fraction': float(same.mean()),
            'correlation': float(((a*b).sum()-sx*sy/count)/denominator) if denominator else None,
            'native_zero_fraction': float((a == 0).mean()),
            'per_channel_mae': np.abs(delta).reshape(-1, CHANNELS).mean(0).tolist(),
            'per_channel_exact_byte_fraction': same.reshape(-1, CHANNELS).mean(0).tolist(),
            'ulp_distance_histogram': {str(i): int(n) for i, n in enumerate(hist) if n}}


def compare(observed, expected):
    if observed.shape != (HEIGHT, WIDTH, CHANNELS):
        raise ValueError('Requires the entire [576,960,32] pooled tensor.')
    return {'all': metrics(observed, expected),
            'image_rows': metrics(observed[:540], expected[:540]),
            'padding_rows': metrics(observed[540:], expected[540:]),
            'mapping_refitted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native, digest = load_pooled(args.trial)
    reference = np.load(args.reference, mmap_mode='r', allow_pickle=False)
    result = {'schema': 1, 'native_pooled_sha256': digest,
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'physical_bit_for_each_logical_channel_bit': list(PHYSICAL_BITS),
              'metrics': compare(native, reference), 'quality_gate_passed': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics']['all']))


if __name__ == '__main__':
    main()
