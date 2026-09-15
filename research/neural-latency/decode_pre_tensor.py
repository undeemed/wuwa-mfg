# SPDX-License-Identifier: Apache-2.0
"""Validate a fixed SM89 first-block tile map against private reference arrays.

This decodes the fenced 32 MiB full-resolution skip prefix, not the separately
addressed pooled tensor. It neither fits a mapping nor accepts model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PHYSICAL_BITS = (0, 4, 5, 1, 3, 6, 7, 8, 2)


def tile_indices():
    """Return physical offsets in logical [inner_y, inner_x, channel] order."""
    physical = np.arange(512, dtype=np.int64)
    logical = sum(((physical >> bit) & 1) << i for i, bit in enumerate(PHYSICAL_BITS))
    if not np.array_equal(np.sort(logical), physical):
        raise ValueError('Tile address map must be a bijection.')
    return np.argsort(logical)


def e4m3_lut():
    codes = np.arange(256)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    values = np.where(codes & 128, -1., 1.) * np.where(
        exponent == 0, mantissa * 2.**-9, (1 + mantissa / 8) * 2.**(exponent - 7))
    values[(exponent == 15) & (mantissa == 7)] = np.nan
    return values


def load_prefix(trial):
    meta = json.loads((trial / 'pre-tensor/metadata.json').read_text())
    required = (meta['complete'] and meta['gpu_completed'] and meta['noise_counter'] == 0
                and [meta['width'], meta['height']] == [1920, 1080]
                and meta['bytes'] == 32 * 1024 * 1024 and meta['frame'] == 1)
    if not required:
        raise ValueError('Requires the completed first-reset 1080p capture contract.')
    filename = Path(meta['file'])
    if filename.name != str(filename):
        raise ValueError('Capture filename must be a basename.')
    raw = (trial / 'pre-tensor' / filename).read_bytes()
    if len(raw) != meta['bytes']:
        raise ValueError('Incomplete prefix.')
    return np.frombuffer(raw, np.uint8).reshape(-1, 512), hashlib.sha256(raw).hexdigest()


def compare(native, reference):
    if reference.dtype != np.uint8 or reference.shape != (1152, 1920, 32):
        raise ValueError('Reference must be uint8 E4M3 [1152,1920,32].')
    inverse, lut = tile_indices(), e4m3_lut()
    slot = np.arange(512)
    count = exact = 0
    sx = sy = sxx = syy = sxy = absolute = squared = maximum = 0.
    per_channel_absolute = np.zeros(32)
    per_channel_squared = np.zeros(32)
    per_channel_exact = np.zeros(32, dtype=np.int64)
    ulp_histogram = np.zeros(256, dtype=np.int64)
    def rank(code):
        code = code.astype(np.int16)
        return np.where(code & 128, 128 - (code & 127), 128 + code)
    for start in range(0, len(native), 2048):
        tiles = np.arange(start, min(start + 2048, len(native)))[:, None]
        expected = reference[(tiles // 480) * 4 + slot // 128,
                             (tiles % 480) * 4 + (slot // 32) % 4, slot % 32]
        observed = native[start:start + len(tiles)][:, inverse]
        a, b = lut[expected], lut[observed]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError('Nonfinite E4M3 value; refusing finite-only selection.')
        delta = a - b
        count += a.size
        same = expected == observed
        exact += int(same.sum())
        sx += a.sum(); sy += b.sum(); sxx += (a*a).sum(); syy += (b*b).sum(); sxy += (a*b).sum()
        absolute += np.abs(delta).sum(); squared += (delta*delta).sum()
        maximum = max(maximum, float(np.abs(delta).max()))
        per_channel_absolute += np.abs(delta).reshape(-1, 32).sum(axis=0)
        per_channel_squared += (delta*delta).reshape(-1, 32).sum(axis=0)
        per_channel_exact += same.reshape(-1, 32).sum(axis=0)
        distance = np.abs(rank(expected) - rank(observed))
        ulp_histogram += np.bincount(distance.ravel(), minlength=256)
    return {'elements': count, 'mae': absolute / count, 'rmse': float(np.sqrt(squared / count)),
            'maximum_absolute_error': maximum,
            'correlation': float((sxy - sx*sy/count) / np.sqrt((sxx - sx*sx/count)*(syy - sy*sy/count))),
            'exact_byte_fraction': exact/count,
            'per_channel_mae': (per_channel_absolute/(count/32)).tolist(),
            'per_channel_rmse': np.sqrt(per_channel_squared/(count/32)).tolist(),
            'per_channel_exact_byte_fraction': (per_channel_exact/(count/32)).tolist(),
            'ulp_distance_histogram': {str(i): int(n) for i, n in enumerate(ulp_histogram) if n},
            'ulp_definition': 'Distance in finite E4M3 numeric codes; signed zeros share rank.',
            'mapping_refitted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native, digest = load_prefix(args.trial)
    reference = np.load(args.reference, mmap_mode='r', allow_pickle=False)
    report = {'schema': 1, 'physical_bit_for_each_logical_bit': list(PHYSICAL_BITS),
              'bijection_exhaustively_verified': True, 'native_prefix_sha256': digest,
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'metrics': compare(native, reference), 'quality_gate_passed': False,
              'limitations': ['First-reset full-resolution skip prefix only, not the pooled output.',
                              'Two-view layout evidence does not prove all model tensor layouts.',
                              'High correlation is not numerical equality or full-image quality acceptance.',
                              'No model-performance benchmark is performed.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report['metrics'][k] for k in ('mae', 'rmse', 'correlation', 'exact_byte_fraction')}))


if __name__ == '__main__':
    main()
