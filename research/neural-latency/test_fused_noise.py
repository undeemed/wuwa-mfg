# SPDX-License-Identifier: Apache-2.0
"""Bound CUDA noise differences against the recovered CPU formula, not NVIDIA."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from fused_norm import FusedNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source / 'python'))
    from mlxdlss.features import deterministic_noise
    kernel = FusedNorm()
    records = []
    with torch.inference_mode():
        for height, width, frame in [(1, 1, 0), (3, 7, 1), (17, 257, 2),
                                     (63, 65, 0xffffffff), (1152, 1920, 0), (1152, 1920, 3)]:
            actual = kernel.noise(height, width, frame)
            repeated = kernel.noise(height, width, frame)
            repeat_equal = torch.equal(actual, repeated)
            half_equal = torch.equal(actual, actual.half().float())
            value = actual.cpu().numpy()
            expected = deterministic_noise(height, width, frame)
            delta = np.abs(value - expected)
            finite = bool(np.isfinite(value).all())
            record = {'shape': [height, width, 3], 'frame_index': frame,
                      'finite': finite, 'repeat_equal': repeat_equal, 'half_rounded': half_equal,
                      'mae': float(delta.mean()), 'max_abs': float(delta.max()),
                      'exact_fraction': float((value == expected).mean())}
            records.append(record)
            if not (finite and repeat_equal and half_equal and record['max_abs'] <= 1/128):
                raise ValueError(record)
    report = {'schema': 1, 'tests': records, 'all_formula_bound_checks_passed': True,
              'native_noise_equality_proven': False, 'quality_gate_passed': False,
              'limitations': ['Compares GPU approximate instructions with the recovered CPU formula.',
                              'No direct native noise tensor or full-model acceptance is provided.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
