# SPDX-License-Identifier: Apache-2.0
"""Read-only feasibility audit of exact zeros in the pinned logical weights.

No weights are changed or exported. Axis checks describe logical storage, not
a proven native sparse-MMA operand mapping. Having at least half the values
zero is necessary, but not sufficient, for lossless 2:4 storage of a tensor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


WEIGHTS_SHA256 = 'fa6ebc71bc6f91347d51b3368b0ef6e952b6157b6d9a164e7db82cffb88d3143'
PTX_URL = 'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-sparse-mma'


def group_statistics(values, axis):
    if values.shape[axis] % 4:
        raise ValueError('Group dimension must be divisible by four.')
    groups = np.moveaxis(values, axis, -1).reshape(-1, 4)
    nonzero = np.count_nonzero(groups, axis=1)
    return {'axis': axis % values.ndim, 'groups': len(groups),
            'nonzero_count_histogram': np.bincount(nonzero, minlength=5).tolist(),
            'groups_with_at_most_two_nonzeros': int(np.count_nonzero(nonzero <= 2)),
            'all_groups_representable': bool(np.all(nonzero <= 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.weights.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError('Requires the exact pinned logical weights.')
    with args.weights.open('rb') as handle:
        header_size = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    base_offset = 8 + header_size
    records = []
    # Hand-computed zero-count examples check the statistic independently of a
    # model or CUDA backend. These do not test hardware sparse arithmetic.
    fixture = np.array([[0, 0, 1, 2], [0, 0, 0, 1], [1, 2, 3, 4], [0, 0, 0, 0]], dtype=np.float32)
    assert group_statistics(fixture, 1)['nonzero_count_histogram'] == [1, 1, 1, 0, 1]
    assert group_statistics(fixture.T, 0)['nonzero_count_histogram'] == [1, 1, 1, 0, 1]
    for name, info in header.items():
        if name == '__metadata__' or 'weight' not in name or len(info['shape']) < 2:
            continue
        dtype = {'F16': '<f2', 'F32': '<f4'}[info['dtype']]
        begin, end = info['data_offsets']
        shape = tuple(info['shape'])
        elements = int(np.prod(shape))
        assert end - begin == elements * np.dtype(dtype).itemsize
        assert base_offset + end <= args.weights.stat().st_size
        values = np.memmap(args.weights, mode='r', dtype=dtype, offset=base_offset + begin, shape=shape)
        assert np.isfinite(values).all()
        zero_count = elements - int(np.count_nonzero(values))
        axes = [axis for axis in (values.ndim - 2, values.ndim - 1) if shape[axis] % 4 == 0]
        records.append({'name': name, 'shape': list(shape), 'elements': elements,
                        'exact_zero_count': zero_count, 'exact_zero_fraction': zero_count / elements,
                        'at_least_half_zero': zero_count * 2 >= elements,
                        'logical_axis_checks': [group_statistics(values, axis) for axis in axes]})
        del values
    elements = sum(row['elements'] for row in records)
    zeros = sum(row['exact_zero_count'] for row in records)
    report = {'schema': 1, 'target_achieved': False, 'quality_gate_passed': False,
              'weights_sha256': digest, 'tensor_selection': 'Names containing weight, rank at least two.',
              'tensor_count': len(records), 'elements': elements, 'exact_zero_count': zeros,
              'exact_zero_fraction': zeros / elements,
              'tensors_with_at_least_half_zero': sum(row['at_least_half_zero'] for row in records),
              'tensors_passing_either_logical_axis': sum(any(axis['all_groups_representable'] for axis in row['logical_axis_checks']) for row in records),
              'tensors': records, 'ptx_reference': PTX_URL,
              'limitations': ['Counts are not a measured speedup or proof of native sparse operand mapping.',
                              'Zeroing additional weights changes the model and requires retraining and quality validation.',
                              'Sparse FP8 accumulator/type support differs from the native half-accumulator path; this audit does not establish arithmetic equivalence.',
                              'No GPU kernels ran and no weights or native binaries changed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('tensors', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
