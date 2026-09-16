# SPDX-License-Identifier: Apache-2.0
"""Export local student parameters for the native DirectML integration.

The output contains learned weights and private reference pixels. Keep it outside
the public repository. This exports the existing model; it does not train it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from progressive_student import load_first
from region_context_student import RegionFeatureRefinement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path, action='append', default=[])
    a = p.parse_args()
    if a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Keep weights and reference pixels outside the public repository.')
    a.output.mkdir(parents=True, exist_ok=False)
    first, _ = load_first(a.first)
    model = RegionFeatureRefinement(first, 'routed')
    checkpoint = a.candidate / 'student-private.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'], strict=True)
    if tuple(state['grade_parameters']) != tuple(first.grade_parameters):
        raise ValueError('Grade contract mismatch.')
    model.eval().half()
    blob = bytearray()
    tensors = {}

    def append(name, array):
        blob.extend(bytes((-len(blob)) % 256))
        tensors[name] = {'offset': len(blob), 'bytes': array.nbytes,
                         'shape': list(array.shape), 'dtype': str(array.dtype)}
        blob.extend(array.tobytes())

    for name, value in model.state_dict().items():
        append(name, value.detach().contiguous().numpy())
    # Spatial masks for the unchanged 1080p bottleneck: 34x60 -> 9x15 regions.
    valid = np.zeros((36, 60), np.float32)
    valid[:34] = 1
    valid = valid.reshape(9, 4, 15, 4).transpose(0, 2, 1, 3).reshape(1, 135, 16, 1)
    append('constant.valid', valid)
    append('constant.denominator', valid.sum(axis=2, keepdims=True))
    diagonal = np.zeros((1, 1, 135, 135), np.float32)
    diagonal[0, 0, np.arange(135), np.arange(135)] = -np.inf
    append('constant.diagonal', diagonal)
    append('constant.own', np.arange(135, dtype=np.uint32).reshape(1, 1, 135, 1))
    append('constant.invalid', np.where(valid > 0, 0, -np.inf).astype(np.float32))
    (a.output / 'weights.bin').write_bytes(blob)
    manifest = {'schema': 1, 'architecture': 'region-broad-v1',
                'width': 1920, 'height': 1080, 'parameters': 381600,
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'weights_sha256': hashlib.sha256(blob).hexdigest(),
                'grade': list(first.grade_parameters), 'tensors': tensors}
    (a.output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Generate matched reference files in the texture layout used by D3D12.
    model.cuda().to(memory_format=torch.channels_last)
    from compare_output_grade import read_capture
    with torch.inference_mode():
        for index, capture in enumerate(a.capture):
            _, arrays, _ = read_capture(capture)
            color = arrays['color']
            if color.shape != (1080, 1920, 3):
                raise ValueError('Reference must be RGB 1920x1080.')
            source = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).cuda().half()
            expected = model(source).squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgba = np.ones((1080, 1920, 4), dtype=np.float16)
            rgba[:, :, :3] = color
            rgba.tofile(a.output / f'input-{index}.bin')
            expected.tofile(a.output / f'expected-{index}.bin')
    print(json.dumps({'complete': True, 'weight_bytes': len(blob), 'references': len(a.capture)}))


if __name__ == '__main__':
    main()
