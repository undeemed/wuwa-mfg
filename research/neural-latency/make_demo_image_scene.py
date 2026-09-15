# SPDX-License-Identifier: MIT
"""Make a private emissive image plane for the existing NVIDIA sample.

This standalone writer emits a small CHK mesh from numeric geometry. It does
not copy sample assets or source, edit binaries, launch a process, or install
anything in a game. Scene outputs and input images must stay outside this repo.
The sample's rendering, filtering and exposure still affect the model input.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plane_mesh():
    """Serialize a two-sided, four-vertex image plane in the x64 CHK format."""
    chunks = []

    def add(kind, data):
        chunks.append((kind, data))
        return len(chunks)

    names = [b'image-plane\0', b'ImageSurface\0', b'ImagePlane\0']
    table = bytearray(struct.pack('<II', 0, len(names)))
    offset = 0
    for name in names:
        table.extend(struct.pack('<QQ', offset, len(name)))
        offset += len(name)
    add(0x110, bytes(table) + b''.join(names))

    def stream(dtype, vary, semantic, values, fmt, count, stride):
        flags = dtype | (vary << 4) | (semantic << 6)
        return add(0x100, struct.pack('<QQQ', flags, count, stride)
                   + struct.pack('<' + fmt, *values))

    # Camera looks along +Z. Its screen-right direction is -X.
    positions = [2, 1.125, 2, -2, 1.125, 2, -2, -1.125, 2, 2, -1.125, 2]
    pos = stream(5, 2, 1, positions, '12f', 4, 12)
    uv = stream(5, 2, 5, [0, 0, 1, 0, 1, 1, 0, 1], '8f', 4, 8)
    normal = stream(3, 2, 2, [0x00810000] * 4, '4I', 4, 4)
    tangent = stream(3, 2, 3, [0x00000081] * 4, '4I', 4, 4)
    bitangent = stream(3, 2, 4, [0x00008100] * 4, '4I', 4, 4)
    indices = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0]
    index = stream(3, 1, 7, indices, '12I', len(indices), 4)
    bounds = [-2, -1.125, 1.99, 2, 1.125, 2.01]
    info = struct.pack('<QQI6fI4I', 0, 1, 0, *bounds, 0, 0, 4, 0, len(indices))
    assert len(info) == 64
    info_id = add(0x201, struct.pack('<II', 0, 1) + info)
    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    instance = struct.pack('<QII12f6f3fI', 2, 0, 0xFFFFFFFF,
                           *identity, *bounds, 0, 0, 2, 0)
    assert len(instance) == 104
    instance_id = add(0x202, struct.pack('<I', 1) + instance)
    descriptor = bytearray(128)
    struct.pack_into('<III4xQ', descriptor, 0, 0, 0, 0, 0)
    streams = [pos, uv, uv, normal, tangent, bitangent, index] + [0xFFFFFFFF] * 9
    struct.pack_into('<16I3I6f', descriptor, 24, *streams, info_id,
                     instance_id, 0xFFFFFFFF, *bounds)
    add(0x200, bytes(descriptor))

    table_start = 24
    cursor = table_start + 32 * len(chunks)
    output = bytearray(cursor)
    struct.pack_into('<8sIII', output, 0, b'NVDACHNK', 0x100, len(chunks), table_start)
    for i, (kind, data) in enumerate(chunks):
        padding = (-len(output)) % 8
        output.extend(b'\0' * padding)
        offset = len(output)
        output.extend(data)
        struct.pack_into('<III4xQQ', output, table_start + i * 32,
                         i + 1, kind, 0x100, offset, len(data))
    return bytes(output)


def fixture_pixels(variant=0):
    """Asymmetric RGB ramps and edges for checking orientation and coverage."""
    y, x = np.mgrid[:1080, :1920].astype(np.float32)
    x /= 1919
    y /= 1079
    pixels = np.stack([.12 + .72*x, .12 + .72*y,
                       .15 + .5*((np.floor(x*24)+np.floor(y*14)) % 2)], axis=-1)
    pixels[:240, :320] = [.85, .15, .10]
    pixels[:240, -320:] = [.10, .85, .15]
    pixels[-240:, :320] = [.10, .15, .85]
    pixels[-240:, -320:] = [.80, .75, .10]
    if variant:
        pixels = pixels[..., [2, 0, 1]]
    return np.rint(pixels * 255).astype(np.uint8)


def png_bytes(pixels):
    if pixels.dtype != np.uint8 or pixels.shape != (1080, 1920, 3):
        raise ValueError('The orientation fixture must be 1920x1080 RGB8.')

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data)))

    rows = b''.join(b'\0' + row.tobytes() for row in pixels)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 1920, 1080, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows, level=6)) + chunk(b'IEND', b''))


def write_scene(output, texture=None, variant=0):
    output = output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) or (texture and texture.resolve().is_relative_to(repo)):
        raise ValueError('Keep private input images and generated assets outside the repository.')
    if output.exists():
        raise FileExistsError('Use a fresh private scene directory.')
    if texture:
        texture = texture.resolve(strict=True)
        if texture.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('Expected a local PNG or JPEG texture.')
        texture_data = texture.read_bytes()
        if not 1 <= len(texture_data) <= 32 * 1024 * 1024:
            raise ValueError('Keep each research texture within 32 MiB.')
        texture_name = 'image' + texture.suffix.lower()
    else:
        texture_name = 'image.png'
        texture_data = png_bytes(fixture_pixels(variant))
    material = {'ImageSurface': {'Diffuse': [0, 0, 0], 'Specular': [0, 0, 0],
        'Emittance': [1, 1, 1], 'Opacity': 1, 'Shininess': 0,
        'Textures': {'Emittance': texture_name}}}
    scene = {'models': [{'file': 'plane.chk', 'materials': 'materials.json', 'metal-rough': False}],
             'cameras': [{'name': 'Camera0', 'pos': [0, 0, 0], 'target': [0, 0, 2],
                          'up': [0, 1, 0], 'fovY': math.degrees(2 * math.atan(1.125/2))}],
             'active_camera': 'Camera0'}
    output.mkdir(parents=True)
    (output / 'plane.chk').write_bytes(plane_mesh())
    (output / texture_name).write_bytes(texture_data)
    for filename, value in (('materials.json', material), ('scene.json', scene)):
        (output / filename).write_text(json.dumps(value, indent=2) + '\n')
    report = {'schema': 1, 'mesh_bytes': (output/'plane.chk').stat().st_size,
              'texture_sha256': sha(texture_data), 'texture_bytes': len(texture_data),
              'fixture_variant': variant if texture is None else None,
              'generated_fixture_resolution': [1920, 1080] if texture is None else None,
              'files_sha256': {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir())},
              'note': 'Texture is rendered through the sample; capture the actual model input. No latency or quality claim.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--texture', type=Path, help='Optional local image with appropriate usage rights.')
    parser.add_argument('--fixture-variant', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    print(json.dumps(write_scene(args.output, args.texture, args.fixture_variant), indent=2))


if __name__ == '__main__':
    main()
