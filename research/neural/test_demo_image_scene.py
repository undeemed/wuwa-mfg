# SPDX-License-Identifier: MIT
"""Check generated mesh/PNG structure without launching the sample."""
import hashlib
import io
import json
import struct
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from make_demo_image_scene import fixture_pixels, plane_mesh, png_bytes, write_scene


def main():
    data = plane_mesh()
    signature, version, count, table = struct.unpack_from('<8sIII', data)
    assert signature == b'NVDACHNK' and version == 0x100 and count == 10
    chunks = {}
    end = table + count * 32
    for i in range(count):
        ident, kind, rev, offset, length = struct.unpack_from('<III4xQQ', data, table+i*32)
        assert ident == i+1 and rev == 0x100 and offset >= end and offset % 8 == 0
        assert 0 < length <= len(data)-offset
        chunks[ident] = (kind, data[offset:offset+length])
        end = offset + length
    assert end == len(data)
    descriptor = chunks[10][1]
    streams = struct.unpack_from('<16I', descriptor, 24)
    expected = [(0, 5, 2, 1, 4, 12), (1, 5, 2, 5, 4, 8), (2, 5, 2, 5, 4, 8),
                (3, 3, 2, 2, 4, 4), (4, 3, 2, 3, 4, 4), (5, 3, 2, 4, 4, 4), (6, 3, 1, 7, 12, 4)]
    for slot, dtype, vary, semantic, elements, stride in expected:
        kind, payload = chunks[streams[slot]]
        flags, n, size = struct.unpack_from('<QQQ', payload)
        assert kind == 0x100 and n == elements and size == stride
        assert flags & 15 == dtype and (flags >> 4) & 3 == vary and (flags >> 6) & 15 == semantic
        assert len(payload) == 24 + n*size
    positions = np.frombuffer(chunks[streams[0]][1][24:], dtype='<f4').reshape(4, 3)
    indices = np.frombuffer(chunks[streams[6]][1][24:], dtype='<u4').reshape(4, 3)
    assert np.isfinite(positions).all() and indices.max() < 4
    crosses = np.cross(positions[indices[:,1]]-positions[indices[:,0]],
                       positions[indices[:,2]]-positions[indices[:,0]])
    assert np.all(np.linalg.norm(crosses, axis=1)>0) and np.sum(crosses[:,2]>0) == 2
    assert struct.unpack_from('<3I', descriptor, 88) == (8, 9, 0xFFFFFFFF)
    assert len(chunks[8][1]) == 72 and len(chunks[9][1]) == 108
    fixture_hashes = []
    for variant in (0, 1):
        pixels = fixture_pixels(variant)
        encoded = png_bytes(pixels)
        with Image.open(io.BytesIO(encoded)) as decoded:
            decoded.load()
            assert decoded.size == (1920, 1080)
            assert np.array_equal(np.array(decoded), pixels)
        fixture_hashes.append(hashlib.sha256(encoded).hexdigest())
    assert fixture_hashes[0] != fixture_hashes[1]
    with tempfile.TemporaryDirectory(prefix='neural-image-scene-') as directory:
        root = Path(directory).resolve()
        default = write_scene(root / 'default')
        explicit = write_scene(root / 'explicit-one', emittance=1.)
        assert default['files_sha256'] == explicit['files_sha256']
        assert json.loads((root / 'default/materials.json').read_text())['ImageSurface']['Emittance'] == [1, 1, 1]
        for gain in (.05, .1, .25):
            generated = write_scene(root / ('gain-' + str(gain)), emittance=gain)
            assert all(generated['files_sha256'][name] == default['files_sha256'][name]
                       for name in ('plane.chk', 'scene.json', 'image.png'))
            assert json.loads((root / ('gain-' + str(gain)) / 'materials.json').read_text())['ImageSurface']['Emittance'] == [gain] * 3
        rejected = [False, True, 0, .049, 1.001, float('nan'), float('inf'), float('-inf')]
        for index, gain in enumerate(rejected):
            target = root / ('invalid-' + str(index))
            try:
                write_scene(target, emittance=gain)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid emission accepted.')
            assert not target.exists()
    print(json.dumps({'mesh_chunks': count, 'validated_stream_slots': len(expected),
        'nondegenerate_triangles': 4, 'png_exact_roundtrips': 2,
        'default_emission_asset_bytes_unchanged': True, 'changed_emission_material_only': 3,
        'invalid_emissions_rejected_before_output': len(rejected),
        'mesh_sha256': hashlib.sha256(data).hexdigest(), 'fixture_sha256': fixture_hashes}, indent=2))


if __name__ == '__main__':
    main()
