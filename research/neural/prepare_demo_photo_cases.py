# SPDX-License-Identifier: MIT
"""Prepare three attributed, private photo cases for unseen-content evaluation.

Downloads are solely renderer input data, never packaged assets or model weights.
Each original is center-cropped/resampled to 1920x1080 without aspect distortion.
The native model subsequently receives the sample's rendered image, not this PNG.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from make_demo_image_scene import write_scene


CASES = [
    {'name': 'portrait', 'author': 'NASA', 'license': 'Public domain in the United States (NASA)',
     'license_url': 'https://www.nasa.gov/nasa-brand-center/images-and-media/',
     'source': 'https://commons.wikimedia.org/wiki/File:Brent_W._Jett_-_Official_Astronaut_Portrait.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/e/e9/Brent_W._Jett_-_Official_Astronaut_Portrait.jpg',
     'source_size': [2400, 3000], 'source_sha1': '1aa64e6c1e05e18b419162c75c26405a1f7322b1'},
    {'name': 'landscape', 'author': 'Hannes Röst', 'license': 'CC BY-SA 3.0',
     'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
     'source': 'https://commons.wikimedia.org/wiki/File:Fronalpstock_big.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg',
     'source_size': [10109, 4542]},
    {'name': 'cat', 'author': 'Babelball', 'license': 'CC0 1.0',
     'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
     'source': 'https://commons.wikimedia.org/wiki/File:3_year_old_calico_cat.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8b/3_year_old_calico_cat.jpg',
     'source_size': [6000, 4000], 'source_sha1': 'aedf4227c4a8d247883f2a54ff5291a8c7cfbebf'},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Keep all photos, textures and scene assets outside the repository.')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'purpose': 'Unseen-content validation only; no fitting or tuning on these three images.',
              'preprocessing': 'Pillow RGB conversion and LANCZOS center-fit to 1920x1080; no aspect-ratio distortion.',
              'model_resolution': [1920, 1080], 'cases': []}
    # Record the full choice before downloading or inspecting any model outputs.
    (output / 'selection.json').write_text(json.dumps({**report, 'selected_sources': CASES}, indent=2) + '\n')
    for case in CASES:
        request = Request(case['url'], headers={'User-Agent': 'wuwa-toolkit-research/1.0 (https://github.com/undeemed/wuwa-toolkit)'})
        with urlopen(request, timeout=30) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if not 1 <= len(data) <= 32 * 1024 * 1024:
            raise ValueError('Unexpected source image size.')
        digest = hashlib.sha1(data).hexdigest()
        if case.get('source_sha1') and digest != case['source_sha1']:
            raise ValueError('Source image changed since its published checksum.')
        with Image.open(io.BytesIO(data)) as opened:
            if list(opened.size) != case['source_size']:
                raise ValueError('Source image dimensions changed.')
            image = ImageOps.fit(opened.convert('RGB'), (1920, 1080),
                                 method=Image.Resampling.LANCZOS, centering=(.5, .5))
        (output / (case['name'] + '-original.jpg')).write_bytes(data)
        texture = output / (case['name'] + '-1080.png')
        image.save(texture)
        scene = write_scene(output / (case['name'] + '-scene'), texture)
        row = {**case, 'split': 'validation', 'source_bytes': len(data),
               'source_sha256': hashlib.sha256(data).hexdigest(),
               'source_sha1_verified': bool(case.get('source_sha1')),
               'texture_sha256': hashlib.sha256(texture.read_bytes()).hexdigest(),
               'scene_manifest': scene}
        report['cases'].append(row)
        (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'prepared': case['name'], 'source_bytes': len(data),
                          'source_sha256': row['source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
