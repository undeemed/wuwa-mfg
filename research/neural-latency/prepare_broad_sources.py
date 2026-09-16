# SPDX-License-Identifier: Apache-2.0
"""Prepare 200 new training and 40 development sources, never final gameplay tests.

Commons metadata, licenses, original hashes and the pre-capture split are pinned.
Sources and generated textures stay private. No application is launched here.
"""
import argparse
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps

from make_demo_image_scene import write_scene


BUCKETS = {
    'forest': 'forest trail photograph',
    'foliage': 'leaves foliage close up photograph',
    'mountains': 'mountains landscape photograph',
    'coast': 'coast beach cliffs photograph',
    'water': 'river waterfall photograph',
    'snow': 'snow winter landscape photograph',
    'desert': 'desert dunes photograph',
    'day-city': 'city street daylight photograph',
    'night-city': 'city night lights photograph',
    'stone': 'castle stone architecture photograph',
    'interior': 'interior room architecture photograph',
    'ornament': 'temple ornate interior photograph',
    'faces': 'person face portrait photograph',
    'people': 'people crowd festival photograph',
    'animals': 'animal fur photograph',
    'flowers': 'flowers garden photograph',
    'reflections': 'glass reflection building photograph',
    'metal': 'metal machinery photograph',
    'sunset': 'sunset clouds landscape photograph',
    'dark': 'cave interior photograph',
}
USER_AGENT = 'wuwa-mfg-research/1.0 (https://github.com/undeemed/wuwa-mfg)'
REPO = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def plain(value):
    return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', value)).split())


def fetch(url, maximum):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('commons.wikimedia.org', 'upload.wikimedia.org'):
        raise ValueError('Unexpected source host.')
    time.sleep(1.5)
    with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=40) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Source exceeds bounded request size.')
    return data


def fingerprints(image):
    grey = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    x = np.arange(32); k = np.arange(8)[:, None]
    basis = np.cos(np.pi * (2*x + 1) * k / 64)
    coefficients = (basis @ grey @ basis.T).flatten()[1:]
    bits = coefficients > np.median(coefficients)
    return sum(int(bit) << i for i, bit in enumerate(bits))


def author_split(author):
    # Group an author's works into one partition, before seeing teacher output.
    return 'development' if int(sha(('broad-v1|' + author.casefold()).encode())[:8], 16) % 6 == 0 else 'train'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve(); lab = args.lab.resolve(strict=True)
    if root.is_relative_to(REPO) or not root.is_relative_to(lab):
        raise ValueError('Use a private child of the explicit research directory.')
    root.mkdir(exist_ok=True); (root/'queries').mkdir(exist_ok=True)
    path = root/'manifest.json'
    old_images = sorted({p for folder in ('photo-validation', 'photo-training-tenth', 'photo-diverse-extension')
                         for p in (lab/folder).glob('*-original.*')})
    excluded_hashes = set(); previous_fingerprints = []
    for old in old_images:
        data = old.read_bytes(); excluded_hashes.add(sha(data))
        with Image.open(io.BytesIO(data)) as image:
            previous_fingerprints.append(fingerprints(ImageOps.exif_transpose(image).convert('RGB')))
    if path.exists():
        manifest = read(path)
        assert manifest['buckets'] == BUCKETS and manifest['excluded_original_sha256'] == sorted(excluded_hashes)
        for row in manifest['cases']:
            case = root/row['id']
            assert sha((case/'original.jpg').read_bytes()) == row['source_sha256']
            assert sha((case/'texture.png').read_bytes()) == row['texture_sha256']
    else:
        manifest = {'schema': 1, 'complete': False, 'buckets': BUCKETS,
                    'target_counts': {'train': 200, 'development': 40},
                    'final_gameplay_test_count': 0, 'final_gameplay_tests_reserved': 300,
                    'excluded_original_sha256': sorted(excluded_hashes),
                    'split_rule': 'SHA256(broad-v1|lowercase author) first32bits modulo6: 0 development, otherwise train. All works by one author stay in one split.',
                    'dedup_rule': 'Exact source SHA256 and 63-bit DCT perceptual hash Hamming distance <= 6 excluded, including old original images. Screening is not proof of scene independence.',
                    'per_bucket': {'train': 10, 'development': 2},
                    'cases': [], 'rejections': []}
        write(path, manifest)
    known = {row['source_sha256'] for row in manifest['cases']} | excluded_hashes
    hashes = previous_fingerprints + [int(row['phash'], 16) for row in manifest['cases']]
    authors = Counter(row['author'].casefold() for row in manifest['cases'])
    processed = {row['page_id'] for row in manifest['cases']} | {row['page_id'] for row in manifest['rejections']}
    for bucket, query in BUCKETS.items():
        for offset in range(0, 400, 10):
            counts = Counter(row['split'] for row in manifest['cases'] if row['bucket'] == bucket)
            if counts['train'] == 10 and counts['development'] == 2:
                break
            cache = root/'queries'/f'{bucket}-{offset:03}.json'
            if cache.exists():
                result = read(cache)
            else:
                search = query if offset < 200 else ' '.join(query.split()[:2])
                params = {'action': 'query', 'format': 'json', 'generator': 'search', 'gsrnamespace': 6,
                          'gsrsearch': search + ' filetype:bitmap -painting -drawing -map -engraving',
                          'gsrlimit': 10, 'gsroffset': offset % 200, 'prop': 'imageinfo',
                          'iiprop': 'url|size|sha1|extmetadata|mime',
                          'iiextmetadatafilter': 'Artist|LicenseShortName|LicenseUrl|ImageDescription|Categories'}
                result = json.loads(fetch('https://commons.wikimedia.org/w/api.php?' + urlencode(params), 4*1024*1024))
                if 'error' in result:
                    raise RuntimeError('Metadata query failed: ' + str(result['error']))
                write(cache, result)
            pages = sorted(result.get('query', {}).get('pages', {}).values(), key=lambda r: r.get('index', 0))
            for page in pages:
                identity = page['pageid']
                if identity in processed:
                    continue
                info = page.get('imageinfo', [{}])[0]; meta = info.get('extmetadata', {})
                author = plain(meta.get('Artist', {}).get('value', ''))
                license_name = plain(meta.get('LicenseShortName', {}).get('value', ''))
                split = author_split(author)
                counts = Counter(row['split'] for row in manifest['cases'] if row['bucket'] == bucket)
                if counts[split] >= manifest['per_bucket'][split]:
                    continue
                reason = None
                if not author or author.casefold() in ('unknown', 'unknown author', 'anonymous'):
                    reason = 'missing distinct author'
                elif not (license_name in ('CC0', 'Public domain') or re.fullmatch(r'CC BY(?:-SA)? [1234]\.0', license_name)):
                    reason = 'license outside selected set'
                elif info.get('mime') != 'image/jpeg' or info.get('width', 0) < 1920 or info.get('height', 0) < 1080:
                    reason = 'format or minimum resolution'
                elif not 0 < info.get('size', 0) <= 12*1024*1024:
                    reason = 'bounded source size'
                elif authors[author.casefold()] >= 4:
                    reason = 'author coverage cap'
                if reason:
                    manifest['rejections'].append({'page_id': identity, 'reason': reason}); processed.add(identity)
                    continue
                if shutil.disk_usage(root).free < 1024**3:
                    raise RuntimeError('Stop preparation with at least 1 GiB free.')
                url = urlsplit(info['url']); url = urlunsplit((url.scheme, url.netloc, url.path, '', ''))
                data = fetch(url, 12*1024*1024)
                digest = sha(data)
                if len(data) != info['size'] or hashlib.sha1(data).hexdigest() != info['sha1']:
                    raise ValueError('Original differs from pinned source metadata.')
                with Image.open(io.BytesIO(data)) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                    fingerprint = fingerprints(image)
                    if digest in known or any((fingerprint ^ previous).bit_count() <= 6 for previous in hashes):
                        manifest['rejections'].append({'page_id': identity, 'reason': 'exact or perceptual duplicate'})
                        processed.add(identity); write(path, manifest); continue
                    texture = ImageOps.fit(image, (1920,1080), method=Image.Resampling.LANCZOS)
                case_id = f'{bucket}-{identity}'; case = root/case_id
                if case.exists():
                    raise FileExistsError('Unrecorded source directory requires inspection before resuming.')
                case.mkdir(); (case/'original.jpg').write_bytes(data); texture.save(case/'texture.png')
                write_scene(case/'scene', case/'texture.png', emittance=.1)
                row = {'id': case_id, 'page_id': identity, 'bucket': bucket, 'split': split,
                       'title': page['title'], 'source': info['descriptionurl'], 'url': url,
                       'author': author, 'license': license_name,
                       'license_url': meta.get('LicenseUrl', {}).get('value', ''),
                       'source_size': [info['width'], info['height']], 'source_bytes': len(data),
                       'source_sha1': info['sha1'], 'source_sha256': digest,
                       'texture_sha256': sha((case/'texture.png').read_bytes()), 'phash': f'{fingerprint:016x}',
                       'scene_manifest_sha256': sha((case/'scene/manifest.json').read_bytes()),
                       'teacher_capture': None, 'visual_review': 'pending'}
                manifest['cases'].append(row); known.add(digest); hashes.append(fingerprint)
                authors[author.casefold()] += 1; processed.add(identity); write(path, manifest)
                print(json.dumps({'prepared': len(manifest['cases']), 'bucket': bucket, 'split': split, 'id': case_id}), flush=True)
        counts = Counter(row['split'] for row in manifest['cases'] if row['bucket'] == bucket)
        if counts != Counter(manifest['per_bucket']):
            write(path, manifest)
            raise RuntimeError('Bucket needs more distinct sources: ' + bucket + ' ' + str(dict(counts)))
    assert Counter(row['split'] for row in manifest['cases']) == Counter(manifest['target_counts'])
    manifest['complete'] = True; write(path, manifest)
    print('Prepared 200 training and 40 development sources; final gameplay testing remains separate.', flush=True)


if __name__ == '__main__':
    main()
