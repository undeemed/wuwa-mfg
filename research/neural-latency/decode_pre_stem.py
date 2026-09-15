# SPDX-License-Identifier: Apache-2.0
"""Decode the complete captured first-block skip and separately laid-out pool.

These private native activations are diagnostic targets, not runnable models.
"""
import hashlib
import json

import numpy as np

from decode_pre_tensor import tile_indices
from decode_pre_pool import channel_indices


SKIP_BYTES = 1152*1920*32
POOL_BYTES = 576*960*32


def read_stem_parts(trial):
    meta = json.loads((trial / 'pre-stem/metadata.json').read_text())
    if not (meta['complete'] and meta['gpu_completed'] and meta['frame'] == 1
            and meta['noise_counter'] == 0 and meta['tensor_kind'] == 'complete_first_block'
            and meta['pointer_argument_offset'] == 216
            and [meta['width'],meta['height']] == [1920,1080]
            and [meta['skip_width'],meta['skip_height']] == [1920,1152]
            and [meta['pool_width'],meta['pool_height']] == [960,576]
            and meta['skip_bytes'] == SKIP_BYTES and meta['pool_bytes'] == POOL_BYTES
            and meta['bytes'] == SKIP_BYTES+POOL_BYTES and meta['file'] == 'pre-stem.raw'):
        raise ValueError('Requires the complete fenced first-block 1080p capture contract.')
    raw = (trial / 'pre-stem/pre-stem.raw').read_bytes()
    if len(raw) != SKIP_BYTES+POOL_BYTES:
        raise ValueError('Truncated first-block capture.')
    skip, pool = memoryview(raw)[:SKIP_BYTES], memoryview(raw)[SKIP_BYTES:]
    hashes = {'stem_sha256': hashlib.sha256(raw).hexdigest(),
              'skip_sha256': hashlib.sha256(skip).hexdigest(),
              'pool_sha256': hashlib.sha256(pool).hexdigest()}
    return skip, pool, hashes


def load_stem(trial):
    skip, pool, hashes = read_stem_parts(trial)
    tiles = np.frombuffer(skip,np.uint8).reshape(288,480,512)
    skip_codes = tiles[...,tile_indices()].reshape(288,480,4,4,32).transpose(0,2,1,3,4).reshape(1152,1920,32)
    planes = np.frombuffer(pool,np.uint8).reshape(2,576,960,16)
    pool_codes = planes[...,channel_indices()].transpose(1,2,0,3).reshape(576,960,32)
    return skip_codes, pool_codes, hashes
