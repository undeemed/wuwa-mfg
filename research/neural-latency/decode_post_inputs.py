# SPDX-License-Identifier: Apache-2.0
"""Read bounded native post-block prefixes and explicit layout hypotheses.

The skip uses the previously recovered first-block tile map. The decoder's
two-plane map is a hypothesis to check against the consumer and paired data.
Captured activations must never be described as an independent renderer.
"""
import hashlib
import json

import numpy as np

from compare_output_grade import read_capture,read_contract
from decode_pre_pool import channel_indices
from decode_pre_tensor import tile_indices
from inspect_pre_kernel import DLL_SHA

DECODER_BYTES=576*960*32
SKIP_BYTES=1152*1920*32


def load_prefixes(trial):
    run=json.loads((trial/'result.json').read_text())
    if run['local_file_sha256']['nvngx_dlssnr.dll']!=DLL_SHA:
        raise ValueError('Unknown native runtime.')
    meta=json.loads((trial/'post-inputs/metadata.json').read_text())
    if not (meta['complete'] and meta['gpu_completed'] and meta['frame']==1 and meta['noise_counter']==0
            and meta['tensor_kind']=='post_block_input_prefixes' and meta['file']=='post-inputs.raw'
            and [meta['width'],meta['height']]==[1920,1080]
            and [meta['network_width'],meta['network_height']]==[1920,1152]
            and meta['bytes']==DECODER_BYTES+SKIP_BYTES and meta['captured_after_chain']==156):
        raise ValueError('Unexpected post-input capture contract.')
    observed=[(x['pointer_argument_offset'],x['file_offset'],x['bytes']) for x in meta['inputs']]
    if observed!=[(0,0,DECODER_BYTES),(8,DECODER_BYTES,SKIP_BYTES)]:
        raise ValueError('Unexpected input-prefix partition.')
    fields=read_contract(trial)
    controls,images,hashes=read_capture(trial/'capture')
    launches=[json.loads(line) for line in (trial/'nr-launch-contract.jsonl').read_text().splitlines()]
    post=[x for x in launches if x.get('kind')=='launch' and x.get('frame')==1
          and x.get('name')=='cc_tinlayout_fused_post_block_swin_1h_32_fp8']
    if len(post)!=1 or post[0]['status']!=0 or post[0]['param_bytes']!=184:
        raise ValueError('Missing successful neural post contract.')
    scalar=post[0]['output_scalar_candidate']
    if scalar['nonzero_64bit_by_offset']['96'] or scalar['float_by_offset']['48']!=.03125:
        raise ValueError('Unsupported history or residual-scale contract.')
    raw=(trial/'post-inputs/post-inputs.raw').read_bytes()
    if len(raw)!=DECODER_BYTES+SKIP_BYTES:raise ValueError('Truncated input capture.')
    decoder,skip=memoryview(raw)[:DECODER_BYTES],memoryview(raw)[DECODER_BYTES:]
    for part in (decoder,skip):
        if np.any((np.frombuffer(part,np.uint8)&127)==127):raise ValueError('Nonfinite E4M3 codes.')
    digests={name:hashlib.sha256(value).hexdigest() for name,value in [('raw',raw),('decoder',decoder),('skip',skip)]}
    return decoder,skip,{'capture_hashes':hashes,'input_sha256':digests,'metadata':meta,
                         'controls':controls,'output_fields':fields,'post_scalar_fields':scalar},images


def decode_skip(raw):
    tiles=np.frombuffer(raw,np.uint8).reshape(288,480,512)
    return tiles[...,tile_indices()].reshape(288,480,4,4,32).transpose(0,2,1,3,4).reshape(1152,1920,32)


def decode_decoder(raw,layout='two-plane-permuted'):
    data=np.frombuffer(raw,np.uint8)
    if layout=='interleaved':return data.reshape(576,960,32)
    if layout=='tiled':
        return data.reshape(144,240,512)[...,tile_indices()].reshape(144,240,4,4,32).transpose(0,2,1,3,4).reshape(576,960,32)
    planes=data.reshape(2,576,960,16)
    if layout=='two-plane-permuted':planes=planes[...,channel_indices()]
    elif layout!='two-plane-identity':raise ValueError('Unknown layout hypothesis.')
    return planes.transpose(1,2,0,3).reshape(576,960,32)
