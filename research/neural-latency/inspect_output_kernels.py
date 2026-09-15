# SPDX-License-Identifier: Apache-2.0
"""Inspect two known output kernels using separately supplied NVIDIA tools.

The DLL is read-only. Extracted code stays outside this repository and must not
be redistributed. No application launch, binary patch or system installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from inspect_pre_kernel import DLL_SHA, read_span, symbols


TARGETS = (
    (0, 913632, 3648048, '42c88626e65d8553cc2728faf77890ef7277b6f0d89d5e7874a13641e52ba0b4',
     '676d04897fe88e3b21f766b89da72b85f468fa414835b02a8d9215aaa4fb3b1e',
     'cc_tinlayout_fused_post_block_swin_1h_32_fp8', 'neural-post'),
    (13, 18021776, 42064, 'b95fbb3b45646cab48680d37e9502b15e195b090846b51fda454e8ec42603b31',
     '0a0d12de5f5ecee455746f56dbe6a836fa4918bb1dfa4e65025f10d4a161dac0',
     'cg2r_post_process_kernel', 'post-process'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--cuobjdump', type=Path, required=True)
    parser.add_argument('--nvdisasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Extracted vendor code must remain outside this repository.')
    data = args.dll.read_bytes()
    if hashlib.sha256(data).hexdigest() != DLL_SHA:
        raise ValueError('Unknown DLL build; refusing to guess module offsets.')
    # Validate every bounded module before creating any output.
    modules = []
    for index, offset, size, digest, cubin_digest, kernel, label in TARGETS:
        blob = read_span(data, offset, size)
        magic, version, header_size, payload_size = struct.unpack('<IHHQ', blob[:16])
        if (magic, version, header_size, payload_size+16) != (0xba55ed50,1,16,size):
            raise ValueError('Unexpected fatbinary header.')
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('Unexpected module hash.')
        modules.append((index, offset, blob, digest, cubin_digest, kernel, label))
    output.mkdir(parents=True)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    report = {'schema': 1, 'dll_sha256': DLL_SHA, 'modules': [],
              'private_code_not_for_redistribution': True}
    for index, offset, blob, digest, cubin_digest, kernel, label in modules:
        fatbin = output / f'module-{index:02}.fatbin'
        cubin = output / f'module-{index:02}.3.sm_89.cubin'
        fatbin.write_bytes(blob)
        run = subprocess.run([str(args.cuobjdump.resolve()), '--extract-elf', cubin.name, fatbin.name],
                             cwd=output, capture_output=True, timeout=60, creationflags=flags)
        if run.returncode:
            raise RuntimeError(run.stderr.decode(errors='replace'))
        image = cubin.read_bytes()
        if hashlib.sha256(image).hexdigest()!=cubin_digest:
            raise ValueError('Unexpected extracted SM89 module hash.')
        symbol = symbols(image, kernel)
        with (output / (label+'.sass')).open('wb') as stdout:
            run = subprocess.run([str(args.nvdisasm.resolve()), '-c', '-fun',
                str(symbol['symbol_index']), '-hex', cubin.name], cwd=output,
                stdout=stdout, stderr=subprocess.PIPE, timeout=60, creationflags=flags)
        if run.returncode:
            raise RuntimeError(run.stderr.decode(errors='replace'))
        report['modules'].append({'fatbin_offset': offset, 'fatbin_bytes':len(blob),
            'fatbin_sha256':digest, 'cubin_sha256':hashlib.sha256(image).hexdigest(),
            'symbol':symbol, 'diagnostics':run.stderr.decode(errors='replace')})
    (output / 'inspection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
