# SPDX-License-Identifier: Apache-2.0
"""Extract one known CUDA module for private inspection with NVIDIA's tools.

The output contains proprietary GPU code: keep it outside the repository and
do not redistribute it. The input is read-only; no runtime patch is performed.
Separately obtain cuobjdump/nvdisasm from NVIDIA. No system installation needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

DLL_SHA = '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
FATBIN_SHA = '42c88626e65d8553cc2728faf77890ef7277b6f0d89d5e7874a13641e52ba0b4'
CUBIN_SHA = '676d04897fe88e3b21f766b89da72b85f468fa414835b02a8d9215aaa4fb3b1e'
KERNEL = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'


def read_span(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError('Binary span outside file.')
    return data[offset:offset + size]


def symbols(data):
    header = struct.unpack('<16sHHIQQQIHHHHHH', read_span(data, 0, 64))
    if header[0][:6] != b'\x7fELF\x02\x01' or header[11] != 64:
        raise ValueError('Expected little-endian ELF64 with normal section headers.')
    sections = [struct.unpack('<IIQQQQIIQQ', read_span(data, header[6] + i*64, 64))
                for i in range(header[12])]
    found = []
    for section in sections:
        if section[1] != 2:
            continue
        if section[9] != 24 or section[6] >= len(sections) or section[5] % 24:
            raise ValueError('Unsupported symbol table.')
        strings_section = sections[section[6]]
        strings = read_span(data, strings_section[4], strings_section[5])
        for index in range(section[5] // 24):
            name, info, other, target, value, size = struct.unpack(
                '<IBBHQQ', read_span(data, section[4] + index*24, 24))
            if name >= len(strings):
                raise ValueError('Invalid symbol name offset.')
            end = strings.find(b'\0', name)
            if end < 0:
                raise ValueError('Unterminated symbol.')
            if strings[name:end].decode('utf-8', errors='strict') == KERNEL:
                if target >= len(sections):
                    raise ValueError('Invalid kernel section.')
                found.append({'symbol_index': index, 'name': KERNEL, 'value': value,
                              'size': size, 'section_index': target,
                              'section_file_offset': sections[target][4]})
    if len(found) != 1:
        raise ValueError('Expected one exact preprocessor symbol.')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--cuobjdump', type=Path, required=True)
    parser.add_argument('--nvdisasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repository = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repository):
        raise ValueError('Private extracted code must be outside the repository.')
    data = args.dll.read_bytes()
    if hashlib.sha256(data).hexdigest() != DLL_SHA:
        raise ValueError('Unknown DLL build; refusing to guess its module contract.')
    offset = data.find(b'\x50\xed\x55\xba')
    magic, version, header_size, payload_size = struct.unpack('<IHHQ', read_span(data, offset, 16))
    if version != 1 or header_size != 16:
        raise ValueError('Unsupported CUDA fatbinary header.')
    fatbin = read_span(data, offset, header_size + payload_size)
    if hashlib.sha256(fatbin).hexdigest() != FATBIN_SHA:
        raise ValueError('Unexpected first module.')
    args.output.mkdir(parents=True)
    (args.output / 'module-00.fatbin').write_bytes(fatbin)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    extract = subprocess.run([str(args.cuobjdump.resolve()), '--extract-elf',
                              'module-00.3.sm_89.cubin', 'module-00.fatbin'],
                             cwd=args.output, capture_output=True, timeout=60, creationflags=flags)
    if extract.returncode:
        raise RuntimeError(extract.stderr.decode(errors='replace'))
    cubin = args.output / 'module-00.3.sm_89.cubin'
    image = cubin.read_bytes()
    if hashlib.sha256(image).hexdigest() != CUBIN_SHA:
        raise ValueError('Unexpected SM89 image.')
    symbol = symbols(image)
    with (args.output / 'pre-sm89.sass').open('wb') as stdout:
        disassembly = subprocess.run([str(args.nvdisasm.resolve()), '-c',
                                      '-fun', str(symbol['symbol_index']), '-hex', cubin.name],
                                     cwd=args.output, stdout=stdout, stderr=subprocess.PIPE,
                                     timeout=60, creationflags=flags)
    if disassembly.returncode:
        raise RuntimeError(disassembly.stderr.decode(errors='replace'))
    metadata = {'schema': 1, 'dll_sha256': DLL_SHA, 'fatbin_sha256': FATBIN_SHA,
                'fatbin_offset': offset, 'fatbin_bytes': len(fatbin), 'cubin_sha256': CUBIN_SHA,
                'symbol': symbol, 'private_code_not_for_redistribution': True,
                'nvdisasm_diagnostics': disassembly.stderr.decode(errors='replace')}
    (args.output / 'inspection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
