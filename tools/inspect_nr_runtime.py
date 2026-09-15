"""Read-only inspection of the pinned NR DLL. Never loads or changes it.

Optional research dependencies: zstandard==0.25.0, capstone==5.0.6.
Not needed by setup, the game, or normal toolkit operation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct


def inspect(path):
    import zstandard
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64

    data = path.read_bytes()
    expected = '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('This inspector supports only the documented SF-v2 runtime.')
    architectures, fp8_objects, ptx_targets = Counter(), Counter(), Counter()
    fatbins = 0
    for match in re.finditer(b'\x50\xed\x55\xba', data):
        start = match.start()
        _, version, header, size = struct.unpack_from('<IHHQ', data, start)
        if version != 1 or header != 16 or start + header + size > len(data):
            continue
        fatbins += 1
        pos, end = start + header, start + header + size
        while pos < end:
            kind, _, header, size = struct.unpack_from('<HHIQ', data, pos)
            if not 32 <= header <= 512 or pos + header + size > end:
                raise ValueError('Unexpected fatbin entry bounds.')
            sm = struct.unpack_from('<I', data, pos + 28)[0]
            payload = data[pos + header:pos + header + size]
            if payload.startswith(b'\x28\xb5\x2f\xfd'):
                payload = zstandard.ZstdDecompressor().decompress(
                    payload, max_output_size=256 * 1024 * 1024)
            if payload[:7] == b'\x7fELF\x02\x01\x01':
                machine = struct.unpack_from('<H', payload, 18)[0]
                flags = struct.unpack_from('<I', payload, 48)[0]
                # The pinned package uses CUDA's newer ELF ABI.
                if machine != 190 or ((flags >> 8) & 0xff) != sm:
                    raise ValueError('ELF/fatbin architecture mismatch.')
                architectures[str(sm)] += 1
                names = re.findall(rb'[\x20-\x7e]{6,}', payload)
                if any(b'fp8' in name.lower() for name in names):
                    fp8_objects[str(sm)] += 1
            elif kind == 1:
                targets = re.findall(rb'\.target[^\r\n]+', payload)
                if len(targets) != 1:
                    raise ValueError('Unexpected PTX target metadata.')
                ptx_targets[targets[0].decode('ascii')] += 1
            else:
                raise ValueError('Unexpected embedded code format.')
            pos += header + size

    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    stores = []
    for rva in (0x18006, 0x1a96a):
        raw = rva - 0x1000 + 0x400
        instruction = next(disassembler.disasm(data[raw:raw+16], 0x180000000+rva))
        if instruction.mnemonic != 'mov' or not instruction.op_str.endswith('0x3f800000'):
            raise ValueError('Expected scaling reset instruction was not found.')
        stores.append(hex(rva))
    return {
        'runtime_sha256': expected,
        'fatbins': fatbins,
        'compiled_objects_by_sm': dict(architectures),
        'objects_with_fp8_names_by_sm': dict(fp8_objects),
        'ptx_targets': dict(ptx_targets),
        'scaling_reset_to_1_native_rvas': stores,
        'limitations': 'Static file inventory only; no active dispatch or performance measurement.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.runtime), indent=2))
