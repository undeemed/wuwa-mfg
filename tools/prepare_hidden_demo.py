"""Keep the exact official DLSS demo window hidden without changing its renderer."""
from pathlib import Path
import argparse
import hashlib
import json
import struct

ORIGINAL_SHA256 = 'd28d05c82fc776189c6049649f67b34b1ed2b48f76b2dcfac4fc3c3c4366af35'
SHOW_WINDOW_RVA = 0x82EC0

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe', type=Path)
    args = parser.parse_args()
    path = args.exe.resolve()
    if path.name.lower() != 'ngx_dlss_demo.exe':
        raise SystemExit('Expected the separate official ngx_dlss_demo.exe.')
    data = path.read_bytes()
    if sha(data) != ORIGINAL_SHA256:
        raise SystemExit('Unsupported or already modified executable; no changes made.')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    count = struct.unpack_from('<H', data, pe + 6)[0]
    sections = pe + 24 + struct.unpack_from('<H', data, pe + 20)[0]
    offset = None
    for i in range(count):
        pos = sections + i * 40
        _, rva, size, raw = struct.unpack_from('<IIII', data, pos + 8)
        if rva <= SHOW_WINDOW_RVA < rva + size:
            offset = raw + SHOW_WINDOW_RVA - rva
            break
    if offset is None or data[offset] != 0x40:
        raise SystemExit('Unexpected function entry; no changes made.')
    # This exact build's void glfwShowWindow entry becomes an immediate return.
    # DeviceManager already creates the window with GLFW_VISIBLE=false.
    # Its render loop tests dimensions, so hidden windows can keep rendering.
    backup = path.with_name(path.name + '.visible-original')
    if backup.exists() and sha(backup.read_bytes()) != ORIGINAL_SHA256:
        raise SystemExit('Existing backup differs; no changes made.')
    if not backup.exists():
        with backup.open('xb') as f:
            f.write(data)
    changed = bytearray(data)
    changed[offset] = 0xC3
    temporary = path.with_name(path.name + '.hidden-new')
    with temporary.open('xb') as f:
        f.write(changed)
    temporary.replace(path)
    record = {'original_sha256': ORIGINAL_SHA256, 'hidden_sha256': sha(changed),
              'rva': hex(SHOW_WINDOW_RVA), 'bytes_changed': 1,
              'purpose': 'Suppress the demo glfwShowWindow call and its focus request.',
              'renderer_or_neural_runtime_modified': False}
    path.with_name('hidden-demo-patch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
