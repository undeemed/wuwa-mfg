"""Reproduce the tested RTXMFG v1.3.3 edit without loading any DLL."""
import hashlib
import io
import struct
import urllib.request
import zipfile

RELEASE_URL = "https://github.com/dashdogy/RTX40MFG-Unlock/releases/download/v1.3.3/RTXMFG-v1.3.3.zip"
ARCHIVE_SHA256 = "c362b3915840463e7dbb14d70c74b5de54dda3f7d6ebe739fb1c36a90e618cfc"
ORIGINAL_SHA256 = "1c0c561f1819b2f37c7f7f2528e9488290967408e51da0573b7b642ba8cfc56e"
PATCHED_SHA256 = "46e931fdc265fcd1875f6c113b689a159466061c13cef239f57ca386d66f148f"
OFFSET = 0x1BD7F
CHECKSUM_OFFSET = 0x180
BEFORE = bytes.fromhex("e85c48010084c07547")
AFTER = bytes.fromhex("e8fc4b0100d1e87447")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pe_checksum(data, offset):
    """IMAGE_OPTIONAL_HEADER.CheckSum: folded 16-bit sum plus file length."""
    total = 0
    for i in range(0, len(data), 2):
        if offset <= i < offset + 4:
            continue
        total += data[i] | ((data[i + 1] if i + 1 < len(data) else 0) << 8)
        total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return (total & 0xFFFF) + len(data)


def patch_dll(data):
    # A whole-file digest prevents applying these offsets to another build.
    if sha256(data) != ORIGINAL_SHA256:
        raise ValueError("Expected the original RTXMFG v1.3.3 DLL; SHA-256 does not match.")
    if data[OFFSET:OFFSET + len(BEFORE)] != BEFORE:
        raise ValueError("Wrapper preparation instructions do not match.")
    result = bytearray(data)
    result[OFFSET:OFFSET + len(AFTER)] = AFTER
    struct.pack_into("<I", result, CHECKSUM_OFFSET, pe_checksum(result, CHECKSUM_OFFSET))
    if sha256(result) != PATCHED_SHA256:
        raise ValueError("Patched output does not match the tested DLL.")
    return bytes(result)


def dll_from_archive(data):
    if sha256(data) != ARCHIVE_SHA256:
        raise ValueError("Upstream archive SHA-256 does not match the pinned release.")
    # Never extract paths supplied by an archive, or execute its contents.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = []
        for entry in archive.infolist():
            if not entry.is_dir() and entry.file_size < 16 * 1024 * 1024:
                if entry.filename.lower().endswith(".dll"):
                    payload = archive.read(entry)
                    if sha256(payload) == ORIGINAL_SHA256:
                        matches.append(payload)
        if not matches or any(item != matches[0] for item in matches):
            raise ValueError("The pinned archive does not contain the expected DLL.")
        return matches[0]


def download_dll():
    request = urllib.request.Request(RELEASE_URL, headers={"User-Agent": "wuwa-toolkit/0.2.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Unexpected download size.")
    return dll_from_archive(data)
