import os
from pathlib import Path
import struct
import unittest

from wuwa_mfg.patch import (AFTER, BEFORE, CHECKSUM_OFFSET, OFFSET, PATCHED_SHA256,
                            dll_from_archive, patch_dll, pe_checksum, sha256)


class PatchTests(unittest.TestCase):
    def test_wrong_build_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            patch_dll(b"not a supported DLL")

    def test_wrong_archive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            dll_from_archive(b"untrusted archive")

    def test_checksum_ignores_checksum_field_and_handles_odd_length(self):
        data = b"\x01\x02\x03\x04" + b"\xff" * 4 + b"\x05"
        self.assertEqual(pe_checksum(data, 4), 0x0201 + 0x0403 + 5 + len(data))

    @unittest.skipUnless(os.environ.get("RTXMFG_TEST_ARCHIVE"), "Set RTXMFG_TEST_ARCHIVE for exact release integration test")
    def test_exact_release_reproduces_verified_binary_and_only_expected_changes(self):
        original = dll_from_archive(Path(os.environ["RTXMFG_TEST_ARCHIVE"]).read_bytes())
        result = patch_dll(original)
        self.assertEqual(len(result), len(original))
        self.assertEqual(sha256(result), PATCHED_SHA256)
        self.assertEqual(original[OFFSET:OFFSET + 9], BEFORE)
        self.assertEqual(result[OFFSET:OFFSET + 9], AFTER)
        changed = [index for index, (a, b) in enumerate(zip(original, result)) if a != b]
        self.assertEqual(changed, [0x180, 0x181, 0x182, 0x1BD80, 0x1BD81, 0x1BD84, 0x1BD85, 0x1BD86])
        self.assertEqual(struct.unpack_from("<I", result, CHECKSUM_OFFSET)[0], pe_checksum(result, CHECKSUM_OFFSET))
        corrupted = bytearray(original)
        corrupted[-1] ^= 1
        with self.assertRaises(ValueError):
            patch_dll(corrupted)


if __name__ == "__main__":
    unittest.main()
