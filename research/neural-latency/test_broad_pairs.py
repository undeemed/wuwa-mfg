# SPDX-License-Identifier: Apache-2.0
"""Diagnostic aliases must not change a training image's sampling domain."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from train_broad_student import old_pairs
from train_student_collection import original_training_names


class PairProvenance(unittest.TestCase):
    def test_duplicate_aliases_and_wrong_teacher_output(self):
        with tempfile.TemporaryDirectory(prefix='broad-pairs-') as folder:
            base = Path(folder)
            scene_names = original_training_names()+[f'light-{i}' for i in range(12)]
            self.assertEqual(len(scene_names), 30)
            (base/'teacher-illumination-manifest.json').write_text(json.dumps({
                'scene_restored': True, 'dll_restored': True,
                'views': [{'label': name, 'split': 'train'} for name in scene_names[-12:]]}))

            def capture(path, index, wrong_output=False):
                path.mkdir(parents=True)
                data = {'color': f'input-{index}'.encode(),
                        'output': f'teacher-{index}{"-wrong" if wrong_output else ""}'.encode()}
                for role, payload in data.items():
                    (path/f'{role}.raw').write_bytes(payload)
                (path/'frame-0.json').write_text(json.dumps({'resources': {
                    role: {'file': f'{role}.raw'} for role in data}}))
                return {role: hashlib.sha256(payload).hexdigest() for role, payload in data.items()}

            hashes = [capture(base/'trials'/name/'capture', i) for i, name in enumerate(scene_names)]
            for i in range(30, 62):
                hashes.append(capture(base/f'photo-{i}-state'/'capture', i))
            capture(base/'aaa-observer-state'/'capture', 0)
            capture(base/'trials'/'aaa-photo-feature-copy'/'capture', 30)
            capture(base/'aaa-wrong-state'/'capture', 31, wrong_output=True)
            record = {'training_capture_hashes': hashes}
            pairs = old_pairs(base, record)
            self.assertEqual([r['capture_hashes'] for r in pairs], hashes)
            self.assertEqual([r['group'] for r in pairs], ['old-scene']*30+['old-photos']*32)
            self.assertEqual(pairs[0]['label'], scene_names[0])
            self.assertEqual(pairs[30]['label'], 'photo-30')
            (base/'photo-31-state'/'capture'/'output.raw').write_bytes(b'changed target')
            with self.assertRaises(ValueError):
                old_pairs(base, record)


if __name__ == '__main__':
    unittest.main()
