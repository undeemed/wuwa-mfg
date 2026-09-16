"""Completion checks must reject unfenced, incomplete, or stale reference data."""
import json
from pathlib import Path
import tempfile
import unittest

from run_neural_demo_trial import capture_ready


class CaptureReadiness(unittest.TestCase):
    def test_capture_publication_boundary(self):
        with tempfile.TemporaryDirectory(prefix='neural-capture-ready-') as folder:
            root = Path(folder)
            self.assertFalse(capture_ready(root))
            frames = []
            for index in range(4):
                frame = dict(index=index, complete=True, gpu_completed=True, evaluate_result=1,
                             controls={'DLSSNR.Reset': int(index == 0), 'DLSSNR.Width': 1920, 'DLSSNR.Height': 1080},
                             resources={})
                for role in ('color', 'output', 'depth', 'motion'):
                    name = f'{role}-{index}.raw'
                    (root/name).write_bytes(b'abcdefgh')
                    frame['resources'][role] = dict(file=name, rows=2, row_bytes=4, width=1920, height=1080)
                frames.append(frame)
                (root/f'frame-{index}.json').write_text(json.dumps(frame))
                self.assertEqual(capture_ready(root), index == 3)
            frames[3]['gpu_completed'] = False
            (root/'frame-3.json').write_text(json.dumps(frames[3]))
            self.assertFalse(capture_ready(root))
            frames[3]['gpu_completed'] = True
            (root/'frame-3.json').write_text(json.dumps(frames[3]))
            (root/'output-3.raw').write_bytes(b'abcdefg')
            self.assertFalse(capture_ready(root))
            (root/'output-3.raw').write_bytes(b'abcdefgh')
            (root/'frame-3.json').write_text('{')
            self.assertFalse(capture_ready(root))
            frames[3]['resources']['output']['file'] = '../outside.raw'
            (root/'frame-3.json').write_text(json.dumps(frames[3]))
            with self.assertRaises(ValueError):
                capture_ready(root)


if __name__ == '__main__':
    unittest.main()
