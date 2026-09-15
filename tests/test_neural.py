import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wuwa_mfg import neural


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.game = self.root / 'WuWa/Client/Binaries/Win64'
        self.game.mkdir(parents=True)
        (self.game / 'Client-Win64-Shipping.exe').write_bytes(b'fake-game')
        (self.game / 'winmm.dll').write_bytes(b'tested-mfg')
        (self.game / 'RTXMFG-Universal.json').write_text('{"followGame":true}')
        self.state = self.root / 'state/neural.json'
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        for name in neural.ALLOWED:
            file = self.bundle / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(('synthetic:' + name).encode())
        self.info = {'base_commit': neural.BASE_COMMIT,
                     'patch_sha256': neural.digest(neural.ROOT / 'patches/optiscaler-wuwa-compat.patch'),
                     'files': {name: neural.digest(self.bundle / name) for name in neural.ALLOWED}}
        self.save_bundle()
        for name, value in [('NR_SHA256', self.info['files']['nvngx_dlssnr.dll']),
                            ('PATCHED_SHA256', neural.digest(self.game / 'winmm.dll'))]:
            context = patch.object(neural, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.closed = lambda: None

    def save_bundle(self):
        (self.bundle / 'bundle.json').write_text(json.dumps(self.info))

    def install(self):
        return neural.install(self.game, self.bundle, self.state, self.closed)

    def test_install_toggle_restore_preserves_mfg(self):
        original = (self.game / 'winmm.dll').read_bytes()
        self.assertFalse(self.install()['nr_enabled'])
        self.assertTrue(neural.set_enabled(self.state, True, self.closed)['nr_enabled_next_launch'])
        text = (self.game / 'OptiScaler.ini').read_text()
        self.assertIn('ToggleKey=0x77', text)
        self.assertIn('[DlssNr]', text)
        self.assertFalse(neural.set_enabled(self.state, False, self.closed)['nr_enabled_next_launch'])
        neural.restore(self.state, self.closed)
        self.assertFalse((self.game / 'dxgi.dll').exists())
        self.assertEqual((self.game / 'winmm.dll').read_bytes(), original)
        self.assertEqual((self.game / 'RTXMFG-Universal.json').read_text(), '{"followGame":true}')
        self.assertFalse(self.state.exists())

    def test_unknown_injector_stops_before_writes(self):
        (self.game / 'dxgi.dll').write_bytes(b'other-mod')
        with self.assertRaises(ValueError): self.install()
        self.assertEqual((self.game / 'dxgi.dll').read_bytes(), b'other-mod')
        self.assertFalse(self.state.exists())

    def test_scale_preserves_enable_and_uses_output(self):
        self.install()
        result = neural.set_enabled(self.state, None, self.closed, scale=0.75)
        self.assertFalse(result['nr_enabled_next_launch'])
        config = (self.game / 'OptiScaler.ini').read_text()
        self.assertIn('WorkingScale=0.75', config)
        self.assertIn('WorkingScaleRelativeToOutput=true', config)
        neural.restore(self.state, self.closed)

    def test_invalid_scale_leaves_config_unchanged(self):
        self.install()
        original = (self.game / 'OptiScaler.ini').read_bytes()
        for scale in (0, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                neural.set_enabled(self.state, None, self.closed, scale=scale)
            self.assertEqual((self.game / 'OptiScaler.ini').read_bytes(), original)

    def test_redirected_backup_refused(self):
        self.install()
        state = json.loads(self.state.read_text())
        state['backup'] = str(self.root / 'elsewhere')
        self.state.write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            neural.set_enabled(self.state, True, self.closed)

    def test_checksum_mismatch_stops(self):
        (self.bundle / 'dxgi.dll').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.install()
        self.assertFalse(self.state.exists())

    def test_manifest_path_escape_rejected(self):
        self.info['files']['../escape.dll'] = '0' * 64
        self.save_bundle()
        with self.assertRaises(ValueError): self.install()

    def test_wrong_base_or_patch_rejected(self):
        self.info['base_commit'] = '0' * 40
        self.save_bundle()
        with self.assertRaises(ValueError): self.install()

    def test_running_game_stops(self):
        def running(): raise RuntimeError('game running')
        with self.assertRaises(RuntimeError): neural.install(self.game, self.bundle, self.state, running)
        self.assertFalse(self.state.exists())

    def test_identical_backend_retained_on_restore(self):
        name = 'OptiScaler/libxess.dll'
        dest = self.game / name
        dest.parent.mkdir()
        dest.write_bytes((self.bundle / name).read_bytes())
        self.install()
        neural.restore(self.state, self.closed)
        self.assertTrue(dest.exists())

    def test_differing_backend_refused(self):
        dest = self.game / 'OptiScaler/libxess.dll'
        dest.parent.mkdir()
        dest.write_bytes(b'other-version')
        with self.assertRaises(ValueError): self.install()
        self.assertFalse(self.state.exists())

    def test_later_config_edit_protected(self):
        self.install()
        (self.game / 'OptiScaler.ini').write_text('user edit')
        with self.assertRaises(ValueError): neural.set_enabled(self.state, True, self.closed)
        with self.assertRaises(ValueError): neural.restore(self.state, self.closed)
        self.assertTrue((self.game / 'dxgi.dll').exists())

    def test_interrupted_config_update_can_restore(self):
        self.install()
        state = json.loads(self.state.read_text())
        data = neural.configure((self.game / 'OptiScaler.ini').read_bytes(), True)
        state['pending_config_hash'] = hashlib.sha256(data).hexdigest()
        self.state.write_text(json.dumps(state))
        (self.game / 'OptiScaler.ini').write_bytes(data)
        neural.restore(self.state, self.closed)
        self.assertFalse((self.game / 'dxgi.dll').exists())

    def test_partial_copy_rolls_back(self):
        copy = neural.copy_verified
        calls = 0
        def fail(source, dest, expected):
            nonlocal calls
            calls += 1
            if calls == 3: raise OSError('simulated disk error')
            copy(source, dest, expected)
        with patch.object(neural, 'copy_verified', fail):
            with self.assertRaises(OSError): self.install()
        self.assertFalse(self.state.exists())
        self.assertFalse((self.game / 'dxgi.dll').exists())
        self.assertTrue((self.game / 'winmm.dll').exists())

    def test_double_install_preserves_backup(self):
        self.install()
        before = self.state.read_bytes()
        with self.assertRaises(ValueError): self.install()
        self.assertEqual(before, self.state.read_bytes())


if __name__ == '__main__':
    unittest.main()
