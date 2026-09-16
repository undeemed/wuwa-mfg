import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import test_core
import test_neural
from wuwa_mfg import core, neural, uninstall


class UninstallTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_neural.NeuralTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.game = self.fixture.game
        self.nr_state = self.fixture.state
        self.mfg_state = self.nr_state.with_name('state.json')
        self.backend = test_core.FakeWindows()
        self.original_profiles = copy.deepcopy(self.backend.profiles)
        self.payload = (self.game / 'winmm.dll').read_bytes()
        context = patch.object(core, 'PATCHED_SHA256', core.sha256(self.payload))
        context.start()
        self.addCleanup(context.stop)

    def install_mfg(self):
        (self.game / 'winmm.dll').unlink()
        state = core.make_plan(self.game, self.backend, self.payload, alias=True)
        core.install(self.mfg_state, state, self.backend)

    def install_both(self):
        self.install_mfg()
        self.fixture.install()

    def restore(self):
        prepared = uninstall.plan(self.mfg_state, self.nr_state, self.backend)
        return uninstall.restore_all(self.mfg_state, self.nr_state, self.backend, prepared)

    def test_both_restore_in_order_and_keep_backups(self):
        self.install_both()
        order = []
        nr_restore, mfg_restore = neural.restore, core.restore
        def nr(*args):
            order.append('neural')
            return nr_restore(*args)
        def mfg(*args):
            order.append('mfg')
            return mfg_restore(*args)
        with patch.object(neural, 'restore', side_effect=nr), patch.object(core, 'restore', side_effect=mfg):
            result = self.restore()
        self.assertEqual(order, ['neural', 'mfg'])
        self.assertTrue(result['restart_required'])
        self.assertFalse((self.game / 'winmm.dll').exists())
        self.assertFalse((self.game / 'dxgi.dll').exists())
        self.assertTrue((self.game / core.EXE).exists())
        self.assertEqual(self.backend.profiles, self.original_profiles)
        self.assertEqual(self.backend.name, 'original')
        self.assertEqual(json.loads(self.mfg_state.read_text())['status'], 'restored')
        self.assertEqual(len(list(self.nr_state.parent.glob('neural-restored-*.json'))), 1)

    def test_mfg_only(self):
        self.install_mfg()
        result = self.restore()
        self.assertTrue(result['mfg_restored'])
        self.assertFalse(result['neural_restored'])

    def test_partial_restore_failure_can_be_retried(self):
        self.install_both()
        self.backend.fail_profile_once = True
        with self.assertRaisesRegex(RuntimeError, 'Neural add-on restored; MFG restore stopped'):
            self.restore()
        self.assertFalse(self.nr_state.exists())
        self.assertFalse((self.game / 'dxgi.dll').exists())
        self.assertEqual(json.loads(self.mfg_state.read_text())['status'], 'restoring')
        result = self.restore()
        self.assertTrue(result['mfg_restored'])
        self.assertFalse(result['neural_restored'])
        self.assertEqual(self.backend.profiles, self.original_profiles)
        self.assertEqual(self.backend.name, 'original')
        self.assertFalse((self.game / 'winmm.dll').exists())

    def test_neural_only_preserves_manual_mfg(self):
        self.fixture.install()
        result = self.restore()
        self.assertTrue(result['neural_restored'])
        self.assertFalse(result['mfg_restored'])
        self.assertEqual((self.game / 'winmm.dll').read_bytes(), self.payload)
        self.assertEqual(self.backend.mutations, 0)

    def test_mfg_conflict_does_not_remove_neural_first(self):
        self.install_both()
        self.backend.profiles['global'][core.FG_COUNT] = 9
        with self.assertRaisesRegex(ValueError, 'changed after setup'):
            self.restore()
        self.assertTrue(self.nr_state.exists())
        self.assertTrue((self.game / 'dxgi.dll').exists())
        self.assertEqual(self.backend.name, 'alias')

    def test_neural_conflict_does_not_change_mfg(self):
        self.install_both()
        (self.game / 'OptiScaler.ini').write_text('external edit')
        mutations = self.backend.mutations
        with self.assertRaisesRegex(ValueError, 'changed after setup'):
            self.restore()
        self.assertEqual(self.backend.mutations, mutations)
        self.assertEqual((self.game / 'winmm.dll').read_bytes(), self.payload)

    def test_running_game_stops_before_changes(self):
        self.install_both()
        self.backend.closed = False
        mutations = self.backend.mutations
        with self.assertRaisesRegex(RuntimeError, 'Game is running'):
            self.restore()
        self.assertEqual(self.backend.mutations, mutations)
        self.assertTrue(self.nr_state.exists())

    def test_cancel_and_launch_during_confirmation(self):
        self.install_both()
        mutations = self.backend.mutations
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(uninstall.run(self.mfg_state, self.nr_state, self.backend, lambda _: 'no'), 0)
            def launch(_):
                self.backend.closed = False
                return 'UNINSTALL'
            with self.assertRaisesRegex(RuntimeError, 'Game is running'):
                uninstall.run(self.mfg_state, self.nr_state, self.backend, launch)
        self.assertEqual(self.backend.mutations, mutations)
        self.assertTrue((self.game / 'dxgi.dll').exists())

    def test_changed_plan_requires_new_review(self):
        self.install_both()
        prepared = uninstall.plan(self.mfg_state, self.nr_state, self.backend)
        state = json.loads(self.mfg_state.read_text())
        state['note'] = 'another process updated the record'
        self.mfg_state.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            uninstall.restore_all(self.mfg_state, self.nr_state, self.backend, prepared)
        self.assertTrue((self.game / 'dxgi.dll').exists())

    def test_no_record_and_repeat_leave_manual_files(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(uninstall.run(self.mfg_state, self.nr_state, self.backend,
                             lambda _: self.fail('No confirmation should be needed')), 0)
        self.assertEqual((self.game / 'winmm.dll').read_bytes(), self.payload)
        self.install_both()
        self.restore()
        mutations = self.backend.mutations
        self.assertFalse(any(self.restore().values()))
        self.assertEqual(self.backend.mutations, mutations)

    def test_different_game_records_stop_before_changes(self):
        self.install_both()
        other = self.fixture.root / 'Other/Client/Binaries/Win64'
        other.mkdir(parents=True)
        (other / core.EXE).write_bytes(b'other game fixture')
        record = json.loads(self.nr_state.read_text())
        record['game'] = str(other)
        self.nr_state.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'different games'):
            self.restore()
        self.assertTrue((self.game / 'dxgi.dll').exists())
        self.assertEqual(self.backend.name, 'alias')


if __name__ == '__main__':
    unittest.main()
