import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wuwa_mfg import core


class FakeWindows:
    def __init__(self):
        self.profiles = {"global": {core.FG_ENABLE: None, core.FG_COUNT: None},
                         "wuwa": {core.FG_ENABLE: 1, core.FG_COUNT: 2}}
        self.closed = True
        self.compatible = True
        self.name = "original"
        self.fail_profile_once = False
        self.fail_names_once = False
        self.mutations = 0

    def assert_closed(self):
        if not self.closed:
            raise RuntimeError("Game is running")

    def check_compatibility(self):
        if not self.compatible:
            raise RuntimeError("Unsupported GPU")

    def read_profiles(self):
        return copy.deepcopy(self.profiles)

    def write_profiles(self, values):
        self.mutations += 1
        if self.fail_profile_once:
            self.fail_profile_once = False
            # Model a post-save error: recovery must handle committed DRS changes.
            self.profiles = copy.deepcopy(values)
            raise RuntimeError("Simulated profile save failure")
        self.profiles = copy.deepcopy(values)

    def name_records(self):
        return [{"before": self.name, "after": "alias"}]

    def check_names(self, records):
        if records and self.name not in (records[0]["before"], records[0]["after"]):
            raise ValueError("Names changed externally")

    def write_names(self, records, restore=False):
        if records:
            self.mutations += 1
            self.name = records[0]["before" if restore else "after"]
            if self.fail_names_once:
                self.fail_names_once = False
                raise RuntimeError("Simulated registry error after write")


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.game = self.root / "Wuthering Waves/Client/Binaries/Win64"
        self.game.mkdir(parents=True)
        (self.game / core.EXE).write_bytes(b"test fixture")
        self.state_path = self.root / "backup/state.json"
        self.backend = FakeWindows()
        self.payload = b"test patched DLL"
        self.sha_patch = patch.object(core, "PATCHED_SHA256", core.sha256(self.payload))
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def plan(self, alias=False):
        return core.make_plan(self.game, self.backend, self.payload, alias)

    def test_install_restore_exact_files_profiles_names(self):
        original_config = b'{"aUserPreference": 123, "mode": "fixed", "multiplier": 3}'
        (self.game / core.CONFIG).write_bytes(original_config)
        initial = self.backend.read_profiles()
        state = self.plan(alias=True)
        core.install(self.state_path, state, self.backend)
        self.assertEqual((self.game / "winmm.dll").read_bytes(), self.payload)
        self.assertEqual(self.backend.profiles, core.desired_profiles())
        config = json.loads((self.game / core.CONFIG).read_text())
        self.assertTrue(config["followGame"])
        self.assertEqual(config["aUserPreference"], 123)
        self.assertEqual(self.backend.name, "alias")
        core.restore(self.state_path, self.backend)
        self.assertFalse((self.game / "winmm.dll").exists())
        self.assertEqual((self.game / core.CONFIG).read_bytes(), original_config)
        self.assertEqual(self.backend.profiles, initial)
        self.assertEqual(self.backend.name, "original")
        self.assertEqual(core.restore(self.state_path, self.backend), "Already restored.")

    def test_prior_official_mod_is_preserved(self):
        (self.game / "winmm.dll").write_bytes(b"official")
        with patch.object(core, "ORIGINAL_SHA256", core.sha256(b"official")):
            state = self.plan()
        core.install(self.state_path, state, self.backend)
        core.restore(self.state_path, self.backend)
        self.assertEqual((self.game / "winmm.dll").read_bytes(), b"official")

    def test_unknown_proxy_rejected_without_writes(self):
        for name in ("winmm.dll", *core.PROXIES):
            with self.subTest(name=name):
                (self.game / name).write_bytes(b"someone else's mod")
                with self.assertRaises(ValueError):
                    self.plan()
                (self.game / name).unlink()
        self.assertEqual(self.backend.mutations, 0)
        self.assertFalse(self.state_path.exists())

    def test_untracked_working_patch_is_not_adopted_or_replaced(self):
        (self.game / "winmm.dll").write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, "already installed"):
            self.plan()

    def test_running_game_blocks_plan(self):
        self.backend.closed = False
        with self.assertRaises(RuntimeError):
            self.plan()
        self.assertEqual(self.backend.mutations, 0)

    def test_unsupported_hardware_blocks_plan(self):
        self.backend.compatible = False
        with self.assertRaises(RuntimeError):
            self.plan()
        self.assertEqual(self.backend.mutations, 0)

    def test_invalid_config_blocks_plan(self):
        (self.game / core.CONFIG).write_text("[]")
        with self.assertRaises(ValueError):
            self.plan()

    def test_game_launch_between_plan_and_install_blocks_changes(self):
        state = self.plan()
        self.backend.closed = False
        with self.assertRaises(RuntimeError):
            core.install(self.state_path, state, self.backend)
        self.assertFalse(self.state_path.exists())

    def test_changed_profile_between_preview_and_install_blocks_changes(self):
        state = self.plan()
        self.backend.profiles["global"][core.FG_COUNT] = 5
        with self.assertRaises(ValueError):
            core.install(self.state_path, state, self.backend)
        self.assertFalse((self.game / "winmm.dll").exists())

    def test_changed_files_between_preview_and_install_blocks_changes(self):
        state = self.plan()
        (self.game / core.CONFIG).write_text("{}")
        with self.assertRaises(ValueError):
            core.install(self.state_path, state, self.backend)
        self.assertFalse(self.state_path.exists())

    def test_profile_save_failure_rolls_back(self):
        original = self.backend.read_profiles()
        state = self.plan()
        self.backend.fail_profile_once = True
        with self.assertRaisesRegex(RuntimeError, "original state restored"):
            core.install(self.state_path, state, self.backend)
        self.assertFalse((self.game / "winmm.dll").exists())
        self.assertEqual(self.backend.read_profiles(), original)
        self.assertEqual(json.loads(self.state_path.read_text())["status"], "restored")

    def test_partial_registry_failure_rolls_back(self):
        state = self.plan(alias=True)
        self.backend.fail_names_once = True
        with self.assertRaisesRegex(RuntimeError, "original state restored"):
            core.install(self.state_path, state, self.backend)
        self.assertEqual(self.backend.name, "original")
        self.assertFalse((self.game / "winmm.dll").exists())

    def test_interrupted_install_can_restore_prepared_manifest(self):
        state = self.plan()
        self.state_path.parent.mkdir()
        core.write_json(self.state_path, state)
        # Only the first file write completed when the process was interrupted.
        (self.game / "winmm.dll").write_bytes(self.payload)
        core.restore(self.state_path, self.backend)
        self.assertFalse((self.game / "winmm.dll").exists())

    def test_external_config_change_stops_restore_before_any_mutation(self):
        core.install(self.state_path, self.plan(), self.backend)
        (self.game / core.CONFIG).write_text('{"mode":"fixed","multiplier":5}')
        mutations = self.backend.mutations
        with self.assertRaisesRegex(ValueError, "changed after setup"):
            core.restore(self.state_path, self.backend)
        self.assertEqual(self.backend.mutations, mutations)
        self.assertEqual((self.game / "winmm.dll").read_bytes(), self.payload)

    def test_equivalent_config_reserialization_can_restore(self):
        core.install(self.state_path, self.plan(), self.backend)
        config_path = self.game / core.CONFIG
        config = json.loads(config_path.read_text())
        config_path.write_text(json.dumps(config, separators=(",", ":")))
        core.restore(self.state_path, self.backend)
        self.assertFalse(config_path.exists())

    def test_late_proxy_conflict_stops_install(self):
        state = self.plan()
        (self.game / "dxgi.dll").write_bytes(b"late conflict")
        with self.assertRaises(ValueError):
            core.install(self.state_path, state, self.backend)
        self.assertFalse(self.state_path.exists())

    def test_external_profile_change_stops_restore(self):
        core.install(self.state_path, self.plan(), self.backend)
        self.backend.profiles["global"][core.FG_COUNT] = 5
        mutations = self.backend.mutations
        with self.assertRaises(ValueError):
            core.restore(self.state_path, self.backend)
        self.assertEqual(self.backend.mutations, mutations)

    def test_backup_is_not_overwritten_by_repeat_install(self):
        state = self.plan()
        core.install(self.state_path, state, self.backend)
        backup = self.state_path.read_bytes()
        with self.assertRaises(ValueError):
            core.install(self.state_path, state, self.backend)
        self.assertEqual(self.state_path.read_bytes(), backup)

    def test_root_and_exe_paths_resolve_to_correct_directory(self):
        self.assertEqual(core.find_game(self.game.parents[2]), self.game)
        self.assertEqual(core.find_game(self.game / core.EXE), self.game)


class TelemetryTests(unittest.TestCase):
    def good(self):
        return dict(pid=123, heartbeat=1000, gameFrameGenerationOn=True, appliedFrameGenerationOn=True,
                    bridgeReady=True, applied=True, setOptionsAccepted=True, setOptionsResult=0,
                    getStateResult=0, stateSampleAgeMs=16, appliedMultiplier=6, actualFramesPresented=6,
                    followGame=True)

    def test_matching_fresh_6x_is_confirmed(self):
        self.assertTrue(core.summarize_status(self.good(), [123], 1001)["verified"])

    def test_stale_previous_process_menu_and_failed_counts_rejected(self):
        alterations = [dict(pid=99), dict(heartbeat=900), dict(heartbeat=1100),
                       dict(gameFrameGenerationOn=False), dict(appliedFrameGenerationOn=False),
                       dict(actualFramesPresented=3), dict(bridgeReady=False), dict(setOptionsResult=38),
                       dict(getStateResult=1), dict(stateSampleAgeMs=9999), dict(setOptionsAccepted=False),
                       dict(applied=False), dict(appliedMultiplier=7, actualFramesPresented=7)]
        for alteration in alterations:
            with self.subTest(alteration=alteration):
                status = self.good() | alteration
                self.assertFalse(core.summarize_status(status, [123], 1001)["verified"])


if __name__ == "__main__":
    unittest.main()
