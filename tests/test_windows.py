"""Exercise actual ctypes DRS translation with an in-memory NVAPI substitute."""
import copy
import ctypes as C
import sys
import unittest

if sys.platform == "win32":
    from wuwa_mfg.windows import Drs, Setting
from wuwa_mfg.core import FG_COUNT, FG_ENABLE, SETTINGS, desired_profiles


@unittest.skipUnless(sys.platform == "win32", "Windows WCHAR/DRS ABI tests")
class DrsTranslationTests(unittest.TestCase):
    def setUp(self):
        self.values = {"global": {FG_ENABLE: None, FG_COUNT: None},
                       "wuwa": {FG_ENABLE: 1, FG_COUNT: 2}}
        self.saved = 0
        self.drs = Drs.__new__(Drs)
        self.drs.session = 10
        self.drs.profiles = {"global": 1, "wuwa": 2}
        self.drs.get = self.get
        self.drs.put = self.put
        self.drs.delete = self.delete
        self.drs.save = self.save
        self.drs.load = lambda session: 0

    def label(self, handle):
        return {1: "global", 2: "wuwa"}[handle]

    def get(self, session, profile, sid, pointer):
        label = self.label(profile)
        key = f"0x{sid:08X}"
        value = self.values[label][key]
        location = 0
        if value is None and label == "wuwa":
            value = self.values["global"][key]
            location = 1
        if value is None:
            return -160
        setting = C.cast(pointer, C.POINTER(Setting)).contents
        setting.type = 0
        setting.location = location
        setting.currentPredefined = 0
        setting.current.u32 = value
        return 0

    def put(self, session, profile, pointer):
        setting = C.cast(pointer, C.POINTER(Setting)).contents
        self.assertEqual(setting.version, C.sizeof(Setting) | (1 << 16))
        self.assertEqual(setting.type, 0)
        self.values[self.label(profile)][f"0x{setting.id:08X}"] = setting.current.u32
        return 0

    def delete(self, session, profile, sid):
        self.values[self.label(profile)][f"0x{sid:08X}"] = None
        return 0

    def save(self, session):
        self.saved += 1
        return 0

    def test_apply_and_restore_preserve_override_ownership(self):
        before = copy.deepcopy(self.values)
        self.drs.update(desired_profiles())
        self.assertEqual(self.drs.snapshot(), desired_profiles())
        self.assertEqual(self.drs.effective()["wuwa"], {FG_ENABLE: 1, FG_COUNT: 0})
        self.drs.update(before)
        self.assertEqual(self.drs.snapshot(), before)
        self.assertEqual(self.saved, 2)

    def test_predefined_conflict_aborts_before_save(self):
        def conflicting_get(session, profile, sid, pointer):
            result = self.get(session, profile, sid, pointer)
            if profile == 2 and sid == int(FG_COUNT, 16) and self.values["wuwa"][FG_COUNT] is None:
                value = C.cast(pointer, C.POINTER(Setting)).contents
                value.location = 0
                value.currentPredefined = 1
                value.current.u32 = 2
            return result
        self.drs.get = conflicting_get
        with self.assertRaisesRegex(RuntimeError, "predefined NVIDIA setting"):
            self.drs.update(desired_profiles())
        self.assertEqual(self.saved, 0)

    def test_cannot_write_unrelated_setting(self):
        snapshot = desired_profiles()
        snapshot["global"]["0x12345678"] = 9
        with self.assertRaises(ValueError):
            self.drs.update(snapshot)
        self.assertEqual(self.saved, 0)


if __name__ == "__main__":
    unittest.main()
