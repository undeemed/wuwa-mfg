"""Windows integration: NVAPI DRS, registry descriptions and read-only discovery."""
import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import winreg

from .core import EXE, SETTINGS, desired_profiles, find_game
from .patch import sha256

U32 = C.c_uint32
Handle = C.c_void_p
Wide = C.c_wchar * 2048
DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
ALIAS = "NVIDIA GeForce RTX 5080"
TESTED_DRIVER = "32.0.16.1692"
TESTED_NVIDIA = {
    "wrapper": ("sl_dlss_g_0", "134656", "190_E658703.dll", "73b8a78a275b5a3db58038a4ff701a1fb4049db5c379eab9473a09fa856ba84e"),
    "provider": ("dlssg", "20318464", "160_E658700.bin", "c64928fdb7c48a57722ea8eef2662171edc323473adea66c29a206a23f1a2bed"),
}


def powershell(script):
    executable = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Windows discovery failed.")
    return result.stdout.strip().lstrip("\ufeff")


def is_admin():
    return bool(C.windll.shell32.IsUserAnAdmin())


def running_pids():
    raw = powershell("[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); @(Get-Process -Name 'Client-Win64-Shipping' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id) | ConvertTo-Json -Compress")
    value = json.loads(raw) if raw else []
    return value if isinstance(value, list) else [value]


def detect_games():
    roots = []
    for hive, key, value in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")):
        try:
            with winreg.OpenKey(hive, key) as opened:
                roots.append(Path(winreg.QueryValueEx(opened, value)[0]))
        except OSError:
            pass
    for root in list(roots):
        libraries = root / "steamapps/libraryfolders.vdf"
        if libraries.exists():
            for value in re.findall(r'"path"\s+"([^"]+)"', libraries.read_text(encoding="utf-8", errors="replace")):
                roots.append(Path(value.replace("\\\\", "\\")))
    games = []
    for root in roots:
        try:
            game = find_game(root / "steamapps/common/Wuthering Waves")
            if game not in games:
                games.append(game)
        except ValueError:
            pass
    return games


class Binary(C.Structure):
    _fields_ = [("length", U32), ("data", C.c_ubyte * 4096)]


class Value(C.Union):
    _pack_ = 4
    _fields_ = [("u32", U32), ("binary", Binary), ("wide", Wide), ("u64", C.c_uint64)]


class Setting(C.Structure):
    _pack_ = 4
    _fields_ = [("version", U32), ("name", Wide), ("id", U32), ("type", U32),
               ("location", U32), ("currentPredefined", U32), ("predefinedValid", U32),
               ("predefined", Value), ("current", Value)]


class Profile(C.Structure):
    _fields_ = [("version", U32), ("name", Wide), ("gpuSupport", U32),
               ("predefined", U32), ("apps", U32), ("settings", U32)]


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}: NVAPI status {code}")


class Drs:
    def __init__(self):
        self.session = Handle()
        self.nv = C.WinDLL(str(Path(os.environ["SystemRoot"]) / "System32/nvapi64.dll"))
        self.query = self.nv.nvapi_QueryInterface
        self.query.argtypes = [U32]
        self.query.restype = Handle
        self.initialize = self.api(0x0150E828)
        self.unload = self.api(0xD22BDD7E)
        self.create = self.api(0x0694D52E, C.POINTER(Handle))
        self.destroy = self.api(0xDAD9CFF8, Handle)
        self.load = self.api(0x375DBD6B, Handle)
        self.save = self.api(0xFCBC7E14, Handle)
        self.get_global = self.api(0x617BFF9F, Handle, C.POINTER(Handle))
        self.find = self.api(0x7E4A9A0B, Handle, C.c_wchar_p, C.POINTER(Handle))
        self.info = self.api(0x61CD6FD6, Handle, Handle, C.POINTER(Profile))
        self.get = self.api(0x73BF8338, Handle, Handle, U32, C.POINTER(Setting))
        self.put = self.api(0x577DD202, Handle, Handle, C.POINTER(Setting))
        self.delete = self.api(0xE4A26362, Handle, Handle, U32)

    def api(self, interface, *args):
        pointer = self.query(interface)
        if not pointer:
            raise RuntimeError(f"Missing NVAPI interface {interface:#x}")
        return C.CFUNCTYPE(C.c_int32, *args)(pointer)

    def __enter__(self):
        check(self.initialize(), "Initialize")
        try:
            check(self.create(C.byref(self.session)), "CreateSession")
            check(self.load(self.session), "LoadSettings")
            self.profiles = {"global": Handle(), "wuwa": Handle()}
            check(self.get_global(self.session, C.byref(self.profiles["global"])), "GetCurrentGlobalProfile")
            global_info = Profile(version=C.sizeof(Profile) | (1 << 16))
            check(self.info(self.session, self.profiles["global"], C.byref(global_info)), "GetGlobalProfileInfo")
            if global_info.name.casefold() != "base profile":
                raise RuntimeError("A custom NVIDIA global profile is active. This recipe expects Base Profile.")
            check(self.find(self.session, "Wuthering Waves", C.byref(self.profiles["wuwa"])), "FindWuWaProfile")
            profile = Profile(version=C.sizeof(Profile) | (1 << 16))
            check(self.info(self.session, self.profiles["wuwa"], C.byref(profile)), "GetProfileInfo")
            if profile.name.casefold() != "wuthering waves":
                raise RuntimeError("Unexpected NVIDIA game profile.")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.session:
            self.destroy(self.session)
        self.unload()

    def setting(self, profile, sid):
        value = Setting(version=C.sizeof(Setting) | (1 << 16))
        code = self.get(self.session, self.profiles[profile], int(sid, 16), C.byref(value))
        if code == -160:
            return None
        check(code, "GetSetting")
        if value.type != 0:
            raise RuntimeError("Expected a DWORD NVIDIA setting.")
        return value

    def snapshot(self):
        result = {}
        for label in self.profiles:
            result[label] = {}
            for sid in SETTINGS:
                value = self.setting(label, sid)
                result[label][sid] = (value.current.u32 if value is not None and value.location == 0
                                      and not value.currentPredefined else None)
        return result

    def effective(self):
        return {label: {sid: (value.current.u32 if (value := self.setting(label, sid)) is not None else None)
                        for sid in SETTINGS} for label in self.profiles}

    def update(self, values):
        if set(values) != {"global", "wuwa"} or any(set(row) != set(SETTINGS) for row in values.values()):
            raise ValueError("Unexpected profile snapshot.")
        existing = self.snapshot()
        for label, row in values.items():
            for sid, number in row.items():
                if number == existing[label][sid]:
                    continue
                if number is None:
                    check(self.delete(self.session, self.profiles[label], int(sid, 16)), "DeleteLocalOverride")
                else:
                    if type(number) is not int or not 0 <= number <= 0xFFFFFFFF:
                        raise ValueError("Invalid DWORD profile value.")
                    setting = Setting(version=C.sizeof(Setting) | (1 << 16), id=int(sid, 16), type=0)
                    setting.current.u32 = number
                    check(self.put(self.session, self.profiles[label], C.byref(setting)), "SetSetting")
        if self.snapshot() != values:
            raise RuntimeError("NVIDIA profile readback failed before save.")
        if values == desired_profiles():
            effective = self.effective()
            if effective["wuwa"] != values["global"]:
                raise RuntimeError("A predefined NVIDIA setting blocks global inheritance. No profile changes saved.")
        check(self.save(self.session), "SaveSettings")
        check(self.load(self.session), "ReloadSettings")
        if self.snapshot() != values:
            raise RuntimeError("NVIDIA profile readback failed after save.")


class WindowsBackend:
    def __init__(self):
        if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("Use 64-bit Windows and 64-bit Python.")

    def assert_closed(self):
        if running_pids():
            raise RuntimeError("Close Wuthering Waves before changing its installation.")

    def hardware(self):
        raw = powershell("[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); @(Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion) | ConvertTo-Json -Compress")
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]

    def tested_gpu(self):
        devices = self.hardware()
        discrete = [d for d in devices if re.search(r"VEN_(10DE|1002)&", d.get("PNPDeviceID", ""), re.I)]
        if len(discrete) != 1 or not re.match(r"PCI\\VEN_10DE&DEV_2782&", discrete[0]["PNPDeviceID"], re.I):
            raise RuntimeError("This release requires one discrete GPU: physical RTX 4070 Ti (PCI 10DE:2782). Other cards need separate validation.")
        return discrete[0]

    def check_compatibility(self):
        gpu = self.tested_gpu()
        if gpu["DriverVersion"] != TESTED_DRIVER:
            raise RuntimeError("This release is validated on NVIDIA 616.92 only. It does not change drivers.")
        cache = Path(os.environ["ProgramData"]) / "NVIDIA/NGX/models"
        for label, (model, version, name, digest) in TESTED_NVIDIA.items():
            path = cache / model / "versions" / version / "files" / name
            if not path.is_file() or sha256(path.read_bytes()) != digest:
                raise RuntimeError(f"The tested NVIDIA {label} cache file is missing or changed. Run the normal game/NVIDIA updater; no NVIDIA files were modified.")
            versions = path.parents[2]
            if any(item.is_dir() and item.name.isdecimal() and int(item.name) > int(version) for item in versions.iterdir()):
                raise RuntimeError(f"A newer NVIDIA {label} cache exists. Its compatibility needs validation.")
        # Explicit FG On-mode caused hangs in the investigation. Do not silently
        # combine this recipe with a different preset/dynamic/Streamline override.
        with Drs() as drs:
            for sid in ("0x10308298", "0x10E41DF1", "0x10562D0F", "0x10CF4125", "0x10E41E06"):
                for label in ("global", "wuwa"):
                    value = drs.setting(label, sid)
                    if value is not None and not value.currentPredefined and value.current.u32 != 0:
                        raise RuntimeError(f"Conflicting NVIDIA {label} FG override {sid}. Return it to application control before installing; setup did not alter it.")
        return "RTX 4070 Ti / 616.92 / tested NVIDIA cache files match."

    def read_profiles(self):
        with Drs() as drs:
            return drs.snapshot()

    def write_profiles(self, values):
        with Drs() as drs:
            drs.update(values)

    def name_records(self):
        gpu = self.tested_gpu()
        device_path = "SYSTEM\\CurrentControlSet\\Enum\\" + gpu["PNPDeviceID"]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, device_path) as key:
            driver = winreg.QueryValueEx(key, "Driver")[0]
        if not re.fullmatch(r"\{4d36e968-e325-11ce-bfc1-08002be10318\}\\\d{4}", driver, re.I):
            raise RuntimeError("Unexpected display driver registry link.")
        class_path = "SYSTEM\\CurrentControlSet\\Control\\Class\\" + driver
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path) as key:
            matching = winreg.QueryValueEx(key, "MatchingDeviceId")[0]
            if not matching.casefold().startswith(r"pci\ven_10de&dev_2782"):
                raise RuntimeError("GPU registry identity changed.")
        records = []
        for path, name in ((class_path, "DriverDesc"), (class_path, "HardwareInformation.AdapterString"), (device_path, "DeviceDesc")):
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
            if kind != winreg.REG_SZ:
                raise RuntimeError("Expected a string GPU description; registry left unchanged.")
            records.append(dict(path=path, name=name, before=value, after=ALIAS))
        return records

    def check_names(self, records):
        if not records:
            return
        current = self.name_records()
        if len(records) != 3 or {(r["path"], r["name"]) for r in records} != {(r["path"], r["name"]) for r in current}:
            raise RuntimeError("GPU registry location changed. Do not restore descriptions into a different driver slot.")
        for record in records:
            value = next(r["before"] for r in current if (r["path"], r["name"]) == (record["path"], record["name"]))
            if record["after"] != ALIAS or value not in (record["before"], record["after"]):
                raise RuntimeError("GPU descriptions changed outside this installer; rollback stopped.")

    def write_names(self, records, restore=False):
        self.check_names(records)
        for record in records:
            expected = record["before" if restore else "after"]
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, record["path"], 0, winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE) as key:
                winreg.SetValueEx(key, record["name"], 0, winreg.REG_SZ, expected)
                winreg.FlushKey(key)
                if winreg.QueryValueEx(key, record["name"])[0] != expected:
                    raise RuntimeError("GPU description readback failed.")
