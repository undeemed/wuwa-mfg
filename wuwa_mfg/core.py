"""Setup planning and recoverable transactions. No Windows imports at import time."""
import base64
import json
import os
from pathlib import Path
import tempfile

from .patch import ORIGINAL_SHA256, PATCHED_SHA256, sha256

EXE = "Client-Win64-Shipping.exe"
CONFIG = "RTXMFG-Universal.json"
FILES = ("winmm.dll", CONFIG)
PROXIES = ("version.dll", "dxgi.dll", "d3d12.dll", "dinput8.dll", "xinput1_3.dll", "xinput9_1_0.dll", "winhttp.dll")
FG_ENABLE = "0x10E41E03"
FG_COUNT = "0x104D6667"
SETTINGS = (FG_ENABLE, FG_COUNT)
CONFIG_DEFAULTS = dict(followGame=True, mode="follow", multiplier=4, dlssgPreset=2,
                       vsyncMode=0, reflexFrameLimitFps=0, version=13, generatedOnlyDebug=False)


def atomic_write(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".wuwa-mfg-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2) + "\n").encode())


def find_game(path):
    path = Path(path).expanduser()
    if path.is_file() and path.name.lower() == EXE.lower():
        path = path.parent
    for directory in (path, path / "Client/Binaries/Win64", path / "Binaries/Win64"):
        if (directory / EXE).is_file():
            directory = directory.resolve()
            if [part.lower() for part in directory.parts[-3:]] != ["client", "binaries", "win64"]:
                raise ValueError("Choose WuWa's Client/Binaries/Win64 folder or its game root.")
            for name in FILES:
                target = directory / name
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"Refusing redirected or non-file target: {name}")
            return directory
    raise ValueError(f"Cannot find {EXE} under that folder.")


def file_snapshot(directory):
    return {name: base64.b64encode((directory / name).read_bytes()).decode()
            if (directory / name).exists() else None for name in FILES}


def decode(value):
    return base64.b64decode(value, validate=True) if value is not None else None


def same_file(name, a, b):
    if a == b:
        return True
    if name == CONFIG and a is not None and b is not None:
        try:
            return json.loads(decode(a).decode("utf-8-sig")) == json.loads(decode(b).decode("utf-8-sig"))
        except (ValueError, UnicodeError):
            pass
    return False


def desired_profiles():
    # None means no local USER override. A predefined or inherited default remains.
    return {"global": {FG_ENABLE: 1, FG_COUNT: 0}, "wuwa": {key: None for key in SETTINGS}}


def config_bytes(previous):
    config = json.loads(previous.decode("utf-8-sig")) if previous is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Existing RTXMFG configuration must be a JSON object.")
    config.update(CONFIG_DEFAULTS)
    return (json.dumps(config, indent=2) + "\n").encode()


def make_plan(directory, backend, patched, alias=False):
    directory = find_game(directory)
    backend.assert_closed()
    backend.check_compatibility()
    if sha256(patched) != PATCHED_SHA256:
        raise ValueError("Installer payload is not the tested patch.")
    for name in PROXIES:
        if (directory / name).exists():
            raise ValueError(f"Existing {name}: resolve the proxy conflict before installing. Nothing was removed.")
    original = file_snapshot(directory)
    old_dll = decode(original["winmm.dll"])
    if old_dll is not None and sha256(old_dll) not in (ORIGINAL_SHA256, PATCHED_SHA256):
        raise ValueError("Existing winmm.dll is unknown; refusing to replace it.")
    # An untracked patched copy is the user's existing working installation.
    if old_dll is not None and sha256(old_dll) == PATCHED_SHA256:
        raise ValueError("The tested patch is already installed outside this installer. Use Verify; no changes needed.")
    profiles = backend.read_profiles()
    names = backend.name_records() if alias else []
    return dict(schema=1, game=str(directory), status="prepared", before_files=original,
                after_files={"winmm.dll": base64.b64encode(patched).decode(),
                             CONFIG: base64.b64encode(config_bytes(decode(original[CONFIG]))).decode()},
                before_profiles=profiles, after_profiles=desired_profiles(), names=names)


def check_restore_conflicts(state, backend):
    directory = find_game(state["game"])
    current = file_snapshot(directory)
    for name in FILES:
        if not any(same_file(name, current[name], candidate) for candidate in
                   (state["before_files"][name], state["after_files"][name])):
            raise ValueError(f"{name} changed after setup. Preserve/reconcile it before retrying rollback.")
    profiles = backend.read_profiles()
    for profile in ("global", "wuwa"):
        for setting in SETTINGS:
            if profiles[profile][setting] not in (state["before_profiles"][profile][setting],
                                                   state["after_profiles"][profile][setting]):
                raise ValueError(f"NVIDIA {profile} setting {setting} changed after setup; rollback stopped.")
    backend.check_names(state["names"])
    return directory


def read_restore_state(state_path):
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    if state.get("schema") != 1 or set(state.get("before_files", {})) != set(FILES) or set(state.get("after_files", {})) != set(FILES):
        raise ValueError("Unsupported backup manifest.")
    return state


def restore(state_path, backend):
    state_path = Path(state_path)
    state = read_restore_state(state_path)
    if state.get("status") == "restored":
        return "Already restored."
    backend.assert_closed()
    directory = check_restore_conflicts(state, backend)
    # Restoring is repeatable: every resource may already be at its original value.
    state["status"] = "restoring"
    write_json(state_path, state)
    backend.write_profiles(state["before_profiles"])
    backend.write_names(state["names"], restore=True)
    for name in FILES:
        payload = decode(state["before_files"][name])
        path = directory / name
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write(path, payload)
    if file_snapshot(directory) != state["before_files"] or backend.read_profiles() != state["before_profiles"]:
        raise RuntimeError("Rollback verification failed; backup retained for retry.")
    state["status"] = "restored"
    write_json(state_path, state)
    return "Original files, the two NVIDIA settings, and any name aliases restored. Backup retained."


def install(state_path, state, backend):
    state_path = Path(state_path)
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") != "restored":
            raise ValueError("A setup backup already exists. Use Verify or Restore before another install.")
        raise ValueError("A restored backup exists. Archive its folder before a new installation.")
    backend.assert_closed()
    directory = find_game(state["game"])
    if any((directory / name).exists() for name in PROXIES):
        raise ValueError("Another proxy appeared since preview. Resolve the conflict before setup.")
    if file_snapshot(directory) != state["before_files"] or backend.read_profiles() != state["before_profiles"]:
        raise ValueError("Files or settings changed since preview. Run setup again.")
    backend.check_names(state["names"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist complete originals before the first game/profile/registry mutation.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for name in FILES:
            atomic_write(directory / name, decode(state["after_files"][name]))
        backend.write_profiles(state["after_profiles"])
        backend.write_names(state["names"], restore=False)
        if file_snapshot(directory) != state["after_files"] or backend.read_profiles() != state["after_profiles"]:
            raise RuntimeError("Installation readback failed.")
        state["status"] = "installed"
        write_json(state_path, state)
    except Exception as error:
        try:
            restore(state_path, backend)
        except Exception as rollback_error:
            raise RuntimeError(f"Setup failed: {error}. Rollback needs attention: {rollback_error}. Backup: {state_path}") from error
        raise RuntimeError(f"Setup failed; original state restored: {error}") from error


def summarize_status(status, running_pids, now):
    """Only fresh, active runtime state counts; a menu label or FPS counter does not."""
    if status.get("pid") not in running_pids:
        return {"verified": False, "reason": "Status belongs to a game process that is no longer running."}
    heartbeat = status.get("heartbeat", 0)
    if not isinstance(heartbeat, (float, int)) or not 0 <= now - heartbeat <= 10:
        return {"verified": False, "reason": "Runtime status is stale."}
    if not status.get("gameFrameGenerationOn") or not status.get("appliedFrameGenerationOn"):
        return {"verified": False, "reason": "Frame Generation is off or gameplay is not active."}
    requested = status.get("appliedMultiplier")
    actual = status.get("actualFramesPresented")
    ok = (status.get("bridgeReady") is True and status.get("applied") is True
          and status.get("setOptionsAccepted") is True and status.get("setOptionsResult") == 0
          and status.get("getStateResult") == 0 and status.get("stateSampleAgeMs", 99999) <= 1000
          and requested in range(2, 7) and actual == requested)
    return {"verified": ok, "requested": requested, "presented": actual,
            "reason": "Runtime reports matching output; also check visible motion and artifacts." if ok
                      else "Requested and presented frame counts are not confirmed.",
            "follow_game": status.get("followGame"), "result": status.get("setOptionsResult"),
            "bridge_ready": status.get("bridgeReady"), "wrapper_patched": status.get("activeWrapperPatched")}
