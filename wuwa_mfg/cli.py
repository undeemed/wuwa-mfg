"""Small interactive installer and read-only runtime verifier."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import __version__
from .core import find_game, install, make_plan, restore, summarize_status
from .patch import download_dll, patch_dll, sha256, PATCHED_SHA256


def choose_game(argument):
    from .windows import detect_games
    if argument:
        return find_game(argument)
    games = detect_games()
    if len(games) == 1:
        print(f"Found WuWa: {games[0]}")
        return games[0]
    for index, path in enumerate(games, 1):
        print(f"  {index}. {path}")
    answer = input("Select a number, or paste your WuWa game folder: ").strip().strip('"')
    if answer.isdecimal() and 1 <= int(answer) <= len(games):
        return games[int(answer) - 1]
    return find_game(answer)


def verify(game, seconds):
    from .windows import running_pids
    pids = running_pids()
    if not pids:
        print("Start WuWa normally, enable Frame Generation and enter gameplay, then run Verify again.")
        return 2
    print("Keep WuWa focused in gameplay. Reading status files only; no process injection or key presses.")
    end = time.monotonic() + seconds
    matched = {}
    failures = {}
    last_key = None
    while time.monotonic() < end:
        candidates = [game / f"RTXMFG-Universal.{pid}.status.json" for pid in pids]
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            failures["No status file for the running WuWa process. Check installation and restart the game."] = 1
        else:
            latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            try:
                status = json.loads(latest.read_text(encoding="utf-8-sig"))
                # Do not count a repeatedly read heartbeat as independent evidence.
                key = (status.get("pid"), status.get("heartbeat"))
                if key != last_key:
                    last_key = key
                    report = summarize_status(status, running_pids(), time.time())
                    if report["verified"]:
                        count = report["presented"]
                        matched[count] = matched.get(count, 0) + 1
                    else:
                        reason = report["reason"]
                        failures[reason] = failures.get(reason, 0) + 1
            except (OSError, ValueError):
                # The upstream file can be in the middle of a write.
                pass
        time.sleep(1)
    output = {"matching_runtime_samples_by_multiplier": matched, "unconfirmed_samples": failures,
              "note": "Runtime counts do not prove frame uniqueness or image quality. Check visible motion too."}
    print(json.dumps(output, indent=2))
    return 0 if matched else 2


def main():
    parser = argparse.ArgumentParser(description="WuWa Toolkit: MFG and experimental neural rendering")
    parser.add_argument("action", choices=("install", "uninstall", "restore", "verify", "doctor", "patch",
        "neural-install", "neural-on", "neural-off", "neural-restore", "neural-status", "neural-scale"))
    parser.add_argument("--game", help="WuWa root, Client/Binaries/Win64, or shipping EXE")
    parser.add_argument("--seconds", type=int, default=30, help="Verify duration (1-300 seconds)")
    parser.add_argument("--input", type=Path, help="Original v1.3.3 DLL for offline patching")
    parser.add_argument("--output", type=Path, help="New output file for offline patching")
    parser.add_argument("--bundle", type=Path, help="Locally built NR bundle folder")
    parser.add_argument("--scale", type=float, help="NR output resolution fraction, 0.25 to 1.0")
    args = parser.parse_args()
    try:
        if args.action == "patch":
            if not args.input or not args.output:
                parser.error("patch requires --input and --output")
            payload = patch_dll(args.input.read_bytes())
            with args.output.open("xb") as stream:
                stream.write(payload)
            print(f"Created {args.output.name}; SHA-256 {sha256(payload)}")
            return 0
        if sys.platform != "win32":
            raise RuntimeError("Game setup requires 64-bit Windows. Offline patching/tests are cross-platform.")
        from .windows import WindowsBackend, is_admin
        backend = WindowsBackend()
        state_path = Path(os.environ["ProgramData"]) / "WuWaMFG/state.json"
        neural_state = state_path.with_name('neural.json')
        if args.action == 'uninstall':
            if not is_admin():
                raise RuntimeError('Use Uninstall.cmd for the normal administrator prompt.')
            from .uninstall import run
            return run(state_path, neural_state, backend)
        if args.action.startswith('neural-'):
            from .neural_cli import run
            return run(args, backend, neural_state, choose_game, is_admin)
        if args.action == "doctor":
            report = {"installer_version": __version__}
            try:
                report["compatibility"] = backend.check_compatibility()
            except RuntimeError as error:
                report["compatibility"] = str(error)
            report["profile_user_overrides"] = backend.read_profiles()
            game = choose_game(args.game)
            dll = game / "winmm.dll"
            report["tested_patch_installed"] = dll.exists() and sha256(dll.read_bytes()) == PATCHED_SHA256
            report["backup_present"] = state_path.exists()
            print(json.dumps(report, indent=2))
            return 0
        if args.action == "verify":
            if not 1 <= args.seconds <= 300:
                parser.error("--seconds must be 1-300")
            return verify(choose_game(args.game), args.seconds)
        if not is_admin():
            raise RuntimeError("Use Setup.cmd to request the normal administrator prompt for Install/Restore.")
        if args.action == "restore":
            if neural_state.exists():
                raise RuntimeError('Restore the neural add-on first with neural-restore, then restore MFG.')
            if not state_path.exists():
                raise RuntimeError("No installer backup found. An existing manual setup is left untouched.")
            state = json.loads(state_path.read_text())
            print(f"Restore the saved setup in: {state['game']}")
            print("Restores backed-up files, the two global/game NVIDIA overrides, and saved description aliases.")
            if input("Type RESTORE to continue: ").strip() != "RESTORE":
                print("Cancelled.")
                return 0
            print(restore(state_path, backend))
            if state.get("names"):
                print("Restart Windows to refresh the GPU descriptions. Setup will not restart it for you.")
            return 0
        game = choose_game(args.game)
        backend.assert_closed()
        print(backend.check_compatibility())
        if state_path.exists():
            raise RuntimeError(f"An installer backup already exists at {state_path}. Use Verify or Restore.")
        print("\nThis experimental mod is not an official NVIDIA feature or an anti-cheat-approved mod.")
        print("Changes: one winmm.dll + Follow game/Preset B config; global FG DLL override ON;")
        print("global frame count application-controlled; remove the same two WuWa local overrides.")
        print("Global NVIDIA settings can affect other compatible games. Originals will be backed up.")
        print("\nThe tested setup also used RTX 5080 display-name aliases to expose WuWa's multiplier menu.")
        print("Optional: changes three Windows descriptions system-wide. Hardware IDs stay unchanged.")
        alias = input("Apply the display-name aliases too? [y/N]: ").strip().lower() == "y"
        print("Downloading the pinned upstream RTXMFG release and verifying its SHA-256...")
        patched = patch_dll(download_dll())
        state = make_plan(game, backend, patched, alias)
        print(f"\nTarget: {game}\nBackup: {state_path}")
        print(f"Display-name aliases: {'yes, Windows restart required' if alias else 'no'}")
        if input("Type INSTALL to apply this setup: ").strip() != "INSTALL":
            print("Cancelled; game, profiles and registry unchanged.")
            return 0
        install(state_path, state, backend)
        print("\nInstalled and read back successfully. WuWa controls the multiplier (2x through 6x).")
        if alias:
            print("Restart Windows before testing. Setup will not restart it for you.")
        else:
            print("Start a fresh WuWa process to clear cached driver overrides.")
        print("Use the native game settings, then Setup.cmd > Verify during gameplay.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"\nStopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
