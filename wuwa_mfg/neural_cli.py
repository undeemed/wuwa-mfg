"""User-facing NR actions, deliberately separate from the MFG installer."""
import json
import re
from . import neural


def run(args, backend, state_path, choose_game, is_admin):
    if args.action == 'neural-status':
        game = choose_game(args.game)
        from .windows import running_pids
        pids = running_pids()
        log = game / 'OptiScaler.log'
        text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
        matching = [pid for pid in pids if re.search(rf'(?:inputPid:|currentPid:){pid}\b', text)]
        report = {'game_running': bool(pids), 'log_matches_running_pid': matching,
                  'managed_install': state_path.exists(), 'toggle': 'F8'}
        if state_path.exists():
            state, _ = neural.read_state(state_path)
            report['nr_enabled_next_launch'] = state['enabled']
        report['recent_nr_lines'] = [line for line in text.splitlines() if re.search(
            r'DlssNr.Enabled:|DlssNr.ToggleKey:|DLSS-NR resolution:|DLSS-NR: feature created|DLSS-NR elapsed:|DLSS-NR descriptor backpressure:', line)][-8:] if matching else []
        report['note'] = 'Startup config and past timings do not prove current toggle state. Use Verify for fresh MFG counts.'
        print(json.dumps(report, indent=2))
        return 0
    if not is_admin():
        raise RuntimeError('Use Setup.cmd for the normal administrator prompt.')
    backend.assert_closed()
    if args.action == 'neural-install':
        backend.check_compatibility()
        game = choose_game(args.game)
        bundle = args.bundle or input('Folder produced by BuildNeural.ps1: ').strip().strip('"')
        neural.inspect_bundle(bundle)
        print('Experimental NR: prior build ran with 6x but GPU-crashed during a settings/window transition.')
        print('The descriptor-lifetime fix passed isolated tests and an initial game run; transition stability remains experimental.')
        print('Installs the locally built OptiScaler bundle; retains RTXMFG and driver settings. NR starts OFF.')
        if input('Type EXPERIMENTAL to install: ').strip() != 'EXPERIMENTAL':
            return 0
        result = neural.install(game, bundle, state_path, backend.assert_closed)
    elif args.action == 'neural-restore':
        result = neural.restore(state_path, backend.assert_closed)
    elif args.action == 'neural-scale':
        scale = args.scale if args.scale is not None else float(input('Output resolution scale (1=100%, 0.75=75%, 0.5=50%): '))
        result = neural.set_enabled(state_path, None, backend.assert_closed, scale=scale)
    else:
        result = neural.set_enabled(state_path, args.action == 'neural-on', backend.assert_closed)
    print(json.dumps(result, indent=2))
    return 0
