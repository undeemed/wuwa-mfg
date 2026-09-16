"""Restore managed NR and MFG in order, using their existing recovery records."""
from pathlib import Path

from . import core, neural


def plan(state_path, neural_state, backend):
    backend.assert_closed()
    mfg = core.read_restore_state(state_path) if Path(state_path).exists() else None
    nr = neural.check_restore_conflicts(neural_state)[0] if Path(neural_state).exists() else None
    active_mfg = mfg is not None and mfg.get('status') != 'restored'
    if active_mfg:
        core.check_restore_conflicts(mfg, backend)
    if active_mfg and nr and Path(mfg['game']).resolve() != Path(nr['game']).resolve():
        raise ValueError('MFG and neural backups refer to different games. Restore each separately.')
    return {'mfg': mfg if active_mfg else None, 'neural': nr}


def restore_all(state_path, neural_state, backend, expected):
    # Recheck both components before changing either, including after confirmation.
    current = plan(state_path, neural_state, backend)
    if current != expected:
        raise ValueError('The uninstall plan changed. Run Uninstall.cmd again to review it.')
    result = {'neural_restored': False, 'mfg_restored': False, 'restart_required': False}
    if current['neural']:
        neural.restore(neural_state, backend.assert_closed)
        result['neural_restored'] = True
    if current['mfg']:
        try:
            core.restore(state_path, backend)
        except (OSError, ValueError, RuntimeError) as error:
            if result['neural_restored']:
                raise RuntimeError('Neural add-on restored; MFG restore stopped. Backups are retained. '
                                   f'Resolve this error and rerun Uninstall.cmd: {error}') from error
            raise
        result['mfg_restored'] = True
        result['restart_required'] = bool(current['mfg'].get('names'))
    return result


def run(state_path, neural_state, backend, confirm=input):
    prepared = plan(state_path, neural_state, backend)
    if not any(prepared.values()):
        print('No active managed installation found. Manual installations are left unchanged.')
        print('Use the original installation backup for manually installed mods.')
        return 0
    state = prepared['neural'] or prepared['mfg']
    print(f"Uninstall managed add-ons from: {state['game']}")
    if prepared['neural']:
        print('  Restore the managed neural add-on first.')
    if prepared['mfg']:
        print('  Restore MFG files, the saved NVIDIA settings, and any GPU-name aliases.')
    print('Recovery records are retained. The game and its saves are not removed.')
    if confirm('Type UNINSTALL to continue: ').strip() != 'UNINSTALL':
        print('Cancelled; no changes made.')
        return 0
    result = restore_all(state_path, neural_state, backend, prepared)
    print('Managed add-ons restored. Backups retained.')
    if result['restart_required']:
        print('Restart Windows to refresh the GPU descriptions. No automatic restart.')
    return 0
