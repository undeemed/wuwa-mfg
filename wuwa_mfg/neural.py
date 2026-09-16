"""Opt-in NR setup. No driver changes; only a locally built, inspected bundle is accepted."""
import configparser
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import uuid

from .core import atomic_write, find_game, write_json
from .patch import PATCHED_SHA256

BASE_COMMIT = '8802b2b470db0462fa1ed03a125e793a7c06d735'
NR_SHA256 = '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
ALLOWED = {
    'dxgi.dll', 'nvngx_dlssnr.dll',
    'OptiScaler/amd_fidelityfx_framegeneration_dx12.dll',
    'OptiScaler/amd_fidelityfx_loader_dx12.dll',
    'OptiScaler/amd_fidelityfx_upscaler_dx12.dll',
    'OptiScaler/amd_fidelityfx_vk.dll', 'OptiScaler/libxell.dll',
    'OptiScaler/libxess_dx11.dll', 'OptiScaler/libxess_fg.dll',
    'OptiScaler/libxess.dll', 'OptiScaler/D3D12_OptiScaler/D3D12Core.dll',
}
ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def target(root, name):
    if name not in ALLOWED | {'OptiScaler.ini'} or PurePosixPath(name).is_absolute():
        raise ValueError(f'Unexpected NR file: {name}')
    path = Path(root) / name
    cursor = path
    while cursor != Path(root):
        # File attributes identify junctions without mistaking Windows 8.3 aliases
        # (for example RUNNER~1 in TEMP) for redirected directories.
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'Redirected NR path: {name}')
        cursor = cursor.parent
    if path.exists() and not path.is_file():
        raise ValueError(f'Not a regular file: {name}')
    return path


def configure(data, enabled, scale=None):
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str
    ini.read_string(data.decode('utf-8-sig'))
    if not ini.has_section('DlssNr'):
        raise ValueError('Missing DlssNr section')
    if enabled is not None:
        ini['DlssNr']['Enabled'] = str(bool(enabled)).lower()
    ini['DlssNr']['ToggleKey'] = '0x77'
    if scale is not None:
        if not math.isfinite(scale) or not 0.25 <= scale <= 1.0:
            raise ValueError('NR scale must be between 0.25 and 1.0 of output width and height')
        ini['DlssNr']['WorkingScale'] = format(scale, '.6g')
        ini['DlssNr']['WorkingScaleRelativeToOutput'] = 'true'
    text = io.StringIO()
    ini.write(text, space_around_delimiters=False)
    return text.getvalue().encode('utf-8')


def copy_verified(source, dest, expected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.wuwa-nr-', dir=dest.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise ValueError('NR copy checksum mismatch')
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_bundle(bundle):
    bundle = Path(bundle).resolve()
    info = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    expected_patch = digest(ROOT / 'patches/optiscaler-wuwa-compat.patch')
    if info.get('base_commit') != BASE_COMMIT or info.get('patch_sha256') != expected_patch:
        raise ValueError('Bundle does not match this toolkit source/patch. Rebuild it.')
    files = info.get('files', {})
    if set(files) != ALLOWED:
        raise ValueError('Incomplete or unexpected NR bundle files')
    for name, expected in files.items():
        if digest(target(bundle, name)) != expected:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    if files['nvngx_dlssnr.dll'] != NR_SHA256:
        raise ValueError('Unrecognized NR runtime; see the pinned-runtime documentation')
    return bundle, files


def install(game, bundle, state_path, assert_closed):
    assert_closed()
    game, state_path = find_game(game), Path(state_path).resolve()
    if state_path.exists():
        raise ValueError('NR backup already exists. Use neural-status or neural-restore.')
    if digest(game / 'winmm.dll') != PATCHED_SHA256:
        raise ValueError('Install and verify the tested WuWa RTXMFG setup first.')
    bundle, files = inspect_bundle(bundle)
    for proxy in ('version.dll', 'd3d12.dll', 'dinput8.dll', 'xinput1_3.dll', 'winhttp.dll'):
        if (game / proxy).exists():
            raise ValueError(f'Existing proxy conflict: {proxy}')
    # Never take ownership of an existing injector or an unrelated NR configuration.
    for name in ('dxgi.dll', 'OptiScaler.ini', 'nvngx_dlssnr.dll'):
        if target(game, name).exists():
            raise ValueError(f'Existing {name}; restore that installation before using NR setup.')
    config = configure((ROOT / 'neural/OptiScaler.ini').read_bytes(), False)
    hashes = dict(files, **{'OptiScaler.ini': hashlib.sha256(config).hexdigest()})
    state_path.parent.mkdir(parents=True, exist_ok=True)
    backup = state_path.parent / ('neural-backup-' + uuid.uuid4().hex)
    entries = {}
    for name, after in hashes.items():
        dest = target(game, name)
        before = digest(dest) if dest.exists() else None
        # Shared backends may be retained if identical, but not silently replaced.
        if before and before != after:
            raise ValueError(f'Existing backend differs: {name}')
        entries[name] = {'before': before, 'after': after}
    backup.mkdir()
    state = {'version': 1, 'game': str(game), 'backup': str(backup), 'phase': 'installing',
             'files': entries, 'enabled': False, 'mfg_hash': digest(game / 'winmm.dll'),
             'mfg_config_hash': digest(game / 'RTXMFG-Universal.json')}
    write_json(state_path, state)
    try:
        for name, entry in entries.items():
            if entry['before'] == entry['after']:
                continue
            dest = target(game, name)
            if name == 'OptiScaler.ini':
                atomic_write(dest, config)
            else:
                copy_verified(target(bundle, name), dest, entry['after'])
        state['phase'] = 'installed'
        write_json(state_path, state)
    except Exception:
        # The journal precedes writes and supports interrupted-install recovery.
        restore(state_path, assert_closed)
        raise
    return {'installed': True, 'nr_enabled': False, 'toggle': 'F8', 'restart_required': True}


def read_state(state_path):
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text(encoding='utf-8'))
    game = find_game(state['game'])
    if state.get('version') != 1 or set(state.get('files', {})) != ALLOWED | {'OptiScaler.ini'}:
        raise ValueError('Unrecognized NR recovery record')
    for name in state['files']:
        target(game, name)
    backup = Path(state['backup'])
    if (backup.parent.resolve() != state_path.parent or
            not backup.name.startswith('neural-backup-') or
            backup.is_symlink() or not backup.is_dir() or
            getattr(backup.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('Invalid NR configuration backup directory')
    return state, game


def check_restore_conflicts(state_path):
    state, game = read_state(state_path)
    for name, entry in state['files'].items():
        dest = target(game, name)
        actual = digest(dest) if dest.exists() else None
        accepted = [entry['before'], entry['after']]
        if name == 'OptiScaler.ini' and state.get('pending_config_hash'):
            accepted.append(state['pending_config_hash'])
        if actual not in accepted:
            raise ValueError(f'{name} changed after setup; restore stopped before modifying anything')
    return state, game


def restore(state_path, assert_closed):
    assert_closed()
    state_path = Path(state_path)
    state, game = check_restore_conflicts(state_path)
    for name, entry in state['files'].items():
        dest = target(game, name)
        if entry['before'] is None and dest.exists():
            dest.unlink()
    # Existing identical backends are deliberately retained. Original game/driver files are untouched.
    state['phase'] = 'restored'
    write_json(state_path, state)
    archive = state_path.with_name('neural-restored-' + uuid.uuid4().hex + '.json')
    state_path.rename(archive)
    return {'restored': True, 'record': str(archive)}


def set_enabled(state_path, enabled, assert_closed, scale=None):
    assert_closed()
    state_path = Path(state_path)
    state, game = read_state(state_path)
    path = target(game, 'OptiScaler.ini')
    entry = state['files']['OptiScaler.ini']
    if state.get('pending_config_hash'):
        raise ValueError('Interrupted config update: use neural-restore first')
    if digest(path) != entry['after']:
        raise ValueError('NR configuration was edited externally; refusing to overwrite it')
    old = path.read_bytes()
    new = configure(old, enabled, scale)
    saved = Path(state['backup']) / ('config-' + uuid.uuid4().hex + '.ini')
    saved.write_bytes(old)
    # Keep an interrupted config change restorable at either checksum.
    state['pending_config_hash'] = hashlib.sha256(new).hexdigest()
    write_json(state_path, state)
    atomic_write(path, new)
    entry['after'] = state.pop('pending_config_hash')
    if enabled is not None:
        state['enabled'] = bool(enabled)
    if scale is not None:
        state['output_scale'] = scale
    write_json(state_path, state)
    return {'nr_enabled_next_launch': state['enabled'], 'output_scale': state.get('output_scale', 1.0),
            'toggle': 'F8', 'restart_required': True}
