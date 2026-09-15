"""Run a bounded trial in the official NVIDIA demo, never in the game."""
from pathlib import Path
import argparse, configparser, hashlib, json, re, statistics, subprocess, time

ARGS = ['-d3d12', '-width', '1920', '-height', '1080']
RX = re.compile(r'DLSS-NR elapsed: ([\d.]+) ms total, ([\d.]+) ms model, ([\d.]+) ms surrounding')

def gpu_snapshot():
    r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.gr,pstate',
                        '--format=csv,noheader'], capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return r.stdout.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--demo-dir', required=True, help='Directory containing the configured ngx_dlss_demo.exe')
    parser.add_argument('--output-dir', required=True, help='Local directory for private logs and trial records')
    parser.add_argument('--mask', choices=['true', 'false'], default='true')
    parser.add_argument('--preset', type=int, choices=range(4), default=0)
    parser.add_argument('--style', type=int, choices=range(3), default=1)
    parser.add_argument('--seconds', type=int, default=45)
    parser.add_argument('--restore-state', action='store_true')
    parser.add_argument('--fps', type=int, choices=[0, 60, 120], default=0)
    parser.add_argument('--isolated-desktop', action='store_true', default=True,
                        help='Enabled by default: contain all demo windows and error dialogs on a desktop that is never activated.')
    opts = parser.parse_args()
    DEMO = Path(opts.demo_dir).resolve()
    ROOT = Path(opts.output_dir).resolve()
    assert (DEMO / 'ngx_dlss_demo.exe').is_file(), 'Expected the official NVIDIA DLSS sample executable'
    background_hash = 'f262742631d02da935649322f220f0490b114287c98a1d75d60e04557f98a9c6'
    if hashlib.sha256((DEMO / 'ngx_dlss_demo.exe').read_bytes()).hexdigest() != background_hash:
        raise SystemExit('Prepare the exact background demo with prepare_hidden_demo.py first; refusing a launch that could open in front.')
    assert re.fullmatch(r'[a-z0-9-]+', opts.label)
    assert 20 <= opts.seconds <= 180
    existing = subprocess.run(['powershell', '-NoProfile', '-Command',
        "@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    assert existing.stdout.strip() == '0', 'Close the demo and game before a controlled trial.'
    out = ROOT / 'trials' / opts.label
    out.mkdir(parents=True, exist_ok=False)
    ini_path = DEMO / 'OptiScaler.ini'
    original = ini_path.read_bytes()
    (out / 'original.ini').write_bytes(original)
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    config.read_string(original.decode('utf-8-sig'))
    config['DlssNr']['AutoMask'] = opts.mask
    config['DlssNr']['Preset'] = str(opts.preset)
    config['DlssNr']['Style'] = str(opts.style)
    if opts.restore_state:
        for key in ('RestoreComputeSignature', 'RestoreGraphicSignature', 'ExtendedStateRestore'):
            config['Hotfix'][key] = 'true'
    config['Framerate']['FramerateLimit'] = str(opts.fps)
    assert config['ProcessFilter']['TargetProcessName'] == 'ngx_dlss_demo.exe'
    assert config['DlssNr']['WorkingScale'] == '1.0'
    assert config['DlssNr']['Passes'] == '1'
    with ini_path.open('w', encoding='utf-8') as f:
        config.write(f, space_around_delimiters=False)
    (out / 'trial.ini').write_bytes(ini_path.read_bytes())
    start = time.monotonic()
    proc = None
    observations = []
    telemetry = []
    desktop_observations = []
    try:
        log_path = DEMO / 'OptiScaler.log'
        if log_path.exists():
            log_path.replace(out / 'pre-existing.log')
        from isolated_demo_process import IsolatedDemoProcess
        proc = IsolatedDemoProcess([str(DEMO / 'ngx_dlss_demo.exe'), *ARGS], cwd=DEMO)
        while time.monotonic() - start < opts.seconds:
            time.sleep(5)
            log_path = DEMO / 'OptiScaler.log'
            lines = log_path.read_text(errors='replace').splitlines() if log_path.exists() else []
            elapsed = time.monotonic() - start
            if opts.isolated_desktop:
                desktop_observations.append({'elapsed_s': round(elapsed, 1), **proc.snapshot()})
            telemetry.append({'elapsed_s': round(elapsed, 1), 'gpu': gpu_snapshot()})
            fresh = [line for line in lines if RX.search(line)]
            for line in fresh:
                if line in [x['line'] for x in observations]:
                    continue
                values = list(map(float, RX.search(line).groups()))
                observations.append({'observed_at_s': round(elapsed, 1), 'line': line,
                    'total_ms': values[0], 'model_ms': values[1], 'surrounding_ms': values[2]})
            if proc.poll() is not None:
                break
        full_log = log_path.read_text(errors='replace') if log_path.exists() else ''
        (out / 'OptiScaler.log').write_text(full_log, encoding='utf-8')
        # Approximate warmup; each record is first observed on a five-second poll.
        kept = [x for x in observations if x['observed_at_s'] >= 15]
        result = {'label': opts.label, 'settings': vars(opts), 'warmup_seconds': 15,
                  'pid': proc.pid, 'exit_code_before_cleanup': proc.poll(),
                  'model_1920x1080_confirmed': 'model 1920x1080, basis output' in full_log,
                  'observations': observations, 'telemetry': telemetry,
                  'retained_count': len(kept), 'timings_are_sparse_gpu_intervals': True}
        result['hidden_launch_requested'] = True
        if opts.isolated_desktop:
            result['isolated_desktop_observations'] = desktop_observations
        result['local_file_sha256'] = {name: hashlib.sha256((DEMO / name).read_bytes()).hexdigest()
            for name in ('ngx_dlss_demo.exe', 'dxgi.dll', 'nvngx_dlssnr.dll')}
        if kept:
            result['summary'] = {field: {'median': statistics.median(x[field] for x in kept),
                'min': min(x[field] for x in kept), 'max': max(x[field] for x in kept)}
                for field in ('total_ms', 'model_ms', 'surrounding_ms')}
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('observations','telemetry')}, indent=2))
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
            if opts.isolated_desktop and proc is not None:
                proc.close()
        finally:
            ini_path.write_bytes(original)

if __name__ == '__main__':
    main()
