"""Package a local GPL source build and user-supplied NR runtime; never uploads binaries."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wuwa_mfg.neural import ALLOWED, BASE_COMMIT, NR_SHA256, ROOT, digest

PACKAGE_SHA256 = '8789912859882e66b3f3a1aa768db947da779dfd65225df69ea919052e73a2e4'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--nr-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip()
    if commit != BASE_COMMIT:
        raise SystemExit('Unexpected source commit')
    patch = ROOT / 'patches/optiscaler-wuwa-compat.patch'
    subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=args.source, check=True)
    if digest(args.package) != PACKAGE_SHA256 or digest(args.nr_runtime) != NR_SHA256:
        raise SystemExit('Package/runtime hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.package) as archive:
        for name in sorted(ALLOWED - {'dxgi.dll', 'nvngx_dlssnr.dll'}):
            # Extract only the explicit allowlist, never arbitrary ZIP paths.
            source = archive.getinfo(name)
            if source.file_size > 200 * 1024 * 1024:
                raise ValueError('Unexpected backend size')
            dest = args.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(source) as data, dest.open('wb') as out:
                shutil.copyfileobj(data, out)
    shutil.copyfile(args.source / 'x64/Release/OptiScaler.dll', args.output / 'dxgi.dll')
    shutil.copyfile(args.nr_runtime, args.output / 'nvngx_dlssnr.dll')
    info = {'base_commit': commit, 'patch_sha256': digest(patch),
            'files': {name: digest(args.output / name) for name in sorted(ALLOWED)}}
    (args.output / 'bundle.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Local NR bundle ready: {args.output.resolve()}')
    print('For local installation only. No NVIDIA runtime or built DLL is uploaded by this tool.')


if __name__ == '__main__':
    main()
