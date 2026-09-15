# SPDX-License-Identifier: Apache-2.0
"""Run a research script using Python's Windows platform-query fallback.

Use only when WMI queries are stalled. This makes the optional _wmi module
unavailable inside this process before platform/PyTorch are imported. Python
then uses its existing environment/Win32 fallbacks. No service or file is changed.
"""
import os
import sys


def main():
    if os.name != 'nt' or len(sys.argv) < 2 or 'platform' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Start a fresh Windows interpreter with a target script argument.')
    if os.environ.get('PROCESSOR_ARCHITECTURE') != 'AMD64' or sys.maxsize <= 2**32:
        raise RuntimeError('This fallback is checked only for a native AMD64 interpreter.')
    sys.modules['_wmi'] = None
    import platform
    assert platform.machine() == 'AMD64'
    import runpy
    from pathlib import Path
    target = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    print('Using process-local Python platform fallback (AMD64); Windows services unchanged.', flush=True)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':main()
