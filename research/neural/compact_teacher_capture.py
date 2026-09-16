# SPDX-License-Identifier: Apache-2.0
"""Lossless Windows file compression for explicitly named private capture files.

No capture format change, deletion, directory recursion or system compression
setting. Call only after the owned capture process has exited and GPU fences
have completed. Hash every logical payload before and after the operation.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def allocated_bytes(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    function=kernel.GetCompressedFileSizeW
    function.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
    function.restype=wintypes.DWORD
    high=wintypes.DWORD();ctypes.set_last_error(0)
    low=function(str(path),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low


def capture_files(capture, allowed_root):
    capture=capture.resolve(strict=True);allowed_root=allowed_root.resolve(strict=True)
    repo=Path(__file__).resolve().parents[2]
    if capture==allowed_root or not capture.is_relative_to(allowed_root) or capture.is_relative_to(repo):
        raise ValueError('Capture must be a private child of its explicit research root.')
    rows=[];seen=set()
    for index in range(4):
        frame=json.loads((capture/f'frame-{index}.json').read_text())
        if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1):
            raise ValueError('Only complete, fenced native captures may be compressed.')
        if set(frame['resources'])!={'color','depth','motion','output'}:
            raise ValueError('Unexpected capture resource set.')
        for role,resource in frame['resources'].items():
            name=resource['file'];path=(capture/name).resolve(strict=True)
            expected=resource['rows']*resource['row_bytes']
            if Path(name).name!=name or path.parent!=capture or path in seen or not path.is_file():
                raise ValueError('Unexpected or repeated capture filename.')
            if not 0<expected<=64*1024*1024 or path.stat().st_size!=expected:
                raise ValueError('Unexpected capture payload size.')
            seen.add(path);rows.append((path,{'frame':index,'role':role,'logical_bytes':expected,
                                            'sha256':digest(path),'allocated_before':allocated_bytes(path)}))
    return rows


def compact_capture(capture, allowed_root):
    files=capture_files(capture,allowed_root)
    free_before=shutil.disk_usage(capture).free
    result=subprocess.run(['compact.exe','/C','/EXE:LZX','/Q',*[str(path) for path,_ in files]],
                          capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=[]
    for path,row in files:
        if path.stat().st_size!=row['logical_bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('Logical capture bytes changed during compression.')
        rows.append({**row,'allocated_after':allocated_bytes(path),'logical_bytes_unchanged':True})
    if result.returncode:
        raise RuntimeError('Compression command failed; logical capture bytes remain verified unchanged.')
    return {'complete':True,'codec':'Windows compact /EXE:LZX','resource_count':len(rows),
            'logical_bytes_unchanged':True,'logical_bytes':sum(r['logical_bytes'] for r in rows),
            'allocated_before':sum(r['allocated_before'] for r in rows),'allocated_after':sum(r['allocated_after'] for r in rows),
            'disk_free_delta':shutil.disk_usage(capture).free-free_before,'resources':rows,
            'scope':'Explicit capture payload files only. No recursion, deletion, format change or OS compression policy change.'}
