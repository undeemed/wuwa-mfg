# SPDX-License-Identifier: Apache-2.0
"""Install or restore the locally built student integration over the tested NR setup.

Requires a closed game and an existing working NR + MFG installation. Writes only
the OptiScaler DLL/INI and the private neural-student folder. Backups are local.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuwa_mfg.core import atomic_write,find_game
from wuwa_mfg.neural import NR_SHA256,digest,inspect_bundle
from wuwa_mfg.patch import PATCHED_SHA256

BASE_DLL='b3b0857a6d94e4745b42f2bb3beb747c527548ceec0abff11afecf89a1f6a590'
DML_DLL='9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1'
FILES={'dxgi.dll','OptiScaler.ini','neural-student/model.json','neural-student/weights.bin','neural-student/DirectML.dll'}


def closed():
    result=subprocess.run(['powershell','-NoProfile','-Command',
        '@(Get-Process -Name Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        check=True,capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip()!='0':raise RuntimeError('Close WuWa before replacing the integration DLL.')


def safe(root,name):
    if name not in FILES:raise ValueError('Unexpected installation file')
    path=root/name
    for cursor in (path,path.parent):
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(),'st_file_attributes',0)&0x400):
            raise ValueError('Redirected installation path')
    if path.exists() and not path.is_file():raise ValueError('Expected a regular file')
    return path


def configure(data):
    text=data.decode('utf-8-sig'); parsed=configparser.ConfigParser(interpolation=None)
    parsed.read_string(text); nr=parsed['DlssNr']
    expected={'Passes':'1','Style':'1','Preset':'0','Intensity':'1','LocalTone':'1','LocalStructure':'1','SkinStructure':'-1'}
    for key,value in expected.items():
        actual=nr.get(key,'auto')
        if actual.lower()!='auto' and float(actual)!=float(value):raise ValueError('Student needs default Natural controls: '+key)
    if nr.get('AutoMask','auto').lower() not in ('auto','true'):raise ValueError('Student needs AutoMask=true')
    # Preserve all unrelated config lines, comments and bindings.
    pattern=r'(?ms)(^\[DlssNr\]\s*\r?\n)(.*?)(?=^\[|\Z)'
    found=re.search(pattern,text)
    if not found:raise ValueError('Missing NR configuration')
    body=found.group(2)
    for key,value in {'Engine':'1','WorkingScale':'0.5','WorkingScaleRelativeToOutput':'true'}.items():
        line=re.compile(r'(?m)^'+key+r'\s*=.*$')
        if line.search(body):body=line.sub(key+'='+value,body)
        else:body=key+'='+value+'\n'+body
    return (text[:found.start(2)]+body+text[found.end(2):]).encode('utf-8')


def install(game,dll,model,backup,existing_bundle=None):
    if backup.exists():raise ValueError('Use a new backup directory')
    if (game/'neural-student').exists():raise ValueError('An existing student package must be preserved/restored first')
    if digest(game/'dxgi.dll')!=BASE_DLL:
        if existing_bundle is None:raise ValueError('Use --existing-bundle to verify a different local compatibility build')
        _,files=inspect_bundle(existing_bundle)
        if digest(game/'dxgi.dll')!=files['dxgi.dll']:raise ValueError('Installed OptiScaler differs from the verified compatibility bundle')
    preserved={name:digest(game/name) for name in ('winmm.dll','nvngx_dlssnr.dll')}
    if preserved!={'winmm.dll':PATCHED_SHA256,'nvngx_dlssnr.dll':NR_SHA256}:raise ValueError('Expected working RTXMFG and native NR runtimes')
    manifest=json.loads((model/'model.json').read_text())
    if manifest.get('architecture')!='region-broad-v1' or (manifest.get('width'),manifest.get('height'))!=(1920,1080):
        raise ValueError('Unsupported student model')
    if digest(model/'weights.bin')!=manifest['weights_sha256'] or digest(model/'DirectML.dll')!=DML_DLL:
        raise ValueError('Student package checksum mismatch')
    payload={'dxgi.dll':dll.read_bytes(),'OptiScaler.ini':configure((game/'OptiScaler.ini').read_bytes())}
    payload.update({'neural-student/'+name:(model/name).read_bytes() for name in ('model.json','weights.bin','DirectML.dll')})
    if not payload['dxgi.dll'].startswith(b'MZ'):raise ValueError('Expected a built Windows DLL')
    original={name:safe(game,name).read_bytes() if safe(game,name).exists() else None for name in FILES}
    backup.mkdir(parents=True)
    record={'schema':1,'game':str(game),'files':{},'preserved':preserved,'complete':False}
    for name,data in original.items():
        if data is not None:
            path=backup/'original'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        record['files'][name]={'before':hashlib.sha256(data).hexdigest() if data is not None else None,
                              'after':hashlib.sha256(payload[name]).hexdigest()}
    (backup/'install.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        for name,data in payload.items():
            dest=safe(game,name);dest.parent.mkdir(parents=True,exist_ok=True);atomic_write(dest,data)
        if any(digest(safe(game,name))!=row['after'] for name,row in record['files'].items()):raise RuntimeError('Installed checksum mismatch')
        if any(digest(game/name)!=value for name,value in preserved.items()):raise RuntimeError('An unrelated runtime changed during installation')
        record['complete']=True
    except Exception:
        for name,data in original.items():
            if data is None:safe(game,name).unlink(missing_ok=True)
            else:atomic_write(safe(game,name),data)
        raise
    finally:(backup/'install.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def restore(backup):
    record=json.loads((backup/'install.json').read_text())
    if record.get('schema')!=1 or set(record['files'])!=FILES:raise ValueError('Unexpected backup manifest')
    game=find_game(record['game'])
    for name,row in record['files'].items():
        if digest(safe(game,name))!=row['after']:raise ValueError('Installed file changed; preserve it before restoring: '+name)
        if row['before'] and digest(backup/'original'/name)!=row['before']:raise ValueError('Backup checksum mismatch')
    for name,row in record['files'].items():
        if row['before']:atomic_write(safe(game,name),(backup/'original'/name).read_bytes())
        else:safe(game,name).unlink()
    folder=game/'neural-student'
    if folder.exists() and not any(folder.iterdir()):folder.rmdir()
    return {'restored':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',type=Path);p.add_argument('--dll',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--restore',action='store_true')
    p.add_argument('--existing-bundle',type=Path,help='Original BuildNeural bundle, needed for a locally rebuilt compatibility DLL.')
    a=p.parse_args();closed()
    if a.restore:result=restore(a.backup.resolve())
    else:
        if not all((a.game,a.dll,a.model)):p.error('Install requires --game, --dll and --model')
        result=install(find_game(a.game),a.dll.resolve(),a.model.resolve(),a.backup.resolve(),a.existing_bundle)
    print(json.dumps({'complete':result.get('complete',result.get('restored')),'backup':str(a.backup.resolve())}))


if __name__=='__main__':main()
