# SPDX-License-Identifier: Apache-2.0
"""Bounded live engine/reload test in the pinned, inactive-desktop DLSS sample.

Restores the original demo DLL/INI in finally. Never opens or edits a game.
Logs/model pixels are private; publish only a sanitized test summary.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from isolated_demo_process import IsolatedDemoProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('demo','dll','model','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); demo=a.demo.resolve(); out=a.output.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Use private output.')
    running=subprocess.run(['powershell','-NoProfile','-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if running.stdout.strip()!='0':raise RuntimeError('Close the demo and game before testing.')
    destination=demo/'neural-student'; request=demo/'neural-engine.request'
    if destination.exists() or request.exists():raise RuntimeError('An existing student package/request needs preservation.')
    out.mkdir(parents=True,exist_ok=False)
    original={name:(demo/name).read_bytes() for name in ('dxgi.dll','OptiScaler.ini')}
    for name,data in original.items():(out/('original-'+name)).write_bytes(data)
    config=configparser.ConfigParser(interpolation=None);config.optionxform=str
    config.read_string(original['OptiScaler.ini'].decode('utf-8-sig'))
    nr=config['DlssNr']
    for key,value in {'Engine':'0','Enabled':'true','WorkingScale':'1.0','WorkingScaleRelativeToOutput':'true',
                       'Passes':'1','Preset':'0','Style':'1','Intensity':'1.0','LocalTone':'1.0',
                       'LocalStructure':'1.0','SkinStructure':'-1.0','AutoMask':'true','AutoCapture':'false'}.items():nr[key]=value
    config['Framerate']['FramerateLimit']='120'
    assert config['ProcessFilter']['TargetProcessName']=='ngx_dlss_demo.exe'
    proc=None; observations=[]; transitions=[]; result={'complete':False}
    started=time.monotonic(); log=demo/'OptiScaler.log'
    def text():return log.read_text(errors='replace') if log.exists() else ''
    def observe():
        if proc.poll() is not None:raise RuntimeError('The hidden sample exited during the test.')
        try:view=proc.snapshot()
        except OSError as error:
            # EnumDesktopWindows can report false/ERROR_SUCCESS before the first window exists.
            if error.winerror==0 and time.monotonic()-started<5:return
            raise
        observations.append(view)
        if not(view['separate_from_input_desktop'] and view['input_desktop_unchanged']):raise RuntimeError('Private desktop containment failed.')
        if any(w['visible_on_private_desktop'] for w in view['windows']):raise RuntimeError('Unexpected visible window on private desktop.')
    def wait_for(needle,offset=0,timeout=25):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            observe()
            if needle in text()[offset:]:return
            time.sleep(.5)
        raise TimeoutError('Missing runtime confirmation: '+needle)
    def select(engine,needle=None):
        offset=len(text()); temporary=demo/'neural-engine.request.tmp'
        temporary.write_text(engine+'\n');temporary.replace(request)
        wait_for(needle or ('active engine: '+engine),offset)
        transitions.append({'engine':engine,'confirmed':True,'at_s':round(time.monotonic()-started,3)})
        print(json.dumps(transitions[-1]),flush=True)
    try:
        if log.exists():shutil.copyfile(log,out/'previous-OptiScaler.log');log.unlink()
        destination.mkdir()
        for name in ('model.json','weights.bin','DirectML.dll'):shutil.copyfile(a.model/name,destination/name)
        shutil.copyfile(a.dll,demo/'dxgi.dll')
        with (demo/'OptiScaler.ini').open('w',encoding='utf-8') as f:config.write(f,space_around_delimiters=False)
        proc=IsolatedDemoProcess([str(demo/'ngx_dlss_demo.exe'),'-d3d12','-width','1920','-height','1080'],cwd=demo)
        wait_for('engine status: NVIDIA active.')
        select('student'); time.sleep(12);observe()
        for _ in range(6):
            select('nvidia');time.sleep(.5)
            select('student');time.sleep(.5)
        select('reload','active engine: student');time.sleep(2)
        # A corrupt/incomplete export must fall back before recording any student work.
        weights=destination/'weights.bin'; saved=destination/'weights.saved'
        weights.rename(saved)
        select('reload','NVIDIA fallback: Invalid student weight file')
        saved.rename(weights)
        select('reload','active engine: student');time.sleep(4);observe()
        result.update(complete=True,confirmed_switches=len(transitions),
                      reload_succeeded=True,missing_weights_fallback_and_recovery=True,
                      isolated_desktop_unchanged=True,unexpected_visible_windows=False,
                      checkpoint_sha256=json.loads((destination/'model.json').read_text())['checkpoint_sha256'],
                      integration_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest())
    except Exception as e:
        result['error']=str(e);raise
    finally:
        result['transitions']=transitions;result['desktop_observations']=observations
        result['elapsed_s']=round(time.monotonic()-started,3)
        if log.exists():shutil.copyfile(log,out/'OptiScaler.log')
        if proc is not None:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
            proc.close()
        for name,data in original.items():(demo/name).write_bytes(data)
        request.unlink(missing_ok=True);(demo/'neural-engine.request.tmp').unlink(missing_ok=True)
        if destination.exists():destination.rename(out/'tested-private-package')
        result['restored']={name:(demo/name).read_bytes()==data for name,data in original.items()}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('transitions','desktop_observations')}),flush=True)


if __name__=='__main__':main()
