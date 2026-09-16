# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('student_install',ROOT/'tools/install_student_neural.py')
install=importlib.util.module_from_spec(spec);spec.loader.exec_module(install)


class StudentInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.game=self.root/'Client/Binaries/Win64';self.game.mkdir(parents=True)
        (self.game/'Client-Win64-Shipping.exe').write_bytes(b'MZfixture')
        self.original={'dxgi.dll':b'MZold','winmm.dll':b'MZmfg','nvngx_dlssnr.dll':b'MZnr',
            'OptiScaler.ini':b'[Menu]\nShortcutKey=0x76\n[DlssNr]\nEnabled=true\nStyle=1\nWorkingScale=0.25\nToggleKey=0x77\n'}
        for name,data in self.original.items():(self.game/name).write_bytes(data)
        self.model=self.root/'model';self.model.mkdir()
        (self.model/'weights.bin').write_bytes(b'fixture-weights');(self.model/'DirectML.dll').write_bytes(b'MZDML')
        (self.model/'model.json').write_text(json.dumps({'architecture':'region-broad-v1','width':1920,'height':1080,
            'weights_sha256':install.digest(self.model/'weights.bin')}))
        self.dll=self.root/'new.dll';self.dll.write_bytes(b'MZstudent')
        for name,value in {'BASE_DLL':install.digest(self.game/'dxgi.dll'),'PATCHED_SHA256':install.digest(self.game/'winmm.dll'),
            'NR_SHA256':install.digest(self.game/'nvngx_dlssnr.dll'),'DML_DLL':install.digest(self.model/'DirectML.dll')}.items():
            change=patch.object(install,name,value);change.start();self.addCleanup(change.stop)
        self.backup=self.root/'backup'

    def assertOriginal(self):
        for name,data in self.original.items():self.assertEqual((self.game/name).read_bytes(),data)

    def test_install_and_exact_restore_preserve_keys_and_vendor_files(self):
        result=install.install(self.game,self.dll,self.model,self.backup);self.assertTrue(result['complete'])
        ini=(self.game/'OptiScaler.ini').read_text()
        for item in ('Engine=1','WorkingScale=0.5','ShortcutKey=0x76','ToggleKey=0x77'):self.assertIn(item,ini)
        for name in ('winmm.dll','nvngx_dlssnr.dll'):self.assertEqual((self.game/name).read_bytes(),self.original[name])
        install.restore(self.backup);self.assertOriginal();self.assertFalse((self.game/'neural-student').exists())

    def test_corrupt_export_is_rejected_before_game_writes(self):
        (self.model/'weights.bin').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):install.install(self.game,self.dll,self.model,self.backup)
        self.assertOriginal();self.assertFalse(self.backup.exists())

    def test_restore_preserves_later_user_edits(self):
        install.install(self.game,self.dll,self.model,self.backup)
        (self.game/'OptiScaler.ini').write_text('[DlssNr]\nEngine=0\n')
        with self.assertRaises(ValueError):install.restore(self.backup)
        self.assertEqual((self.game/'OptiScaler.ini').read_text(),'[DlssNr]\nEngine=0\n')

    def test_unsupported_controls_are_not_silently_overridden(self):
        (self.game/'OptiScaler.ini').write_text('[DlssNr]\nStyle=2\n')
        with self.assertRaises(ValueError):install.install(self.game,self.dll,self.model,self.backup)
        self.assertEqual((self.game/'dxgi.dll').read_bytes(),self.original['dxgi.dll'])
        self.assertFalse(self.backup.exists())

if __name__=='__main__':unittest.main()
