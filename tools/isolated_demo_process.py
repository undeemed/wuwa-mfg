# SPDX-License-Identifier: MIT
"""Run the pinned sample on a private Windows desktop that is never activated.

This also contains modal error dialogs, unlike hiding only the main window.
The helper supports only the exact previously hidden NVIDIA sample executable.
"""
import ctypes as c
from ctypes import wintypes as w
import hashlib
from pathlib import Path
import subprocess
import uuid


class Startup(c.Structure):
    _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR),
                ('title', w.LPWSTR), ('x', w.DWORD), ('y', w.DWORD),
                ('width', w.DWORD), ('height', w.DWORD), ('chars_x', w.DWORD),
                ('chars_y', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                ('show', w.WORD), ('reserved_size', w.WORD), ('reserved_bytes', c.c_void_p),
                ('stdin', w.HANDLE), ('stdout', w.HANDLE), ('stderr', w.HANDLE)]


class ProcessInfo(c.Structure):
    _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]


class IsolatedDemoProcess:
    def __init__(self, args, cwd):
        exe = Path(args[0]).resolve()
        expected = 'f262742631d02da935649322f220f0490b114287c98a1d75d60e04557f98a9c6'
        if exe.name != 'ngx_dlss_demo.exe' or hashlib.sha256(exe.read_bytes()).hexdigest() != expected:
            raise ValueError('Only the exact hidden sample is supported.')
        self.user = c.WinDLL('user32', use_last_error=True)
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self.callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        signatures = [
            (self.user, 'CreateDesktopW', [w.LPCWSTR, c.c_void_p, c.c_void_p, w.DWORD, w.DWORD, c.c_void_p], w.HANDLE),
            (self.user, 'CloseDesktop', [w.HANDLE], w.BOOL),
            (self.user, 'OpenInputDesktop', [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            (self.user, 'GetUserObjectInformationW', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)], w.BOOL),
            (self.user, 'EnumDesktopWindows', [w.HANDLE, self.callback_type, w.LPARAM], w.BOOL),
            (self.user, 'EnumChildWindows', [w.HWND, self.callback_type, w.LPARAM], w.BOOL),
            (self.user, 'GetWindowThreadProcessId', [w.HWND, c.POINTER(w.DWORD)], w.DWORD),
            (self.user, 'GetWindowTextW', [w.HWND, w.LPWSTR, c.c_int], c.c_int),
            (self.user, 'IsWindowVisible', [w.HWND], w.BOOL),
            (self.kernel, 'CreateProcessW', [w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(Startup), c.POINTER(ProcessInfo)], w.BOOL),
            (self.kernel, 'GetExitCodeProcess', [w.HANDLE, c.POINTER(w.DWORD)], w.BOOL),
            (self.kernel, 'WaitForSingleObject', [w.HANDLE, w.DWORD], w.DWORD),
            (self.kernel, 'TerminateProcess', [w.HANDLE, w.UINT], w.BOOL),
            (self.kernel, 'CloseHandle', [w.HANDLE], w.BOOL),
        ]
        for library, name, argtypes, restype in signatures:
            function = getattr(library, name)
            function.argtypes, function.restype = argtypes, restype
        self.desktop_name = 'NeuralDemo-' + uuid.uuid4().hex
        self.input_before = self.input_desktop_name()
        # No DESKTOP_SWITCHDESKTOP access and no call to SwitchDesktop.
        self.desktop = self.user.CreateDesktopW(self.desktop_name, None, None, 0, 0xFF, None)
        if not self.desktop:
            raise c.WinError(c.get_last_error())
        self.process = None
        startup = Startup()
        startup.cb = c.sizeof(startup)
        startup.desktop = self.desktop_name
        startup.flags = 0x81  # STARTF_USESHOWWINDOW | STARTF_FORCEOFFFEEDBACK
        startup.show = 0
        info = ProcessInfo()
        command = c.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in args]))
        if not self.kernel.CreateProcessW(str(exe), command, None, None, False,
                subprocess.CREATE_NO_WINDOW, None, str(cwd), c.byref(startup), c.byref(info)):
            error = c.get_last_error()
            self.user.CloseDesktop(self.desktop)
            self.desktop = None
            raise c.WinError(error)
        self.process, self.pid = info.process, info.pid
        self.kernel.CloseHandle(info.thread)

    def input_desktop_name(self):
        desktop = self.user.OpenInputDesktop(0, False, 1)
        if not desktop:
            raise c.WinError(c.get_last_error())
        try:
            name, needed = c.create_unicode_buffer(256), w.DWORD()
            if not self.user.GetUserObjectInformationW(desktop, 2, name, c.sizeof(name), c.byref(needed)):
                raise c.WinError(c.get_last_error())
            return name.value
        finally:
            self.user.CloseDesktop(desktop)

    def poll(self):
        code = w.DWORD()
        if not self.kernel.GetExitCodeProcess(self.process, c.byref(code)):
            raise c.WinError(c.get_last_error())
        return None if code.value == 259 else code.value

    def terminate(self):
        if self.poll() is None and not self.kernel.TerminateProcess(self.process, 1):
            raise c.WinError(c.get_last_error())

    def wait(self, timeout):
        result = self.kernel.WaitForSingleObject(self.process, int(timeout * 1000))
        if result == 258:
            raise subprocess.TimeoutExpired('isolated NVIDIA sample', timeout)
        if result != 0:
            raise c.WinError(c.get_last_error())
        return self.poll()

    def snapshot(self):
        windows = []

        def title(hwnd):
            text = c.create_unicode_buffer(2048)
            self.user.GetWindowTextW(hwnd, text, len(text))
            return text.value

        @self.callback_type
        def visit(hwnd, _):
            pid = w.DWORD()
            self.user.GetWindowThreadProcessId(hwnd, c.byref(pid))
            if pid.value == self.pid:
                item = {'title': title(hwnd), 'visible_on_private_desktop': bool(self.user.IsWindowVisible(hwnd))}
                children = []

                @self.callback_type
                def child(window, _):
                    value = title(window)
                    if value:
                        children.append(value)
                    return True

                self.user.EnumChildWindows(hwnd, child, 0)
                item['child_text'] = children
                windows.append(item)
            return True

        if not self.user.EnumDesktopWindows(self.desktop, visit, 0):
            raise c.WinError(c.get_last_error())
        current = self.input_desktop_name()
        if current == self.desktop_name:
            raise RuntimeError('Research desktop unexpectedly became active.')
        return {'separate_from_input_desktop': True, 'input_desktop_unchanged': current == self.input_before,
                'windows': windows}

    def close(self):
        if self.process:
            if self.poll() is None:
                self.terminate()
                self.wait(15)
            self.kernel.CloseHandle(self.process)
            self.process = None
        if self.desktop:
            if not self.user.CloseDesktop(self.desktop):
                raise c.WinError(c.get_last_error())
            self.desktop = None
