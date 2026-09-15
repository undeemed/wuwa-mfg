# Isolated WMI lock investigation control

This original, dependency-free Windows program checks the lock-pairing mechanism
described in [the WMI hang investigation](../../docs/neural-crashes.md#provider-dump-malformed-lock-in-windows-health-accounting).
It operates on one lock in its own process. It does not query, open, modify or
restart WMI, WuWa, services or other processes.

From an x64 Visual Studio Developer Command Prompt, in this directory:

```bat
cl /nologo /EHsc /W4 srw_pairing_control.cpp /Fe:srw_pairing_control.exe
srw_pairing_control.exe matched
srw_pairing_control.exe mismatch
```

`matched` acquires and releases exclusively. `mismatch` deliberately uses the
wrong release function, as a negative control. Both use the nonblocking
`TryAcquireSRWLockExclusive` for the subsequent attempt; neither waits for the
malformed lock. The observed results are recorded in
[sanitized evidence](../../evidence/neural-wmi-hang.json).

The malformed value is an implementation-specific observation on the tested
Windows build, not a guaranteed API result on other versions. This control is
neither a complete reproduction of the WMI failure nor a fix. It contains no
Windows binary, disassembly, dump or patch. No administrator privileges are
needed.
