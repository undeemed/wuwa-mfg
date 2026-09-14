param([ValidateSet('menu','install','verify','restore','doctor')][string]$Action = 'menu')
$ErrorActionPreference = 'Stop'
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    if ($Action -eq 'menu') {
        Write-Host "`nWuWa MFG 0.1.0 - experimental setup`n"
        Write-Host '1. Install (game must be closed)'
        Write-Host '2. Verify actual runtime frame counts'
        Write-Host '3. Restore the previous setup'
        Write-Host '4. Diagnose compatibility (read only)'
        Write-Host 'Q. Exit'
        $choice = Read-Host 'Choose'
        $Action = switch ($choice) { '1' {'install'} '2' {'verify'} '3' {'restore'} '4' {'doctor'} default { exit 0 } }
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if ($Action -in @('install','restore') -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        # Only mutating actions request the standard Windows UAC prompt.
        $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Action {1}' -f $PSCommandPath, $Action
        $child = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -Verb RunAs -Wait -PassThru
        exit $child.ExitCode
    }
    $runtime = Join-Path $PSScriptRoot '.runtime'
    $archive = Join-Path $runtime 'python-3.12.10-embed-amd64.zip'
    $expected = '4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3'
    if (-not (Test-Path -LiteralPath $runtime)) { New-Item -ItemType Directory -Path $runtime | Out-Null }
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host 'Downloading the official Python runtime into this setup folder (no system installation)...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'Python archive checksum mismatch. Remove .runtime and retry.' }
    # Refresh from the verified archive; do not trust a cached executable on its own.
    Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
    & (Join-Path $runtime 'python.exe') -I (Join-Path $PSScriptRoot 'setup.py') $Action
    $result = $LASTEXITCODE
    Read-Host 'Press Enter to close' | Out-Null
    exit $result
} catch {
    Write-Host "Stopped: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host 'Press Enter to close' | Out-Null
    exit 1
}
