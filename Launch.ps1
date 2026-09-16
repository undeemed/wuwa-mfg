param([ValidateSet('menu','install','uninstall','verify','restore','doctor','neural-install','neural-on','neural-off','neural-restore','neural-status','neural-scale')][string]$Action = 'menu')
$ErrorActionPreference = 'Stop'
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    if ($Action -eq 'menu') {
        Write-Host "`nWuWa Experience Toolkit 0.2.1`n"
        Write-Host '1. Install MFG unlock (game must be closed)'
        Write-Host '2. Verify actual runtime frame counts'
        Write-Host '3. Restore the previous MFG setup'
        Write-Host '4. Diagnose compatibility (read only)'
        Write-Host '5. Install experimental neural rendering from a local build'
        Write-Host '6. Enable neural rendering at next launch (game closed)'
        Write-Host '7. Disable neural rendering at next launch (game closed)'
        Write-Host '8. Restore/remove the neural add-on (game closed)'
        Write-Host '9. Neural rendering status (read only)'
        Write-Host '10. Set neural model resolution as a fraction of output (game closed)'
        Write-Host 'U. Uninstall all managed add-ons (game closed)'
        Write-Host 'Q. Exit'
        $choice = Read-Host 'Choose'
        $Action = switch ($choice) { '1' {'install'} '2' {'verify'} '3' {'restore'} '4' {'doctor'} '5' {'neural-install'} '6' {'neural-on'} '7' {'neural-off'} '8' {'neural-restore'} '9' {'neural-status'} '10' {'neural-scale'} 'U' {'uninstall'} default { exit 0 } }
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if ($Action -in @('install','uninstall','restore','neural-install','neural-on','neural-off','neural-restore','neural-scale') -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
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
