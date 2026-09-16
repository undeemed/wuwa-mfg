param([Parameter(Mandatory)][string]$Dependencies,[Parameter(Mandatory)][string]$JsonInclude,[Parameter(Mandatory)][string]$Output)
$ErrorActionPreference='Stop'
. 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$Dependencies=(Resolve-Path -LiteralPath $Dependencies).Path
$JsonInclude=(Resolve-Path -LiteralPath $JsonInclude).Path
$Output=(Resolve-Path -LiteralPath $Output).Path
Push-Location $Output
try {
    & cl.exe /nologo /EHsc /std:c++17 /O2 /MD /DNOMINMAX /DDML_TARGET_VERSION=0x6400 "/I$Dependencies" "/I$JsonInclude" "/I$PSScriptRoot" "$PSScriptRoot\StudentProbe.cpp" "$PSScriptRoot\StudentRuntime.cpp" /Fe:StudentProbe.exe /link "/LIBPATH:$Dependencies" DirectML.lib d3d12.lib dxgi.lib
    if ($LASTEXITCODE -ne 0) { throw 'Native student build failed.' }
    Copy-Item -LiteralPath "$Dependencies\DirectML.dll" -Destination "$Output\DirectML.dll" -Force
} finally { Pop-Location }
