param(
    [Parameter(Mandatory=$true)][string]$NrRuntime,
    [string]$WorkDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) '.build')
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
function RunGit { & git @args; if ($LASTEXITCODE -ne 0) { throw 'Git command failed.' } }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git for Windows is required for this source build.' }
$NrRuntime = (Resolve-Path -LiteralPath $NrRuntime).Path
if ((Get-FileHash -LiteralPath $NrRuntime).Hash -ne '6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927') { throw 'NR runtime hash mismatch. See docs/neural-rendering.md.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio 2022 C++ desktop build tools and Windows SDK first.' }
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ tools are unavailable.' }
& (Join-Path $vs 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (Test-Path -LiteralPath $WorkDirectory) { throw 'Choose a new WorkDirectory; existing source/builds are preserved.' }
New-Item -ItemType Directory -Path $WorkDirectory | Out-Null
$WorkDirectory = (Resolve-Path -LiteralPath $WorkDirectory).Path
$source = Join-Path $WorkDirectory 'compat-source'
RunGit -c core.autocrlf=false clone --depth 1 --branch v0.8.4 https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass.git $source
if ((& git -C $source rev-parse HEAD).Trim() -ne '8802b2b470db0462fa1ed03a125e793a7c06d735') { throw 'Upstream tag moved; stopped.' }
RunGit -C $source submodule update --init --recursive --depth 1 --jobs 4
$headers = Join-Path $WorkDirectory 'directx-headers'
RunGit clone --depth 1 --branch v1.619.5 https://github.com/microsoft/DirectX-Headers.git $headers
if ((& git -C $headers rev-parse HEAD).Trim() -ne 'ee479f0bd5f7b884f202bcf0c3f076cc050dd256') { throw 'DirectX-Headers tag moved; stopped.' }
$patch = Join-Path $repoRoot 'patches\optiscaler-wuwa-compat.patch'
RunGit -C $source apply --check $patch
RunGit -C $source apply $patch
& (Join-Path $source 'tests\run_nr_dispatch_slots.ps1') -LegacyControl
& (Join-Path $source 'tests\run_nr_dispatch_slots.ps1')
& (Join-Path $source 'tests\run_nr_gpu_lifetime.ps1')
& (Join-Path $source 'tests\run_nr_working_extent.ps1')
& (Join-Path $source 'tests\run_external_fg_lifetime.ps1')
$props = Join-Path $WorkDirectory 'local-build.props'
@'
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemDefinitionGroup><ClCompile>
    <AdditionalIncludeDirectories>$(SolutionDir)..\directx-headers\include\directx;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
  </ClCompile></ItemDefinitionGroup>
</Project>
'@ | Set-Content -LiteralPath $props -Encoding utf8
& (Join-Path $vs 'MSBuild\Current\Bin\MSBuild.exe') (Join-Path $source 'OptiScaler.sln') /m:4 /p:Configuration=Release /p:Platform=x64 /p:CL_MPCount=4 /p:PostBuildEventUseInBuild=false "/p:ForceImportBeforeCppTargets=$props" /v:minimal /nologo
if ($LASTEXITCODE -ne 0) { throw 'OptiScaler build failed.' }
$package = Join-Path $WorkDirectory 'OptiScaler-NR-v0.8.4.zip'
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/releases/download/v0.8.4/OptiScaler-NR-v0.8.4.zip' -OutFile $package
if ((Get-FileHash -LiteralPath $package).Hash -ne '8789912859882E66B3F3A1AA768DB947DA779DFD65225DF69EA919052E73A2E4') { throw 'Upstream package checksum mismatch.' }
$runtime = Join-Path $repoRoot '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$pythonZip = Join-Path $runtime 'python-3.12.10-embed-amd64.zip'
if (-not (Test-Path -LiteralPath $pythonZip)) { Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $pythonZip }
if ((Get-FileHash -LiteralPath $pythonZip).Hash -ne '4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3') { throw 'Python archive checksum mismatch.' }
Expand-Archive -LiteralPath $pythonZip -DestinationPath $runtime -Force
& (Join-Path $runtime 'python.exe') -I (Join-Path $repoRoot 'tools\make_neural_bundle.py') --source $source --package $package --nr-runtime $NrRuntime --output (Join-Path $WorkDirectory 'bundle')
if ($LASTEXITCODE -ne 0) { throw 'NR bundle validation failed.' }
Write-Host 'Build complete. Open Setup.cmd, choose 5, and select the bundle folder printed above.'
