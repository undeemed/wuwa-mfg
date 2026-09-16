param(
    [Parameter(Mandatory)][string]$CompatibilitySource,
    [Parameter(Mandatory)][string]$StudentModelDirectory,
    [Parameter(Mandatory)][string]$WorkDirectory,
    [string]$Python = 'python'
)
$ErrorActionPreference='Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source=(Resolve-Path -LiteralPath $CompatibilitySource).Path
$model=(Resolve-Path -LiteralPath $StudentModelDirectory).Path
if ((& git -C $source rev-parse HEAD).Trim() -ne '8802b2b470db0462fa1ed03a125e793a7c06d735') { throw 'Unexpected OptiScaler base commit.' }
if (Test-Path -LiteralPath $WorkDirectory) { throw 'Choose a new work directory; prior artifacts are preserved.' }
$manifest=Get-Content -LiteralPath (Join-Path $model 'model.json') -Raw | ConvertFrom-Json
if ($manifest.architecture -ne 'region-broad-v1' -or $manifest.width -ne 1920 -or $manifest.height -ne 1080) { throw 'Unsupported private student export.' }
if ((Get-FileHash -LiteralPath (Join-Path $model 'weights.bin')).Hash -ne $manifest.weights_sha256) { throw 'Model checksum mismatch.' }
$patch=Join-Path $repoRoot 'patches\optiscaler-neural-student.patch'
if (Test-Path -LiteralPath (Join-Path $source 'OptiScaler\dlssnr\DlssNr_Student.cpp')) {
    & git -C $source apply --reverse --check $patch
    if ($LASTEXITCODE -ne 0) { throw 'Existing student source differs from this patch.' }
} else {
    & git -C $source apply --reverse --check (Join-Path $repoRoot 'patches\optiscaler-wuwa-compat.patch')
    if ($LASTEXITCODE -ne 0) { throw 'Prepare the compatibility source with BuildNeural.ps1 first.' }
    & git -C $source apply --check $patch
    if ($LASTEXITCODE -ne 0) { throw 'Student patch does not apply cleanly.' }
    & git -C $source apply $patch
    if ($LASTEXITCODE -ne 0) { throw 'Student patch failed.' }
}
New-Item -ItemType Directory -Path $WorkDirectory | Out-Null
$work=(Resolve-Path -LiteralPath $WorkDirectory).Path
$deps=Join-Path $work 'directml'
& $Python (Join-Path $repoRoot 'tools\prepare_directml.py') --output $deps
if ($LASTEXITCODE -ne 0) { throw 'DirectML dependency preparation failed.' }
$native=Join-Path $source 'OptiScaler\dlssnr\student'
New-Item -ItemType Directory -Path $native -Force | Out-Null
foreach($name in @('StudentRuntime.cpp','StudentRuntime.h','StudentGraph.h')) {
    Copy-Item -LiteralPath (Join-Path $repoRoot "research\neural\backend\$name") -Destination (Join-Path $native $name) -Force
}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ desktop tools are required.' }
& (Join-Path $vs 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$props=Join-Path $work 'student-build.props'
$escapedDeps=[Security.SecurityElement]::Escape($deps)
@"
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemDefinitionGroup><ClCompile>
    <AdditionalIncludeDirectories>$escapedDeps;`$(SolutionDir)..\directx-headers\include\directx;`$(SolutionDir)external;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
  </ClCompile></ItemDefinitionGroup>
</Project>
"@ | Set-Content -LiteralPath $props -Encoding utf8
& (Join-Path $vs 'MSBuild\Current\Bin\MSBuild.exe') (Join-Path $source 'OptiScaler.sln') /m:4 /p:Configuration=Release /p:Platform=x64 /p:CL_MPCount=4 /p:PostBuildEventUseInBuild=false "/p:ForceImportBeforeCppTargets=$props" /v:minimal /nologo
if ($LASTEXITCODE -ne 0) { throw 'Student integration build failed.' }
$bundle=Join-Path $work 'bundle'
$package=Join-Path $bundle 'neural-student'
New-Item -ItemType Directory -Path $package -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'x64\Release\OptiScaler.dll') -Destination (Join-Path $bundle 'OptiScaler.dll')
foreach($name in @('model.json','weights.bin')) { Copy-Item -LiteralPath (Join-Path $model $name) -Destination (Join-Path $package $name) }
foreach($name in @('DirectML.dll','LICENSE.txt')) { Copy-Item -LiteralPath (Join-Path $deps $name) -Destination (Join-Path $package $name) }
Write-Host "Local student bundle: $bundle"
Write-Host 'Use tools/install_student_neural.py to install over the tested compatibility build with a backup. Keep the learned model package private.'
