param([string]$VcVars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
node scripts/prepare-native.mjs
if ($LASTEXITCODE -ne 0) { throw 'Fixture generation failed' }
if (!(Test-Path -LiteralPath $VcVars)) { throw 'Pass -VcVars pointing to vcvars64.bat' }
$taskBuildCommand = 'call "' + $VcVars + '" && nvcc -O2 --fmad=false -std=c++17 -arch=native build/native/runner.cu -o build/native/runner.exe'
& cmd.exe /d /c $taskBuildCommand
if ($LASTEXITCODE -ne 0) { throw 'NVCC compilation failed' }
& ./build/native/runner.exe
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA execution failed' }
