param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Microsoft\WindowsApps"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

$launcher = Join-Path $InstallDir "novel.cmd"
$startScript = Join-Path $Root "scripts\start.ps1"

@"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "$startScript" %*
"@ | Set-Content -Path $launcher -Encoding ASCII

Write-Host "已安装 Novel-factory 启动命令：$launcher"

$pathParts = ($env:PATH -split ";") | ForEach-Object { $_.TrimEnd("\") }
$normalizedInstallDir = $InstallDir.TrimEnd("\")
if ($pathParts -contains $normalizedInstallDir) {
    Write-Host "现在可以直接输入：novel"
} else {
    Write-Host ""
    Write-Warning "$InstallDir 还不在 PATH 中。"
    Write-Host "你可以把它加入用户 PATH，或者用完整路径运行："
    Write-Host $launcher
}
