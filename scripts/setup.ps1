$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Find-Python {
    $candidates = @(
        @{ Command = "py"; Args = @("-3", "-c", "import sys; print(sys.executable)") },
        @{ Command = "python"; Args = @("-c", "import sys; print(sys.executable)") },
        @{ Command = "python3"; Args = @("-c", "import sys; print(sys.executable)") }
    )

    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }
        try {
            $output = & $candidate.Command @($candidate.Args) 2>$null
            if ($LASTEXITCODE -eq 0 -and $output) {
                return ($output | Select-Object -Last 1).Trim()
            }
        } catch {
            continue
        }
    }

    throw "未找到 Python 3。请先安装 Python 3.11+，并勾选 Add python.exe to PATH。"
}

$llamaServer = Get-Command "llama-server" -ErrorAction SilentlyContinue
if (-not $llamaServer) {
    $llamaServer = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
}
if (-not $llamaServer) {
    Write-Warning "未在 PATH 中找到 llama-server。请先安装 llama.cpp 的 Windows 版本，或在 .env 中设置 LLAMA_SERVER_BIN=C:\path\to\llama-server.exe。"
}

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $python = Find-Python
    Write-Host "正在创建 Python 虚拟环境：.venv"
    & $python -m venv ".venv"
}

Write-Host "正在安装 Python 依赖..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r "requirements-dev.txt"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已从 .env.example 创建 .env"
}

Write-Host ""
Write-Host "准备完成。运行 .\scripts\start.ps1 启动 Novel-factory。"
