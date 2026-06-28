$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "尚未安装项目依赖，正在运行 setup.ps1..."
    & (Join-Path $Root "scripts\setup.ps1")
}

$env:PYTHONUTF8 = "1"

$resolvedHost = (& $venvPython -c "from backend.config import get_settings; print(get_settings().app_host)").Trim()
$resolvedPort = [int](& $venvPython -c "from backend.config import get_settings; print(get_settings().app_port)")
$openBrowser = $env:NOVEL_FACTORY_OPEN_BROWSER
if ([string]::IsNullOrWhiteSpace($openBrowser)) {
    $openBrowser = "true"
}

$portState = & $venvPython (Join-Path $Root "scripts\check_app_port.py") $resolvedHost $resolvedPort
$portCode = $LASTEXITCODE

if ($portCode -eq 10) {
    Write-Host "Novel-factory 已经在运行：http://$resolvedHost`:$resolvedPort"
    Write-Host "无需重复启动，直接在浏览器中打开上面的地址即可。"
    if ($openBrowser -notin @("false", "0")) {
        & $venvPython (Join-Path $Root "scripts\open_browser.py") $resolvedHost $resolvedPort | Out-Null
    }
    exit 0
}

if ($portCode -eq 11) {
    Write-Error "无法启动：$resolvedHost`:$resolvedPort 已被其他程序占用。"
    Write-Host "可以查看占用程序：netstat -ano | findstr :$resolvedPort"
    Write-Host "或临时换一个端口：`$env:APP_PORT=$($resolvedPort + 1); .\scripts\start.ps1"
    exit 1
}

if ($portCode -ne 0) {
    Write-Error "检查应用端口失败：$portState"
    exit $portCode
}

if ($openBrowser -notin @("false", "0")) {
    Start-Process -FilePath $venvPython -ArgumentList @((Join-Path $Root "scripts\open_browser.py"), $resolvedHost, $resolvedPort) -WindowStyle Hidden
}

& $venvPython -m uvicorn backend.app:app --host $resolvedHost --port $resolvedPort
exit $LASTEXITCODE
