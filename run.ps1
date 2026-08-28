<#
  fence_lite —— 本地启动脚本（Windows）.

  双击运行，或 powershell -File run.ps1。服务起在 http://127.0.0.1:<端口>，
  端口取环境变量 FENCE_LITE_PORT，默认 5060。

  为什么先杀旧实例：webapp.py / job.py / steps 都是 import 期求值的模块级常量，
  旧进程还占着端口时改了后端代码也不会生效。启动前按端口 stop 一次是唯一
  可靠的做法（与 5051 的 stop_web.bat 同法）。
  解释器默认复用 fence_detector 的 venv（依赖完全一致），可用 FENCE_PYTHON 覆盖。
  GEMINI_API_KEY 由 core/config.py 从本目录的 .env 读，不在这里处理。
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [string]$Listen = '',
    [switch]$LoopbackOnly,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not $Port) {
    $Port = if ($env:FENCE_LITE_PORT) { [int]$env:FENCE_LITE_PORT } else { 5060 }
}
# 默认绑 0.0.0.0：浏览器常常不在这台机器上（远程桌面/另一台笔记本/Tailscale），
# 只绑 127.0.0.1 的话对方拿到的就是「无法访问此页面」。要退回只听本机用
# -LoopbackOnly。注意本服务**没有任何鉴权**，且 /api/project DELETE 会删项目 ——
# 只在可信网络里这么开。
if (-not $Listen) {
    $Listen = if ($LoopbackOnly) { '127.0.0.1' }
              elseif ($env:FENCE_LITE_HOST) { $env:FENCE_LITE_HOST }
              else { '0.0.0.0' }
}
$env:FENCE_LITE_PORT = "$Port"
$env:FENCE_LITE_HOST = $Listen
$env:PYTHONUTF8 = '1'          # 被管道/编辑器捕获时 Python 会退回 gbk，中文全乱码

& (Join-Path $PSScriptRoot 'stop.ps1') -Port $Port -NoPause

$python = if ($env:FENCE_PYTHON) { $env:FENCE_PYTHON } `
          else { 'C:\Users\Administrator\fence_detector\venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "No Python interpreter at $python"
    Write-Host 'Set FENCE_PYTHON to a Python 3.10+ interpreter with the requirements installed.'
    if (-not $NoPause -and $Host.Name -eq 'ConsoleHost') {
        Write-Host 'Press Enter to close…' -NoNewline
        try { Read-Host | Out-Null } catch { }
    }
    exit 1
}

Write-Host "Interpreter: $python"
Write-Host "Starting fence_lite (bind $Listen`:$Port, Ctrl+C to stop)"
Write-Host "  local : http://127.0.0.1:$Port/"
if ($Listen -ne '127.0.0.1') {
    # 局域网/Tailscale 地址列出来，省得对方猜自己该访问哪个 IP。
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -ne '127.0.0.1' -and
                       $_.IPAddress -notlike '169.254.*' } |
        ForEach-Object { Write-Host "  remote: http://$($_.IPAddress):$Port/  ($($_.InterfaceAlias))" }
}
& $python -B -u (Join-Path $PSScriptRoot 'webapp.py')
$code = $LASTEXITCODE

if (-not $NoPause -and $Host.Name -eq 'ConsoleHost') {
    Write-Host ''
    Write-Host "webapp.py exited with code $code. Press Enter to close…" -NoNewline
    try { Read-Host | Out-Null } catch { }
}
exit $code
