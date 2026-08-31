<#
  fence_lite —— 按端口停服务（暴力停，不等「正在取消」）.

  双击运行，或 powershell -File stop.ps1 [-Port 5060] [-NoPause]。
  端口默认取环境变量 FENCE_LITE_PORT，没设就是 5060。
  说明：进程一杀，正在跑的作业立刻中断；下次启动时 job.resume_interrupted()
  只会把它标成 interrupted，绝不自动重跑（避免重启偷偷烧钱）。
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'

if (-not $Port) {
    $Port = if ($env:FENCE_LITE_PORT) { [int]$env:FENCE_LITE_PORT } else { 5060 }
}

function Get-ListenerPids([int]$p) {
    try {
        return @(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction Stop |
                 Select-Object -ExpandProperty OwningProcess -Unique)
    } catch {
        # 老系统 / 没有 NetTCPIP 模块时退回 netstat（与 5051 的 stop_web.bat 同法）
        return @(netstat -ano |
                 Select-String -Pattern ":$p\s" |
                 Select-String -Pattern 'LISTENING' |
                 ForEach-Object { ($_.ToString().Trim() -split '\s+')[-1] } |
                 Sort-Object -Unique)
    }
}

$killed = 0
foreach ($processId in Get-ListenerPids $Port) {
    $id = 0
    if (-not [int]::TryParse("$processId", [ref]$id)) { continue }
    if ($id -le 4) { continue }        # 0/4 是系统进程，永远不动
    try {
        # /T is essential: arrow/line-type workers are child process trees.
        # Killing only the listening Python PID can leave Node/Python sidecars
        # alive with an old PDF and an inherited pipe.
        & taskkill.exe /PID $id /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "taskkill exited with code $LASTEXITCODE"
        }
        Write-Host "Stopped PID $id and its process tree (port $Port)."
        $killed++
    } catch {
        Write-Host "Could not stop PID ${id}: $($_.Exception.Message)"
    }
}

if ($killed -eq 0) { Write-Host "Nothing was listening on port $Port." }

if (-not $NoPause -and $Host.Name -eq 'ConsoleHost') {
    Write-Host ''
    Write-Host 'Press Enter to close…' -NoNewline
    try { Read-Host | Out-Null } catch { }
}
