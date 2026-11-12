<#
 一键板上验证：传模型 -> 版本/md5 校验 -> 单图推理 -> 三核 benchmark -> PASS/FAIL 汇总

用法（在仓库根目录）：
  powershell -ExecutionPolicy Bypass -File 01_vision\deploy\board_test.ps1 `
    -Rknn "E:\best_fp16_rk3588.rknn" -Image "01_vision\assets\predict_demo.jpg"

参数：
  -Board     ssh 目标，默认 cat@192.168.137.2
  -Key       ssh 私钥，默认 D:\deepseek harness work space\_deploy_key
  -Rknn      本地 .rknn 路径（必填）
  -Image     本地测试图（默认 01_vision\assets\predict_demo.jpg）
  -RemoteDir 板上目录，默认 ~/uav_test
  -Bench     benchmark 次数，默认 200
  -SkipMd5   跳过 md5 校验
#>
param(
    [string]$Board = "cat@192.168.137.2",
    [string]$Key = "D:\deepseek harness work space\_deploy_key",
    [Parameter(Mandatory = $true)][string]$Rknn,
    [string]$Image = "",
    [string]$RemoteDir = "~/uav_test",
    [int]$Bench = 200,
    [switch]$SkipMd5
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$knownHosts = Join-Path $env:TEMP "uav_board_known_hosts"
$results = New-Object System.Collections.ArrayList

function Add-Result($item, $ok, $detail) {
    [void]$results.Add([pscustomobject]@{ 项目 = $item; 结果 = $(if ($ok) { "PASS" } else { "FAIL" }); 说明 = $detail })
}

$sshBase = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=$knownHosts", "-o", "ConnectTimeout=8")

function Invoke-Remote([string]$cmd) {
    $out = & ssh.exe @sshBase $Board $cmd 2>&1 | Out-String
    return @{ Out = $out; Code = $LASTEXITCODE }
}

Write-Host "== [1/5] 连通性与免密 ==" -ForegroundColor Cyan
$r = Invoke-Remote "echo READY; uname -m"
if ($r.Code -ne 0 -or $r.Out -notmatch "READY") {
    Write-Host $r.Out
    Write-Host "连不上或没有免密。先做 deploy/README.md 的 8.0 两步（装公钥 + 验证 key-ok）。" -ForegroundColor Red
    Add-Result "免密登录" $false $Board
    $results | Format-Table -AutoSize
    exit 1
}
$arch = ($r.Out -split "`n" | Where-Object { $_ -match "aarch64|arm" } | Select-Object -First 1)
Add-Result "免密登录" $true "$Board  $($arch)"

Write-Host "== [2/5] 推送文件 ==" -ForegroundColor Cyan
if (-not (Test-Path $Rknn)) { Write-Host "找不到 $Rknn" -ForegroundColor Red; exit 1 }
$img = if ($Image) { $Image } else { Join-Path $repo "01_vision\assets\predict_demo.jpg" }
$pyf = Join-Path $repo "01_vision\deploy\rknn_inference.py"
foreach ($f in @($Rknn, $img, $pyf)) { if (-not (Test-Path $f)) { Write-Host "缺文件: $f" -ForegroundColor Red; exit 1 } }

& ssh.exe @sshBase $Board "mkdir -p $RemoteDir" | Out-Null
& scp.exe -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o "UserKnownHostsFile=$knownHosts" `
    $Rknn $pyf $img "${Board}:${RemoteDir}/" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "scp 失败" -ForegroundColor Red
    Add-Result "scp 推送" $false $RemoteDir
    $results | Format-Table -AutoSize
    exit 1
}
$rknnName = Split-Path -Leaf $Rknn
$imgName = Split-Path -Leaf $img
Add-Result "scp 推送" $true "$rknnName / rknn_inference.py / $imgName"

Write-Host "== [3/5] 环境与文件校验 ==" -ForegroundColor Cyan
$r = Invoke-Remote "cat /sys/kernel/debug/rknpu/version 2>/dev/null | head -1; strings /usr/lib/librknnrt.so 2>/dev/null | grep -E '^2\.[0-9]+\.[0-9]+' | head -1; python3 -c 'import rknnlite; print(\"rknnlite ok\")' 2>&1 | tail -1"
Write-Host $r.Out.Trim()
$verOk = ($r.Out -match "2\.[0-9]+\.[0-9]+") -and ($r.Out -match "rknnlite ok")
Add-Result "板端环境" $verOk $(if ($verOk) { "runtime/lite 可加载" } else { "见上方输出" })

if (-not $SkipMd5) {
    $local = (Get-FileHash -LiteralPath $Rknn -Algorithm MD5).Hash.ToLower()
    $r = Invoke-Remote "md5sum $RemoteDir/$rknnName | cut -d' ' -f1"
    $remote = ($r.Out -split "`n" | Where-Object { $_ -match "^[0-9a-f]{32}$" } | Select-Object -First 1)
    $md5Ok = ($remote -and $remote.Trim() -eq $local)
    Add-Result "模型 md5" $md5Ok $(if ($md5Ok) { $local } else { "本地 $local / 板端 $remote" })
}

Write-Host "== [4/5] 单图推理 ==" -ForegroundColor Cyan
$r = Invoke-Remote "cd $RemoteDir && LANG=C.UTF-8 PYTHONIOENCODING=utf-8 python3 rknn_inference.py --model $rknnName --source $imgName --json board.json 2>&1 | tail -12"
Write-Host $r.Out.Trim()
$ndet = ([regex]::Matches($r.Out, "\(\d+,\d+\)-\(\d+,\d+\)")).Count
Add-Result "单图推理" ($r.Code -eq 0) "检出 $ndet 个目标（应与 PC 端 onnx 一致）"

Write-Host "== [5/5] 三核 benchmark（$Bench 次）==" -ForegroundColor Cyan
$r = Invoke-Remote "cd $RemoteDir && python3 rknn_inference.py --model $rknnName --source $imgName --core 012 --benchmark $Bench 2>&1 | grep -E 'bench' | tail -1"
Write-Host $r.Out.Trim()
$ms = $null
if ($r.Out -match "([0-9]+\.?[0-9]*)\s*ms") { $ms = [double]$Matches[1] }
$speedOk = ($ms -ne $null -and $ms -le 40)
Add-Result "FP16 三核速度" $speedOk $(if ($ms) { ("{0:N1} ms/帧 ≈ {1:N0} fps（阈值 ≤40ms）" -f $ms, (1000 / $ms)) } else { "未解析到耗时" })

Write-Host "`n================ 汇总 ================" -ForegroundColor Cyan
$results | Format-Table -AutoSize
$fail = @($results | Where-Object { $_.结果 -eq "FAIL" }).Count
if ($fail -eq 0) {
    Write-Host "全部通过：模型可以在板上用了。" -ForegroundColor Green
    exit 0
} else {
    Write-Host "有 $fail 项未通过，按 deploy/README.md 8.5 的顺序排查。" -ForegroundColor Yellow
    exit 2
}
