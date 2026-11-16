<#
  统一入口：自动挑一个装了 ultralytics 的 Python 来跑脚本。

  为什么需要：这台机器上 `python` 默认指向 C:\Program Files\Python313（没装 ultralytics），
  而训练用的是 conda 环境 yolo（Python 3.11 + torch cu128）。忘记激活就会报
  ModuleNotFoundError: ultralytics。

  用法：
    .\run.ps1 01_vision/scripts/camera_demo.py
    .\run.ps1 01_vision/scripts/train.py --device 0 --epochs 200 --cache
    .\run.ps1 01_vision/scripts/check_dataset.py

  也可以直接用环境里的解释器（等价）：
    D:\miniconda\envs\yolo\python.exe 01_vision/scripts/camera_demo.py
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$candidates = @(
    "D:\miniconda\envs\yolo\python.exe"
)

$py = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $py) {
    Write-Host "找不到可用的 python，请先装 Python 或 conda 环境" -ForegroundColor Red
    exit 1
}

# 快速自检：没有 ultralytics 就别硬跑，直接告诉他该装哪儿
& $py -c "import ultralytics" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "这个解释器没有 ultralytics：$py" -ForegroundColor Yellow
    Write-Host "请在对应环境里 pip install ultralytics，或修改 run.ps1 里的路径" -ForegroundColor Yellow
    exit 1
}

Write-Host "python: $py" -ForegroundColor DarkGray
& $py @Args
exit $LASTEXITCODE
