$ErrorActionPreference = 'Stop'
$dir = 'D:\internship block diagram'
$svgPath = Join-Path $dir 'hr-100-1.svg'
if (-not (Test-Path -LiteralPath $svgPath)) {
    Write-Output 'svg missing'
    exit 1
}
$raw = Get-Content -LiteralPath $svgPath -Raw
Write-Output ("bytes: " + (Get-Item -LiteralPath $svgPath).Length)
$m = [regex]::Match($raw, 'width="([\d.]+)"[^>]*height="([\d.]+)"')
if ($m.Success) { Write-Output ("dims: " + $m.Groups[1].Value + " x " + $m.Groups[2].Value) }
$checks = @(
    '100× AXIS Cameras',
    '80× AXIS P3275-LVE',
    '20× AXIS V5925',
    'Jetson AGX Thor',
    'Jetson AGX Orin 64GB',
    '×8 NODES',
    'YOLO11n',
    'TensorRT 10.3',
    'NvDCF',
    'Qwen3-VL-8B',
    'vLLM',
    'RTX 6000 Ada 48 GB',
    'EMQX',
    'Kafka',
    'PostgreSQL 16',
    'coverage_map.yaml',
    'drone_rules.yaml',
    'ONVIF',
    'Kalman',
    'handoff',
    'PTZ',
    'recovery'
)
foreach ($t in $checks) {
    if ($raw.Contains($t)) { Write-Output ("OK   " + $t) } else { Write-Output ("MISS " + $t) }
}
