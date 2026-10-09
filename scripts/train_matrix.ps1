<#
.SYNOPSIS
    Run the 14-run experiment matrix in order, resumable.

.DESCRIPTION
    Executes `anti-uav matrix run` for the selected models and combos, in the
    order the planner produces: single-dataset runs first, cumulative ones last.
    That order matters - if a run fails you immediately know which source caused
    it, instead of discovering it three cumulative runs later.

    Each run is skipped when its weights/best.pt already exists, so this is safe
    to re-run after an interruption or a crash.

    This script does not create an environment. It targets whichever
    interpreter resolves `anti-uav`; if that fails, pass -Python or set
    ANTI_UAV_PYTHON.

.PARAMETER Profile
    GPU tier: pascal, ampere, ada, blackwell, cpu. Drives the torch build check
    and the per-GPU batch/imgsz overrides. Default: auto-detected.

.PARAMETER Models
    Comma list of model families. Default: yolo11n,rtdetr_x2 (all 14 runs).

.PARAMETER Combos
    Comma list of combo slugs, or "all". Default: all.

.PARAMETER Python
    Interpreter to use. Defaults to $env:ANTI_UAV_PYTHON, then the resolved
    `anti-uav` entry point, then `python` on PATH.

.PARAMETER DryRun
    Print the plan and exit. Costs nothing and catches a wrong -Models or
    -Combos before it costs a GPU day.

.PARAMETER StopOnFailure
    Abort on the first failing run instead of continuing. Default is to continue
    and report at the end, so one bad combo does not cost you the other thirteen.

.EXAMPLE
    .\scripts\train_matrix.ps1 -DryRun
    Show all 14 runs and their resolved settings. Run this first.

.EXAMPLE
    .\scripts\train_matrix.ps1 -Profile blackwell
    Run everything on the RTX 5090 box.

.EXAMPLE
    .\scripts\train_matrix.ps1 -Profile ada -Models yolo11n -Combos dvb,all4
    Just the two runs that answer the two questions worth answering first:
    does bird suppression work (dvb), and is the combination worth deploying (all4).
#>
[CmdletBinding()]
param(
    [ValidateSet("pascal", "ampere", "ada", "blackwell", "cpu", "auto")]
    [string] $Profile = "auto",
    [string] $Models = "yolo11n,rtdetr_x2",
    [string] $Combos = "all",
    [string] $Python = $env:ANTI_UAV_PYTHON,
    [switch] $DryRun,
    [switch] $StopOnFailure
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

# ---------------------------------------------------------------------------
# resolve the CLI
# ---------------------------------------------------------------------------
function Test-HasCli {
    param([string] $Exe, [string[]] $Prefix)

    # `anti-uav --version` succeeds only when the package is importable in that
    # interpreter, which is the thing we actually care about. Probing beats
    # picking `python` off PATH and failing three steps later with a
    # ModuleNotFoundError that reads like a broken install.
    #
    # A failing probe is the normal case, so it must not be able to terminate the
    # script: under $ErrorActionPreference = 'Stop', a native command's stderr
    # promoted by 2>&1 becomes a terminating error, which would abort resolution
    # at the first interpreter without anti_uav instead of moving on to the next.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @($Prefix + @("--help")) *> $null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Resolve-CLI {
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python)) {
            throw "no interpreter at $Python"
        }
        $candidate = @{ Exe = $Python; Prefix = @("-m", "anti_uav.cli") }
        if (-not (Test-HasCli $candidate.Exe $candidate.Prefix)) {
            throw "anti_uav is not importable by $Python. Run scripts\setup_env.ps1 -EnvName <that env>."
        }
        return $candidate
    }

    $candidates = @()

    $script = Get-Command anti-uav -ErrorAction SilentlyContinue
    if ($script) { $candidates += @{ Exe = $script.Source; Prefix = @() } }

    # Conda envs first: this project is meant to live in one, and PATH usually
    # points at the system interpreter.
    $condaRoots = @(
        "$env:USERPROFILE\.conda\envs",
        "$env:USERPROFILE\anaconda3\envs",
        "$env:USERPROFILE\miniconda3\envs",
        "C:\ProgramData\miniconda3\envs"
    )
    foreach ($root in $condaRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        foreach ($envDir in (Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) {
            $exe = Join-Path $envDir.FullName "python.exe"
            if (Test-Path -LiteralPath $exe) {
                $candidates += @{ Exe = $exe; Prefix = @("-m", "anti_uav.cli") }
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { $candidates += @{ Exe = $python.Source; Prefix = @("-m", "anti_uav.cli") } }

    foreach ($candidate in $candidates) {
        if (Test-HasCli $candidate.Exe $candidate.Prefix) { return $candidate }
    }

    throw ("cannot find an interpreter with anti_uav installed. Tried {0} candidate(s)." -f $candidates.Count +
        " Run scripts\setup_env.ps1, or pass -Python with the right interpreter.")
}

$cli = Resolve-CLI

function Invoke-AntiUav {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)

    # Out-Host is load-bearing: a PowerShell function returns every uncaptured
    # pipeline value, so without it the caller's `if ($verifyExit -ne 0)` would
    # compare an array of log lines against 0 and always look like a failure.
    & $cli.Exe @($cli.Prefix + $Arguments) 2>&1 | Out-Host
    return $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# preflight: refuse to start a 14-run job on a machine that cannot run it
# ---------------------------------------------------------------------------
Write-Host "==> Verifying the environment before committing GPU hours" -ForegroundColor Cyan
$verifyExit = Invoke-AntiUav "verify-env"
if ($verifyExit -ne 0) {
    Write-Host ""
    Write-Host "verify-env found blocking problems (exit $verifyExit)." -ForegroundColor Red
    Write-Host "Fix them before training - a 14-run matrix on a machine with the wrong" -ForegroundColor Red
    Write-Host "torch build wastes the whole job." -ForegroundColor Red
    exit $verifyExit
}

if ($Profile -eq "auto") {
    # Let the project detect the tier from the live device rather than guessing
    # here; `matrix run` already resolves "auto" to the detected profile.
    $Profile = ""
}

# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
$matrixArgs = @("matrix", "run", "--models", $Models, "--combos", $Combos)
if ($Profile) { $matrixArgs += @("--profile", $Profile) }
if ($DryRun) { $matrixArgs += "--dry-run" }

Write-Host ""
Write-Host "==> $Models x $Combos$(if ($Profile) { " on the $Profile profile" })" -ForegroundColor Cyan
if ($DryRun) { Write-Host "    (dry run - nothing will be trained)" -ForegroundColor Yellow }

$started = Get-Date
$exit = Invoke-AntiUav @matrixArgs
$elapsed = (Get-Date) - $started

Write-Host ""

if ($DryRun) {
    Write-Host "Dry run complete. Nothing was trained." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The 14 runs and their resolved settings are also listed by:" -ForegroundColor Yellow
    Write-Host "  anti-uav matrix list" -ForegroundColor Yellow
    exit 0
}

if ($exit -eq 0) {
    Write-Host "Matrix finished in $([math]::Round($elapsed.TotalHours, 2)) h." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next:" -ForegroundColor Green
    Write-Host "  anti-uav eval --run artifacts/runs/yolo11n/all4 --cross"
    Write-Host "  anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav"
}
else {
    Write-Host "Matrix exited with code $exit after $([math]::Round($elapsed.TotalHours, 2)) h." -ForegroundColor Red
    Write-Host ""
    Write-Host "Completed runs are kept. Re-run the same command to resume; the" -ForegroundColor Yellow
    Write-Host "planner skips any run whose weights/best.pt already exists." -ForegroundColor Yellow
    if (-not $StopOnFailure) {
        Write-Host "" -ForegroundColor Yellow
        Write-Host "Runs that failed:" -ForegroundColor Yellow
        Get-ChildItem -LiteralPath (Join-Path $projectRoot "artifacts\runs") -Directory -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-Path -LiteralPath (Join-Path $_.FullName "weights\best.pt")) } |
            ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Yellow }
    }
}

exit $exit