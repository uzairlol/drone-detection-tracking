<#
.SYNOPSIS
    Prepare an existing Python environment to run anti-uav.

.DESCRIPTION
    Installs this project in editable mode plus the optional extras, then runs
    `anti-uav verify-env` to report whether the machine can actually do what you
    are asking of it.

    It deliberately does NOT create a virtual environment. This project is run
    on shared workstations and on two supervisor GPU boxes, and a second env per
    machine is where dependency drift comes from. Pass -EnvName to choose which
    existing conda env to install into.

    Torch is the one package it will not upgrade on its own: see -Profile.

.PARAMETER EnvName
    Existing conda environment to install into. Defaults to $env:CONDA_DEFAULT_ENV
    and falls back to "ml".

.PARAMETER Profile
    GPU tier whose torch build is required. One of:
      pascal    sm_61  - GTX 1070. Needs cu126 or older; CUDA 12.8 removed sm_61.
      ampere    sm_80/86 - RTX 3090
      ada       sm_89  - RTX 4090
      blackwell sm_120 - RTX 5090. Needs cu128 or newer.
      cpu       no GPU. Data prep, the API, the UI and the whole test suite work.
    Default: cpu, because guessing a CUDA wheel for someone else's GPU is how you
    end up with a box that reports torch 2.13.0+cpu and no explanation.

.PARAMETER Extras
    Comma list of extras to install, from pyproject.toml:
      download  gdown / kaggle / modelscope - dataset acquisition only
      train     ultralytics
      export    onnx / onnxruntime / TensorRT tooling
      dev       pytest, ruff, mypy - the test suite
      all       every extra above
    Default "all". There is no reid extra: the appearance model is plain
    torch + torchvision, which are already hard dependencies.

.PARAMETER UpgradeTorch
    Actually install the torch build this profile requires. Off by default: it
    downloads ~2.5 GB and replacing a working torch is rarely what you want
    without asking.

.PARAMETER UseLockFile
    Install requirements-lock.txt before the project. The lock is a verified-good
    snapshot of a working environment; without it, pip re-resolves loose ranges
    from pyproject.toml against whatever PyPI serves today. Turn it on when you
    are reproducing a known-good setup - a new office machine, or a box where
    something drifted. Leave it off when you deliberately want the newest
    resolution.

    The lock deliberately omits torch and torchvision, because one file cannot
    pin five GPU tiers. -Profile still governs those, so the two work together
    rather than conflicting.

.PARAMETER WheelDir
    Directory holding pre-downloaded wheels. Installed with --find-links and
    --no-index for any package the lock names, so a machine with a slow or
    blocked route to PyPI can still be provisioned offline. Packages not present
    locally fall back to the index.

.PARAMETER DataRoot
    Where data/ and artifacts/ live. Defaults to <project>/data. Must be on a
    volume with room for the datasets - see docs/DATASETS.md for the totals.

.EXAMPLE
    .\scripts\setup_env.ps1
    Installs into the current conda env with the CPU profile. Safe default.

.EXAMPLE
    .\scripts\setup_env.ps1 -Profile blackwell -UpgradeTorch
    On the RTX 5090 box: installs the cu128 torch build first, then the project.

.EXAMPLE
    .\scripts\setup_env.ps1 -EnvName anti-uav -DataRoot D:\drone-data
    A dedicated env on a machine where data lives on another volume.
#>
[CmdletBinding()]
param(
    [string] $EnvName = $(if ($env:CONDA_DEFAULT_ENV) { $env:CONDA_DEFAULT_ENV } else { "ml" }),
    [ValidateSet("pascal", "ampere", "ada", "blackwell", "cpu")]
    [string] $Profile = "cpu",
    [string] $Extras = "all",
    [switch] $UpgradeTorch,
    [switch] $UseLockFile,
    [string] $WheelDir,
    [string] $DataRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Write-Step([string] $Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Fail([string] $Message) {
    Write-Host "  [FAIL] $Message" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# resolve the interpreter
# ---------------------------------------------------------------------------
Write-Step "Locating the Python for conda env '$EnvName'"

$python = $null
$conda = Get-Command conda -ErrorAction SilentlyContinue

if ($conda) {
    # `conda run` would work but is slow and swallows exit codes; ask for the
    # prefix and use the interpreter directly instead.
    $prefix = & conda run -n $EnvName python -c "import sys; print(sys.prefix)" 2>$null
    if ($prefix -and (Test-Path -LiteralPath "$prefix\python.exe")) {
        $python = "$prefix\python.exe"
    }
}

if (-not $python) {
    $guesses = @(
        "$env:USERPROFILE\.conda\envs\$EnvName\python.exe",
        "$env:USERPROFILE\anaconda3\envs\$EnvName\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$EnvName\python.exe"
    )
    foreach ($guess in $guesses) {
        if (Test-Path -LiteralPath $guess) { $python = $guess; break }
    }
}

if (-not $python) {
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) {
        Write-Warning "conda env '$EnvName' not found; falling back to $($onPath.Source)."
        Write-Warning "Pass -EnvName to target a specific environment."
        $python = $onPath.Source
    }
}

if (-not $python) {
    Write-Fail "no Python found. Install miniconda, or pass -EnvName."
    exit 1
}

Write-Host "  python    : $python"
& $python --version

# ---------------------------------------------------------------------------
# pip hygiene: keep the cache and temp off a small system drive
# ---------------------------------------------------------------------------
Write-Step "Pinning pip cache and temp"

$cacheRoot = if ($DataRoot) { Join-Path $DataRoot ".pip" } else { Join-Path $projectRoot ".cache\pip" }
$tempRoot = Join-Path (Split-Path -Parent $cacheRoot) ".tmp"
foreach ($dir in @($cacheRoot, $tempRoot)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}
$env:PIP_CACHE_DIR = $cacheRoot
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
Write-Host "  cache     : $cacheRoot"
Write-Host "  temp      : $tempRoot"

& $python -m pip install --upgrade pip --quiet --disable-pip-version-check
Write-Host "  pip       : $(& $python -m pip --version)"

# ---------------------------------------------------------------------------
# torch, only when asked
# ---------------------------------------------------------------------------
# CUDA 12.8 dropped sm_61 and CUDA 13 dropped more, so "latest torch" is simply
# wrong for a Pascal card and quietly wrong for Blackwell. The index URL per
# profile is the whole reason this script exists.
$torchIndex = switch ($Profile) {
    "pascal" { "https://download.pytorch.org/whl/cu126" }
    "ampere" { "https://download.pytorch.org/whl/cu124" }
    "ada" { "https://download.pytorch.org/whl/cu128" }
    "blackwell" { "https://download.pytorch.org/whl/cu128" }
    default { $null }
}

if ($UpgradeTorch -and $torchIndex) {
    Write-Step "Installing the '$Profile' torch build from $torchIndex"
    & $python -m pip install torch torchvision --index-url $torchIndex --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { Write-Fail "torch install failed"; exit 1 }
}
elseif ($torchIndex) {
    Write-Step "Skipping torch"
    Write-Host "  profile '$Profile' wants the wheels at $torchIndex"
    Write-Host "  Re-run with -UpgradeTorch to install it, or install it yourself:"
    Write-Host "    $python -m pip install torch torchvision --index-url $torchIndex"
}
else {
    Write-Step "Skipping torch (cpu profile keeps whatever is installed)"
}

# ---------------------------------------------------------------------------
# the locked dependency set, before the project
# ---------------------------------------------------------------------------
# Ordering matters: torch first (so the lock's packages resolve against the
# build this GPU actually needs and pip leaves it alone), then the lock, then
# the project itself. The lock omits torch and torchvision precisely so it
# cannot fight -Profile here.
if ($UseLockFile) {
    $lockPath = Join-Path $projectRoot "requirements-lock.txt"
    if (-not (Test-Path -LiteralPath $lockPath)) {
        Write-Fail "requirements-lock.txt not found at $lockPath"
        exit 1
    }

    Write-Step "Installing the pinned set from requirements-lock.txt"
    $lockArgs = @("install", "-r", $lockPath, "--disable-pip-version-check")
    if ($WheelDir) {
        if (-not (Test-Path -LiteralPath $WheelDir)) {
            Write-Fail "WheelDir '$WheelDir' does not exist"
            exit 1
        }
        # --find-links without --no-index: prefer a local wheel, fall back to the
        # index for anything not cached. A hard --no-index turns one missing
        # wheel into a failed install on a machine that had fine connectivity.
        $lockArgs += @("--find-links", (Resolve-Path -LiteralPath $WheelDir).Path)
        Write-Host "  local wheels: $WheelDir"
    }
    & $python -m pip @lockArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail "locked install failed"; exit 1 }

    # torch is intentionally absent from the lock. If this profile needs one and
    # -UpgradeTorch was not passed, say so now rather than at the first train.
    if ($torchIndex -and -not $UpgradeTorch) {
        $torchPresent = & $python -c "import importlib.metadata as m; print(m.version('torch'))" 2>$null
        if ($torchPresent -and $torchPresent -notmatch "\+cu") {
            Write-Host ""
            Write-Warning "profile '$Profile' wants wheels from $torchIndex"
            Write-Warning "installed torch is $torchPresent (no CUDA suffix) - this box will train on CPU."
            Write-Warning "Add -UpgradeTorch to fix, or install manually:"
            Write-Warning "  $python -m pip install torch torchvision --index-url $torchIndex"
        }
    }
}

# ---------------------------------------------------------------------------
# the project
# ---------------------------------------------------------------------------
Write-Step "Installing anti-uav in editable mode"

# One specifier carries both the path and the extras: `-e ".[all]"`.
# Passing `-e .` *and* `--all-extras` as separate arguments makes pip resolve
# two requirements for the same project - one editable, one not - and fail with
# ResolutionImpossible.
$known = @("download", "train", "export", "dev", "all")
$requested = @($Extras -split "," | Where-Object { $_ })
foreach ($extra in $requested) {
    if ($known -notcontains $extra) {
        Write-Fail "unknown extra '$extra'; known extras: $($known -join ', ')"
        exit 1
    }
}

$specifier = $projectRoot
if ($requested) {
    $specifier = "${projectRoot}[$($requested -join ',')]"
}

Write-Step "Installing $specifier (editable)"
& $python -m pip install --editable $specifier --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Fail "editable install failed"; exit 1 }

# ---------------------------------------------------------------------------
# data root
# ---------------------------------------------------------------------------
if ($DataRoot) {
    Write-Step "Recording ANTI_UAV_DATA_DIR"
    [Environment]::SetEnvironmentVariable(
        "ANTI_UAV_DATA_DIR", (Resolve-Path -LiteralPath $DataRoot).Path, "User")
    $env:ANTI_UAV_DATA_DIR = (Resolve-Path -LiteralPath $DataRoot).Path
    Write-Host "  ANTI_UAV_DATA_DIR = $env:ANTI_UAV_DATA_DIR (set for your user account)"
}

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
Write-Step "Verifying"
& $python -m anti_uav.cli verify-env
$verifyExit = $LASTEXITCODE

Write-Step "Done"
if ($verifyExit -ne 0) {
    Write-Host "  The install succeeded but verify-env found problems (exit $verifyExit)." -ForegroundColor Yellow
    Write-Host "  Read the output above; each line says what to do about it."
}
else {
    Write-Host "  Ready. Next: anti-uav download --dataset dvb --dry-run" -ForegroundColor Green
}

exit $verifyExit