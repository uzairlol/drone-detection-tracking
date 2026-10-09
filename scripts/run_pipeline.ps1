<#
.SYNOPSIS
    Run the whole anti-UAV pipeline end to end, resumable, one stage at a time.

.DESCRIPTION
    Wraps the ordered chain in docs/COMMANDS.md into a single command:

        verify-env -> fetch-weights -> download -> convert -> stats
                   -> splits -> build -> sanity
                   -> train (matrix) -> evaluate -> track-eval
                   -> export -> deploy render

    Two things make this worth having as a script rather than a list to paste.

    1. It refuses to start on a machine that cannot run the job. verify-env and
       sanity are hard gates: a non-zero exit from either stops the run there,
       because a 14-run matrix on a box with the wrong torch build, or a training
       set with a sequence leaked across the train/val boundary, wastes days in
       ways that are obvious only in hindsight.

    2. It is resumable. Every stage skips itself when its output already exists
       - the same trick train_matrix.ps1 uses for weights/best.pt - so running
       it again after an interruption picks up where it stopped. This matters
       most for the two large downloads and the 14 training runs, which is to
       say it matters every time.

    Stage order is not interchangeable. convert -> splits -> build -> sanity is a
    data dependency chain: build reads each frame record's `split` field, which
    splits writes. Building before splitting yields an empty train set and a
    sanity failure, not a subtle degradation.

    It creates no environment. Run scripts\setup_env.ps1 first.

.PARAMETER Profile
    GPU tier: pascal, ampere, ada, blackwell, cpu. Drives the torch build check
    and the per-GPU batch/imgsz overrides. Default: auto-detect.

.PARAMETER Datasets
    Comma list of dataset aliases to acquire and convert: dvb, mavvid, antiuav,
    mmuav. Default: all four.

.PARAMETER Combos
    Comma list of combo slugs to build and train, or "all".
    Default: all4 - the candidate production model. Building all seven copies
    every frame seven times, so ask for "all" only when you mean to run the whole
    matrix and have the disk for it.

.PARAMETER Models
    Comma list of model families. Default: yolo11n,rtdetr_x2 (all 14 runs).

.PARAMETER Stage
    Run exactly one stage and exit. For iterating on a single step.

.PARAMETER From
    Run this stage and everything after it, then stop.

.PARAMETER Skip
    Comma list of stage names to skip even when their output is missing.

.PARAMETER ExportFormats
    Comma list for `anti-uav export`. Default: onnx. Add tensorrt on a machine
    with TensorRT installed; FP16 is refused on pascal.

.PARAMETER Python
    Interpreter to use. Defaults to $env:ANTI_UAV_PYTHON, then the resolved
    `anti-uav` entry point, then `python` on PATH.

.PARAMETER DryRun
    Print every stage, its command and whether it would be skipped, then exit.
    Costs nothing. Run this first, every time, on a new machine.

.PARAMETER StopOnFailure
    Abort on the first failing stage. Default is to stop anyway for the gated
    stages (verify-env, sanity) and continue otherwise, so one bad combo does not
    cost you the other thirteen.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -DryRun
    Show the whole plan and exit. Nothing is downloaded, built or trained.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Profile pascal -Combos all4
    The full chain for the production candidate on a GTX 1070 box.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Combos dvb -Models yolo11n -Skip download
    You already have dvb on disk. Just convert, split, build and train it.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -From train
    Data is ready. Start training and carry on through evaluation and export.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Stage sanity
    Re-run the invariant checks on one combo after editing something.
#>
[CmdletBinding()]
param(
    [ValidateSet("pascal", "ampere", "ada", "blackwell", "cpu", "auto")]
    [string] $Profile = "auto",
    [string] $Datasets = "dvb,mavvid,antiuav,mmuav",
    [string] $Combos = "all4",
    [string] $Models = "yolo11n,rtdetr_x2",
    [string] $Stage,
    [string] $From,
    [string] $Skip = "",
    [string] $ExportFormats = "onnx",
    [string] $Python = $env:ANTI_UAV_PYTHON,
    [switch] $DryRun,
    [switch] $StopOnFailure
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Write-Step([string] $Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Skip([string] $Message) {
    Write-Host "    [skip] $Message" -ForegroundColor DarkGray
}

function Write-Fail([string] $Message) {
    Write-Host "    [FAIL] $Message" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# resolve the CLI
# ---------------------------------------------------------------------------
function Test-HasCli {
    param([string] $Exe, [string[]] $Prefix)

    # A probe must never be able to terminate the script. Under
    # $ErrorActionPreference = 'Stop', a native command's stderr promoted by 2>&1
    # becomes a terminating error - so probing an interpreter that does NOT have
    # anti_uav installed aborts resolution instead of falling through to the next
    # candidate. That is exactly backwards: a failing probe is the normal case.
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
        if (-not (Test-Path -LiteralPath $Python)) { throw "no interpreter at $Python" }
        $candidate = @{ Exe = $Python; Prefix = @("-m", "anti_uav.cli") }
        if (-not (Test-HasCli $candidate.Exe $candidate.Prefix)) {
            throw "anti_uav is not importable by $Python. Run scripts\setup_env.ps1 -EnvName <that env>."
        }
        return $candidate
    }

    $candidates = @()

    # The active conda env is the strongest signal about intent. Two envs on this
    # machine both have anti_uav installed, and picking the wrong one silently
    # trains against a different numpy and a different torch build than the one
    # verify-env just approved.
    if ($env:CONDA_PREFIX -and (Test-Path -LiteralPath "$env:CONDA_PREFIX\python.exe")) {
        $candidates += @{ Exe = "$env:CONDA_PREFIX\python.exe"; Prefix = @("-m", "anti_uav.cli") }
    }

    $entry = Get-Command anti-uav -ErrorAction SilentlyContinue
    if ($entry) { $candidates += @{ Exe = $entry.Source; Prefix = @() } }

    # Then the standard per-user conda trees.
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

    # Then drive-local env trees. This project lives on D:, and `conda env list`
    # can show an identically-named env under two roots, so scan these AFTER the
    # active env but keep them ahead of bare `python` on PATH.
    foreach ($root in @("D:\envs", "D:\conda\envs", "E:\envs")) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        foreach ($envDir in (Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) {
            $exe = Join-Path $envDir.FullName "python.exe"
            if (Test-Path -LiteralPath $exe) {
                $candidates += @{ Exe = $exe; Prefix = @("-m", "anti_uav.cli") }
            }
        }
    }

    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += @{ Exe = $onPath.Source; Prefix = @("-m", "anti_uav.cli") } }

    foreach ($candidate in $candidates) {
        if (Test-HasCli $candidate.Exe $candidate.Prefix) { return $candidate }
    }
    throw ("cannot find an interpreter with anti_uav installed. Tried {0} candidate(s)." -f $candidates.Count +
        " Run scripts\setup_env.ps1, or pass -Python with the right interpreter.")
}

# ---------------------------------------------------------------------------
# validate arguments BEFORE hunting for an interpreter. A mistyped -Stage should
# say "unknown stage", not spend twenty seconds probing conda trees first.
# ---------------------------------------------------------------------------
$stageNames = @(
    "verify-env", "fetch-weights", "download", "convert", "stats",
    "splits", "build", "sanity", "train", "evaluate", "track-eval",
    "export", "deploy"
)

foreach ($arg in @($Stage, $From)) {
    if ($arg -and ($stageNames -notcontains $arg)) {
        throw "unknown stage '$arg'. Known stages: $($stageNames -join ', ')"
    }
}

$cli = Resolve-CLI

function Invoke-AntiUav {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)

    # Not `| Out-Host` directly: under $ErrorActionPreference = 'Stop' a native
    # command's stderr promoted by 2>&1 can terminate the script, and ultralytics
    # and rich both write progress to stderr. ForEach-Object { Write-Host $_ } does
    # the same job - keeps pipeline values out of the function's return, so the
    # caller's `if ($code -ne 0)` compares an exit code and not an array of log
    # lines - without the terminating-error path.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $cli.Exe @($cli.Prefix + $Arguments) 2>&1 | ForEach-Object { Write-Host $_ }
    }
    finally {
        $ErrorActionPreference = $previous
    }
    return $LASTEXITCODE
}

function Invoke-Stage {
    param([string[]] $Arguments)

    if ($DryRun) {
        Write-Host "    anti-uav $($Arguments -join ' ')" -ForegroundColor DarkGray
        return 0
    }
    return (Invoke-AntiUav @Arguments)
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
$datasetList = @($Datasets -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$comboList = @($Combos -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$modelList = @($Models -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$skipList = @($Skip -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })

# Each dataset needs its variant named explicitly. A bare `anti-uav download
# --dataset antiuav` would silently pick default_variant 300, and for mmuav it
# would pick `subset` - which is right, but only by coincidence, and a run whose
# numbers cannot be traced back to a variant is not a result.
function Get-VariantArgs([string] $Dataset) {
    switch ($Dataset) {
        "antiuav" { return @("--variant", "300") }
        "mmuav"   { return @("--variant", "subset") }
        default   { return @() }
    }
}

function Test-Skipped([string] $Name) {
    return ($skipList -contains $Name.ToLower())
}

# ---------------------------------------------------------------------------
# resumability predicates. Each returns $true when the stage's work is already
# on disk. These are what make re-running the script safe.
# ---------------------------------------------------------------------------
function Test-DatasetDownloaded([string] $Dataset) {
    $root = Join-Path $projectRoot "data\raw\$Dataset"
    if (-not (Test-Path -LiteralPath $root)) { return $false }
    # A populated tree, not merely an existing directory - the placeholder
    # README.md counts as nothing.
    $payload = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "README.md" -and $_.Extension -ne ".example" }
    return ($null -ne $payload -and @($payload).Count -gt 0)
}

function Test-DatasetConverted([string] $Dataset) {
    $root = Join-Path $projectRoot "data\interim\$Dataset"
    if (-not (Test-Path -LiteralPath $root)) { return $false }
    foreach ($variant in @("full", "300", "subset")) {
        if (Test-Path -LiteralPath (Join-Path $root "$variant\index.jsonl")) { return $true }
    }
    return $false
}

function Test-ComboSplit([string] $Combo) {
    return (Test-Path -LiteralPath (Join-Path $projectRoot "data\interim"))
}

function Test-ComboBuilt([string] $Combo) {
    $root = Join-Path $projectRoot "data\processed\$Combo"
    if (-not (Test-Path -LiteralPath (Join-Path $root "data.yaml"))) { return $false }
    $report = Join-Path $root "build_report.json"
    if (-not (Test-Path -LiteralPath $report)) { return $false }
    # ok:false on disk means a source was missing or a split was empty. Rebuild.
    try { return ((Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).ok -eq $true) }
    catch { return $false }
}

function Get-TrainedRunDirs {
    $runsRoot = Join-Path $projectRoot "artifacts\runs"
    if (-not (Test-Path -LiteralPath $runsRoot)) { return @() }
    return @(Get-ChildItem -LiteralPath $runsRoot -Recurse -Directory -Filter "weights" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Parent.FullName })
}

# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
# $stageNames and its validation already happened above, before Resolve-CLI.
# $gates marks the two hard stops: a non-zero exit from either ends the run.
$gates = @("verify-env", "sanity")

$selected = if ($Stage) {
    @($Stage)
}
elseif ($From) {
    @($stageNames[$stageNames.IndexOf($From)..($stageNames.Count - 1)])
}
else {
    $stageNames
}

Write-Host "anti-uav pipeline" -ForegroundColor White
Write-Host "  project  : $projectRoot"
Write-Host "  python   : $($cli.Exe)"
Write-Host "  datasets : $($datasetList -join ', ')"
Write-Host "  combos   : $($comboList -join ', ')"
Write-Host "  models   : $($modelList -join ', ')"
if ($Profile -ne "auto") { Write-Host "  profile  : $Profile" }
if ($DryRun) { Write-Host "  DRY RUN  - nothing will be downloaded, built or trained" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
Write-Step "Verifying the environment"
$verifyExit = Invoke-Stage @("verify-env")
if (-not $DryRun -and $verifyExit -ne 0) {
    Write-Fail "verify-env found blocking problems (exit $verifyExit)."
    Write-Host "Fix them before anything else. Each line above says what to do." -ForegroundColor Red
    exit $verifyExit
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
$failed = @()

foreach ($name in $selected) {
    if (Test-Skipped $name) {
        Write-Step "$name  [skipped by -Skip]"
        continue
    }

    switch ($name) {

        "verify-env" {
            # already run above as the preflight; do not run it twice.
            Write-Step "verify-env  [done above]"
        }

        "fetch-weights" {
            Write-Step "fetch-weights"
            # Missing checkpoints are a warning from verify-env, not an error, so
            # they are handled here where they actually block: training.
            $code = Invoke-Stage @("fetch-weights")
            if (-not $DryRun -and $code -ne 0) {
                Write-Fail "base weights missing; training cannot start."
                $failed += $name
                if ($StopOnFailure) { exit 1 }
                # Nothing downstream of train can succeed, so stop here rather
                # than burn hours converting and building first.
                Write-Host "    stopping before download: every later stage needs this." -ForegroundColor Yellow
                exit 1
            }
        }

        "download" {
            Write-Step "download  ($($datasetList -join ', '))"
            foreach ($dataset in $datasetList) {
                if (Test-DatasetDownloaded $dataset) {
                    Write-Skip "$dataset already on disk"
                    continue
                }
                Write-Host "  $dataset" -ForegroundColor White
                $args = @("download", "--dataset", $dataset) + (Get-VariantArgs $dataset)
                if ($DryRun) {
                    Write-Host "    anti-uav $($args -join ' ') --dry-run" -ForegroundColor DarkGray
                    continue
                }
                Write-Host "    (large - expect it to take a while)" -ForegroundColor DarkGray
                $code = Invoke-AntiUav @args
                if ($code -ne 0) {
                    Write-Fail "$dataset download failed (exit $code)"
                    $failed += "download:$dataset"
                    if ($StopOnFailure) { exit 1 }
                    # Not fatal. A missing source is skipped by `build` with a
                    # warning and the rest of the matrix still runs - but the
                    # combo will be smaller than its name suggests, so say so.
                    Write-Host "    continuing: build will skip this source and record it in" -ForegroundColor Yellow
                    Write-Host "    build_report.json. Do not read a combo name as proof of" -ForegroundColor Yellow
                    Write-Host "    dataset membership." -ForegroundColor Yellow
                }
            }
        }

        "convert" {
            Write-Step "convert  ($($datasetList -join ', '))"
            foreach ($dataset in $datasetList) {
                if (Test-DatasetConverted $dataset) {
                    Write-Skip "$dataset already converted"
                    continue
                }
                Write-Host "  $dataset" -ForegroundColor White
                $code = Invoke-Stage (@("convert", "--dataset", $dataset) + (Get-VariantArgs $dataset))
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "$dataset convert failed (exit $code)"
                    $failed += "convert:$dataset"
                    if ($StopOnFailure) { exit 1 }
                }
            }
        }

        "stats" {
            Write-Step "stats  ($($datasetList -join ', '))"
            foreach ($dataset in $datasetList) {
                # Cheap, and the bird-negative table is the thing that tells you
                # which precision numbers mean anything. Always run it.
                Write-Host "  $dataset" -ForegroundColor White
                $code = Invoke-Stage (@("stats", "--dataset", $dataset) + (Get-VariantArgs $dataset))
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "$dataset stats failed (exit $code)"
                    $failed += "stats:$dataset"
                }
            }
        }

        "splits" {
            # One call per combo. splits writes the `split` field that build reads,
            # so this must precede build even when the combo is unchanged, because
            # build's skipping decision depends on the report this produces.
            Write-Step "splits  ($($comboList -join ', '))"
            foreach ($combo in $comboList) {
                Write-Host "  $combo" -ForegroundColor White
                $code = Invoke-Stage @("splits", "--combo", $combo)
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "$combo splits failed (exit $code)"
                    $failed += "splits:$combo"
                    if ($StopOnFailure) { exit 1 }
                    Write-Host "    build would produce an empty train set without this." -ForegroundColor Yellow
                }
            }
        }

        "build" {
            Write-Step "build  ($($comboList -join ', '))"
            foreach ($combo in $comboList) {
                if (Test-ComboBuilt $combo) {
                    Write-Skip "$combo already built (build_report.json ok:true)"
                    continue
                }
                Write-Host "  $combo" -ForegroundColor White
                $code = Invoke-Stage @("build", "--combo", $combo)
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "$combo build failed (exit $code)"
                    $failed += "build:$combo"
                    if ($StopOnFailure) { exit 1 }
                }
            }
        }

        "sanity" {
            Write-Step "sanity  ($($comboList -join ', '))"
            foreach ($combo in $comboList) {
                Write-Host "  $combo" -ForegroundColor White
                $code = Invoke-Stage @("sanity", "--combo", $combo)
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "$combo FAILED its invariants (exit $code)."
                    Write-Host "    Do not train this combo. A sequence shared across the" -ForegroundColor Red
                    Write-Host "    train/val boundary inflates mAP by double digits, and an empty" -ForegroundColor Red
                    Write-Host "    split is worse than no run at all." -ForegroundColor Red
                    $failed += "sanity:$combo"
                    # Hard stop, always. This is the gate the whole script exists
                    # to enforce; -StopOnFailure is not needed to honour it.
                    exit 1
                }
            }
        }

        "train" {
            Write-Step "train  ($($modelList -join ', ') x $($comboList -join ', '))"
            $existing = @(Get-TrainedRunDirs)
            if ($existing.Count -gt 0) {
                Write-Host "    $($existing.Count) run(s) already have weights and will be skipped:" -ForegroundColor DarkGray
                foreach ($dir in $existing) {
                    Write-Host "      $dir" -ForegroundColor DarkGray
                }
            }
            $args = @("matrix", "run", "--models", ($modelList -join ","), "--combos", ($comboList -join ","))
            if ($Profile -ne "auto") { $args += @("--profile", $Profile) }
            $code = Invoke-Stage $args
            if (-not $DryRun -and $code -ne 0) {
                Write-Fail "matrix run exited $code"
                $failed += "train"
                if ($StopOnFailure) { exit 1 }
                Write-Host "    completed runs are kept. Re-run the same command to resume." -ForegroundColor Yellow
            }
        }

        "evaluate" {
            # --cross is the whole point: it scores the run against each
            # dataset's held-out split separately, which is the only way to see
            # that a model trained on everything did not just learn one
            # dataset's target scale.
            foreach ($dir in @(Get-TrainedRunDirs)) {
                Write-Step "evaluate $(Split-Path -Leaf $dir)"
                $code = Invoke-Stage @("evaluate", "--run", $dir, "--cross")
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "evaluate failed for $dir (exit $code)"
                    $failed += "evaluate:$dir"
                }
            }
        }

        "track-eval" {
            foreach ($dir in @(Get-TrainedRunDirs)) {
                Write-Step "track-eval $(Split-Path -Leaf $dir)"
                # mmuav only: it is the only dataset with multi-object identity to
                # score against. antiuav has track ids but no birds.
                $code = Invoke-Stage @("track-eval", "--run", $dir, "--dataset", "mmuav")
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "track-eval failed for $dir (exit $code)"
                    $failed += "track-eval:$dir"
                }
            }
        }

        "export" {
            foreach ($dir in @(Get-TrainedRunDirs)) {
                Write-Step "export $(Split-Path -Leaf $dir)"
                $code = Invoke-Stage @("export", "--run", $dir, "--formats", $ExportFormats)
                if (-not $DryRun -and $code -ne 0) {
                    Write-Fail "export failed for $dir (exit $code)"
                    $failed += "export:$dir"
                }
            }
        }

        "deploy" {
            Write-Step "deploy render"
            $code = Invoke-Stage @("deploy", "render", "--output", "artifacts/deploy")
            if (-not $DryRun -and $code -ne 0) {
                Write-Fail "deploy render failed (exit $code)"
                $failed += "deploy"
            }
            else {
                Write-Host "    engines are built on the target device; the render lists the" -ForegroundColor DarkGray
                Write-Host "    exact trtexec command per node when one is missing." -ForegroundColor DarkGray
            }
        }
    }
}

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
Write-Step "Done"

if ($DryRun) {
    Write-Host "Dry run complete. Nothing was downloaded, built or trained." -ForegroundColor Yellow
    exit 0
}

if ($failed.Count -eq 0) {
    Write-Host "  every requested stage completed." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next:" -ForegroundColor Green
    Write-Host "  anti-uav evaluate   --run artifacts/runs/yolo11n/all4 --cross"
    Write-Host "  anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav"
    Write-Host "  anti-uav replay     --run artifacts/runs/yolo11n/all4 --dataset mmuav --sequence 0007"
    Write-Host "  anti-uav serve"
    exit 0
}

Write-Host "  $($failed.Count) stage(s) failed:" -ForegroundColor Red
foreach ($item in $failed) { Write-Host "    $item" -ForegroundColor Red }
Write-Host ""
Write-Host "Re-run the same command to resume. Stages whose output already exists" -ForegroundColor Yellow
Write-Host "are skipped, so nothing that succeeded is repeated." -ForegroundColor Yellow
exit 1