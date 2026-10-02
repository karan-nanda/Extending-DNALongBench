# Resumable sweep of analysis/eqtl_v2_cnn.py. Runs whose .json already exists are skipped,
# so re-running after an interruption picks up where it stopped.
#
#   powershell -ExecutionPolicy Bypass -File analysis\run_cnn_sweep.ps1          # siamese pair, 5 folds
#   powershell -ExecutionPolicy Bypass -Command "& .\analysis\run_cnn_sweep.ps1 -Modes published -Folds 0 -Seeds 1,2,4 -Lr 0.005"
#
# Pass array arguments through -Command, not -File: -File hands "1,2,4" over as one string.
#
# Siamese modes use lr 0.001 (default here); the published mode keeps its published lr 0.005.
#
# Stop it with Ctrl+C (or kill the python process); the finished runs are kept.
param(
    [string[]]$Modes = @("refalt", "ref_only"),
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int[]]$Seeds = @(0),
    [string]$Lr = "0.001",
    [string]$Scheme = "chrom",
    [string]$Version = "matched",
    [string]$Out = "analysis\results\cnn"
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$out = $Out
New-Item -ItemType Directory -Force "$out\logs" | Out-Null

foreach ($f in $Folds) {
    foreach ($s in $Seeds) {
        foreach ($m in $Modes) {
            $name = "${Version}_${Scheme}_f${f}_${m}_s${s}"
            if (Test-Path "$out\$name.json") { "skip  $name (done)"; continue }
            "$(Get-Date -Format s) start $name"
            cmd /c "python -X utf8 -W ignore analysis\eqtl_v2_cnn.py --version $Version --scheme $Scheme --fold $f --mode $m --seed $s --lr $Lr --out $out > $out\logs\$name.log 2>&1"
            "$(Get-Date -Format s) done  $name (exit $LASTEXITCODE)"
        }
    }
}
