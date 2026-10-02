# GPU robustness queue (matrix §14). Each step is resumable; rerunning skips finished work.
#
#   powershell -ExecutionPolicy Bypass -File analysis\run_robustness_gpu.ps1
#
# Order (most informative first):
#   R6  variant-window embeddings (all layers; mean / +-1 kb / +-64 bp / variant position;
#       ref, alt and a random-alt control): HyenaDNA on Windows, then Caduceus in WSL
#   R7  published-CNN collapse on the PUBLISHED split (Whole_Blood, Nerve_Tibial),
#       10 init seeds, each trained one full epoch
#   R8  CNN refalt with test-time ablation, seeds 1 and 2 on all 5 chrom folds
#       (seed 0 exists in analysis\results\cnn_ablation)
# Pause Wallpaper Engine / games while this runs.
Set-Location (Split-Path $PSScriptRoot -Parent)
$res = "analysis\results\robustness"
New-Item -ItemType Directory -Force $res | Out-Null

"$(Get-Date -Format s) R6 HyenaDNA variant-window embeddings"
do {
    cmd /c "python -u -X utf8 -W ignore analysis\variant_window_embed.py --model hyena --max-shards 3"
    $code = $LASTEXITCODE
} while ($code -eq 3)
"$(Get-Date -Format s) R6 HyenaDNA exit $code"

"$(Get-Date -Format s) R6 Caduceus variant-window embeddings (WSL)"
wsl -d Ubuntu-24.04 -u root --cd "/mnt/d/Extending DNALongBench" -- /root/cad/bin/python -u -W ignore analysis/variant_window_embed.py --model caduceus
"$(Get-Date -Format s) R6 Caduceus exit $LASTEXITCODE"

foreach ($t in @("Whole_Blood", "Nerve_Tibial")) {
    $log = "$res\R7_collapse_published_$t.log"
    if (Test-Path $log) { "skip R7 $t (done)"; continue }
    "$(Get-Date -Format s) R7 CNN collapse, published split, $t"
    cmd /c "python -u -X utf8 -W ignore analysis\cnn_dead_head_check.py --published-tissue $t --seeds 10 --train-seeds 0,1,2,3,4,5,6,7,8,9 --train-steps 0 > $log.tmp 2>&1"
    if ($LASTEXITCODE -eq 0) { Move-Item -Force "$log.tmp" $log }
}

"$(Get-Date -Format s) R8 CNN refalt + ablation, seeds 1,2"
& "$PSScriptRoot\run_cnn_sweep.ps1" -Modes refalt -Seeds 1,2 -Lr 0.001 -Out analysis\results\cnn_ablation
"$(Get-Date -Format s) GPU robustness queue finished"
