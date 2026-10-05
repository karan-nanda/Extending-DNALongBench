# R6 only: variant-window embeddings for both models, then the probe analysis (matrix §14).
# Resumable: finished shards are skipped, so this can be stopped and rerun.
#
#   powershell -ExecutionPolicy Bypass -File analysis\run_r6.ps1
#
# ~4.5 h for HyenaDNA (rev) + ~1.5 h for Caduceus (fwd, WSL), then a few minutes of CPU.
# Peak GPU ~4.8 GB; close other GPU apps.
Set-Location (Split-Path $PSScriptRoot -Parent)

"$(Get-Date -Format s) R6 HyenaDNA"
do {
    cmd /c "python -u -X utf8 -W ignore analysis\variant_window_embed.py --model hyena --max-shards 3"
    $code = $LASTEXITCODE
    if ($code -notin 0, 3) { "HyenaDNA stopped with exit $code"; break }
} while ($code -eq 3)
"$(Get-Date -Format s) R6 HyenaDNA exit $code"

"$(Get-Date -Format s) R6 Caduceus (WSL)"
do {
    wsl -d Ubuntu-24.04 -u root --cd "/mnt/d/Extending DNALongBench" -- /root/cad/bin/python -u -W ignore analysis/variant_window_embed.py --model caduceus --max-shards 6
    $code = $LASTEXITCODE
    if ($code -notin 0, 3) { "Caduceus stopped with exit $code"; break }
} while ($code -eq 3)
"$(Get-Date -Format s) R6 Caduceus exit $code"

"$(Get-Date -Format s) R6 probes"
cmd /c "python -u -X utf8 -W ignore analysis\variant_window_probe.py > analysis\results\robustness\R6_probe.log 2>&1"
"$(Get-Date -Format s) R6 finished (exit $LASTEXITCODE)"
