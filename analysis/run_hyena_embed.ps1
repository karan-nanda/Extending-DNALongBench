# Runs analysis/hyena_embed.py a few shards at a time, each batch in a fresh Python
# process, until every shard exists and the embeddings are merged. Finished shards are
# kept, so this can be stopped and rerun at any point.
#
#   powershell -ExecutionPolicy Bypass -File analysis\run_hyena_embed.ps1
param(
    [string]$Version = "matched",
    [int]$ChunkShards = 3,
    [double]$MemFraction = 0.5
)
Set-Location (Split-Path $PSScriptRoot -Parent)
do {
    cmd /c "python -u -X utf8 -W ignore analysis\hyena_embed.py --version $Version --max-shards $ChunkShards --mem-fraction $MemFraction"
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) python exited $code"
    if ($code -ne 0 -and $code -ne 3) { "stopping: unexpected exit code $code"; break }
} while ($code -eq 3)
