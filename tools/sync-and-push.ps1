$ErrorActionPreference = 'Stop'

# The script lives in tools/, so the parent directory is the repository root.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

try {
    Write-Host "Repository: $repoRoot" -ForegroundColor Cyan
    Write-Host "Collecting all changes..." -ForegroundColor Cyan
    Invoke-Git -Arguments @('add', '--all')

    $stagedFiles = @(git diff --cached --name-only)
    if ($stagedFiles.Count -eq 0) {
        Write-Host 'No changes to commit.' -ForegroundColor Yellow
        return
    }

    Write-Host "`nChanges to commit:" -ForegroundColor Cyan
    & git diff --cached --stat

    do {
        $commitMessage = Read-Host "`nEnter commit message"
        if ([string]::IsNullOrWhiteSpace($commitMessage)) {
            Write-Host 'Commit message cannot be empty.' -ForegroundColor Yellow
        }
    } while ([string]::IsNullOrWhiteSpace($commitMessage))

    Invoke-Git -Arguments @('commit', '-m', $commitMessage)
    Invoke-Git -Arguments @('push')

    Write-Host "`nPush completed successfully." -ForegroundColor Green
}
catch {
    Write-Host "`nSync failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Read-Host "`nPress Enter to close"
}
