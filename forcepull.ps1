<#
.SYNOPSIS
Force-reset local repository to origin/main.
.DESCRIPTION
Runs 'git fetch --all' and 'git reset --hard origin/main'.
If -Clean is specified, also removes all untracked files.
#>

param(
    [switch]$Clean  # If specified, run 'git clean -fd' after reset
)

# 1. Verify current directory is a Git repository
try {
    $isRepo = git rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Current directory is not a Git repository!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error: Git command not found. Please ensure Git is installed." -ForegroundColor Red
    exit 1
}

# 2. Display current branch name
$currentBranch = git branch --show-current
Write-Host "Current branch: $currentBranch" -ForegroundColor Yellow

# 3. Check if remote branch origin/main exists
$remoteCheck = git show-ref --verify --quiet refs/remotes/origin/main
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Remote branch origin/main does not exist! Check your remote configuration." -ForegroundColor Red
    exit 1
}

# 4. Ask for confirmation to prevent accidental data loss
Write-Host "WARNING: This operation will discard ALL local changes and unpushed commits!" -ForegroundColor Red
$confirmation = Read-Host "Are you sure? (type 'yes' to proceed)"
if ($confirmation -ne "yes") {
    Write-Host "Operation cancelled." -ForegroundColor Green
    exit 0
}

# 5. Execute git fetch
Write-Host "Fetching latest updates from remote..." -ForegroundColor Cyan
git fetch --all
if ($LASTEXITCODE -ne 0) {
    Write-Host "Fetch failed. Please check your network." -ForegroundColor Red
    exit 1
}

# 6. Execute git reset
Write-Host "Force-resetting to origin/main ..." -ForegroundColor Cyan
git reset --hard origin/main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Reset successful!" -ForegroundColor Green
} else {
    Write-Host "Reset failed. Check the error messages above." -ForegroundColor Red
    exit 1
}

# 7. (Optional) Clean untracked files
if ($Clean) {
    Write-Host "Cleaning untracked files..." -ForegroundColor Cyan
    git clean -fd
    Write-Host "Cleanup completed." -ForegroundColor Green
}

# 8. Show final status
git status