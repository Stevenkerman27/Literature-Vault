<#
.SYNOPSIS
Force-reset local repository to origin/main.
.DESCRIPTION
Runs 'git fetch --all' and 'git reset --hard origin/main'.
If -Clean is specified, also removes all untracked files.
If -UseProxy (or -Y) is specified, uses the built-in proxy (http://192.168.3.9:7890) for git fetch.
If -Proxy is provided, it overrides the built-in proxy.
#>

param(
    [switch]$Clean,                     # Run 'git clean -fd' after reset
    [string]$Proxy = $null,             # Optional custom proxy (e.g., "http://proxy:port")
    [switch]$UseProxy,                  # Alias: -Y, use built-in proxy
    [Alias('Y')]
    [switch]$Y                          # Alias for -UseProxy, so you can just type -Y
)

$defaultProxy = "http://192.168.3.9:7890"

# Determine which proxy to use (if any)
$proxyToUse = $null
if ($Proxy) {
    $proxyToUse = $Proxy
    Write-Host "Using supplied proxy: $proxyToUse" -ForegroundColor Cyan
} elseif ($UseProxy -or $Y) {
    $proxyToUse = $defaultProxy
    Write-Host "Using built-in proxy: $proxyToUse" -ForegroundColor Cyan
}

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

# 5. Execute git fetch (with optional proxy)
$oldHttpProxy = $env:HTTP_PROXY
$oldHttpsProxy = $env:HTTPS_PROXY
$oldAllProxy = $env:ALL_PROXY

try {
    if ($proxyToUse) {
        $env:HTTP_PROXY = $proxyToUse
        $env:HTTPS_PROXY = $proxyToUse
        $env:ALL_PROXY = $proxyToUse
        Write-Host "Fetching latest updates from remote (via proxy)..." -ForegroundColor Cyan
    } else {
        Write-Host "Fetching latest updates from remote..." -ForegroundColor Cyan
    }

    git fetch --all
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Fetch failed. Please check your network." -ForegroundColor Red
        exit 1
    }
} finally {
    # Restore proxy environment variables (or remove if they were not set before)
    if ($null -eq $oldHttpProxy) {
        Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    } else {
        $env:HTTP_PROXY = $oldHttpProxy
    }
    if ($null -eq $oldHttpsProxy) {
        Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
    } else {
        $env:HTTPS_PROXY = $oldHttpsProxy
    }
    if ($null -eq $oldAllProxy) {
        Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
    } else {
        $env:ALL_PROXY = $oldAllProxy
    }
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