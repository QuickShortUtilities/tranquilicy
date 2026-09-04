# Start the Cloudflare quick tunnel that fronts the local GPU server, capture
# the hostname it is assigned, and health-check it before reporting success.
#
# Why this exists: a *quick* tunnel gets a new random hostname every time
# cloudflared starts, so after any reboot or crash the deployed Worker is
# pointing at a dead host and every visitor sees "GPU tunnel unreachable".
# Recovery is then: read the new hostname out of cloudflared's banner, and put
# it somewhere the Worker reads.
#
# Set the GPU_BACKEND variable in the Cloudflare dashboard
# (Workers -> tranquilicy -> Settings -> Variables) to the URL this prints.
# That applies instantly with no redeploy. worker.js only falls back to its
# hardcoded constant when that variable is unset.
#
# The durable fix is a NAMED tunnel (stable hostname, survives restarts), which
# needs an interactive `cloudflared tunnel login` against a domain on your
# Cloudflare account -- see the notes printed at the end.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_tunnel.ps1
#   ... -Port 8000 -TimeoutSec 60

param(
    [int]$Port = 8000,
    [int]$TimeoutSec = 60,
    [string]$LogPath = "D:\musicgen_data\cloudflared.log",
    [string]$UrlFile = "D:\musicgen_data\tunnel_url.txt"
)

$ErrorActionPreference = "Stop"
$exe = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

if (-not (Test-Path $exe)) { Write-Error "cloudflared not found at $exe"; exit 1 }

# A tunnel to a dead server is worse than no tunnel: it returns 502s that look
# like the app is broken rather than not running.
Write-Host "Checking local server on port $Port ..." -NoNewline
try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/capacity" -TimeoutSec 8 -UseBasicParsing
    Write-Host " up"
} catch {
    Write-Host " DOWN"
    Write-Error "Local server is not responding on port $Port. Start 09_api_server.py first."
    exit 1
}

Get-Process cloudflared -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping existing cloudflared (PID $($_.Id)) ..."
    Stop-Process -Id $_.Id -Force
    Start-Sleep -Milliseconds 800
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
if (Test-Path $LogPath) { Remove-Item $LogPath -Force }

# Detached via WMI: Start-Process children are grouped into this session's job
# object and die when it closes (confirmed repeatedly on this machine).
$cmd = "cmd.exe /c `"`"$exe`" tunnel --url http://localhost:$Port > `"$LogPath`" 2>&1`""
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
if ($res.ReturnValue -ne 0) { Write-Error "failed to launch cloudflared (code $($res.ReturnValue))"; exit 1 }
Write-Host "Launched cloudflared (PID $($res.ProcessId)); waiting for hostname ..."

$url = $null
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 700
    if (-not (Test-Path $LogPath)) { continue }
    $text = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
    if ($text -and $text -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
        $url = $Matches[0]
        break
    }
}

if (-not $url) {
    Write-Error "no tunnel hostname appeared within $TimeoutSec s. See $LogPath"
    exit 1
}

# Assigned is not the same as reachable; edge propagation takes a few seconds.
Write-Host "Assigned $url - verifying it serves the app ..." -NoNewline
$ok = $false
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "$url/capacity" -TimeoutSec 10 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
Write-Host $(if ($ok) { " reachable" } else { " NOT reachable yet" })

Set-Content -Path $UrlFile -Value $url -Encoding utf8

Write-Host ""
Write-Host "======================================================================"
Write-Host " TUNNEL URL:  $url"
Write-Host " saved to:    $UrlFile"
Write-Host "======================================================================"
Write-Host ""
if (-not $ok) {
    Write-Host " WARNING: the hostname was assigned but did not serve /capacity yet."
    Write-Host " Give it another ~30s, then re-test:  curl $url/capacity"
    Write-Host ""
}
Write-Host " NEXT: set the Worker variable so the live site uses this hostname."
Write-Host "   Cloudflare dashboard -> Workers & Pages -> tranquilicy"
Write-Host "     -> Settings -> Variables -> GPU_BACKEND = $url  -> Deploy"
Write-Host "   (applies instantly; no git push, no code change)"
Write-Host ""
Write-Host " To stop needing this after every restart, create a NAMED tunnel once:"
Write-Host "   1. `"$exe`" tunnel login          # browser; pick a domain you own"
Write-Host "   2. `"$exe`" tunnel create tranquilicy"
Write-Host "   3. `"$exe`" tunnel route dns tranquilicy gpu.<your-domain>"
Write-Host "   then run it with:  `"$exe`" tunnel run tranquilicy"
Write-Host "   That hostname is stable forever, so GPU_BACKEND never changes again."
Write-Host ""
