param(
    [int]$BackendPort = 8000,
    [int]$MlPort = 8010,
    [int]$FrontendPort = 5173,
    [string]$BackendUrl = "http://127.0.0.1:$BackendPort",
    [string]$MlUrl = "http://127.0.0.1:$MlPort",
    [string]$FrontendUrl = "http://127.0.0.1:$FrontendPort"
)

$ErrorActionPreference = "Stop"
$failures = [System.Collections.Generic.List[string]]::new()

function Test-HttpReady {
    param([string]$Name, [string]$Url, [scriptblock]$Validator)
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 10
        if (-not (& $Validator $response)) { $failures.Add("$Name failed readiness validation: $Url") }
    } catch { $failures.Add("$Name is unavailable: $Url ($($_.Exception.Message))") }
}

function Test-PortListening {
    param([string]$Name, [int]$Port)
    if (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue)) {
        $failures.Add("$Name is not listening on port $Port")
    }
}

Test-PortListening "Frontend" $FrontendPort
Test-PortListening "Backend" $BackendPort
Test-PortListening "ML" $MlPort
Test-HttpReady "Frontend" $FrontendUrl { param($response) $true }
Test-HttpReady "Backend health" "$BackendUrl/health" { param($response) $response.status -eq "ok" }
Test-HttpReady "ML health" "$MlUrl/health" { param($response) $response.ready -eq $true -and $response.speaker_ready -eq $true }

$aasistCheckpoint = Join-Path $PSScriptRoot "..\ml\spoof\vendor\aasist\AASIST.pth"
$ecapaCheckpoint = Join-Path $PSScriptRoot "..\ml\speaker\vendor\ecapa\embedding_model.ckpt"
if (-not (Test-Path -LiteralPath $aasistCheckpoint -PathType Leaf)) { $failures.Add("Required AASIST checkpoint is missing: $aasistCheckpoint") }
if (-not (Test-Path -LiteralPath $ecapaCheckpoint -PathType Leaf)) { $failures.Add("Required ECAPA checkpoint is missing: $ecapaCheckpoint") }

$servicePatterns = @("uvicorn app.main:app", "uvicorn runtime.server:app", "vite.*--port $FrontendPort")
foreach ($pattern in $servicePatterns) {
    $matches = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match $pattern }
    if ($matches.Count -gt 1) { $failures.Add("Duplicate service processes match '$pattern': $($matches.Count)") }
}

if ($failures.Count -gt 0) {
    Write-Host "VOXSENTINEL DEMO NOT READY" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "FAIL: $_" }
    exit 1
}

Write-Host "VOXSENTIN DEMO READY" -ForegroundColor Green
