#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Dev
)

$DEV_MODE = $Dev.IsPresent

$ErrorActionPreference = "Stop"
$IMAGE = if ($Env:BENCHMARK_IMAGE) { $Env:BENCHMARK_IMAGE } else { "mablospate/tfg-bench:latest" }
$DOCKER_STARTED = $false   # we started Docker Desktop from scratch
$CONTAINER_NAME = "tfg-bench-$PID"

$proc         = Get-CimInstance Win32_Processor | Select-Object -First 1
$CPU_MODEL    = $proc.Name.Trim()
$CPU_PHYSICAL = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
$CPU_LOGICAL  = $proc.NumberOfLogicalProcessors
$CPU_FREQ_MHZ = $proc.MaxClockSpeed
$RAM_BYTES    = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$RAM_GB_I     = [int]($RAM_BYTES / 1GB)
$RAM_GB_F     = [math]::Round($RAM_BYTES / 1GB, 1)

$DOCKER_MEM_GB = if ($RAM_GB_I -gt 6) { $RAM_GB_I - 2 } else { 4 }
$DOCKER_CPUS   = $CPU_PHYSICAL

$HAS_NVIDIA = $false
$GPU_FLAGS  = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nv = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nv) {
        $GPU_MODEL  = $nv | Select-Object -First 1
        $HAS_NVIDIA = $true
        $GPU_FLAGS  = @("--gpus", "all")
    }
}

function Ensure-Docker {
    $null = docker info 2>$null
    if ($LASTEXITCODE -eq 0) { return }

    $dockerExe = Join-Path $Env:ProgramFiles "Docker\Docker\Docker.exe"
    if (Test-Path $dockerExe) {
        Write-Host "-> Opening Docker Desktop (waiting up to 90s)..."
        Start-Process $dockerExe
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep 1
            $null = docker info 2>$null
            if ($LASTEXITCODE -eq 0) { $script:DOCKER_STARTED = $true; return }
        }
        Write-Error "Docker Desktop did not respond in time."
        exit 1
    }

    Write-Error "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

function Restore-Docker {
    if ($script:DOCKER_STARTED) {
        Write-Host "-> Stopping Docker Desktop (was not running before)..."
        $dockerExe = Join-Path $Env:ProgramFiles "Docker\Docker\Docker.exe"
        & $dockerExe --quit-on-logout 2>$null
        Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
    }
}

Ensure-Docker

$inspect = docker image inspect $IMAGE 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "-> Pulling $IMAGE..."
    docker pull $IMAGE
}

Write-Host "Hardware: $CPU_MODEL | ${CPU_PHYSICAL}p/${CPU_LOGICAL}l cores | ${CPU_FREQ_MHZ}MHz | ${RAM_GB_F}GB RAM"
Write-Host "Docker:   --memory ${DOCKER_MEM_GB}g --cpus $DOCKER_CPUS"

$extraArgs = @()
if ($DEV_MODE) { $extraArgs += "--dev" }
if ($args -notcontains "--contributor") {
    if ($DEV_MODE) {
        $contributor = "dev"
    } else {
        $contributor = Read-Host "Contributor name"
    }
    $extraArgs += @("--contributor", $contributor)
}

$TIME_BUDGET_MINS = 0  # 0 = unlimited
if (-not $DEV_MODE -and $args -notcontains "--time-budget") {
    $ans = Read-Host "Tiempo límite en minutos (Enter = sin límite)"
    if ($ans -match '^\d+$' -and [int]$ans -gt 0) {
        $TIME_BUDGET_MINS = [int]$ans
    }
}

$dockerArgs = @(
    "run", "--rm",
    "--name", $CONTAINER_NAME,
    "--memory", "${DOCKER_MEM_GB}g",
    "--cpus", "$DOCKER_CPUS"
) + $GPU_FLAGS + @(
    "-e", "BENCH_HOSTNAME=$Env:COMPUTERNAME",
    "-e", "BENCH_CPU_MODEL=$CPU_MODEL",
    "-e", "BENCH_CPU_CORES_PHYSICAL=$DOCKER_CPUS",
    "-e", "BENCH_CPU_CORES_LOGICAL=$DOCKER_CPUS",
    "-e", "BENCH_CPU_FREQ_MHZ=$CPU_FREQ_MHZ",
    "-e", "BENCH_RAM_GB=$DOCKER_MEM_GB",
    "-e", "BENCH_OS=Windows",
    "-e", "BENCH_OS_VERSION=$([System.Environment]::OSVersion.Version)",
    "-v", "${PWD}\results:/app/results",
    $IMAGE
) + $extraArgs + $args

Write-Host "(Pulsa 'q' para detener el benchmark)"
$dockerProc = Start-Process -FilePath "docker" -ArgumentList $dockerArgs -NoNewWindow -PassThru

$timerJob = $null
if ($TIME_BUDGET_MINS -gt 0) {
    $containerName = $CONTAINER_NAME
    $timerJob = Start-Job -ScriptBlock {
        param($secs, $name)
        Start-Sleep -Seconds $secs
        docker stop $name 2>$null
    } -ArgumentList ($TIME_BUDGET_MINS * 60), $containerName
}

while (-not $dockerProc.HasExited) {
    if ([Console]::KeyAvailable) {
        $key = [Console]::ReadKey($true)
        if ($key.Key -eq [ConsoleKey]::Q) {
            Write-Host ""
            Write-Host "-> Benchmark detenido por el usuario."
            & docker stop $CONTAINER_NAME 2>$null
            break
        }
    }
    Start-Sleep -Milliseconds 200
}
$dockerProc.WaitForExit()
if ($timerJob) { Stop-Job $timerJob; Remove-Job $timerJob }

if ($Env:KEEP_IMAGE -ne "1") {
    Write-Host "-> Removing image $IMAGE..."
    docker rmi $IMAGE 2>$null
}

Restore-Docker
