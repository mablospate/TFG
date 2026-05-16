#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Dev
)

$DEV_MODE = $Dev.IsPresent

$ErrorActionPreference = "Stop"
$IMAGE = if ($Env:BENCHMARK_IMAGE) { $Env:BENCHMARK_IMAGE } else { "mablospate/tfg-bench:latest" }
$DOCKER_STARTED = $false   # we started Docker Desktop from scratch
$script:PASS_NUM = 0
$script:BENCH_CONTAINER_NAME = ""

# --- Hardware detection ---
$proc         = Get-CimInstance Win32_Processor | Select-Object -First 1
$CPU_MODEL    = $proc.Name.Trim()
$CPU_PHYSICAL = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
$CPU_LOGICAL  = $proc.NumberOfLogicalProcessors
$CPU_FREQ_MHZ = $proc.MaxClockSpeed
$RAM_BYTES    = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$RAM_GB_I     = [int]($RAM_BYTES / 1GB)
$RAM_GB_F     = [math]::Round($RAM_BYTES / 1GB, 1)
$HOST_ARCH    = $env:PROCESSOR_ARCHITECTURE   # "AMD64" or "ARM64" on Windows

# --- Compute resources ---
$DOCKER_MEM_GB = if ($RAM_GB_I -gt 6) { $RAM_GB_I - 2 } else { 4 }
$DOCKER_CPUS   = $CPU_PHYSICAL

# --- NVIDIA detection ---
$HAS_NVIDIA = $false
$GPU_FLAGS  = @()
$GPU_MODEL  = ""
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nv = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nv) {
        $GPU_MODEL  = ($nv | Select-Object -First 1).Trim()
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

function Cap-ToDockerLimits {
    try {
        $dockerNcpu = & docker info --format '{{.NCPU}}' 2>$null
        if ($LASTEXITCODE -eq 0 -and $dockerNcpu -match '^\d+$') {
            $dockerNcpuInt = [int]$dockerNcpu
            if ($dockerNcpuInt -gt 0 -and $dockerNcpuInt -lt $script:DOCKER_CPUS) {
                $script:DOCKER_CPUS = $dockerNcpuInt
            }
        }
    } catch { }

    try {
        $dockerMem = & docker info --format '{{.MemTotal}}' 2>$null
        if ($LASTEXITCODE -eq 0 -and $dockerMem -match '^\d+$') {
            $dockerMemGb = [int]([int64]$dockerMem / 1GB)
            if ($dockerMemGb -gt 0 -and $dockerMemGb -lt $script:DOCKER_MEM_GB) {
                $script:DOCKER_MEM_GB = $dockerMemGb
            }
        }
    } catch { }
}

function Test-QemuAvailable {
    $result = & docker run --rm --platform linux/amd64 alpine uname -m 2>$null
    return ($LASTEXITCODE -eq 0 -and "$result" -match 'x86_64')
}

function Pull-Image {
    $imageExists = $false
    try {
        $null = docker image inspect $IMAGE 2>$null
        $imageExists = ($LASTEXITCODE -eq 0)
    } catch {
        $imageExists = $false
    }

    if (-not $imageExists) {
        Write-Host "-> Pulling $IMAGE from Docker Hub..."
        docker pull $IMAGE
        if ($LASTEXITCODE -ne 0) { Write-Error "docker pull failed."; exit 1 }
    }
}

function ConvertTo-ProcessArg([string]$s) {
    # Implements CommandLineToArgvW quoting rules (MSVCRT / Go runtime).
    if ($s -eq '') { return '""' }
    if ($s -notmatch '[ \t\n\v"]') { return $s }
    # Escape backslashes that immediately precede a quote, then the quote itself.
    $e = [regex]::Replace($s, '(\\*)"', { '\\' * ($args[0].Groups[1].Length * 2 + 1) + '"' })
    # Double trailing backslashes before the closing quote.
    $e = [regex]::Replace($e, '(\\+)$', { '\\' * ($args[0].Groups[1].Length * 2) })
    return '"' + $e + '"'
}

function Run-Benchmark {
    [CmdletBinding()]
    param(
        [switch]$NoGpu,
        [switch]$Emulated,
        [string]$DockerPlatform = "",
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PassthroughArgs
    )

    $script:PASS_NUM += 1
    $script:BENCH_CONTAINER_NAME = "tfg-bench-$PID-$($script:PASS_NUM)"

    $dockerRunArgs = @(
        "run", "--rm",
        "--name", $script:BENCH_CONTAINER_NAME
    )

    if ($DockerPlatform) {
        $dockerRunArgs += @("--platform", $DockerPlatform)
    }

    $dockerRunArgs += @(
        "--memory", "${DOCKER_MEM_GB}g",
        "--cpus", "$DOCKER_CPUS"
    )

    if (-not $NoGpu) {
        $dockerRunArgs += $GPU_FLAGS
    }

    $dockerRunArgs += @(
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
    )

    if ($Emulated) {
        $dockerRunArgs += "--emulated"
    }

    if ($NoGpu) {
        $dockerRunArgs += "--no-gpu"
    }

    if ($PassthroughArgs) {
        $dockerRunArgs += $PassthroughArgs
    }

    # Build a properly-quoted command line (CommandLineToArgvW rules) so that
    # values with spaces, embedded quotes, or trailing backslashes survive
    # Start-Process's verbatim pass-through to CreateProcess unchanged.
    $argLine = ($dockerRunArgs | ForEach-Object { ConvertTo-ProcessArg $_ }) -join ' '

    Write-Host "(Pulsa 'q' para detener el benchmark)"
    $dockerProc = Start-Process -FilePath "docker" -ArgumentList $argLine -NoNewWindow -PassThru

    $timerJob = $null
    if ($TIME_BUDGET_MINS -gt 0) {
        $containerName = $script:BENCH_CONTAINER_NAME
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
                & docker stop $script:BENCH_CONTAINER_NAME 2>$null
                break
            }
        }
        Start-Sleep -Milliseconds 200
    }
    $dockerProc.WaitForExit()
    if ($timerJob) {
        Stop-Job $timerJob -ErrorAction SilentlyContinue
        Remove-Job $timerJob -ErrorAction SilentlyContinue
    }

    $script:BENCH_CONTAINER_NAME = ""
}

# --- Main flow ---

Ensure-Docker
Cap-ToDockerLimits

# Collect run args (contributor, time budget, dev)
$extraArgs = @()

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
    $ans = Read-Host "Tiempo limite en minutos (Enter = sin limite)"
    if ($ans -match '^\d+$' -and [int]$ans -gt 0) {
        $TIME_BUDGET_MINS = [int]$ans
    }
}

if ($DEV_MODE) { $extraArgs += "--dev" }

Pull-Image

Write-Host "Hardware: $CPU_MODEL | ${CPU_PHYSICAL}p/${CPU_LOGICAL}l cores | ${CPU_FREQ_MHZ}MHz | ${RAM_GB_F}GB RAM"
Write-Host "Docker:   --memory ${DOCKER_MEM_GB}g --cpus $DOCKER_CPUS"

if ($HAS_NVIDIA) {
    Write-Host "-> Primera pasada: con GPU (CUDA)..."
    Run-Benchmark @extraArgs @args

    Write-Host ""
    Write-Host "-> Segunda pasada: sin GPU (CPU only)..."
    $GPU_FLAGS = @()
    Run-Benchmark -NoGpu @extraArgs @args
} elseif ($HOST_ARCH -eq "ARM64") {
    Write-Host "-> Primera pasada: build nativo arm64..."
    Run-Benchmark @extraArgs @args

    if (Test-QemuAvailable) {
        Write-Host ""
        Write-Host "-> Segunda pasada: build amd64 via emulacion..."
        Write-Host "  Descargando variante amd64..."
        & docker pull --platform linux/amd64 $IMAGE 2>$null
        if ($LASTEXITCODE -eq 0) {
            Run-Benchmark -DockerPlatform linux/amd64 -Emulated -NoGpu @extraArgs @args
        } else {
            Write-Host "  [WARN] No se pudo descargar la imagen amd64, omitiendo segunda pasada."
        }
    } else {
        Write-Host "  [INFO] Emulacion amd64 no disponible, omitiendo segunda pasada."
    }
} else {
    Run-Benchmark @extraArgs @args
}

if ($Env:KEEP_IMAGE -ne "1") {
    Write-Host "-> Removing image $IMAGE..."
    docker rmi $IMAGE 2>$null
}

Restore-Docker
