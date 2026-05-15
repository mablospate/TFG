param([Parameter(ValueFromRemainingArguments)]$PassArgs)

$Image = $env:BENCHMARK_IMAGE ?? "ghcr.io/mablospate/tfg-bench"
$Vol   = "-v ${PWD}\results:/app/results"

function Probe-ContainerGpu {
    docker run --gpus all --rm --entrypoint nvidia-smi $Image 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Install-NvidiaContainerToolkit {
    Write-Host "Configurando soporte GPU en WSL2 (puede pedir contraseña)..."
    wsl --update
    wsl -- bash -c @'
set -e
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -qq && sudo apt-get install -y -q nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
'@
}

New-Item -ItemType Directory -Force results | Out-Null

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$hostHasGpu = $nvidiaSmi -and (nvidia-smi 2>$null; $LASTEXITCODE -eq 0)

if ($hostHasGpu) {
    if (-not (Probe-ContainerGpu)) {
        Install-NvidiaContainerToolkit
        if (-not (Probe-ContainerGpu)) {
            Write-Warning "Configuración GPU fallida — ejecutando solo CPU"
            $hostHasGpu = $false
        }
    }
}

if ($hostHasGpu) {
    Write-Host "GPU NVIDIA detectada."
    Write-Host "Dos pasadas: CPU y GPU (~2x tiempo). ¡Gracias!"
    docker run --rm -it $Vol $Image @PassArgs
    docker run --rm -it --gpus all $Vol $Image @PassArgs
} else {
    docker run --rm -it $Vol $Image @PassArgs
}
