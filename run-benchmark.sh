#!/usr/bin/env bash
set -e
IMAGE="${BENCHMARK_IMAGE:-ghcr.io/mablospate/tfg-bench}"
VOL="-v $(pwd)/results:/app/results"

_run()       { docker run --rm -it $1 $VOL "$IMAGE" "${@:2}"; }
_probe_gpu() { docker run --gpus all --rm --entrypoint nvidia-smi "$IMAGE" &>/dev/null 2>&1; }

mkdir -p results

if command -v nvidia-smi &>/dev/null 2>&1 && nvidia-smi &>/dev/null 2>&1; then
    if _probe_gpu; then
        echo "GPU NVIDIA detectada."
        echo "Se ejecutarán dos pasadas: CPU y GPU (~2x tiempo). ¡Gracias!"
        echo ""
        _run "" "$@"
        _run "--gpus all" "$@"
    else
        echo "GPU NVIDIA en el host pero sin acceso desde Docker."
        echo "Instala nvidia-container-toolkit y vuelve a intentarlo, o ejecuta solo CPU."
        _run "" "$@"
    fi
else
    _run "" "$@"
fi
