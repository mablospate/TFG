#!/usr/bin/env bash
set -e

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  ARCH_STR="x86_64" ;;
  aarch64) ARCH_STR="aarch64" ;;
  *)       ARCH_STR="x86_64" ;;
esac

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    PLATFORM="linux-${ARCH_STR}-nvidia"
    echo "GPU: NVIDIA detectada → plataforma: $PLATFORM"
else
    PLATFORM="linux-${ARCH_STR}-cpu"
    echo "GPU: no detectada → plataforma: $PLATFORM"
fi

for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        exec uv run --frozen python -u -m python.smoke_test
    fi
done

exec uv run --frozen python -u run.py --platform "$PLATFORM" "$@"
