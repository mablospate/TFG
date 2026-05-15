# ── Stage 1: Rust builder — always runs natively on build machine ─────────────
FROM --platform=$BUILDPLATFORM rust:slim-bookworm AS rust-builder
ARG TARGETARCH
WORKDIR /build

# arm64: install cross-compilation toolchain; amd64: only OpenCL headers (for qcgpu)
RUN apt-get update && \
    apt-get install -y ocl-icd-opencl-dev clang pkg-config libssl-dev \
        $([ "$TARGETARCH" = "arm64" ] && echo "gcc-aarch64-linux-gnu" || true) && \
    rm -rf /var/lib/apt/lists/*

# arm64: register Rust target + configure cross-linker
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        rustup target add aarch64-unknown-linux-gnu && \
        printf '[target.aarch64-unknown-linux-gnu]\nlinker = "aarch64-linux-gnu-gcc"\n' \
            >> /usr/local/cargo/config.toml; \
    fi

COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/

# Build and collect binaries into /binaries/
# arm64 excludes qcgpu (OpenCL headers not available for cross-compile; qcgpu excluded from
# linux-aarch64-* PLATFORM_CONFIGS anyway)
RUN mkdir -p /binaries && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        cargo build --release --target aarch64-unknown-linux-gnu --workspace --exclude qcgpu && \
        cp target/aarch64-unknown-linux-gnu/release/q1tsim-grover   /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/q1tsim-shor      /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantr-grover    /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantr-shor       /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantrs2-grover  /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantrs2-shor     /binaries/; \
    else \
        cargo build --release && \
        cp target/release/q1tsim-grover   /binaries/ && \
        cp target/release/q1tsim-shor      /binaries/ && \
        cp target/release/quantr-grover    /binaries/ && \
        cp target/release/quantr-shor       /binaries/ && \
        cp target/release/quantrs2-grover  /binaries/ && \
        cp target/release/quantrs2-shor     /binaries/ && \
        cp target/release/qcgpu-grover     /binaries/ && \
        cp target/release/qcgpu-shor        /binaries/; \
    fi

# ── Stage 2: CPU runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS cpu
ARG TARGETARCH
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /binaries/ ./bin/
ENV PATH="/app/bin:$PATH"

# Install uv
RUN pip install --no-cache-dir uv

# Python deps: x86only extra for amd64 (projectq + cudaq), base only for arm64
COPY pyproject.toml uv.lock ./
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        uv sync --no-dev --extra x86only; \
    else \
        uv sync --no-dev; \
    fi

COPY python/ ./python/
COPY run.py ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/app/results"]
LABEL org.opencontainers.image.source="https://github.com/mablospate/TFG"

# DOCKER_IMAGE env var lets run.py record which image produced the results
ARG DOCKER_IMAGE_TAG=dev
ENV DOCKER_IMAGE=${DOCKER_IMAGE_TAG}

ENTRYPOINT ["/entrypoint.sh"]

# ── Stage 3: CUDA runtime ─────────────────────────────────────────────────────
# Builds on CUDA base; adds cuda-quantum-cu13 on top of x86only deps.
# Use: docker build --target cuda ...
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS cuda
ARG TARGETARCH=amd64
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 curl python3.12 python3.12-dev python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY --from=rust-builder /binaries/ ./bin/
ENV PATH="/app/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra x86only --extra gpu

COPY python/ ./python/
COPY run.py ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG DOCKER_IMAGE_TAG=dev
ENV DOCKER_IMAGE=${DOCKER_IMAGE_TAG}

VOLUME ["/app/results"]
ENTRYPOINT ["/entrypoint.sh"]
