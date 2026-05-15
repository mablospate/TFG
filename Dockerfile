# ── Stage 1: Rust builder ────────────────────────────────────────────────────
FROM rust:1.85-slim-bookworm AS rust-builder
WORKDIR /build
# OpenCL headers for qcgpu compilation (runtime OpenCL provided by NVIDIA driver)
RUN apt-get update && apt-get install -y ocl-icd-opencl-dev clang pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/
RUN cargo build --release

# ── Stage 2: CPU runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS cpu
ARG TARGETARCH
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Rust binaries
COPY --from=rust-builder /build/target/release/q1tsim-grover    ./bin/
COPY --from=rust-builder /build/target/release/q1tsim-shor      ./bin/
COPY --from=rust-builder /build/target/release/quantr-grover    ./bin/
COPY --from=rust-builder /build/target/release/quantr-shor      ./bin/
COPY --from=rust-builder /build/target/release/quantrs2-grover  ./bin/
COPY --from=rust-builder /build/target/release/quantrs2-shor    ./bin/
COPY --from=rust-builder /build/target/release/qcgpu-grover     ./bin/
COPY --from=rust-builder /build/target/release/qcgpu-shor       ./bin/

ENV PATH="/app/bin:$PATH"

# Install uv
RUN pip install --no-cache-dir uv

# Python deps: x86only extra for amd64 (projectq + cudaq), base only for arm64
COPY pyproject.toml uv.lock ./
RUN if [ "$TARGETARCH" = "amd64" ] || [ "$(uname -m)" = "x86_64" ]; then \
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

COPY --from=rust-builder /build/target/release/q1tsim-grover    ./bin/
COPY --from=rust-builder /build/target/release/q1tsim-shor      ./bin/
COPY --from=rust-builder /build/target/release/quantr-grover    ./bin/
COPY --from=rust-builder /build/target/release/quantr-shor      ./bin/
COPY --from=rust-builder /build/target/release/quantrs2-grover  ./bin/
COPY --from=rust-builder /build/target/release/quantrs2-shor    ./bin/
COPY --from=rust-builder /build/target/release/qcgpu-grover     ./bin/
COPY --from=rust-builder /build/target/release/qcgpu-shor       ./bin/
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
