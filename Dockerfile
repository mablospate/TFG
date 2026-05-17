# syntax=docker/dockerfile:1
# Global ARG — injected by buildx per platform; also used for FROM substitution
ARG TARGETARCH

# ── Stage 0: qcgpu — always built natively for linux/amd64 ───────────────────
# Runs via QEMU when host is ARM. Only the amd64 runtime stage uses these bins.
FROM --platform=linux/amd64 rust:slim-bookworm AS qcgpu-amd64
WORKDIR /build
RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-opencl-dev clang pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/
RUN mkdir -p /qcgpu-bins && \
    cargo build --release -p qcgpu-bench && \
    cp target/release/qcgpu-grover /qcgpu-bins/ && \
    cp target/release/qcgpu-shor   /qcgpu-bins/

# ── Stage 1: Rust builder — always runs natively on build machine ─────────────
FROM --platform=$BUILDPLATFORM rust:slim-bookworm AS rust-builder
ARG TARGETARCH
# Limit parallel codegen units to reduce peak memory during compilation
ENV CARGO_BUILD_JOBS=4
WORKDIR /build

# Install cross-compilation toolchain for the target architecture when cross-compiling
RUN apt-get update && \
    apt-get install -y ocl-icd-opencl-dev clang pkg-config libssl-dev \
        $([ "$TARGETARCH" = "arm64" ] && echo "gcc-aarch64-linux-gnu" || true) \
        $([ "$TARGETARCH" = "amd64" ] && [ "$(uname -m)" != "x86_64" ] && echo "gcc-x86-64-linux-gnu" || true) && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        dpkg --add-architecture arm64 && \
        apt-get update && \
        apt-get install -y libssl-dev:arm64 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# arm64: register Rust target + configure cross-linker + pkg-config sysroot for openssl-sys
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        rustup target add aarch64-unknown-linux-gnu && \
        printf '[target.aarch64-unknown-linux-gnu]\nlinker = "aarch64-linux-gnu-gcc"\n' \
            >> /usr/local/cargo/config.toml; \
    fi

# amd64: register Rust target + configure cross-linker when building on arm64 host
RUN if [ "$TARGETARCH" = "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then \
        rustup target add x86_64-unknown-linux-gnu && \
        printf '[target.x86_64-unknown-linux-gnu]\nlinker = "x86_64-linux-gnu-gcc"\n' \
            >> /usr/local/cargo/config.toml; \
    fi

ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH_aarch64_unknown_linux_gnu=/usr/lib/aarch64-linux-gnu/pkgconfig
ENV OPENSSL_DIR_aarch64_unknown_linux_gnu=/usr
ENV OPENSSL_INCLUDE_DIR_aarch64_unknown_linux_gnu=/usr/include/aarch64-linux-gnu

COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/

# Build and collect binaries into /binaries/
# arm64 excludes qcgpu (OpenCL headers not available for cross-compile)
# amd64 cross-compiled from arm64 host also excludes qcgpu (no OpenCL headers for foreign arch)
RUN mkdir -p /binaries && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        cargo build --release --target aarch64-unknown-linux-gnu --workspace --exclude qcgpu-bench && \
        cp target/aarch64-unknown-linux-gnu/release/q1tsim-grover   /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/q1tsim-shor      /binaries/ && \
        find target/aarch64-unknown-linux-gnu/release -maxdepth 2 -name 'libq1tsim*.so' -exec cp {} /binaries/ \; && \
        cp target/aarch64-unknown-linux-gnu/release/quantr-grover    /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantr-shor       /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantrs2-grover  /binaries/ && \
        cp target/aarch64-unknown-linux-gnu/release/quantrs2-shor     /binaries/; \
    elif [ "$TARGETARCH" = "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then \
        cargo build --release --target x86_64-unknown-linux-gnu --workspace --exclude qcgpu-bench && \
        cp target/x86_64-unknown-linux-gnu/release/q1tsim-grover   /binaries/ && \
        cp target/x86_64-unknown-linux-gnu/release/q1tsim-shor      /binaries/ && \
        find target/x86_64-unknown-linux-gnu/release -maxdepth 2 -name 'libq1tsim*.so' -exec cp {} /binaries/ \; && \
        cp target/x86_64-unknown-linux-gnu/release/quantr-grover    /binaries/ && \
        cp target/x86_64-unknown-linux-gnu/release/quantr-shor       /binaries/ && \
        cp target/x86_64-unknown-linux-gnu/release/quantrs2-grover  /binaries/ && \
        cp target/x86_64-unknown-linux-gnu/release/quantrs2-shor     /binaries/; \
    else \
        cargo build --release && \
        cp target/release/q1tsim-grover   /binaries/ && \
        cp target/release/q1tsim-shor      /binaries/ && \
        find target/release -maxdepth 2 -name 'libq1tsim*.so' -exec cp {} /binaries/ \; && \
        cp target/release/quantr-grover    /binaries/ && \
        cp target/release/quantr-shor       /binaries/ && \
        cp target/release/quantrs2-grover  /binaries/ && \
        cp target/release/quantrs2-shor     /binaries/ && \
        cp target/release/qcgpu-grover     /binaries/ && \
        cp target/release/qcgpu-shor        /binaries/; \
    fi

# When cross-compiling to amd64 from an ARM host, supplement with qcgpu binaries
# built natively in the qcgpu-amd64 stage (via QEMU).
COPY --from=qcgpu-amd64 /qcgpu-bins/ /tmp/qcgpu-bins/
RUN if [ "$TARGETARCH" = "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then \
        cp /tmp/qcgpu-bins/qcgpu-grover /binaries/ && \
        cp /tmp/qcgpu-bins/qcgpu-shor   /binaries/; \
    fi

# ── Stage 2a: amd64 base — CUDA runtime (enables cudaq-nvidia + qcgpu OpenCL) ─
FROM --platform=linux/amd64 nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base-amd64

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y libgomp1 python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# ── Stage 2b: arm64 base — Python slim (CPU only; no CUDA wheels for arm64) ───
FROM --platform=linux/arm64 python:3.12-slim-bookworm AS base-arm64

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# ── Stage 3: Python dependencies — runs in parallel with rust-builder ─────────
FROM base-${TARGETARCH} AS python-deps
ARG TARGETARCH
WORKDIR /app

COPY pyproject.toml uv.lock ./
# amd64: install dependency extras only; project source is copied in runtime
# arm64: base dependencies only
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        uv sync --no-dev --no-install-project --extra x86only --extra gpu --python python3.12; \
    else \
        uv sync --no-dev --no-install-project; \
    fi

# ── Stage 4: runtime — assembles Python venv + Rust binaries ─────────────────
FROM base-${TARGETARCH} AS runtime
ARG TARGETARCH
WORKDIR /app

COPY --from=python-deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./

COPY --from=rust-builder /binaries/ ./bin/
ENV PATH="/app/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/bin"
RUN echo "/app/bin" > /etc/ld.so.conf.d/rust-bins.conf && ldconfig

COPY python/ ./python/
COPY run.py ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/app/results"]
LABEL org.opencontainers.image.source="https://github.com/mablospate/TFG"

ARG DOCKER_IMAGE_TAG=dev
ENV DOCKER_IMAGE=${DOCKER_IMAGE_TAG}

ENTRYPOINT ["/entrypoint.sh"]
