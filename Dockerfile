# syntax=docker/dockerfile:1

# Arguments for base image selection
ARG TARGETPLATFORM
ARG CUDA_VERSION="12.6.3"
ARG UBUNTU_VERSION="24.04"
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# First stage - Builder
FROM ${BASE_CUDA_DEV_CONTAINER} AS builder

# Install common build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libcurl4-openssl-dev \
    python3.12 \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp

WORKDIR /build/llama.cpp

# Set up build arguments
ARG GGML_CPU_ARM_ARCH=armv8-a
ARG CUDA_COMPUTE_LEVEL="86"

# Detect architecture and configure build
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        if command -v nvcc >/dev/null 2>&1; then \
            # CUDA build for AMD64
            cmake -B build \
                -DGGML_NATIVE=OFF \
                -DGGML_CUDA=ON \
                -DLLAMA_CUDA=ON \
                -DLLAMA_CURL=ON \
                -DBUILD_SHARED_LIBS=ON \
                -DCMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE_LEVEL}; \
        else \
            # CPU-only build for AMD64
            cmake -B build \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLAMA_CUDA=OFF \
                -DLLAMA_CURL=ON \
                -DGGML_NATIVE=OFF \
                -DGGML_CPU_ALL_VARIANTS=ON; \
        fi; \
    elif [ "$ARCH" = "aarch64" ]; then \
        if [ "$(uname)" = "Darwin" ]; then \
            # macOS Metal build
            cmake -B build \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLAMA_CURL=ON \
                -DLLAMA_METAL=ON \
                -DGGML_NATIVE=OFF; \
        else \
            # Linux ARM build
            cmake -B build \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLAMA_CURL=ON \
                -DGGML_NATIVE=OFF \
                -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH}; \
        fi; \
    else \
        echo "Unsupported architecture"; \
        exit 1; \
    fi && \
    cmake --build build -j$(nproc)

# Create directory for shared libraries and copy them
RUN mkdir -p build/all_libs && \
    find build -name "*.so*" -exec cp {} build/all_libs/ \; || true && \
    find build -name "*.dylib" -exec cp {} build/all_libs/ \; || true

# Clean up unnecessary files, keeping only llama-server
RUN find . -maxdepth 1 \( -name "llama-*" -o -name "ggml" -o -name "examples" -o -name "models" \) ! -name "llama-server" -exec rm -rf {} +

# Second stage - Runtime
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libcurl4-openssl-dev \
    libgomp1 \
    curl \
    ocrmypdf \
    tesseract-ocr-deu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy server binary and shared libraries
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/all_libs/* /usr/local/lib/

# Configure library path and update cache
RUN ldconfig /usr/local/lib

WORKDIR /app

ENV PYTHONPATH=/app/.venv/lib/python3.12/site-packages
ENV PATH="/root/.local/bin:$PATH"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install wheel setuptools

COPY requirements.txt .
RUN . .venv/bin/activate && \
    uv pip install -r requirements.txt

COPY . .

ENV LLAMA_ARG_HOST=0.0.0.0
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

EXPOSE 5000

CMD [".venv/bin/python", "app.py", "--server_path", "/usr/local/bin/llama-server", "--model_path", "/models"]