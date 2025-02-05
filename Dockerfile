# syntax=docker/dockerfile:1

ARG TARGETARCH
ARG CUDA_VERSION="12.6.3"
ARG UBUNTU_VERSION="24.04"

# Select base image based on architecture
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder-amd64
FROM ubuntu:${UBUNTU_VERSION} AS builder-arm64
FROM ${TARGETARCH:+builder-${TARGETARCH}} AS builder

# Common build arguments
ARG TARGETARCH
ARG GGML_CPU_ARM_ARCH=arm64
ARG CUDA_COMPUTE_LEVEL="86"

# Install build dependencies
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
RUN git clone https://github.com/ggerganov/llama.cpp
WORKDIR /build/llama.cpp

# Configure build based on architecture
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        cmake -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_CUDA=ON \
            -DLLAMA_CURL=ON \
            -DGGML_CUDA=ON \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE_LEVEL}; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        cmake -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_CURL=ON \
            -DGGML_NATIVE=OFF \
            -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH}; \
    fi && \
    cmake --build build -j$(nproc)

RUN mkdir -p build/all_libs && \
    find build -name "*.so*" -exec cp {} build/all_libs/ \; || true && \
    find build -name "*.dylib" -exec cp {} build/all_libs/ \; || true && \
    find . -maxdepth 1 \( -name "llama-*" -o -name "ggml" -o -name "examples" -o -name "models" \) ! -name "llama-server" -exec rm -rf {} +

# Runtime stage - select base image based on architecture
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime-amd64
FROM ubuntu:${UBUNTU_VERSION} AS runtime-arm64
FROM ${TARGETARCH:+runtime-${TARGETARCH}}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pluggy \
    libcurl4-openssl-dev \
    libgomp1 \
    curl \
    tesseract-ocr-deu \
    ocrmypdf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/all_libs/* /usr/local/lib/
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