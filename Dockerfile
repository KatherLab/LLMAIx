ARG CUDA_VERSION="12.6.3"
ARG UBUNTU_VERSION="24.04"
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Builder Stage
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# CUDA architecture to build for
ARG CUDA_COMPUTE_LEVEL="86"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libcurl4-openssl-dev \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE_LEVEL} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release --target llama-server -j$(nproc) && \
    # Keep only the server binary and necessary files
    find . -maxdepth 1 \( -name "llama-*" -o -name "ggml" -o -name "examples" -o -name "models" \) ! -name "llama-server" -exec rm -rf {} +

# Runtime Stage
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    libcurl4-openssl-dev \
    libgomp1 \
    curl \
    ocrmypdf \
    tesseract-ocr-deu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the build directory structure
WORKDIR /build
COPY --from=build /build/llama.cpp .

# Copy the specific shared libraries
COPY --from=build /build/llama.cpp/build/ggml/src/libggml.so /libggml.so
COPY --from=build /build/llama.cpp/build/src/libllama.so /libllama.so

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application
COPY . .

# Configure server host
ENV LLAMA_ARG_HOST=0.0.0.0

# Add health check
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

EXPOSE 5000

CMD ["python", "app.py", "--server_path", "/build/llama-server", "--model_path", "/models"]