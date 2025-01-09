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

# Clone llama.cpp and build with shared libraries
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE_LEVEL} \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release --target llama-server -j$(nproc) && \
    # Find and copy all .so files to a common directory for easier copying
    mkdir -p build/all_libs && \
    find build -name "*.so*" -exec cp {} build/all_libs/ \; && \
    # Keep only the server binary and necessary files
    find . -maxdepth 1 \( -name "llama-*" -o -name "ggml" -o -name "examples" -o -name "models" \) ! -name "llama-server" -exec rm -rf {} +

# Runtime Stage
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

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

# Copy server binary and all shared libraries
COPY --from=build /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /build/llama.cpp/build/all_libs/* /usr/local/lib/

# Configure library path and update cache
RUN ldconfig /usr/local/lib

# Set up Python environment with uv
WORKDIR /app

# Install uv and set up Python environment
ENV PYTHONPATH=/app/.venv/lib/python3.12/site-packages
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install wheel setuptools

# Copy requirements and install dependencies
COPY requirements.txt .
RUN . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Configure server host
ENV LLAMA_ARG_HOST=0.0.0.0

# Add library path to environment
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Add health check
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

EXPOSE 5000

# Update the CMD to use the virtual environment's Python
CMD [".venv/bin/python", "app.py", "--server_path", "/usr/local/bin/llama-server", "--model_path", "/models"]