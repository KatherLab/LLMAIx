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

WORKDIR /app

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp .

# Configure and build with cmake
RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE_LEVEL} \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release --target llama-server -j$(nproc) && \
    mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

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

# Copy libraries and server binary
COPY --from=build /app/lib/ /
COPY --from=build /app/build/bin/llama-server /llama-server

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

CMD ["python", "app.py", "--server_path", "/llama-server", "--model_path", "/models"]