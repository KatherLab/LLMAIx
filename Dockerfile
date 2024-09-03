# Use ARG for the compute level in the builder stage
ARG CUDA_VERSION="12.6.0"
ARG OS="ubuntu24.04"
ARG COMPUTE_LEVEL="86"

ARG CUDA_BUILDER_IMAGE="${CUDA_VERSION}-devel-${OS}"
ARG CUDA_RUNTIME_IMAGE="${CUDA_VERSION}-runtime-${OS}"

FROM nvidia/cuda:${CUDA_BUILDER_IMAGE} AS builder

# Install build dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    cmake \
    libpoppler-cpp-dev \
    pkg-config \
    git \
    poppler-utils \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Use ARG directly in the make command
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    CUDA_DOCKER_ARCH="compute_${COMPUTE_LEVEL}" make GGML_CUDA=1 -j 8


# Runtime Stage: Setting up the runtime environment
FROM nvidia/cuda:${CUDA_RUNTIME_IMAGE} AS runtime

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    ocrmypdf \
    tesseract-ocr-deu \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /build

# Copy the built artifacts from the builder stage
COPY --from=builder /build/llama.cpp .

# Set the working directory for the application
WORKDIR /app

# Copy the requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py", "--server_path", "/build/llama.cpp/llama-server", "--model_path", "/models"]
