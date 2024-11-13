# Use ARG for the compute level in the builder stage
ARG CUDA_VERSION="12.6.2"
ARG OS="ubuntu24.04"
ARG COMPUTE_LEVEL="86"

ARG CUDA_BUILDER_IMAGE="${CUDA_VERSION}-devel-${OS}"
ARG CUDA_RUNTIME_IMAGE="${CUDA_VERSION}-runtime-${OS}"

# Builder Stage: Compiling and building
FROM nvidia/cuda:${CUDA_BUILDER_IMAGE} AS builder

# Set the compute level as an environment variable to be used later
ARG COMPUTE_LEVEL
ENV COMPUTE_LEVEL=${COMPUTE_LEVEL}

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

# Clone and build the project
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    # Echo the value to debug the variable substitution
    echo "Using compute level: compute_${COMPUTE_LEVEL}" && \
    CUDA_DOCKER_ARCH="compute_${COMPUTE_LEVEL}" make GGML_CUDA=1 -j 8

RUN find . -maxdepth 1 \( -name "llama-*" -o -name "ggml" -o -name "examples" -o -name "models" \) ! -name "llama-server" -exec rm -rf {} +

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
CMD ["python", "app.py", "--server_path", "/build/llama-server", "--model_path", "/models"]
