ARG CUDA_VERSION="12.6.0"
ARG OS="ubuntu24.04"
ARG COMPUTE_LEVEL="86"

ARG CUDA_BUILDER_IMAGE="${CUDA_VERSION}-devel-${OS}"
ARG CUDA_RUNTIME_IMAGE="${CUDA_VERSION}-runtime-${OS}"


FROM nvidia/cuda:${CUDA_BUILDER_IMAGE} AS builder

# Install system dependencies
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

# Clone and build llama.cpp - adjust the compute level to your GPU
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    CUDA_DOCKER_ARCH=compute_${COMPUTE_LEVEL} make GGML_CUDA=1 -j 8


FROM nvidia/cuda:${CUDA_RUNTIME_IMAGE} AS runtime

RUN apt update && \
    apt install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    ocrmypdf \
    tesseract-ocr-deu \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY --from=builder /build/llama.cpp .

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py", "--server_path", "/build/llama.cpp/llama-server", "--model_path", "/models"]
