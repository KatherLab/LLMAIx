# Use the official Python image from the Docker Hub
# FROM python:3.12-slim
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    ocrmypdf \
    tesseract-ocr-deu \
    build-essential \
    cmake \
    libpoppler-cpp-dev \
    pkg-config \
    git \
    poppler-utils \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    CUDA_DOCKER_ARCH=compute_89 make GGML_CUDA=1


# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Hopefully with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py", "--server_path", "/build/llama.cpp/llama-server", "--model_path", "/models"]
