# Changelog

All notable changes to LLMAIx are documented here.

## 0.4.0

**LLMAIx no longer compiles llama.cpp** — Docker images are built on the
official prebuilt `ghcr.io/ggml-org/llama.cpp` server images — and **models now
load directly from Hugging Face by default**, so no models directory is needed
to get started.

### Action required for admins

- **New image names:** `ghcr.io/katherlab/llmaix-cpu` (`linux/amd64` +
  `linux/arm64`), `ghcr.io/katherlab/llmaix-cuda` (NVIDIA, `linux/amd64`),
  `ghcr.io/katherlab/llmaix-api` (external OpenAI-compatible API only). The
  **Metal image is gone** — on macOS, Docker runs the Linux ARM64 CPU image
  without GPU acceleration; for Metal, run LLMAIx natively (see README).
- **Compose defaults changed to zero-setup mode:** `docker compose up` now uses
  the bundled `config.yml` with `google/gemma-4-E4B-it-qat-q4_0-gguf` from
  Hugging Face (~5 GB download on first use, cached in the new `llmaix_models`
  volume). To keep using your own models directory, bind-mount it to `/models`
  and set `CONFIG_FILE=/models/config.yml` (see comments in the compose files).
  For NVIDIA GPUs use the new `docker-compose-cuda.yml`.
- **`SERVER_PATH` default is now `/app/llama-server`** — update any custom
  overrides.
- **Native (non-Docker) setups:** surya OCR now needs a llama.cpp
  `llama-server` binary (`brew install llama.cpp` or set `LLAMA_CPP_BINARY`).

### Configuration

- `config.yml` can load models from Hugging Face via `hf_repo` (+ optional
  `hf_quant` / `hf_file`) instead of a local `file_name`; set `HF_TOKEN` for
  gated/private repos.

### Internal

- All Python dependencies upgraded to latest (transformers 5, torch 2.13,
  pandas 3, surya-ocr 0.22 with reworked full-page OCR); bounded,
  health-checked llama.cpp startup (`LLAMA_STARTUP_TIMEOUT`, default 600 s);
  expanded automated tests; added `.dockerignore`.
