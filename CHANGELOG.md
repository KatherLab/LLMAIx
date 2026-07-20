# Changelog

All notable changes to LLMAIx are documented here.

## 0.4.0

**LLMAIx no longer compiles llama.cpp.** Images are now built on the official
prebuilt `ghcr.io/ggml-org/llama.cpp` server images, pinned to a tested build.
The server binary lives at **`/app/llama-server`** and stays internal to the
container.

### Action required for admins

- **Metal image is gone.** The `llmaix-metal` image and `Dockerfile_metal` no
  longer exist. On macOS, Docker runs the **Linux ARM64 CPU image** — there is
  **no Metal/GPU acceleration inside Docker**. For Metal, run LLMAIx natively
  (see README, `--server_path` / `LLAMA_SERVER_PATH`).
- **New image names:**
  - `ghcr.io/katherlab/llmaix-cpu` — CPU, `linux/amd64` + `linux/arm64` (incl.
    Docker Desktop on Apple Silicon)
  - `ghcr.io/katherlab/llmaix-cuda` — NVIDIA GPU, `linux/amd64` (needs NVIDIA
    Container Toolkit)
  - `ghcr.io/katherlab/llmaix-api` — external OpenAI-compatible API only
- **Compose files:** default `docker-compose.yml` now uses the **CPU** image.
  For NVIDIA GPUs use the new **`docker-compose-cuda.yml`**
  (`docker compose -f docker-compose-cuda.yml up`).
- **`SERVER_PATH` default is now `/app/llama-server`** — update any custom
  overrides.

### New configuration possibilities

- **Load models straight from Hugging Face** — in `config.yml`, use `hf_repo`
  (+ optional `hf_quant` / `hf_file`) instead of a local `file_name`;
  llama-server downloads on demand. Set **`HF_TOKEN`** for gated/private repos.
  Downloads cache in `/models/.llama_cache` (override with `LLAMA_CACHE`).
- **Pin/upgrade llama.cpp** via the `LLAMACPP_IMAGE` build arg in
  `Dockerfile_cpu` / `Dockerfile_cuda` (bump the `server-b*` / `server-cuda-b*`
  tag).
- **Bounded, health-checked startup** — LLMAIx now polls `/health` with a
  timeout (`LLAMA_STARTUP_TIMEOUT`, default 600s), reports the model-load log on
  failure (OOM vs. model-loading vs. startup), and shuts the server down cleanly
  on exit.

### Also in this release

- All Python dependencies upgraded to latest (transformers 5, torch 2.13,
  pandas 3, etc.); Pillow is held `<11` due to `surya-ocr`.
- Expanded automated tests (Docker smoke tests, CLI-flag validation,
  startup/API integration, anonymization & redaction-metric units).
