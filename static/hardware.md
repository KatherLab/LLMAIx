# Hardware Requirements

LLMAIx is a web application to support scientific evaluation of information extraction using LLMs.

The application itself can run on almost any hardware, as it does not use a lot of resources. When you want to perform preprocessing using OCR (e.g. tesseract or surya-ocr), it is recommended to have at least a few GB of system memory and in case of surya, a GPU (a small one is okay) is beneficial.

The resource-intensive part is to run LLMs. LLMAIx either runs its own llama.cpp server or you can use any OpenAI-compatible LLM API (e.g. OpenAI, or another self-hosted LLM service providing a OpenAI-compatible API like ollama).

## LLama.cpp Hardware Requirements

Llama.cpp is a LLM inference engine which runs on a variety of hardware (including CPUs, GPUs (Nvidia, AMD and others) and also on Apple Silicon processors). While all models can also run on CPU it is recommended to run them on a GPU (or Apple Silicon).

The prompt processing throughput (how many tokens of your prompt, including the documents) can be processed per second depends on the GPU architecture and its compute performance. The token generation throughput however almost linearly scales with the (GPU) memory bandwidth.

For maximum performance your model should fit entirely into the GPU memory (can be 1 or multiple GPUs). The memory requirements can be estimated by taking the LLM weights (roughly the size of the downlaoded gguf file) plus some more space for the kv cache (depending on the type and size of it).


### Example Configurations

| Model | Quantization | KV Cache Size | KV Cache Type | GPU Memory required |
|-------|--------------|---------------|---------------|--------------------|
| Llama 3.3 70B Instruct Q4_K_M | q4_k_m | 32768 | q8_0 | 48GB |

> [!IMPORTANT]
> Please note that using a quantization less than Q8_0 might lead to slightly decreased quality. Below Q4_K_M the quality might decrease a lot.

> [!NOTE]
> The KV Cache Size determines how many tokens you can process (input & output). You can split the KV cache into multiple slots to parallely process requests.