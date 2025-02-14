# Hardware Requirements

LLMAIx is a web application to support scientific evaluation of information extraction using LLMs.

The application itself can run on almost any hardware, as it does not use a lot of resources. When you want to perform preprocessing using OCR (e.g. tesseract or surya-ocr), it is recommended to have at least a few GB of system memory and in case of surya, a GPU (a small one is okay) is beneficial.

The resource-intensive part is to run LLMs. LLMAIx either runs its own llama.cpp server or you can use any OpenAI-compatible LLM API (e.g. OpenAI, or another self-hosted centrally provided LLM service providing a OpenAI-compatible API).

## LLama.cpp Hardware Requirements

Llama.cpp is a LLM inference engine which runs on a variety of hardware (including CPUs, GPUs (Nvidia, AMD and others) and also on Apple Silicon processors). 