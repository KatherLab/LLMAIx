[![Build and Push Docker Image](https://github.com/KatherLab/LLMAnonymizer/actions/workflows/docker-image.yml/badge.svg)](https://github.com/KatherLab/LLMAnonymizer/actions/workflows/docker-image.yml)

# LLM-AIx - Information Extraction & Anonymization

> [!Important]
> This tool is a prototype which is in active development and is still undergoing major changes. Please always check the results!
> 
> **Use for research purposes only!**

![Information Extraction](static/ie_front_image.png)

Web-based tool to extract structured information from medical reports, anonymize documents.

**Features**:

- Supports various input formats: pdf, png, jpg, jpeg, txt, csv, xlsx and docx (only if Word is installed on your system)
- Performs OCR if necessary (_tesseract_ and _surya-ocr_)
- Extracts (personal) information from medical reports in JSON format (enforced by a grammar)

**Information Extraction**:

- Structured information extraction and comparison against a ground truth. 
- Detailed metrics on label- and document-level.

![Label Annotation Report](static/image_labelannotation_report.png)

**New: Annotation Helper**:

- Speed up your annotation process by using the LLM output as a starting point and only curate the LLM output.

![Annotation Helper](static/image_annotationhelper.png)

**Anonymizer**:

- Matches the extracted personal information in the reports using a fuzzy matching algorithm based on the Levenshtein distance (configurable)
- Compare documents and calculate metrics using annotated pdf files as a ground truth ([Inception](https://inception-project.github.io/))

![Redaction View of the Tool. Side-by-side documents, left side original, right side redacted view](static/image_redaction_view.png)


## Examples

Examples of doctoral reports in various formats as well as grammar examples and annotations can be found in the `examples` directory.


## LLM Models and Model Config

LLM-AIx supports all models which are supported by llama-cpp at the time. Please download models in the **gguf** format.

In addition, create a config.yml file inside of the model directoy and configure your downloaded models according to the following example.


Example config.yml file:
```yaml
models:
  - name: "llama3.1_8b_instruct_q5km"
    display_name: "LLaMA 3.1 8B Instruct Q5_K_M"
    file_name: "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
    model_context_size: 128000 # Right now only informative.
    kv_cache_size: 16000 # Which size should the llama.cpp KV Cache have?
    kv_cache_quants: "q8_0" # e.g. "q_8" or "q_4" - requires flash attention
    flash_attention: true # does not work for some models
    mlock: true
    server_slots: 2 # How many requests should be processed in parallel. Please note: The size of each slot is kv_cache_size / server_slots!
    seed: 42 # Random initialization
    n_gpu_layers: 33 # How many layers to offload to the GPU, fastest if all are offloaded
```

## Run with Docker

1. Download/Clone this repository: `git clone https://github.com/KatherLab/LLMAIx.git`
2. Go to the repository directory: `cd LLMAIx`
3. Edit `docker-compose.yml` with the correct model path. Inside of this model path should be the _.gguf_ files as well as the adapted `config.yml` file.
4. Run the docker image: `docker-compose up` (add `-d` to run in detached mode)

Now access in your browser via `http://localhost:19999`

Alternatively to the first step, create a `docker-compose.yml` and `config.yml` file. Edit the files according to this repository, no need to download the whole repository!

### Build Docker Image

Run `docker compose build` inside of the repository.

> [!Tip]
> You can specify the compute level of your CUDA-capable GPU in the docker-compose file. 
>
> Use `86` for compute level 8.6.
>
> Look up here for your GPU: [GPU Compute Capabilites](https://developer.nvidia.com/cuda-gpus)


## Manual Setup

1. Download and extract or build [llama-cpp](https://github.com/ggerganov/llama.cpp) for your operating system.
2. Download desired models (must be compatible with llama-cpp, in gguf format)
3. Update the config.yml file with the downloaded models accordingly.
4. If you intend to use OCR: Install [OCRmyPDF](https://ocrmypdf.readthedocs.io/en/latest/installation.html#)
5. Create a python venv or a conda environment (tested with *Python 3.12.1*) with requirements.txt:
  - `python -m venv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

## Launch LLM-AIx

Run:
`python app.py`


|Parameter|Description|Example|
|---|---|---|
|--model_path|Directory with downloaded model files which can be processed by llama.cpp|/path/to/models|
|--server_path|Path of llama cpp executable (on Windows: server.exe).|/path/to/llamacpp/executable/server|
|--n_gpu_layers|How many layers of the model to offload to the GPU. Adjust according to model and GPU memory. Default: 80|-1 for all, otherwise any number|
|--host|Hostname of the server. Default: 0.0.0.0|0.0.0.0 or localhost|
|--port|Port on which this web app should be started on. Default: 5001|5001|
|--config_file|Custom path to the configuration file.|config.yml|
|--llamacpp_port|On which port to run the llama-cpp server. Default: 2929|2929|
|--debug|When set, the web app will be started in debug mode and with auto-reload, for debugging and development|
|--mode|Which mode to run (`choice` will interactively ask the user). Can be `anonymizer`, `informationextraction`, `choice`. Default: 'choice'|choice|
|--disable_parallel|Disable parallel llama.cpp processing. Default: False|False|
|--no_parallel_preprocessing|Disable parallel preprocessing. Default: False|False|
|--verbose_llama|Enable verbose logging of llama.cpp. Default: False|False|
|--no_password|Disable password protection. Default: False|False|

## Usage

View one of the tutorials:

[Information Extraction Tutorial](static/information_extraction.md)

[Anonymizer Tutorial](static/anonymization.md)


## Additional Notes

> [!NOTE] 
> An active internet connection is currently required. This is because some javascript and CSS libraries are loaded directly from CDNs. To change that please download them and replace the respective occurrences in the html files.


## Contributions

Pull requests are welcome!

## Citation

This repository is part of the paper [LLM-AIx: An open source pipeline for Information Extraction from unstructured medical text based on privacy pre-serving Large Language Models](https://doi.org/10.1101/2024.09.02.24312917)
