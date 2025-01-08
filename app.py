from datetime import datetime
import logging
import os
from argparse import ArgumentParser

import yaml
from webapp import create_app, socketio

os.makedirs("logs", exist_ok=True)

log_file_name = datetime.now().strftime(
    os.path.join("logs", "llmaix_%H_%M_%d_%m_%Y.log")
)
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file_name,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.debug("Starting LLM-AIx ...")

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Web app for llama-cpp')
    parser.add_argument("--model_path", type=str, default=os.getenv('MODEL_PATH', "models"), help="Path where the models are stored which llama cpp can load.")
    parser.add_argument("--server_path", type=str, default=os.getenv('SERVER_PATH', r""), help="Path to the llama server executable.")
    parser.add_argument("--port", type=int, default=int(os.getenv('PORT', 5000)), help="On which port the Web App should be available.")
    parser.add_argument("--host", type=str, default=os.getenv('HOST', "localhost"))
    parser.add_argument("--config_file", type=str, default=os.getenv('CONFIG_FILE', "config.yml"))
    parser.add_argument("--llamacpp_port", type=int, default=int(os.getenv('LLAMACPP_PORT', 2929)))
    parser.add_argument("--gpu", type=str, default=os.getenv('GPU', "all"), help="Which GPU to use?", choices=["all", "0", "1", "2", "3", "4", "5", "6", "7", "mps", "row"])
    parser.add_argument("--debug", action="store_true", default=os.getenv('DEBUG', 'false') == 'true')
    parser.add_argument("--mode", type=str, default=os.getenv('MODE', "choice"), choices=["anonymizer", "informationextraction", "choice"], help="Which mode to run")
    parser.add_argument("--disable_parallel", action="store_true", default=os.getenv('DISABLE_PARALLEL', 'false') == 'true', help="Disable parallel llama-cpp processing. If not set, the number of parallel server slot is determined by the model config file.")
    parser.add_argument("--no_parallel_preprocessing", action="store_true", default=os.getenv('NO_PARALLEL_PREPROCESSING', 'false') == 'true', help="Disable parallel preprocessing")
    parser.add_argument("--verbose_llama", action="store_true", default=os.getenv('VERBOSE_LLAMA', 'false') == 'true', help="Verbose llama cpp")
    parser.add_argument("--password", type=str, default=os.getenv('PASSWORD', ""), help="Password for password protection")
    return parser


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None
        
def is_path(string):
    # Check for directory separators
    if '/' in string or '\\' in string:
        return True

    # Check if it's an absolute path
    if os.path.isabs(string):
        return True

    # Check if the directory part of the path exists
    if os.path.dirname(string) and os.path.exists(os.path.dirname(string)):
        return True

    return False

def check_model_config(model_dir, model_config_file):
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist")
    if is_path(model_config_file):
        if not os.path.isfile(model_config_file):
            raise ValueError(f"Model config file {model_config_file} does not exist")
    else:
        if not os.path.isfile(os.path.join(model_dir, model_config_file)):
            raise ValueError(
                f"Model config file {model_config_file} does not exist in model directory {model_dir}. Did you mount the model directory to the container?")
        model_config_file = os.path.join(model_dir, model_config_file)

    model_config = load_yaml_file(model_config_file)
    if model_config is None:
        raise ValueError(f"Error loading model config file {model_config_file}")

    print("Model config loaded: ", model_config)

    # check model config yaml. Example entry:
    # - name: "llama3.1_8b_instruct_q8"
    #     display_name: "LLaMA 3.1 8B Instruct Q8_0"
    #     file_name: "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    #     model_context_size: 128000
    #     kv_cache_size: 128000
    #     kv_cache_quants: "q8_0"  # e.g. "q_8" or "q_4" - requires flash attention
    #     flash - attention: true  # does not work for some models
    #     mlock: true
    #     server_slots: 8
    #     seed: 42
    #     n_gpu_layers: 33

    for model_dict in model_config['models']:
        if 'kv_cache_size' not in model_dict:
            raise ValueError(f"Model config for {model_dict['name']} is missing 'kv_cache_size'")
        if 'server_slots' not in model_dict:
            raise ValueError(f"Model config for {model_dict['name']} is missing 'server_slots'")
        if 'file_name' not in model_dict:
            raise ValueError(f"Model config for {model_dict['name']} is missing 'file_name'")
        if 'n_gpu_layers' not in model_dict:
            raise ValueError(f"Model config for {model_dict['name']} is missing 'n_gpu_layers'")
        if 'flash_attention' not in model_dict:
            print("Flash attention not found in model config, setting to False")
            # model_dict['flash_attention'] = False
        if 'mlock' not in model_dict:
            print("Mlock not found in model config, setting to False")
            # model_dict['mlock'] = False
        if 'seed' not in model_dict:
            print("Seed not found in model config, setting to 42")
            # model_dict['seed'] = 42
        if 'kv_cache_quants' not in model_dict:
            print("KV cache quants not found in model config, setting to None")
            # model_dict['kv_cache_quants'] = None
        elif model_dict['kv_cache_quants'] == "":
            print("KV cache quants is empty, setting to f16")
            # model_dict['kv_cache_quants'] = None
        elif model_dict['kv_cache_quants'] != "" and 'flash_attention' not in model_dict:
            raise ValueError(
                f"Model config for {model_dict['name']} is missing 'flash_attention' when kv_cache_quants is set")
        elif model_dict['kv_cache_quants'] != "" and not model_dict['flash_attention']:
            raise ValueError(
                f"Model config for {model_dict['name']} has kv_cache_quants set but flash_attention is False")
        elif model_dict['kv_cache_quants'] not in ["q4_0", "q8_0", "f16", "f32", "q5_0", "q5_1", "q4_1", "iq4_nl", "bf16"]:
            raise ValueError(
                f"Model config for {model_dict['name']} has invalid kv_cache_quants value. Please use one of: q4_0, q8_0, f16, f32, q5_0, q5_1, q4_1, iq4_nl, bf16")
        if 'model_context_size' not in model_dict:
            raise ValueError(f"Model config for {model_dict['name']} is missing 'model_context_size'")
        if 'display_name' not in model_dict:
            print("Display name not found in model config, setting to file name")
            # model_dict['display_name'] = model_dict['file_name']

        model_file = os.path.join(model_dir, model_dict['file_name'])

        if not os.path.isfile(model_file):
            raise ValueError(f"Model file {model_file} not found")

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    app = create_app(auth_required=True if args.password else False, password=args.password)

    app.config["MODEL_PATH"] = args.model_path
    app.config["SERVER_PATH"] = args.server_path
    app.config["SERVER_PORT"] = args.port
    app.config["CONFIG_FILE"] = args.config_file
    app.config["GPU"] = args.gpu
    app.config["LLAMACPP_PORT"] = args.llamacpp_port
    app.config["DEBUG"] = args.debug
    app.config["NO_PARALLEL"] = args.disable_parallel
    app.config["VERBOSE_LLAMA"] = args.verbose_llama
    app.config["PARALLEL_PREPROCESSING"] = not args.no_parallel_preprocessing

    app.config["MODE"] = args.mode

    if args.gpu != "all":
        print("Using GPU " + args.gpu)

    # if model path is relative, make it absolute
    if not os.path.isabs(app.config["MODEL_PATH"]):
        app.config["MODEL_PATH"] = os.path.abspath(app.config["MODEL_PATH"])

    # if server path is relative, make it absolute
    if not os.path.isabs(app.config["SERVER_PATH"]):
        app.config["SERVER_PATH"] = os.path.abspath(app.config["SERVER_PATH"])

    check_model_config(app.config["MODEL_PATH"], app.config["CONFIG_FILE"],)

    if not os.path.isfile(app.config["SERVER_PATH"]):
        print("WARNING: Llama.cpp server executable not found in '"  + app.config["SERVER_PATH"] + "'. Did you specify --server_path correctly?")

    print("Start Webserver on http://" + args.host + ":" + str(args.port))
    if args.host == "0.0.0.0":
        print("Please use http://localhost:" + str(args.port) + " to access the web app locally or the IP / hostname of your server to access the web app in your local network.")
    if args.host != "localhost":
        print("Requires authentication")

    socketio.run(app, debug=args.debug, use_reloader=args.debug, port=args.port, host=args.host, allow_unsafe_werkzeug=True)
