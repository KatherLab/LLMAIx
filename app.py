from datetime import datetime
import logging
import os
from argparse import ArgumentParser
from webapp import create_app, socketio

app = create_app()

os.makedirs("logs", exist_ok=True)

log_file_name = datetime.now().strftime(
    os.path.join("logs", "llmanonymizer_%H_%M_%d_%m_%Y.log")
)
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file_name,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.debug("Start LLM Anonymizer")

if __name__ == "__main__":
    parser = ArgumentParser(description="Parameters to run the KatherLab LLM Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"D:\LLM-Pipeline\models",
        help="Path where the models are stored which llama cpp can load.",
    )
    parser.add_argument(
        "--server_path",
        type=str,
        default=r"D:\LLMAnonymizer\llama-b3177-bin-win-cuda-cu12.2.0-x64\llama-server.exe",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="On which port the Web App should be available.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--config_file", type=str, default="config.yml")
    parser.add_argument("--n_gpu_layers", type=int, default=80)
    parser.add_argument("--llamacpp_port", type=int, default=2929)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, default="choice", choices=["anonymizer", "informationextraction", "choice"], help="Which mode to run")
    parser.add_argument("--enable-parallel", action="store_true", help="Parallel llama-cpp processing.")
    parser.add_argument("--parallel-slots", type=int, default=1, help="Number of parallel slots for llama processing")
    parser.add_argument("--context-size", type=int, default=-1, help="Set custom context size for llama cpp")
    parser.add_argument("--verbose-llama", action="store_true", help="Verbose llama cpp")

    args = parser.parse_args()

    app.config["MODEL_PATH"] = args.model_path
    app.config["SERVER_PATH"] = args.server_path
    app.config["SERVER_PORT"] = args.port
    app.config["CONFIG_FILE"] = args.config_file
    app.config["N_GPU_LAYERS"] = args.n_gpu_layers
    app.config["LLAMACPP_PORT"] = args.llamacpp_port
    app.config["DEBUG"] = args.debug
    app.config["NO_PARALLEL"] = not args.enable_parallel
    app.config["PARALLEL_SLOTS"] = args.parallel_slots
    app.config["CTX_SIZE"] = args.context_size
    app.config["VERBOSE_LLAMA"] = args.verbose_llama

    app.config["MODE"] = args.mode

    print("Start Server on http://" + args.host + ":" + str(args.port))

    socketio.run(app, debug=args.debug, use_reloader=args.debug, port=args.port)
