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
        default=r"D:\LLMAnonymizer\llama-b2808-bin-win-cuda-cu12.2.0-x64\server.exe",
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
    parser.add_argument("--n_predict", type=int, default=384)
    parser.add_argument("--llamacpp_port", type=int, default=2929)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    app.config["MODEL_PATH"] = args.model_path
    app.config["SERVER_PATH"] = args.server_path
    app.config["SERVER_PORT"] = args.port
    app.config["CONFIG_FILE"] = args.config_file
    app.config["N_GPU_LAYERS"] = args.n_gpu_layers
    app.config["N_PREDICT"] = args.n_predict
    app.config["LLAMACPP_PORT"] = args.llamacpp_port
    app.config["DEBUG"] = args.debug

    print("Start Server on http://" + args.host + ":" + str(args.port))

    socketio.run(app, debug=args.debug, use_reloader=args.debug, port=args.port)
