"""End-to-end integration test against a real llama-server + small GGUF model.

This exercises the same code path LLMAIx uses in production: build the
llama-server command, launch the process, wait for the /health endpoint to
report readiness, then call the OpenAI-compatible /v1/chat/completions endpoint
that ``fetch_chat_result`` uses and confirm llama.cpp produces a response.

The test is skipped unless a llama-server binary is available, so it is a no-op
in the default CI matrix and only runs where a binary (and network access to
Hugging Face, or a local model) is present:

    LLAMA_SERVER_PATH=/app/llama-server \
        python -m pytest webapp/llm_processing/tests/test_api_integration.py

By default it loads a tiny model straight from Hugging Face via ``-hf`` so no
local file is required. Override the model with the env vars below, or point
LLMAIX_TEST_MODEL at a local .gguf to run fully offline.
"""

import os
import shutil
import socket
import unittest

import requests

from webapp.llm_processing.tests.test_server_lifecycle import make_job

# A very small GGUF that loads quickly on CPU. Overridable for offline runs.
HF_REPO = os.getenv("LLMAIX_TEST_HF_REPO", "ggml-org/gemma-3-270m-GGUF")
HF_QUANT = os.getenv("LLMAIX_TEST_HF_QUANT", "Q8_0")
LOCAL_MODEL = os.getenv("LLMAIX_TEST_MODEL", "")  # path to a local .gguf, optional


def find_llama_server():
    return (
        os.getenv("LLAMA_SERVER_PATH")
        or os.getenv("SERVER_PATH")
        or shutil.which("llama-server")
        or ("/app/llama-server" if os.path.exists("/app/llama-server") else None)
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@unittest.skipUnless(find_llama_server(), "llama-server binary not available")
class TestApiIntegration(unittest.TestCase):
    proc = None
    job = None

    @classmethod
    def setUpClass(cls):
        server = find_llama_server()
        port = free_port()
        if LOCAL_MODEL:
            cls.job = make_job(
                server_path=server, llamacpp_port=port, n_gpu_layers=0,
                flash_attention=False, kv_cache_type="", mlock=False,
                model_path=os.path.dirname(LOCAL_MODEL) or ".",
                model_name=os.path.basename(LOCAL_MODEL),
            )
            command = cls.job.build_server_command(model_path=LOCAL_MODEL)
        else:
            cls.job = make_job(
                server_path=server, llamacpp_port=port, n_gpu_layers=0,
                flash_attention=False, kv_cache_type="", mlock=False,
                hf_repo=HF_REPO, hf_quant=HF_QUANT,
            )
            command = cls.job.build_server_command()

        import subprocess
        cls.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            # Model download (first run) can be slow; allow a generous timeout.
            cls.job.server_startup_timeout = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "600"))
            cls.job._await_server_ready(cls.proc)
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.proc is not None:
            cls.job._shutdown_server(cls.proc)
            cls.proc = None

    def test_health_reports_ok(self):
        r = requests.get(f"http://localhost:{self.job.llamacpp_port}/health", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    def test_chat_completion_returns_content(self):
        # Mirrors CancellableJob.fetch_chat_result's request shape.
        r = requests.post(
            f"http://localhost:{self.job.llamacpp_port}/v1/chat/completions",
            headers={"Authorization": "Bearer no-key"},
            json={
                "model": "llmaix",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with the single word: pong"},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "max_tokens": 16,
            },
            timeout=120,
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        content = body["choices"][-1]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertTrue(content.strip(), "model returned empty content")
        self.assertIn(body["choices"][-1]["finish_reason"], {"stop", "length"})

    def test_models_endpoint_lists_a_model(self):
        r = requests.get(f"http://localhost:{self.job.llamacpp_port}/v1/models", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("data"), "no models reported by /v1/models")


if __name__ == "__main__":
    unittest.main()
