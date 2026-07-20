"""Validation for the llama-server CLI arguments LLMAIx generates.

These tests guard against silent breakage when the pinned llama.cpp build
changes flag semantics (e.g. b10068 made -fa/--flash-attn take a value, so a
bare "-fa" aborts server startup).

When a llama-server binary is available (LLAMA_SERVER_PATH / SERVER_PATH env or
/app/llama-server, as in the Docker images) the generated flags are also
checked against that binary's actual ``--help`` output.
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from webapp.llm_processing.routes import CancellableJob

# Flags LLMAIx is allowed to generate, verified against the pinned
# llama.cpp server build (server-b10068) `llama-server --help`.
KNOWN_FLAGS = {
    "--model", "--ctx-size", "--n-gpu-layers", "--port", "--metrics",
    "-np", "-b", "-ub", "-t", "--seed", "--verbose", "--mlock",
    "-ctk", "-ctv", "-sm", "-mg", "-fa",
    "-hf", "-hff", "-hft",
}


def make_job(**overrides) -> CancellableJob:
    defaults = dict(
        df=pd.DataFrame(),
        model_name="model.gguf",
        api_model=False,
        prompt="",
        symptoms=[],
        temperature=0.0,
        grammar="",
        json_schema="",
        model_path="/models",
        server_path="/app/llama-server",
        ctx_size=4096,
        n_gpu_layers=99,
        n_predict=256,
        job_id=1,
        zip_file_path="",
        llamacpp_port=2929,
    )
    defaults.update(overrides)
    return CancellableJob(**defaults)


def flags_in(command) -> list[str]:
    return [tok for tok in command if isinstance(tok, str) and tok.startswith("-")]


def find_llama_server():
    return (
        os.getenv("LLAMA_SERVER_PATH")
        or os.getenv("SERVER_PATH")
        or shutil.which("llama-server")
        or ("/app/llama-server" if os.path.exists("/app/llama-server") else None)
    )


class TestServerCommand(unittest.TestCase):
    MODEL = Path("/models/model.gguf")

    def test_flash_attention_flag_takes_value(self):
        # b10068+ requires -fa to be followed by on/off/auto; a bare -fa makes
        # the server exit with "expected value for argument".
        cmd = make_job(flash_attention=True).build_server_command(self.MODEL)
        self.assertIn("-fa", cmd)
        idx = cmd.index("-fa")
        self.assertLess(idx + 1, len(cmd), "-fa must not be the trailing argument")
        self.assertIn(cmd[idx + 1], {"on", "off", "auto"})

    def test_no_flash_attention_flag_when_disabled(self):
        cmd = make_job(flash_attention=False).build_server_command(self.MODEL)
        self.assertNotIn("-fa", cmd)

    def test_all_generated_flags_are_known(self):
        # Exercise the flag-producing options together (gpu="0" yields -sm/-mg).
        cmd = make_job(
            flash_attention=True,
            verbose_llama=True,
            mlock=True,
            kv_cache_type="q8_0",
            gpu="0",
        ).build_server_command(self.MODEL)
        for flag in flags_in(cmd):
            self.assertIn(flag, KNOWN_FLAGS, f"Unexpected llama-server flag generated: {flag}")

    def test_hf_repo_with_quant(self):
        cmd = make_job(hf_repo="ggml-org/gemma-3-4b-it-GGUF", hf_quant="Q4_K_M").build_server_command()
        self.assertIn("-hf", cmd)
        self.assertEqual(cmd[cmd.index("-hf") + 1], "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")
        # HF loading must not also pass a local --model.
        self.assertNotIn("--model", cmd)

    def test_hf_repo_quant_not_double_appended(self):
        # If the repo already carries a :quant, hf_quant must not be re-appended.
        cmd = make_job(hf_repo="ggml-org/gemma-3-4b-it-GGUF:Q8_0", hf_quant="Q4_K_M").build_server_command()
        self.assertEqual(cmd[cmd.index("-hf") + 1], "ggml-org/gemma-3-4b-it-GGUF:Q8_0")

    def test_hf_file_overrides_quant(self):
        cmd = make_job(
            hf_repo="ggml-org/gemma-3-4b-it-GGUF",
            hf_file="gemma-3-4b-it-Q4_K_M.gguf",
        ).build_server_command()
        self.assertIn("-hff", cmd)
        self.assertEqual(cmd[cmd.index("-hff") + 1], "gemma-3-4b-it-Q4_K_M.gguf")

    def test_local_model_still_uses_model_flag(self):
        cmd = make_job().build_server_command(self.MODEL)
        self.assertIn("--model", cmd)
        self.assertNotIn("-hf", cmd)

    def test_generated_flags_exist_in_pinned_help(self):
        server = find_llama_server()
        if not server:
            self.skipTest("llama-server binary not available; set LLAMA_SERVER_PATH to enable")
        result = subprocess.run([server, "--help"], capture_output=True, text=True)
        help_text = result.stdout + result.stderr
        local_cmd = make_job(
            flash_attention=True,
            verbose_llama=True,
            mlock=True,
            kv_cache_type="q8_0",
            gpu="0",
        ).build_server_command(self.MODEL)
        hf_cmd = make_job(
            hf_repo="ggml-org/gemma-3-4b-it-GGUF",
            hf_quant="Q4_K_M",
            hf_file="x.gguf",
            hf_token="t",
        ).build_server_command()
        for flag in set(flags_in(local_cmd)) | set(flags_in(hf_cmd)):
            self.assertIn(flag, help_text, f"Flag {flag} not found in llama-server --help")


if __name__ == "__main__":
    unittest.main()
