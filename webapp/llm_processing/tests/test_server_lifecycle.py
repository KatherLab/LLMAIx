"""Tests for llama-server startup, readiness, shutdown and error handling.

These cover the lifecycle logic added when LLMAIx moved to the official
prebuilt llama.cpp server images: readiness is decided by the /health endpoint
(not merely by the process being alive), startup failures are classified and
reported with the captured server log, and the child process is shut down
cleanly. All tests here are hermetic - the llama-server process and its HTTP
health endpoint are mocked, so no binary or model is required.
"""

import subprocess
import tempfile
import unittest
from unittest import mock

import pandas as pd
import requests

from webapp.llm_processing import routes
from webapp.llm_processing.routes import CancellableJob


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


class TestClassifyStartupFailure(unittest.TestCase):
    def test_out_of_memory(self):
        for log in ("ggml_cuda: CUDA error: out of memory",
                    "cudaMalloc failed",
                    "failed to allocate buffer"):
            self.assertEqual(
                CancellableJob._classify_startup_failure(log), "out-of-memory", log
            )

    def test_model_loading(self):
        for log in ("error loading model architecture",
                    "failed to load model",
                    "gguf_init_from_file failed",
                    "No such file or directory"):
            self.assertEqual(
                CancellableJob._classify_startup_failure(log), "model-loading", log
            )

    def test_generic_startup(self):
        self.assertEqual(
            CancellableJob._classify_startup_failure("some unrelated crash"), "startup"
        )


class TestReadServerLog(unittest.TestCase):
    def test_returns_tail(self):
        job = make_job()
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as f:
            f.write("A" * 100 + "TAIL")
            job._server_log = f
        self.assertTrue(job._read_server_log(max_chars=4).endswith("TAIL"))
        self.assertEqual(len(job._read_server_log(max_chars=4)), 4)

    def test_no_log_returns_empty(self):
        self.assertEqual(make_job()._read_server_log(), "")


class TestShutdownServer(unittest.TestCase):
    def test_terminates_running_process(self):
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        CancellableJob._shutdown_server(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    def test_escalates_to_kill_on_timeout(self):
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="llama-server", timeout=10), 0]
        CancellableJob._shutdown_server(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_noop_when_already_exited(self):
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = 0
        CancellableJob._shutdown_server(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()


class TestAwaitServerReady(unittest.TestCase):
    def _proc(self, poll_value):
        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = poll_value
        proc.wait.return_value = 0
        return proc

    def test_ready_when_health_ok(self):
        job = make_job(server_startup_timeout=30)
        proc = self._proc(None)
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"status": "ok"}
        with mock.patch.object(routes.requests, "get", return_value=resp), \
                mock.patch.object(routes.time, "sleep"):
            self.assertIsNone(job._await_server_ready(proc))

    def test_keeps_polling_until_ok(self):
        job = make_job(server_startup_timeout=30)
        proc = self._proc(None)
        loading = mock.Mock(status_code=503)
        loading.json.return_value = {"status": "loading model"}
        ready = mock.Mock(status_code=200)
        ready.json.return_value = {"status": "ok"}
        with mock.patch.object(
            routes.requests, "get",
            side_effect=[requests.exceptions.ConnectionError(), loading, ready],
        ), mock.patch.object(routes.time, "sleep"):
            job._await_server_ready(proc)

    def test_raises_when_process_dies(self):
        job = make_job(server_startup_timeout=30)
        proc = self._proc(1)  # exited with code 1
        with mock.patch.object(routes.time, "sleep"), \
                mock.patch.object(routes, "warning_job"), \
                mock.patch.object(job, "_read_server_log", return_value="out of memory"):
            with self.assertRaises(RuntimeError) as ctx:
                job._await_server_ready(proc)
        self.assertIn("exit code 1", str(ctx.exception))
        self.assertIn("out-of-memory", str(ctx.exception))

    def test_raises_and_shuts_down_on_timeout(self):
        job = make_job(server_startup_timeout=10)
        proc = self._proc(None)  # still alive but never healthy
        with mock.patch.object(routes.time, "sleep"), \
                mock.patch.object(routes, "warning_job"), \
                mock.patch.object(routes.time, "monotonic", side_effect=[1000.0, 2000.0]), \
                mock.patch.object(job, "_read_server_log", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                job._await_server_ready(proc)
        self.assertIn("did not become ready", str(ctx.exception))
        proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
