from datetime import datetime
import json
import tempfile
import traceback
import zipfile
from . import llm_processing
from .. import socketio
from flask import (
    render_template,
    current_app,
    flash,
    request,
    redirect,
    send_file,
    url_for,
    session,
    jsonify
)
from .forms import LLMPipelineForm
import requests
import pandas as pd
from pathlib import Path
import subprocess
import time
import ast
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Dict, Any
import os
from .read_strange_csv import read_and_save_csv
import secrets
from concurrent import futures
import io
import asyncio
import aiohttp
from tqdm import tqdm
from prometheus_client.parser import text_string_to_metric_families
from .utils import (
    read_preprocessed_csv_from_zip,
    replace_personal_info,
    is_empty_string_nan_or_none,
)
from io import BytesIO
from .. import set_mode

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None
model_active = False

JobID = str
llm_jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)


new_model = False
llm_progress = {}

start_times = {}


def format_time(seconds):
    if seconds < 120:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def update_progress(job_id, progress: tuple[int, int, bool, bool]):
    global llm_progress

    # Initialize llm_progress dictionary if not already initialized
    if "llm_progress" not in globals():
        llm_progress = {}

    # Update progress dictionary
    llm_progress[job_id] = progress

    # Calculate elapsed time since the job started

    if job_id not in start_times or progress[0] == 0:
        start_times[job_id] = time.time()
    elapsed_time = time.time() - start_times[job_id]

    # Calculate average time per progress step
    if progress[0] > 0:
        avg_time_per_step = elapsed_time / progress[0]
    else:
        avg_time_per_step = 0

    # Calculate estimated remaining time
    if progress[0] < progress[1]:
        remaining_steps = progress[1] - progress[0]
        estimated_remaining_time = avg_time_per_step * remaining_steps
    else:
        estimated_remaining_time = 0

    estimated_remaining_time = format_time(estimated_remaining_time)

    print(
        "Progress: ",
        progress[0],
        " Total: ",
        progress[1],
        " Estimated Remaining Time: ",
        estimated_remaining_time,
    )

    # Emit progress update via socketio
    socketio.emit(
        "llm_progress_update",
        {
            "job_id": job_id,
            "progress": progress[0],
            "total": progress[1],
            "remaining_time": estimated_remaining_time,
            "canceled": progress[3],
        },
    )


def warning_job(job_id, message):
    global job_progress
    socketio.emit("progress_warning", {"job_id": job_id, "message": message})


@socketio.on("connect")
def handle_connect():
    print("Client Connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client Disconnected")


@llm_processing.before_request
def before_request():
    set_mode(session, current_app.config["MODE"])


def fetch_metrics(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.text


def parse_metrics(metrics_text):
    metrics_dict = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            # Sample name is in sample.name, value in sample.value
            metrics_dict[sample.name] = sample.value
    return metrics_dict


@llm_processing.route("/metrics")
def get_metrics():
    try:
        url = f"http://localhost:{current_app.config['LLAMACPP_PORT']}/metrics"
        metrics_text = fetch_metrics(url)
        metrics_dict = parse_metrics(metrics_text)
        return jsonify(metrics_dict)
    except Exception as e:
        # Log the error if needed
        # print(f"Error fetching or parsing metrics: {e}")
        return jsonify({"error": "Probably llama-cpp is not running."})

@dataclass
class CancellableJob:
    df: pd.DataFrame
    model_name: str
    prompt: str
    symptoms: Iterable[str]
    temperature: float
    grammar: str
    model_path: str
    server_path: str
    ctx_size: int
    n_gpu_layers: int
    n_predict: int
    job_id: int
    zip_file_path: str
    llamacpp_port: int
    debug: bool = False
    model_name_name: str = ""
    parallel_slots: int = 1
    verbose_llama: bool = False
    kv_cache_type: str = "q8_0"
    mlock: bool = True
    gpu: str = "ALL"
    flash_attention: bool = False
    buffer_slots: int = 2
    mode: str = "informationextraction"
    chat_endpoint: bool = False
    system_prompt: str = ""
    job_name: str = ""
    _canceled: bool = field(default=False, init=False)
    results: Dict = field(default_factory=dict, init=False)
    skipped: int = field(default=0, init=False)


    def cancel(self) -> None:
        print("CANCELING JOB")
        self._canceled = True

    def is_canceled(self) -> bool:
        return self._canceled

    async def fetch_tokenized_result(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        async with session.post(
            f"http://localhost:{self.llamacpp_port}/tokenize", 
            json={"content": prompt_formatted}
        ) as response:
            return await response.json()
        
    async def fetch_chat_result(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        if self._canceled:
            raise asyncio.CancelledError()

        url = f"http://localhost:{self.llamacpp_port}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key"
        }
        data = {
            # "response_format": TODO implement this instead of the grammar to make the pipeline compatible with OpenAI API
            "model": "llmaix",
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_formatted
                }
            ]
        }

        if self.grammar and self.grammar not in [" ", None, "\n", "\r", "\r\n"]:
            data["grammar"] = self.grammar

        async def watch_cancellation():
            while True:
                if self._canceled:
                    print("Watching cancellation received cancel signal")
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)

        try:
            cancel_task = asyncio.create_task(watch_cancellation())
            
            async with session.post(
                headers=headers,
                url=url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=20 * 60)
            ) as response:
                # Create a task for the response
                response_task = asyncio.create_task(response.json())
                
                try:
                    # Wait for either the response or cancellation
                    done, pending = await asyncio.wait(
                        [response_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    print("Done")
                    
                    # If we were cancelled, ensure it propagates
                    if cancel_task in done and self._canceled:
                        response.close()
                        raise asyncio.CancelledError()
                    
                    return await response_task
                finally:
                    # Clean up tasks and connection
                    cancel_task.cancel()
                    if not response_task.done():
                        response_task.cancel()
                        response.close()
        
        except asyncio.CancelledError:
            if 'response' in locals():
                response.close()
            raise

    async def fetch_completion_result(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        if self._canceled:
            raise asyncio.CancelledError()
            
        json_data = {
            "prompt": prompt_formatted,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "cache_prompt": True
        }

        if self.grammar and self.grammar not in [" ", None, "\n", "\r", "\r\n"]:
            json_data["grammar"] = self.grammar

        async def watch_cancellation():
            while True:
                if self._canceled:
                    print("Watching cancellation received cancel signal")
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)
        
        try:
            cancel_task = asyncio.create_task(watch_cancellation())
            
            async with session.post(
                f"http://localhost:{self.llamacpp_port}/completion", 
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=20 * 60)
            ) as response:
                # Create a task for the response
                response_task = asyncio.create_task(response.json())
                
                try:
                    # Wait for either the response or cancellation
                    done, pending = await asyncio.wait(
                        [response_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    print("Done")
                    
                    # If we were cancelled, ensure it propagates
                    if cancel_task in done and self._canceled:
                        response.close()
                        raise asyncio.CancelledError()
                    
                    return await response_task
                finally:
                    # Clean up tasks and connection
                    cancel_task.cancel()
                    if not response_task.done():
                        response_task.cancel()
                        response.close()
        
        except asyncio.CancelledError:
            if 'response' in locals():
                response.close()
            raise

    async def process_report(self, session: aiohttp.ClientSession, report: str, id: Any, index: int, 
                       progress_bar: tqdm) -> None:
        if self._canceled:
            raise asyncio.CancelledError()  # Changed from return to raise

        try:
            if is_empty_string_nan_or_none(report):
                print("SKIPPING EMPTY REPORT!")
                self.skipped += 1
                update_progress(
                    job_id=self.job_id, 
                    progress=(progress_bar.n + 1 - self.skipped, len(self.df) - self.skipped, True, False)
                )
                progress_bar.update(1)
                return
            
            for symptom in self.symptoms:
                if self._canceled:
                    raise asyncio.CancelledError()  # Changed from return to raise

                prompt_formatted = self.prompt.format(symptom=symptom, report="".join(report))
                
                if self._canceled:
                    raise asyncio.CancelledError()
                
                if id not in self.results:
                    self.results[id] = {}

                self.results[id]["report"] = report
                self.results[id]["symptom"] = symptom
                

                if self.chat_endpoint:  
                    summary = await self.fetch_chat_result(session, prompt_formatted)

                    # self.results[id]["summary"] = summary
                    self.results[id]['content'] = summary['choices'][-1]['message']['content']

                    if summary['choices'][-1]['finish_reason'] == "length":
                        warning_job(
                            job_id=self.job_id,
                            message=f"Report {id}: Generation stopped after {summary['usage']['completion_tokens']} tokens "
                                    f"(limit reached), the results might be incomplete! Please increase n_predict!",
                        )

                else:
                    summary = await self.fetch_completion_result(session, prompt_formatted)

                    # self.results[id]["summary"] = summary
                    self.results[id]["content"] = summary['content']

                    if summary['stop_type'] == "limit":
                        warning_job(
                            job_id=self.job_id,
                            message=f"Report {id}: Generation stopped after {summary['tokens_predicted']} tokens "
                                    f"(limit reached), the results might be incomplete! Please increase n_predict!",
                        )

                # if num_prompt_tokens >= (self.ctx_size / self.parallel_slots) - self.n_predict:
                #     print(
                #         f"PROMPT MIGHT BE TOO LONG. PROMPT: {num_prompt_tokens} Tokens. "
                #         f"CONTEXT SIZE PER SLOT: {self.ctx_size / self.parallel_slots} Tokens. "
                #         f"N-PREDICT: {self.n_predict} Tokens.", flush=True
                #     )
                #     warning_job(
                #         job_id=self.job_id,
                #         message=f"Report: {id}. Prompt might be too long. Prompt: {num_prompt_tokens} Tokens. "
                #                 f"Context size per Slot: {self.ctx_size / self.parallel_slots} Tokens. "
                #                 f"N-Predict: {self.n_predict} Tokens.",
                #     )

                # summary = await self.fetch_completion_result(session, prompt_formatted)


            progress_bar.update(1)
            update_progress(
                job_id=self.job_id, 
                progress=(progress_bar.n - self.skipped, len(self.df) - self.skipped, True, False)
            )
            
        except asyncio.CancelledError:
            update_progress(
                job_id=self.job_id, 
                progress=(progress_bar.n - self.skipped, len(self.df) - self.skipped, False, True)
            )
            raise
        except Exception as e:
            print("REPORT ERROR - ", id)
            print(e)


    
    async def process_all_reports(self) -> None:
        if self._canceled:
            return

        MAX_CONCURRENT_REQUESTS = self.parallel_slots + self.buffer_slots
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def process_report_limited(session, report, id, i, progress_bar):
            if self._canceled:
                raise asyncio.CancelledError()
            async with semaphore:
                await self.process_report(session, report, id, i, progress_bar)

        async def main():
            tasks = []
            try:
                async with aiohttp.ClientSession() as session:
                    with tqdm(total=len(self.df), desc="Processing Reports") as progress_bar:
                        for i, (report, id) in enumerate(zip(self.df.report, self.df.id)):
                            if self._canceled:
                                raise asyncio.CancelledError()
                            task = asyncio.create_task(
                                process_report_limited(session, report, id, i, progress_bar)
                            )
                            tasks.append(task)
                        
                        # Modified to handle cancellation properly
                        try:
                            await asyncio.gather(*tasks)
                        except asyncio.CancelledError:
                            print("Cancelling all tasks...")
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            # Wait for tasks to finish cancelling
                            await asyncio.gather(*tasks, return_exceptions=True)
                            raise
                            
            except asyncio.CancelledError:
                print("Main task cancelled")
                raise
            finally:
                # Ensure all tasks are cancelled
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            update_progress(job_id=self.job_id, progress=(0, len(self.df), True, False))
            await main()
        except asyncio.CancelledError:
            update_progress(job_id=self.job_id, progress=(0, len(self.df), True, True))
            print("Process cancelled")
            raise

    def start_server(self) -> None:
        global new_model, server_connection, current_model, model_active

        model_dir = Path(self.model_path)
        model_path = model_dir / self.model_name
        assert model_path.absolute().parent == model_dir

        print("Current model:", current_model, "Load new model:", new_model)
        
        if current_model != self.model_name:
            if server_connection:
                server_connection.kill()

            print("Starting new server for model", self.model_name)

            new_model = True
            server_connection = subprocess.Popen(
                [
                    self.server_path,
                    "--model",
                    str(model_path),
                    "--ctx-size",
                    str(self.ctx_size),
                    "--n-gpu-layers",
                    str(self.n_gpu_layers),
                    "--port",
                    str(self.llamacpp_port),
                    "--metrics",
                    "-np",
                    str(self.parallel_slots),
                    "-b",
                    "2048",
                    "-ub",
                    "512",
                    "-t",
                    "8",
                ] + 
                (["--verbose"] if self.verbose_llama else []) +
                (["--mlock"] if self.mlock else []) +
                (["-ctk", self.kv_cache_type, "-ctv", self.kv_cache_type] if self.kv_cache_type != "" else []) +
                (["-sm", "none", "-mg", str(self.gpu)] if self.gpu not in ["all", "ALL", "mps", ""] else []) +
                (["-fa"] if self.flash_attention else []),
            )
            
            current_model = self.model_name
            model_active = True
            time.sleep(5)

            try:
                os.environ.pop("HTTP_PROXY", None)
                os.environ.pop("HTTPS_PROXY", None)
            except KeyError:
                print("No proxy set")

            while not self._canceled:
                try:
                    response = requests.get(f"http://localhost:{self.llamacpp_port}/health")
                    if response.status_code == 200 and response.json()["status"] == "ok":
                        break
                    time.sleep(2)
                except requests.exceptions.ConnectionError:
                    warning_job(
                        job_id=self.job_id,
                        message="Server connection error, will keep retrying ...",
                    )
                    print("Server connection error, will keep retrying ...")
                    time.sleep(5)

        else:
            model_active = True

        print("Server running")
        new_model = False
        socketio.emit("load_complete")

    def process(self) -> Tuple[Tuple[pd.DataFrame, int], str]:
        try:
            self.start_server()
            if self._canceled:
                update_progress(job_id=self.job_id, progress=(0, len(self.df), True, True))
                socketio.emit(
                    "llm_progress_canceled", 
                    {"job_id": self.job_id, "total_steps": len(self.df) - self.skipped}
                )
                return (pd.DataFrame(), 0), self.zip_file_path

            try:
                asyncio.run(self.process_all_reports())
            except asyncio.CancelledError:
                print("Processing cancelled")
                update_progress(job_id=self.job_id, progress=(0, len(self.df), True, True))
                socketio.emit(
                    "llm_progress_canceled", 
                    {"job_id": self.job_id, "total_steps": len(self.df) - self.skipped}
                )
                return (pd.DataFrame(), 0), self.zip_file_path

            if self._canceled:
                return (pd.DataFrame(), 0), self.zip_file_path


            socketio.emit(
                "llm_progress_complete", 
                {"job_id": self.job_id, "total_steps": len(self.df) - self.skipped}
            )

            llm_metadata = {
                "job_id": self.job_id,
                "job_name": self.job_name,
                "model_name": self.model_name_name if self.model_name_name else self.model_name,
                "system_prompt": self.system_prompt if self.chat_endpoint else "",
                "mode": "chat" if self.chat_endpoint else "completion",
                "prompt": self.prompt,
                "symptoms": self.symptoms,
                "temperature": self.temperature,
                "n_predict": self.n_predict,
                "ctx_size": self.ctx_size,
                "grammar": self.grammar,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            global model_active
            model_active = False

            result_df, errors = postprocess_grammar(self.results, self.df, llm_metadata, self.debug)

            if errors:
                warning_job(
                    job_id=self.job_id, 
                    message=f"Postprocessing: {errors} reports failed! The LLM did probably not generate valid JSON."
                )

            return (result_df, errors), self.zip_file_path

        except Exception as e:
            print(f"Error in process: {e}")
            return (pd.DataFrame(), 0), self.zip_file_path


def submit_llm_job(
    llm_jobs: Dict[int, Tuple[futures.Future, CancellableJob]],  # Modified type hint
    executor: futures.ThreadPoolExecutor,
    job_id: int,
    **kwargs
) -> None:
    job = CancellableJob(job_id=job_id, **kwargs)
    future = executor.submit(job.process)
    llm_jobs[job_id] = (future, job)

def postprocess_grammar(result, df, llm_metadata, debug=False, mode="informationextraction"):
    print("POSTPROCESSING")

    extracted_data = []

    error_count = 0

    # Iterate over each report and its associated data
    for i, (id, info) in enumerate(result.items()):
        print(f"Processing report {i+1} of {len(result)}")
        # Get the first key in the dictionary (here assumed to be the relevant field)

        # Extract the content of the first field
        content = info["content"]

        # Parse the content string into a dictionary
        try:
            if content.endswith("<|eot_id|>"):
                # print("Remove eot_id")
                content = content[: -len("<|eot_id|>")]
            if content.endswith("</s>"):
                # print("Remove </s>")
                content = content[: -len("</s>")]
            # search for last } in content and remove anything after that
            content = content[: content.rfind("}") + 1]
            # replace space null comma with space "null" comma

            # replace all backslash in the content string with nothing
            content = content.replace("\n","")
            content = content.replace("\r","")
            content = content.replace("\\", "")

            try:
                info_dict_raw = ast.literal_eval(content)
            except Exception:
                try:
                    content = content.replace(" null,", '' ,' "null",')
                    # print("REPLACE NULL")
                
                    info_dict_raw = ast.literal_eval(content)
                except Exception as e:
                    print("Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. ({content=})", flush=True)
                    print("RAW LLM OUTPUT: '" + info["content"] + "'", flush=True)
                    print("Error:", e, flush=True)
                    print("TRACEBACK:", traceback.format_exc(), flush=True)
                    info_dict_raw = {}
                    error_count += 1

            info_dict = {}
            for key, value in info_dict_raw.items():
                if is_empty_string_nan_or_none(value):
                    info_dict[key] = ""
                else:
                    info_dict[key] = value

            # print(f"Successfully parsed LLM output. ({content=})")
        except Exception as e:
            print(
                f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. (Output: {content=})", flush=True
            )
            print("Error:", e, flush=True)
            print("TRACEBACK:", traceback.format_exc(), flush=True)
            print(f"Will ignore the error for report {i} and continue.", flush=True)
            # if debug:
            #     breakpoint()
            info_dict = {}
            error_count += 1

            # raise Exception(f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. ({content=})") from e

        # get metadata from df by looking for row where id == id and get the column metadata

        metadata = df[df["id"] == id]["metadata"].iloc[0]

        metadata = ast.literal_eval(metadata)
        metadata["llm_processing"] = llm_metadata

        import json

        # Construct a dictionary containing the report and extracted information
        extracted_info = {
            "report": info["report"],
            "id": id,
            "metadata": json.dumps(metadata),
        }
        for key, value in info_dict.items():
            extracted_info[key] = value

        # Append the extracted information to the list
        extracted_data.append(extracted_info)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(extracted_data)

    # id without the extension for splitted reports
    def extract_base_id(id):
        parts = id.split("$")
        base_id = parts[0]  # The part before the dollar sign

        if len(parts) > 1:  # If there's a dollar sign in the ID
            subparts = parts[1].split("_")
            if len(subparts) > 1 and subparts[-1].isdigit():
                # If there's an underscore followed by a number after the dollar sign
                return base_id + "$" + "_".join(subparts[:-1])

        return id  # Return the original ID if no underscore followed by a number is found after the dollar sign

    df["base_id"] = df["id"].apply(extract_base_id)

    # Group by base_id and aggregate reports and other columns into lists
    if mode == "anonymizer":
        aggregated_df = (
            df.groupby("base_id")
            .agg(lambda x: x.tolist() if x.name != "report" else " ".join(x))
            .reset_index()
        )
    

        aggregated_df["personal_info_list"] = aggregated_df.apply(
            lambda row: [
                item
                for list in row.drop(["id", "base_id", "report", "metadata"])
                for item in list
            ],
            axis=1,
        )

        aggregated_df["masked_report"] = aggregated_df["report"].apply(
            lambda x: replace_personal_info(x, aggregated_df["personal_info_list"][0], [])
        )
    else:
        aggregated_df = df

    aggregated_df.drop(columns=["id"], inplace=True)
    aggregated_df.rename(columns={"base_id": "id"}, inplace=True)

    if mode == "anonymizer":
        aggregated_df["metadata"] = aggregated_df["metadata"].apply(lambda x: x[0])

    print("POSTPROCESSING DONE")

    return aggregated_df, error_count

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

def get_model_config(model_dir, config_file, model_file_path):

    # Example model config:
    # models:
    #     - name: "llama3.1_8b_instruct_q8"
    #         display_name: "LLaMA 3.1 8B Instruct Q8_0"
    #         file_name: "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    #         model_context_size: 128000
    #         kv_cache_size: 128000
    #         kv_cache_quants: "q8_0" # e.g. "q_8" or "q_4" - requires flash attention
    #         flash_attention: true # does not work for some models
    #         mlock: true
    #         server_slots: 8
    #         seed: 42
    #         n_gpu_layers: 33

    import yaml

    if not is_path(config_file):
        config_file = os.path.join(model_dir, config_file)

    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

        for model_dict in config_data["models"]:
            if model_dict["file_name"] == model_file_path:
                return model_dict

@llm_processing.route("/llm", methods=["GET", "POST"])
def main():

    try:
        import yaml

        config_file_path = current_app.config["CONFIG_FILE"]

        if not is_path(config_file_path):
            config_file_path = os.path.join(current_app.config["MODEL_PATH"], config_file_path)

        with open(config_file_path, 'r') as file:
            _ = yaml.safe_load(file)
    except Exception as e:
        flash(f"Cannot open LLM View - Cannot load config.yaml file. Error: {str(e)}", "danger")
        return redirect(request.referrer)


    form = LLMPipelineForm(
        config_file_path, current_app.config["MODEL_PATH"]
    )
    form.variables.render_kw = {"disabled": "disabled"}

    if form.validate_on_submit():
        file = request.files["file"]

        if "{report}" not in form.prompt.data:
            flash(
                "The prompt must contain the placeholder: {report}",
                "danger",
            )
            return redirect(request.referrer)

        if file.filename.endswith(".csv"):
            try:
                # print(file)
                df = pd.read_csv(file)
            except pd.errors.ParserError as e:
                # print the error message in console
                print(e)
                print(
                    "The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed. Usually, this is caused by a line break in a field. The file will be fixed and then read again."
                )
                # fix the file
                fixed_file = BytesIO()
                read_and_save_csv(file, fixed_file)
                fixed_file.seek(0)
                df = pd.read_csv(fixed_file)

        elif file.filename.endswith(".xlsx"):
            try:
                df = pd.read_excel(file)
                # print(df.head())
                # ValueError: Excel file format cannot be determined, you must specify an engine manually.
            except ValueError as e:
                print(e)
                print(
                    "The error message indicates that the Excel file format cannot be determined. This means that the Excel file is not properly formatted and needs to be fixed. The file will be fixed and then read again."
                )
                # fix the file
                flash("Excel file is not properly formatted!", "danger")
                return render_template("llm_processing.html", form=form)

        elif file.filename.endswith(".zip"):
            zip_buffer = BytesIO()
            file.save(zip_buffer)
            zip_buffer.seek(0)

            temp_dir = tempfile.mkdtemp()

            # Save the uploaded file to the temporary directory
            zip_file_path = os.path.join(temp_dir, file.filename)
            with open(zip_file_path, "wb") as f:
                f.write(zip_buffer.getvalue())
                # print("Zip file saved:", zip_file_path)

            # Verify the integrity of the saved file (optional)
            if os.path.exists(zip_file_path):
                saved_file_size = os.path.getsize(zip_file_path)
                print(f"Saved file size: {saved_file_size} bytes")

                # Check if the saved file is a valid ZIP file
                try:
                    with zipfile.ZipFile(zip_file_path, "r") as test_zip:
                        test_zip.testzip()
                    # print("File is a valid ZIP file")
                except zipfile.BadZipFile:
                    print("File is not a valid ZIP file")

            else:
                print("File not found:", zip_file_path)

            # Now you can proceed to read the contents of the ZIP file
            df = read_preprocessed_csv_from_zip(zip_file_path)

            if df is None:
                flash(
                    "Preprocessed zip file seems to be malformed or in a not supported format! Is there a csv file in it? Please note that the zip file must not contain a directory!",
                    "danger",
                )
                return render_template("llm_processing.html", form=form)

        else:
            flash("File format not supported!", "danger")
            return render_template("llm_processing.html", form=form)

        model_name = ""

        for filename, name in form.model.choices:
            if filename == form.model.data:
                model_name = name

        variables = [var.strip() for var in form.variables.data.split(",")]

        current_datetime = datetime.now()
        prefix = current_datetime.strftime("%Y%m%d%H%M")

        job_id = (
            form.job_name.data
            + "-"
            + model_name.replace(" ", "").replace("_", "-")
            + "_"
            + prefix
            + "_"
            + secrets.token_urlsafe(8)
        ) if form.job_name.data else (
            model_name.replace(" ", "").replace("_", "-")
            + "_"
            + prefix
            + "_"
            + secrets.token_urlsafe(8)
        )

        if not os.path.isfile(current_app.config["SERVER_PATH"]):
            flash(
                "Llama CPP Server executable not found. Did you specify --server_path correctly?",
                "danger",
            )
            return render_template("llm_processing.html", form=form)

        # print("Run job!")
        global llm_jobs

        model_config = get_model_config(current_app.config["MODEL_PATH"], current_app.config["CONFIG_FILE"], form.model.data)

        update_progress(job_id=job_id, progress=(0, len(df), True, False))

        submit_llm_job(
            llm_jobs=llm_jobs,
            executor=executor,
            job_id=job_id,
            df=df,
            model_name=form.model.data,
            prompt=form.prompt.data,
            symptoms=variables,
            temperature=float(form.temperature.data),
            grammar=form.grammar.data.replace("\r\n", "\n"),
            model_path=current_app.config["MODEL_PATH"],
            server_path=current_app.config["SERVER_PATH"],
            n_predict=form.n_predict.data,
            ctx_size=model_config['kv_cache_size'],
            n_gpu_layers=model_config['n_gpu_layers'],
            zip_file_path=zip_file_path or None,
            llamacpp_port=current_app.config["LLAMACPP_PORT"],
            debug=current_app.config["DEBUG"],
            model_name_name=model_name,
            parallel_slots=model_config['server_slots'],
            verbose_llama=current_app.config['VERBOSE_LLAMA'],
            gpu=current_app.config['GPU'],
            flash_attention=model_config['flash_attention'],
            mlock=model_config['mlock'],
            kv_cache_type=model_config['kv_cache_quants'],
            mode=current_app.config['MODE'],
            chat_endpoint=form.chat_endpoint.data,
            system_prompt=form.system_prompt.data,
            job_name=form.job_name.data
        )

        print(f"Started job {job_id} successfully!")

        return redirect(url_for("llm_processing.llm_results"))

    return render_template("llm_processing.html", form=form)


@llm_processing.route("/cancel_job")
def cancel_job():
    job_id = request.args.get("job")
    if job_id in llm_jobs:
        future, job = llm_jobs[job_id]  # Unpack both the Future and Job
        if not future.done():
            job.cancel()  # Call cancel on the actual job object
            future.cancel()
            socketio.emit(
                    "llm_progress_canceled", 
                    {"job_id": job_id, "total_steps": 0}
                )
            print(f"Cancelled job {job_id}")
            flash(f"Cancelled job {job_id}", "success")
        else:
            print(f"Job {job_id} already finished")
            flash(f"Job {job_id} already finished", "warning")
        del llm_jobs[job_id]
    else:
        print(f"Job {job_id} not found")
        flash(f"Job {job_id} not found!", "danger")
    return redirect(url_for("llm_processing.llm_results"))


@llm_processing.route("/llm_results", methods=["GET"])
def llm_results():
    global llm_progress
    return render_template(
        "llm_results.html", llm_progress=llm_progress, model_loaded=not new_model
    )


@llm_processing.route("/llm_download", methods=["GET"])
def llm_download():
    job_id = request.args.get("job")

    if job_id not in llm_jobs:
        flash("LLM Job not found!", "danger")
        return redirect(url_for("llm_processing.llm_results"))

    future, job = llm_jobs[job_id]

    if future.done():
        try:
            (result_df, error_count), zip_file_path = future.result()
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for("llm_processing.llm_results"))
        
        if error_count > 0:
            print("LLM output contains {} errors.".format(error_count), flush=True)

        if not zip_file_path or not os.path.exists(zip_file_path):
            # print("Download only the csv.")
            result_io = BytesIO()
            # breakpoint()

            result_df.to_csv(result_io, index=False)
            result_io.seek(0)
            return send_file(
                result_io,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"llm-output-{job_id}.csv",
            )

        with zipfile.ZipFile(zip_file_path, "r") as existing_zip:
            # Create an in-memory BytesIO object to hold the updated ZIP file
            updated_zip_buffer = io.BytesIO()

            # Create a new ZIP file
            with zipfile.ZipFile(
                updated_zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as updated_zip:
                # Add all existing files from the original ZIP
                for existing_file in existing_zip.filelist:
                    updated_zip.writestr(
                        existing_file.filename,
                        existing_zip.read(existing_file.filename),
                    )

                # Add the DataFrame as a CSV file to the ZIP
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                updated_zip.writestr(f"llm-output-{job_id}.csv", csv_buffer.getvalue())

        # Reset the BytesIO object to the beginning
        updated_zip_buffer.seek(0)

        # Send the updated ZIP file
        return send_file(
            updated_zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"llm-output-{job_id}.zip",
        )

    else:
        flash(
            f"Job {job}: An unknown error occurred! Probably the model did not predict anything / the output is empty and / or the code ran into a breakpoint!",
            "danger",
        )
        return redirect(url_for("llm_processing.llm_results"))
