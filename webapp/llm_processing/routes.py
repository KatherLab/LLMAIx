from . import llm_processing
from .. import socketio
from flask import render_template, current_app, flash, request, redirect, send_file, url_for
from flask_socketio import emit
from .forms import LLMPipelineForm
import requests
import pandas as pd
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Optional
import ast
import os
from .read_strange_csv import read_and_save_csv
import secrets
from concurrent import futures
import subprocess
import io
import math

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None

JobID = str
llm_jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

llm_progress = {}
new_model = False

def update_progress(job_id, progress: tuple[int, int, bool]):
    global llm_progress
    llm_progress[job_id] = progress    

    print("Progress: ", progress[0], " total: ", progress[1])
    socketio.emit('llm_progress_update', {'job_id': job_id, 'progress': progress[0], 'total': progress[1]})

@socketio.on('connect')
def handle_connect():
    print("Client Connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client Disconnected")

def extract_from_report(
        df: pd.DataFrame,
        model_name: str,
        prompt: str,
        symptoms: Iterable[str],
        temperature: float,
        grammar: str,
        model_path: str,
        server_path: str,
        ctx_size: int,
        n_gpu_layers: int,
        n_predict: int,
        job_id: int
) -> dict[Any]:
    print("Extracting from report")
    # Start server with correct model if not already running
    model_dir = Path(model_path)

    model_path = model_dir / model_name
    assert model_path.absolute().parent == model_dir

    global new_model
    global server_connection, current_model
    if current_model != model_name:
        server_connection and server_connection.kill()
        
        new_model = True
        server_connection = subprocess.Popen(
            [
                server_path,
                "--model",
                str(model_path),
                "--ctx-size",
                str(ctx_size),
                "--n-gpu-layers",
                str(n_gpu_layers),
                # "--verbose",
            ],
        )
        current_model = model_name
        time.sleep(5)

    try:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
    except KeyError:
        print("No proxy set")
        pass

    for _ in range(16):
        # wait until server is running
        try:
            requests.post(
                url="http://localhost:8080/completion",
                json={"prompt": "foo", "n_predict": 1}
            )
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)

    try:
        requests.post(
            url="http://localhost:8080/completion",
            json={"prompt": "foo", "n_predict": 1}
        )
    except requests.exceptions.ConnectionError:
        socketio.emit('load_failed')
        return
    
    print("Server running")
    
    new_model = False
    socketio.emit('load_complete')

    results = {}

    # socketio.emit('llm_progress_update', {'job_id': job_id, 'progress': 0, 'total_steps': len(df)})

    def is_empty_string_nan_or_none(variable):
        if variable is None:
            return True
        elif isinstance(variable, str) and variable.strip() == "":
            return True
        elif isinstance(variable, float) and math.isnan(variable):
            return True
        else:
            return False
        
    skipped = 0

    for i, (report, id) in enumerate(zip(df.report, df.id)):
        print("parsing report: ", i)
        if is_empty_string_nan_or_none(report):
            print("SKIPPING EMPTY REPORT!")
            skipped += 1
            update_progress(job_id=job_id, progress=(i + 1 - skipped, len(df) - skipped, True))
            continue
        for symptom in symptoms:
            result = requests.post(
                url="http://localhost:8080/completion",
                json={
                    "prompt": prompt.format(
                        symptom=symptom, report="".join(report)
                    ),
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "grammar": grammar,
                },
                timeout=20 * 60,
            )

            summary = result.json()
            if report not in results:
                results[report] = {}
            results[report][symptom] = summary
            results[report]["id"] = id

        print(f"Report {i} completed.")
        update_progress(job_id=job_id, progress=(i+1 - skipped, len(df) - skipped, True))

    socketio.emit('llm_progress_complete', {'job_id': job_id,'total_steps': len(df) - skipped})

    return postprocess_grammar(results, symptoms)

def postprocess_grammar(result, grammar):

    extracted_data = []

    # Iterate over each report and its associated data
    for report, info in result.items():
        # Get the first key in the dictionary (here assumed to be the relevant field)
        first_key = next(iter(info))
        
        # Extract the content of the first field
        content = info.get(first_key, {}).get('content', '')
        
        # Parse the content string into a dictionary
        try:
            info_dict = ast.literal_eval(content)
        except:
            raise Exception("Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little.")
        
        # Construct a dictionary containing the report and extracted information
        extracted_info = {'report': report, 'id': info['id']}
        for key, value in info_dict.items():
            extracted_info[key] = value
        
        # Append the extracted information to the list
        extracted_data.append(extracted_info)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(extracted_data)

    df['base_id'] = df['id'].str.split('_').str[0]

    # Group by base_id and aggregate reports and other columns into lists
    aggregated_df = df.groupby('base_id').agg(lambda x: x.tolist() if x.name != 'report' else ' '.join(x)).reset_index()
    # print(aggregated_df)

    # breakpoint()

    aggregated_df['personal_info_list'] = aggregated_df.apply(lambda row: [item for list in row.drop(["id", "base_id", "report"]) for item in list], axis=1)

    aggregated_df['masked_report'] = df['report'].apply(lambda x: replace_personal_info(x, aggregated_df['personal_info_list'][0]))

    aggregated_df.drop(columns=['id'], inplace=True)
    aggregated_df.rename(columns={'base_id': 'id'}, inplace=True)

    # breakpoint()

    # # Reorder the columns to have 'report' as the first column
    # columns = ['report', 'id'] + [col for col in df.columns if col != 'report' or col != 'id']
    # df = df[columns]

    # # list with the variables from the grammar excluding those the model did not predict anything for.
    # personal_info_list = list(filter(lambda x: x != '', df.drop(columns=["report", "id"]).values.flatten().tolist()))

    # breakpoint()
    # df["report_masked"] = df["report"].apply(lambda x: replace_personal_info(x, personal_info_list))

    return aggregated_df

from thefuzz import process

def is_empty_string_nan_or_none(variable):
        if variable is None:
            return True
        elif isinstance(variable, str) and variable.strip() == "":
            return True
        elif isinstance(variable, float) and math.isnan(variable):
            return True
        else:
            return False

def replace_personal_info(text, personal_info_list):
    # remove redundant items
    personal_info_list = list(set(personal_info_list))
    personal_info_list = [item for item in personal_info_list if item != ""]
    masked_text = text

    for info in personal_info_list:
        if is_empty_string_nan_or_none(info):
            print("SKIPPING EMPTY INFO!")
            continue
        # Get a list of best matches for the current personal information from the text
        best_matches = process.extract(info, text.split())
        best_score = best_matches[0][1]
        for match, score in best_matches:
            if score == best_score:
                # Replace best matches with asterisks (*) of the same length as the personal information
                masked_text = masked_text.replace(match, '*' * len(match))

    return masked_text

@llm_processing.route("/llm", methods=['GET', 'POST'])
def main():

    form = LLMPipelineForm(current_app.config['MODEL_PATH'])
    form.variables.render_kw = {'disabled': 'disabled'}

    if form.validate_on_submit():
        file = request.files["file"]

        if file.filename.endswith('.csv'):
            try:
                print(file)
                df = pd.read_csv(file)
            except pd.errors.ParserError as e:
                # print the error message in console
                print(e)
                print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed. Usually, this is caused by a line break in a field. The file will be fixed and then read again.")
                # fix the file
                fixed_file = io.BytesIO()
                read_and_save_csv(file, fixed_file)
                fixed_file.seek(0)
                df = pd.read_csv(fixed_file)
        
        elif file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file)
                print(df.head())
                # ValueError: Excel file format cannot be determined, you must specify an engine manually.
            except ValueError as e:
                print(e)
                print("The error message indicates that the Excel file format cannot be determined. This means that the Excel file is not properly formatted and needs to be fixed. The file will be fixed and then read again.")
                # fix the file
                flash("Excel file is not properly formatted!", "danger")
                return render_template("llm_processing.html", form=form)

        variables = [var.strip() for var in form.variables.data.split(",")]
        job_id = secrets.token_urlsafe()

        if not os.path.exists(current_app.config['SERVER_PATH']):
            flash("Llama CPP Server executable not found. Did you specify --server_path correctly?", "danger")
            return render_template("llm_processing.html", form=form)

        print("Run job!")
        global llm_jobs

        # extract_from_report(
        #     df=df,
        #     model_name=form.model.data,
        #     prompt=form.prompt.data,
        #     symptoms=variables,
        #     temperature=float(form.temperature.data),
        #     grammar=form.grammar.data.replace("\r\n", "\n"),
        #     model_path=current_app.config['MODEL_PATH'],
        #     server_path=current_app.config['SERVER_PATH'],
        #     n_predict=current_app.config['N_PREDICT'],
        #     ctx_size=current_app.config['CTX_SIZE'],
        #     n_gpu_layers=current_app.config['N_GPU_LAYERS'],
        #     job_id=job_id
        # )

        update_progress(job_id=job_id, progress=(0, len(df), True))

        llm_jobs[job_id] = executor.submit(
            extract_from_report,
            df=df,
            model_name=form.model.data,
            prompt=form.prompt.data,
            symptoms=variables,
            temperature=float(form.temperature.data),
            grammar=form.grammar.data.replace("\r\n", "\n"),
            model_path=current_app.config['MODEL_PATH'],
            server_path=current_app.config['SERVER_PATH'],
            n_predict=current_app.config['N_PREDICT'],
            ctx_size=current_app.config['CTX_SIZE'],
            n_gpu_layers=current_app.config['N_GPU_LAYERS'],
            job_id=job_id
        )

        print("Started job successfully!")

        return redirect(url_for('llm_processing.llm_results'))

    return render_template("llm_processing.html", form=form)

@llm_processing.route("/llm_results", methods=['GET'])
def llm_results():

    global llm_progress
    return render_template("llm_results.html", llm_progress=llm_progress, model_loaded=not new_model)

@llm_processing.route("/llm_download", methods=['GET'])
def llm_download():
    job_id = request.args.get('job')

    if job_id not in llm_jobs:
        flash("Job not found!", "danger")
        return redirect(url_for('llm_processing.llm_results'))

    job = llm_jobs[job_id]

    if job.done():
        try:
            result_df = job.result()
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for('llm_processing.llm_results'))
        
        breakpoint()
        result_io = io.BytesIO()
        result_df.to_csv(result_io, index=False)
        result_io.seek(0)
        return send_file(
            result_io,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"lllm-output-{job_id}.csv",
        )
    else:
        flash(f"Job {job}: An unknown error occurred! Probably the model did not predict anything / the output is empty!", "danger")
        return redirect(url_for('llm_processing.llm_results'))