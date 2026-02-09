from concurrent import futures
from datetime import datetime
import io
import json
import os
import secrets
import tempfile
import time
import traceback
import zipfile
from cassis import load_cas_from_json
import fitz
from flask import render_template, session, current_app
import pandas as pd
from webapp.llm_processing.utils import (
    anonymize_pdf,
    convert_personal_info_list,
    replace_personal_info,
)
from webapp.report_redaction.utils import (
    InceptionAnnotationParser,
    find_llm_output_csv,
    generate_confusion_matrix_from_counts,
    generate_score_dict,
    get_pymupdf_text_wordwise,
    find_fuzzy_matches
)
from . import report_redaction
from flask import abort, request, redirect, send_file, url_for, flash
from .forms import ReportRedactionForm
from .. import socketio, set_mode

JobID = str
report_redaction_jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(5)

job_progress = {}
client_connected = False
client_connect_timeout = 0.5

@report_redaction.before_request
def before_request():
    set_mode(session, current_app.config['MODE'])

def update_progress(job_id, progress: tuple[int, int, bool]):
    global job_progress
    job_progress[job_id] = progress

    print("Progress: ", progress)
    socketio.emit(
        "progress_update",
        {"job_id": job_id, "progress": progress[0], "total": progress[1]},
    )

def wait_for_client():
    passed_time = 0
    global client_connect_timeout
    global client_connected
    if not client_connected:
        while not client_connected:
            passed_time += 0.2
            time.sleep(0.2)
            print("Wait for client to connect")

            if passed_time > client_connect_timeout:
                break
    return
            

def failed_job(job_id):
    print("FAILED")

    wait_for_client()

    global job_progress
    # wait for 1s
    socketio.emit("progress_failed", {"job_id": job_id})


def warning_job(job_id, message):
    
    wait_for_client()

    global job_progress
    socketio.emit("progress_warning", {"job_id": job_id, "message": message})


def complete_job(job_id):
    print("COMPLETE")

    global job_progress
    # set the job progress tuple [2] to true
    job_progress[job_id] = (job_progress[job_id][0], job_progress[job_id][1], True)

    wait_for_client()

    socketio.emit("progress_complete", {"job_id": job_id})


@socketio.on("connect")
def handle_connect():
    print("Client Connected")
    global client_connected
    client_connected = True

@socketio.on("disconnect")
def handle_disconnect():
    print("Client Disconnected")
    global client_connected
    client_connected = False


@report_redaction.route("/reportredaction", methods=["GET", "POST"])
def main():
    form = ReportRedactionForm()

    if form.validate_on_submit():
        # Check if the POST request has the file part
        if "file" not in request.files:
            flash("No file was sent!", "danger")
            return redirect(request.url)

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty part without filename
        if file.filename == "":
            flash("No selected file!", "danger")
            return redirect(request.url)

        # Save the file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        session["annotation_file"] = None

        # If annotation_file is sent, save it to a temporary directory
        if "annotation_file" in request.files:
            annotation_file = request.files["annotation_file"]
            if annotation_file.filename != "":
                annotation_file_path = os.path.join(temp_dir, annotation_file.filename)
                annotation_file.save(annotation_file_path)
                session["annotation_file"] = annotation_file_path

        # Extract the content from the uploaded file and save it to a new temporary directory
        content_temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(content_temp_dir)

        # Save the path to the extracted content directory to the session variable
        session["pdf_file_zip"] = content_temp_dir

        # Optionally, save other form data to session variables
        session["enable_fuzzy"] = form.enable_fuzzy.data
        session["threshold"] = form.threshold.data
        session["exclude_single_chars"] = form.exclude_single_chars.data
        session["scorer"] = form.scorer.data
        
        session["ignore_labels"] = form.ignore_labels.data.split(";")
        print("Ignore labels: ", session["ignore_labels"])

        session["redacted_pdf_filename"] = None
        session["annotation_pdf_filepath"] = None

        session["report_list"] = []

        global job_progress

        # First report id

        df = find_llm_output_csv(content_temp_dir)
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded LLM output file!", "danger")
            return redirect(request.url)

        if "submit-viewer" in request.form:
            report_id = df["id"].iloc[0]

            session['current_redaction_job'] = None

            return redirect(
                url_for("report_redaction.report_redaction_viewer", report_id=report_id)
            )
        
        if 'submit-redaction-download' in request.form:

            def convert_personal_info_dict(df, report_id, ignore_labels:list[str] = []):
                try:
                    # Extract all column values for the current report ID, except column 'id', 'report', 'metadata' and 'report_redacted', put them in a dict with the column name as the key

                    personal_info_dict = {}
                    for column_name in df.columns:
                        if (
                            column_name != "id"
                            and column_name != "report"
                            and column_name != "metadata"
                            and column_name != "report_redacted"
                            and column_name != "masked_report"
                            and column_name not in ignore_labels
                        ):
                            personal_info_dict[column_name] = df[df["id"] == report_id][
                                column_name
                            ].item()

                    # personal_info_list = df[df['id'] == report_id]['personal_info_list'].item()
                except Exception as e:
                    flash("Error Loading personal info from llm output file: " + str(e), "danger")
                    return redirect(request.url)

                personal_info_dict = {
                    key: convert_personal_info_list(value)
                    for key, value in personal_info_dict.items()
                }

                return personal_info_dict
            
            def create_yaml_settings(enable_fuzzy, threshold, exclude_single_chars, scorer):
                import yaml
                settings = {
                    'enable_fuzzy': enable_fuzzy,
                    'threshold': threshold,
                    'exclude_single_chars': exclude_single_chars,
                    'scorer': scorer
                }
                yaml_data = yaml.dump(settings)
                return yaml_data


            def create_zip_from_dataframe(df, pdf_folder_path):
                # Create an in-memory BytesIO object to hold the ZIP file
                zip_buffer = io.BytesIO()

                # Create a new ZIP file in memory
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:

                    yaml_settings = create_yaml_settings(
                        enable_fuzzy=session.get("enable_fuzzy", False),
                        threshold=session.get("threshold", 90),
                        exclude_single_chars=session.get("exclude_single_chars", False),
                        scorer=session.get("scorer", None)
                    )
                    zip_file.writestr('redaction_config.yaml', yaml_settings)


                    for idx, row in df.iterrows():
                        pdf_id = str(row['id'])
                        possible_filenames = [pdf_id, pdf_id + '.pdf']

                        for filename in possible_filenames:
                            file_path = os.path.join(pdf_folder_path, filename)
                            if os.path.isfile(file_path):

                                file_path_redacted, _, _, _ = load_redacted_pdf(
                                    convert_personal_info_dict(df, row['id']),
                                    file_path,
                                    df,
                                    row['id'],
                                    enable_fuzzy=session.get("enable_fuzzy", False),
                                    threshold=session.get("threshold", 90),
                                    exclude_single_chars=session.get("exclude_single_chars", False),
                                    scorer=session.get("scorer", None),
                                    apply_redaction=True
                                )

                                with open(file_path_redacted, 'rb') as pdf_file:
                                    zip_file.writestr(os.path.basename(file_path_redacted), pdf_file.read())
                                break

                zip_buffer.seek(0)
                return zip_buffer
            
            # Create the ZIP file from the DataFrame
            zip_buffer = create_zip_from_dataframe(df, content_temp_dir)

            llm_output_id = file.filename.split(".zip")[0]
            if llm_output_id.startswith("llm-output"):
                llm_output_id = llm_output_id.split("llm-output-")[1]
            download_filename = f'redacted-reports-{llm_output_id}.zip'

            # Send the ZIP file without saving it to disk
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=download_filename)


        elif "submit-metrics" in request.form:
            
            if session["annotation_file"] is None:
                form.annotation_file.errors.append(
                    "For the metrics summary page, please upload an annotation file."
                )  

                return render_template("report_redaction_form.html", form=form, progress=job_progress)

            print("Metrics Page")

            df = find_llm_output_csv(session["pdf_file_zip"])
            if df is None or len(df) == 0:
                flash("No CSV file found in the uploaded file!", "danger")
                return redirect(request.url)

            current_datetime = datetime.now()
            prefix = current_datetime.strftime("%Y%m%d%H%M")

            import ast
            import json

            # Helper function to parse metadata - try JSON first, fallback to ast.literal_eval
            def parse_metadata(metadata_str):
                try:
                    return json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    try:
                        return ast.literal_eval(metadata_str)
                    except (ValueError, SyntaxError):
                        return None

            # breakpoint()

            metadata = parse_metadata(df["metadata"].iloc[0])
            model_name = metadata["llm_processing"]["model_name"]

            job_id = (
                "reportredaction_"
                + model_name.replace("_", "-").replace(" ", "")
                + "_"
                + prefix
                + "_"
                + secrets.token_urlsafe(8)
            )
            update_progress(job_id=job_id, progress=(0, len(df), True))

            print("Start Redaction Job")
            global report_redaction_jobs
            report_redaction_jobs[job_id] = executor.submit(
                generate_report_list,
                df=df,
                job_id=job_id,
                pdf_file_zip=session["pdf_file_zip"],
                annotation_file=session["annotation_file"],
                enable_fuzzy=session.get('enable_fuzzy', False), 
                threshold=session.get('threshold', 90), 
                exclude_single_chars=session.get('exclude_single_chars', False), 
                scorer=session.get('scorer', None),
                ignore_labels=session.get('ignore_labels', [])
            )

            # wait 0.5s
            time.sleep(0.5)
            # check if the job is aborted
            if job_id in report_redaction_jobs:
                if report_redaction_jobs[job_id].cancelled():
                    flash("Job Aborted!", "danger")
                elif report_redaction_jobs[job_id].done():
                    flash("Job Finished!", "success")
                else:
                    flash("Upload Successful! Job is running!", "success")

    
    return render_template(
        "report_redaction_form.html", form=form, progress=job_progress
    )


def generate_report_list(df, job_id, pdf_file_zip, annotation_file, enable_fuzzy=False, threshold=90, exclude_single_chars=False, scorer=None,ignore_labels:list[str]=[]):
    # print("Run Report List Generator")
    report_list = []

    try:
        for index, row in df.iterrows():
            # print("Calculate Metrics for Report ", index, row["id"])
            report_dict = {}
            report_dict["id"] = row["id"]

            if "personal_info_list" not in row:
                print("No personal info list found for report, reconstruct from csv.", row["id"])
                
                # reconstruct personal info list from csv by making the row['personal_info_list'] a string with all of the relevant column values in a python list: ['value of first column', 'value of second column']
                personal_info_list = "["
                for column_name in df.columns:
                    if column_name != "id" and column_name != "report" and column_name != "metadata" and column_name != "report_redacted" and column_name != "masked_report":
                        personal_info_list += "'" + str(row[column_name]) + "', "
                personal_info_list = personal_info_list[:-2] + "]"
                row["personal_info_list"] = personal_info_list

            try:
                report_dict["personal_info_list"] = convert_personal_info_list(
                    row["personal_info_list"]
                )
            except Exception as e:
                print("Error converting personal info list: ", e)
                breakpoint()

            try:
                # Extract all column values for the current report ID, except column 'id', 'report', 'metadata' and 'report_redacted', put them in a dict with the column name as the key

                personal_info_dict = {}
                for column_name in df.columns:
                    if (
                        column_name != "id"
                        and column_name != "report"
                        and column_name != "metadata"
                        and column_name != "report_redacted"
                        and column_name != "masked_report"
                    ):
                        personal_info_dict[column_name] = df[df["id"] == row["id"]][
                            column_name
                        ].item()
                    
                if "personal_info_list" not in personal_info_dict:
                    personal_info_dict["personal_info_list"] = row["personal_info_list"]

                # personal_info_list = df[df['id'] == report_id]['personal_info_list'].item()
            except Exception as e:
                print("Error extracting personal info dict: ", e)
                breakpoint()

            personal_info_dict = {
                key: convert_personal_info_list(value)
                for key, value in personal_info_dict.items()
                if key not in ignore_labels
            }

            orig_pdf_path = os.path.join(pdf_file_zip, f"{row['id']}.pdf")

            # print("Load Annotated PDF")
            (
                report_dict["annotated_pdf_filepath"],
                report_dict["annotated_text_labelwise"],
                report_dict["original_text"],
                report_dict["colormap"],
                sofastring,
            ) = load_annotated_pdf(row["id"], orig_pdf_path, annotation_file)

            # print("Load Redacted PDF")
            (
                report_dict["redacted_pdf_filepath"],
                report_dict["dollartext_redacted_dict"],
                report_dict["personal_info_dict"],
                report_dict["fuzzy_matches_dict"],
            ) = load_redacted_pdf(
                personal_info_dict, orig_pdf_path, df, row["id"], text=sofastring, enable_fuzzy=enable_fuzzy, threshold=threshold, exclude_single_chars=exclude_single_chars, scorer=scorer
            )

            report_dict["scores"] = {}

            # Generate Report Dict
            for key in report_dict["annotated_text_labelwise"].keys():
                if key not in report_dict["dollartext_redacted_dict"].keys():
                    warning_job(
                        job_id,
                        "For Report "
                        + row["id"]
                        + ", '"
                        + key
                        + "' was not found in the llm output. Is the grammar compatible with the annotation?",
                    )

            for key in report_dict["dollartext_redacted_dict"].keys():
                if key not in report_dict["annotated_text_labelwise"].keys():
                    warning_job(
                        job_id,
                        "For Report "
                        + row["id"]
                        + ", '"
                        + key
                        + "' was not found in the annotation. Is the grammar compatible with the annotation? Skipping.",
                    )

            for key, value in report_dict["dollartext_redacted_dict"].items():
                if key in report_dict["annotated_text_labelwise"].keys():
                    report_dict["scores"][key] = generate_score_dict(
                        report_dict["annotated_text_labelwise"][key],
                        value,
                        report_dict["original_text"],
                    )
                elif key in ignore_labels:
                    report_dict["scores"][key] = {}
                else:
                    print(
                        "Key not found in annotated_text_labelwise: ",
                        key,
                        " in report: ",
                        row["id"],
                    )

            report_list.append(report_dict)

            update_progress(job_id=job_id, progress=(index + 1, len(df), False))

        if len(report_list) == len(df):
            complete_job(job_id)
        else:
            failed_job(job_id)

        import ast
        import json

        if "metadata" in df.columns:
            metadata_str = df["metadata"].iloc[0]
            try:
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                try:
                    metadata = ast.literal_eval(metadata_str)
                except (ValueError, SyntaxError):
                    metadata = None
        else:
            metadata = None

        report_summary_dict = {
            "report_list": report_list,
            "total_reports": len(report_list),
            "metadata": metadata,
        }

        report_summary_dict["accumulated_metrics"] = accumulate_metrics(
            report_summary_dict["report_list"]
        )

        for label, value in report_summary_dict["accumulated_metrics"].items():
            # report_summary_dict['accumulated_metrics'][label] = accumulate_metrics(value)

            confusion_matrix_filepath = os.path.join(
                tempfile.mkdtemp(), f"confusion_matrix_{label}.svg"
            )
            generate_confusion_matrix_from_counts(
                report_summary_dict["accumulated_metrics"][label]["true_positives"],
                report_summary_dict["accumulated_metrics"][label]["true_negatives"],
                report_summary_dict["accumulated_metrics"][label]["false_positives"],
                report_summary_dict["accumulated_metrics"][label]["false_negatives"],
                confusion_matrix_filepath,
            )

            report_summary_dict["accumulated_metrics"][label] = {
                "confusion_matrix_filepath": confusion_matrix_filepath,
                "metrics": report_summary_dict["accumulated_metrics"][label],
            }

    except Exception:
        print(traceback.print_exc())
        failed_job(job_id)
        return
        # breakpoint()

    return report_summary_dict


def accumulate_metrics(report_list):
    labelwise_metrics = {}

    num_samples = len(report_list)

    metrics = [
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
    ]

    for index, report_dict in enumerate(report_list):
        if "scores" not in report_dict:
            print("No scores in report ", index, ". Skip")
            breakpoint()
            continue
        for label, score_tuple in report_dict["scores"].items():
            score_dict = score_tuple[0]
            if label not in labelwise_metrics:
                labelwise_metrics[label] = {metric: 0 for metric in metrics}

            for metric in metrics:
                labelwise_metrics[label][metric] += score_dict[metric]

    for label, score_dict in labelwise_metrics.items():
        for metric in metrics:
            if metric in [
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]:
                labelwise_metrics[label][metric] = int(score_dict[metric])
            else:
                labelwise_metrics[label][metric] = score_dict[metric] / num_samples

        labelwise_metrics[label]["micro_precision"] = (
            score_dict["true_positives"]
            / (score_dict["true_positives"] + score_dict["false_positives"])
            if (score_dict["true_positives"] + score_dict["false_positives"]) != 0
            else 0
        )
        labelwise_metrics[label]["micro_recall"] = (
            score_dict["true_positives"]
            / (score_dict["true_positives"] + score_dict["false_negatives"])
            if (score_dict["true_positives"] + score_dict["false_negatives"]) != 0
            else 0
        )
        labelwise_metrics[label]["micro_accuracy"] = (
            (score_dict["true_positives"] + score_dict["true_negatives"])
            / (
                score_dict["true_positives"]
                + score_dict["true_negatives"]
                + score_dict["false_positives"]
                + score_dict["false_negatives"]
            )
            if (
                score_dict["true_positives"]
                + score_dict["true_negatives"]
                + score_dict["false_positives"]
                + score_dict["false_negatives"]
            )
            != 0
            else 0
        )
        labelwise_metrics[label]["micro_f1_score"] = (
            2
            * labelwise_metrics[label]["micro_precision"]
            * labelwise_metrics[label]["micro_recall"]
            / (
                labelwise_metrics[label]["micro_precision"]
                + labelwise_metrics[label]["micro_recall"]
            )
            if (
                labelwise_metrics[label]["micro_precision"]
                + labelwise_metrics[label]["micro_recall"]
            )
            != 0
            else 0
        )
        labelwise_metrics[label]["micro_specificity"] = (
            score_dict["true_negatives"]
            / (score_dict["true_negatives"] + score_dict["false_positives"])
            if (score_dict["true_negatives"] + score_dict["false_positives"]) != 0
            else 0
        )
        labelwise_metrics[label]["micro_false_positive_rate"] = (
            score_dict["false_positives"]
            / (score_dict["false_positives"] + score_dict["true_negatives"])
            if (score_dict["false_positives"] + score_dict["true_negatives"]) != 0
            else 0
        )
        labelwise_metrics[label]["micro_false_negative_rate"] = (
            score_dict["false_negatives"]
            / (score_dict["false_negatives"] + score_dict["true_positives"])
            if (score_dict["false_negatives"] + score_dict["true_positives"]) != 0
            else 0
        )

    for label, score_dict in labelwise_metrics.items():
        for m, value in score_dict.items():
            labelwise_metrics[label][m] = round(value, 4)

    return labelwise_metrics


@report_redaction.route("/reportredactionmetrics/<string:job_id>", methods=["GET"])
def report_redaction_metrics(job_id: str):
    # Check if the job exists
    if job_id not in report_redaction_jobs:
        flash(f"Job {job_id} not found!", "danger")
        return redirect(request.url)

    # Get the result from the job
    result = report_redaction_jobs[job_id].result()

    if result is None:
        flash(f"Job {job_id} not found or not finished yet!", "danger")
        return redirect(request.url)
    
    session['current_redaction_job'] = job_id

    return render_template(
        "report_redaction_metrics.html",
        total_reports=len(result["report_list"]),
        report_list=result,
        job_id=job_id,
        metadata=result["metadata"],
    )


def generate_export_df(result_dict: list):
    # Iterate over every report in result_list['report_list'] and add all scores in ['scores'] as one row to the dataframe, use ['id'] as id column
    # df = pd.DataFrame()

    scores_to_include = []

    for label in result_dict["report_list"][0]["scores"].keys():
        for metric in result_dict["report_list"][0]["scores"][label][0].keys():
            if label != "personal_info_list":
                scores_to_include.append("{}${}".format(label, metric))
            else:
                scores_to_include.append(metric)
    # scores_to_include = ['f1_score', 'accuracy', 'precision', 'recall', 'specificity', 'true_positives',
    #                  'false_positives', 'true_negatives', 'false_negatives',
    #                  'false_positive_rate', 'false_negative_rate']

    # Initialize a dictionary to store the extracted scores
    data = {"id": []}
    for score in scores_to_include:
        data[score] = []

    macro_scores = {}
    micro_scores = {}

    accumulated_metrics = result_dict.get("accumulated_metrics", {})

    # Iterate over the list of dictionaries
    for entry in result_dict["report_list"]:
        # Append ID to the 'id' list
        data["id"].append(entry["id"])
        # Iterate over the scores to include
        for score in scores_to_include:
            if "$" in score:
                label = score.split("$")[0]
                metric = score.split("$")[1]
            else:
                label = "personal_info_list"
                metric = score
            # If the score exists in the entry, append it to the corresponding list
            if label in entry["scores"] and metric in entry["scores"][label][0]:
                data[score].append(entry["scores"][label][0][metric])
            else:
                print("Score {} not found in entry {}".format(score, entry["id"]))
                data[score].append(None)  # Append None if score doesn't exist

            macro_scores[score] = accumulated_metrics[label]["metrics"][metric]
            if metric in [
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            ]:
                micro_scores[score] = accumulated_metrics[label]["metrics"][metric]
            else:
                micro_scores[score] = accumulated_metrics[label]["metrics"][
                    f"micro_{metric}"
                ]

    # Append macro and micro scores to the DataFrame
    data["id"].append("macro_scores")
    for score in scores_to_include:
        data[score].append(macro_scores.get(score, None))

    data["id"].append("micro_scores")
    for score in scores_to_include:
        data[score].append(micro_scores.get(score, None))

    # Create a DataFrame using the extracted values
    df = pd.DataFrame(data)

    return df


@report_redaction.route("/downloadall", methods=["GET"])
def download_all():
    job_id = request.args.get("job_id", None)
    if not job_id:
        flash("No job ID found!", "danger")
        return redirect(request.url)

    job_result = report_redaction_jobs[job_id].result()
    df = generate_export_df(job_result)

    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add the CSV file to the zip
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False, float_format="%.4f")
        csv_buffer.seek(0)
        zip_file.writestr(f"{job_id}.csv", csv_buffer.getvalue())

        # Loop through redacted PDFs
        for report_dict in job_result["report_list"]:  # Corrected variable name
            redacted_pdf_filename = report_dict["redacted_pdf_filepath"]
            # Apply redactions using PyMuPDF
            with fitz.open(redacted_pdf_filename) as pdf:
                for page in pdf:
                    page.apply_redactions()

                # Store the redacted PDF content in memory
                redacted_pdf_buffer = io.BytesIO()
                pdf.save(redacted_pdf_buffer)
                redacted_pdf_buffer.seek(0)

                # Add redacted PDF to the zip
                zip_file.writestr(
                    os.path.basename(redacted_pdf_filename),
                    redacted_pdf_buffer.getvalue(),
                )

    # Send the zip file as an attachment
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"{job_id}.zip",
        mimetype="application/zip",
    )


@report_redaction.route("/reportredactionviewer/<string:report_id>", methods=["GET"])
def report_redaction_viewer(report_id):
    # Get the DataFrame from the uploaded file
    pdf_file_zip = session.get("pdf_file_zip", None)
    if not pdf_file_zip:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID

    df = find_llm_output_csv(session.get("pdf_file_zip", None))
    if df is None or len(df) == 0:
        flash(
            "No CSV file (llm-output***********.csv) found in the uploaded zip file or the csv is empty!",
            "danger",
        )
        return redirect(request.url)

    # Check if the current report ID exists in the DataFrame
    if report_id not in df["id"].values:
        flash(f"Report ID {report_id} not found!", "danger")
        return redirect(request.url)

    # Find the index of the current report ID
    current_index = df[df["id"] == report_id].index[0]

    try:
        # Extract all column values for the current report ID, except column 'id', 'report', 'metadata' and 'report_redacted', put them in a dict with the column name as the key

        # print("Viewer: IGNORE ", session.get("ignore_labels", []))
        personal_info_dict = {}
        for column_name in df.columns:
            if (
                column_name != "id"
                and column_name != "report"
                and column_name != "metadata"
                and column_name != "report_redacted"
                and column_name != "masked_report"
                and column_name not in session.get("ignore_labels", [])
            ):
                personal_info_dict[column_name] = df[df["id"] == report_id][
                    column_name
                ].item()            

        if "personal_info_list" not in personal_info_dict:
            print("Reconstructing personal info list from the relevant columns as python list in a string with the values: ['value1', 'value2']")
            personal_info_list = "["
            for column_name in df.columns:
                if column_name != "id" and column_name != "report" and column_name != "metadata" and column_name != "report_redacted" and column_name != "masked_report":
                    personal_info_list += "'" + str(df[df["id"] == report_id][column_name].item()) + "', "
            personal_info_list = personal_info_list[:-2] + "]"
            personal_info_dict["personal_info_list"] = personal_info_list

        # personal_info_list = df[df['id'] == report_id]['personal_info_list'].item()
    except Exception as e:
        flash("Error Loading personal info from llm output file: " + str(e), "danger")
        return redirect(request.url)

    personal_info_dict = {
        key: convert_personal_info_list(value)
        for key, value in personal_info_dict.items()
        if key not in session.get("ignore_labels", [])
    }

    # print("personal_info_dict", personal_info_dict)

    # Find the previous and next report IDs if they exist
    previous_id = df.at[current_index - 1, "id"] if current_index > 0 else None
    next_id = df.at[current_index + 1, "id"] if current_index < len(df) - 1 else None

    orig_pdf_path = os.path.join(pdf_file_zip, f"{report_id}.pdf")
    if session.get("annotation_file", None) is None:
        (
            session["redacted_pdf_filename"],
            dollartext_redacted_dict,
            personal_info_dict,
            fuzzy_matches_dict,
        ) = load_redacted_pdf(
            personal_info_dict,
            orig_pdf_path,
            df,
            report_id,
            enable_fuzzy=session.get("enable_fuzzy", False),
            threshold=session.get("threshold", 90),
            exclude_single_chars=session.get("exclude_single_chars", False),
            scorer=session.get("scorer", None),
        )

    if session.get("annotation_file", None):
        try:
            (
                session["annotation_pdf_filepath"],
                dollartext_annotated_labelwise,
                original_text,
                colormap,
                sofastring,
            ) = load_annotated_pdf(report_id, pdf_file_zip, session["annotation_file"])
            (
                session["redacted_pdf_filename"],
                dollartext_redacted_dict,
                personal_info_dict,
                fuzzy_matches_dict,
            ) = load_redacted_pdf(
                personal_info_dict,
                orig_pdf_path,
                df,
                report_id,
                enable_fuzzy=session.get("enable_fuzzy", False),
                threshold=session.get("threshold", 90),
                exclude_single_chars=session.get("exclude_single_chars", False),
                scorer=session.get("scorer", None),
                text=sofastring,
            )
            # Calculate scores labelwise

            scores_dict = {}

            for key, value in dollartext_redacted_dict.items():
                if key in dollartext_annotated_labelwise:
                    scores_dict[key] = generate_score_dict(
                        dollartext_annotated_labelwise[key], value, original_text
                    )
                else:
                    print("Key not found in Annotations, SKIP: " + key)

            if "personal_info_list" in scores_dict:
                session["confusion_matrix_filepath"] = scores_dict[
                    "personal_info_list"
                ][1]
            else:
                session["confusion_matrix_filepath"] = None
                print("Confusion Matrix not found ...")
            # scores, session["confusion_matrix_filepath"] = generate_score_dict(dollartext_annotated, dollartext_redacted_dict['personal_info_list'], original_text)
        except FileNotFoundError as e:
            flash("File Not Found: " + str(e), "danger")
            print("File Not Found: " + str(e))
            scores_dict = {}
            colormap = {}

            session["annotation_pdf_filepath"] = None
            session["confusion_matrix_filepath"] = None

    else:
        scores_dict = {}
        colormap = {}
        fuzzy_matches_dict = {}

    import ast
    import json

    # Check if metadata key exists in df[df['id'] == report_id]['metadata']

    if "metadata" not in df[df["id"] == report_id].columns:
        metadata = None
    else:
        metadata_str = df[df["id"] == report_id]["metadata"].item()
        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            try:
                metadata = ast.literal_eval(metadata_str)
            except (ValueError, SyntaxError) as e:
                print("Error parsing Metadata from llm output file: " + str(e))
                breakpoint()

    return render_template(
        "report_redaction_viewer.html",
        report_id=report_id,
        previous_id=previous_id,
        next_id=next_id,
        report_number=current_index + 1,
        total_reports=len(df),
        personal_info_dict=personal_info_dict,
        enable_fuzzy=session.get("enable_fuzzy", False),
        threshold=session.get("threshold", 90),
        colormap=colormap,
        scores=scores_dict,
        fuzzy_matches_dict=fuzzy_matches_dict,
        metadata=metadata,
    )


def load_annotated_pdf(report_id, pdf_file, annotation_zip_file):
    # print("load_annotated_pdf")
    json_filename = ".".join(report_id.split("$")[0].split(".")[:-1]) + ".json"

    # print("Filename: ", json_filename)
    # Open zip file and get filename file in any level of this zip file
    with zipfile.ZipFile(annotation_zip_file, "r") as zipf:
        for file_info in zipf.infolist():
            if file_info.filename == json_filename:
                with zipf.open(file_info, "r") as annotation_file:
                    cas = load_cas_from_json(annotation_file)
                with zipf.open(file_info, "r") as annotation_file:
                    annotation = json.load(annotation_file)
                break
        else:
            raise FileNotFoundError(
                f"JSON file {json_filename} not found in annotation zip file."
            )

    annoparser = InceptionAnnotationParser(annotation, cas)

    # Construct the filename based on the session variable and the ID
    if not pdf_file.endswith(".pdf"):
        pdf_file = os.path.join(pdf_file, f"{report_id}.pdf")

    # Check if the file exists
    if not os.path.isfile(pdf_file):
        print("File not found: ", pdf_file)
        raise FileNotFoundError(f"File {pdf_file} not found.")

    # print("Apply Annotations to PDF")
    (
        annotation_pdf_filepath,
        dollartext_annotated,
        original_text,
        dollartext_annotated_labelwise,
    ) = annoparser.apply_annotations_to_pdf(pdf_file)

    dollartext_annotated_labelwise["personal_info_list"] = dollartext_annotated

    return (
        annotation_pdf_filepath,
        dollartext_annotated_labelwise,
        original_text,
        annoparser.colormap,
        annoparser.get_sofastring(),
    )


@report_redaction.route("/reportredactionfileannotation/<string:id>")
def reportredactionfileannotation(id):
    return send_file(session["annotation_pdf_filepath"], mimetype="application/pdf")


@report_redaction.route("/reportredactionfileoriginal/<string:id>")
def reportredactionfileoriginal(id):
    # Retrieve the path to the zip file from the session
    pdf_file_zip = session.get("pdf_file_zip", None)
    if not pdf_file_zip:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID
    filename = os.path.join(pdf_file_zip, f"{id}.pdf")

    # Check if the file exists
    if not os.path.isfile(filename):
        abort(404)

    # Serve the PDF file
    return send_file(filename, mimetype="application/pdf")


def load_redacted_pdf(
    personal_info_dict,
    filename,
    df,
    id,
    exclude_single_chars=False,
    enable_fuzzy=False,
    threshold=90,
    scorer="WRatio",
    text=None,
    apply_redaction:bool=False,
):

    if exclude_single_chars:
        for key, value in personal_info_dict.items():
            personal_info_dict[key] = [item for item in value if len(item) > 1]

    fuzzy_matches_dict = {}
    if enable_fuzzy:
        for key, value in personal_info_dict.items():
            fuzzy_matches_dict[key] = find_fuzzy_matches(
                df.loc[df["id"] == id, "report"].iloc[0],
                value,
                threshold=int(threshold),
                scorer=scorer,
            )
        # fuzzy_matches = find_fuzzy_matches(df.loc[df['id'] == id, 'report'].iloc[0], personal_info_list, threshold=int(threshold), scorer=scorer)
    else:
        for key, value in personal_info_dict.items():
            fuzzy_matches_dict[key] = []
        # fuzzy_matches = None

    dollartext_redacted_dict = {}
    for key, value in personal_info_dict.items():
        dollartext_redacted_dict[key] = generated_dollartext_stringlist(
            filename,
            value,
            fuzzy_matches_dict[key],
            ignore_short_sequences=1 if exclude_single_chars else 0,
            text=text,
        )

    # dollartext_redacted = generated_dollartext_stringlist(filename, personal_info_list, fuzzy_matches, ignore_short_sequences=1 if exclude_single_chars else 0, text=text)

    # regenerate personal_info_dict["personal_info_list"] with the other keys in the dict, if they are there

    personal_info_dict["personal_info_list"] = []
    for key, value in personal_info_dict.items():
        if key != "personal_info_list":
            personal_info_dict["personal_info_list"].extend(value)

    fuzzy_matches_dict["personal_info_list"] = []
    for key, value in fuzzy_matches_dict.items():
        if key != "personal_info_list":
            fuzzy_matches_dict["personal_info_list"].extend(value)


    # Redact with full personal info

    anonymize_pdf(
        filename,
        personal_info_dict["personal_info_list"],
        filename.replace(".pdf", "_redacted.pdf"),
        fuzzy_matches_dict["personal_info_list"],
        apply_redaction=apply_redaction,
    )

    return (
        filename.replace(".pdf", "_redacted.pdf"),
        dollartext_redacted_dict,
        personal_info_dict,
        fuzzy_matches_dict,
    )


@report_redaction.route("/reportredactionfileredacted/<string:id>")
def reportredactionfileredacted(id):
    return send_file(session["redacted_pdf_filename"], mimetype="application/pdf")


def generated_dollartext_stringlist(
    filename,
    information_list,
    fuzzy_matches,
    ignore_short_sequences: int = 0,
    text=None,
):
    """Replace all occurrences of the strings in information_list with dollar signs in the pdf text"""

    if not text:
        text = get_pymupdf_text_wordwise(filename, add_spaces=True)

    return replace_personal_info(
        text,
        information_list,
        fuzzy_matches,
        generate_dollarstring=True,
        ignore_short_sequences=ignore_short_sequences,
    )


@report_redaction.route("/reportredactionconfusionmatrix")
def reportredactionconfusionmatrix():
    job_id = request.args.get("job_id")

    if job_id:
        # If job_id is provided, serve the confusion matrix from the specific job
        result = report_redaction_jobs.get(job_id).result()

        if result is None or "accumulated_metrics" not in result:
            # Handle the case where the result or confusion matrix filepath is not found
            print("No result found for job_id:", job_id)
            abort(404)

        # Serve the confusion matrix SVG file from the specific job
        label = request.args.get("label")

        if label:
            return send_file(
                result["accumulated_metrics"][label]["confusion_matrix_filepath"],
                mimetype="image/svg+xml",
            )

        return send_file(
            result["accumulated_metrics"]["personal_info_list"][
                "confusion_matrix_filepath"
            ],
            mimetype="image/svg+xml",
        )
    else:
        # If no job_id provided, serve the default confusion matrix SVG file
        confusion_matrix_svg_filepath = session.get("confusion_matrix_filepath", None)

        if not confusion_matrix_svg_filepath:
            # Handle the case where the path to the confusion matrix SVG file is not found
            abort(404)

        # Serve the default confusion matrix SVG file
        return send_file(confusion_matrix_svg_filepath, mimetype="image/svg+xml")
