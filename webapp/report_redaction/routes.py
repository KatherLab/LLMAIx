from concurrent import futures
import io
import json
import os
import secrets
import tempfile
import time
import traceback
import zipfile
import os
from flask import render_template, session
import pandas as pd
from webapp.llm_processing.utils import anonymize_pdf, convert_personal_info_list, find_fuzzy_matches, replace_personal_info
from webapp.report_redaction.utils import InceptionAnnotationParser, find_llm_output_csv, calculate_metrics, generate_confusion_matrix_from_counts, generate_score_dict, get_pymupdf_text_wordwise
from . import report_redaction
from flask import abort, render_template, request, redirect, send_file, url_for, session, flash
from .forms import ReportRedactionForm
from .. import socketio

JobID = str
report_redaction_jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

job_progress = {}

def update_progress(job_id, progress: tuple[int, int, bool]):
    global job_progress
    job_progress[job_id] = progress    

    print("Progress: ", progress)
    socketio.emit('progress_update', {'job_id': job_id, 'progress': progress[0], 'total': progress[1]})

def failed_job(job_id):
    time.sleep(2)
    print("FAILED")
    global job_progress
    # wait for 1s
    socketio.emit('progress_failed', {'job_id': job_id})

def complete_job(job_id):
    print("COMPLETE")
    global job_progress
    # set the job progress tuple [2] to true 
    job_progress[job_id] = (job_progress[job_id][0], job_progress[job_id][1], True)
    
    socketio.emit('progress_complete', {'job_id': job_id})

@socketio.on('connect')
def handle_connect():
    print("Client Connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client Disconnected")

@report_redaction.route("/reportredaction", methods=['GET', 'POST'])
def main():
    form = ReportRedactionForm()

    if form.validate_on_submit():

        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file was sent!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty part without filename
        if file.filename == '':
            flash('No selected file!', 'danger')
            return redirect(request.url)

        # Save the file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        session['annotation_file'] = None

        # If annotation_file is sent, save it to a temporary directory
        if 'annotation_file' in request.files:
            annotation_file = request.files['annotation_file']
            if annotation_file.filename != '':
                annotation_file_path = os.path.join(temp_dir, annotation_file.filename)
                annotation_file.save(annotation_file_path)
                session['annotation_file'] = annotation_file_path

        # Extract the content from the uploaded file and save it to a new temporary directory
        content_temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(content_temp_dir)

        # Save the path to the extracted content directory to the session variable
        session['pdf_file_zip'] = content_temp_dir

        # Optionally, save other form data to session variables
        session['enable_fuzzy'] = form.enable_fuzzy.data
        session['threshold'] = form.threshold.data
        session['exclude_single_chars'] = form.exclude_single_chars.data
        session['scorer'] = form.scorer.data

        session['redacted_pdf_filename'] = None
        session['annotation_pdf_filepath'] = None

        session['report_list'] = []

        # First report id

        df = find_llm_output_csv(content_temp_dir)
        if df is None or len(df) == 0:
            flash('No CSV file found in the uploaded file!', 'danger')
            return redirect(request.url)

        if 'submit-viewer' in request.form:

            report_id = df['id'].iloc[0]

            return redirect(url_for('report_redaction.report_redaction_viewer', report_id=report_id))
        
        elif 'submit-metrics' in request.form:
            if session['annotation_file'] is None:
                flash('No annotation file was sent!', 'danger')
                return redirect(request.url)
            
            print("Metrics Page")

            df = find_llm_output_csv(session['pdf_file_zip'])
            if df is None or len(df) == 0:
                flash('No CSV file found in the uploaded file!', 'danger')
                return redirect(request.url)
            
            job_id = secrets.token_urlsafe()
            update_progress(job_id=job_id, progress=(0, len(df), True))

            # report_list = generate_report_list(df, job_id, session['pdf_file_zip'], session['annotation_file'])

            print("Start Job")

            global report_redaction_jobs
            report_redaction_jobs[job_id] = executor.submit(
                generate_report_list,
                df = df,
                job_id = job_id, 
                pdf_file_zip = session['pdf_file_zip'],
                annotation_file = session['annotation_file']
            )

            # wait 0.5s
            time.sleep(0.5)
            # check if the job is aborted
            if job_id in report_redaction_jobs:
                if report_redaction_jobs[job_id].cancelled():
                    flash('Job Aborted!', "danger")
                elif report_redaction_jobs[job_id].done():
                    flash('Job Finished!', "success")
                else:
                    flash('Upload Successful! Job is running!', "success")

    global job_progress
    return render_template("report_redaction_form.html", form=form, progress=job_progress)


def generate_report_list(df, job_id, pdf_file_zip, annotation_file):
    print("Run Report List Generator")
    report_list = []

    try:
        for index, row in df.iterrows():
            print("Calculate Metrics for Report ", index, row['id'])
            report_dict = {}
            report_dict['id'] = row['id']

            report_dict['personal_info_list'] = convert_personal_info_list(row['personal_info_list'])

            orig_pdf_path = os.path.join(pdf_file_zip, f"{row['id']}.pdf")

            print("Load Redacted PDF")
            report_dict['redacted_pdf_filepath'], report_dict['redacted_text'], report_dict['fuzzy_matches'] = load_redacted_pdf(report_dict['personal_info_list'], orig_pdf_path, df, row['id'])
            print("Load Annotated PDF")
            report_dict['annotated_pdf_filepath'], report_dict['annotated_text'], report_dict['original_text'], report_dict['colormap'] = load_annotated_pdf(row['id'], orig_pdf_path, annotation_file)

            print("Generate Score Dict")
            report_dict['scores'], report_dict["confusion_matrix_filepath"] = generate_score_dict(report_dict['annotated_text'], report_dict['redacted_text'], report_dict['original_text'])

            report_list.append(report_dict)

            update_progress(job_id=job_id, progress=(index + 1, len(df), False))

        if len(report_list) == len(df):
            complete_job(job_id)
        else:
            failed_job(job_id)

        import ast
        if 'metadata' in df.columns:
            metadata = df['metadata'].iloc[0]
            metadata = ast.literal_eval(metadata)
        else:
            metadata = None

        report_summary_dict = {
            'report_list': report_list,
            'total_reports': len(report_list),
            'accumulated_metrics': accumulate_metrics(report_list),
            'metadata': metadata
        }

        confusion_matrix_filepath = os.path.join(tempfile.mkdtemp(), "confusion_matrix.svg")
        generate_confusion_matrix_from_counts(report_summary_dict['accumulated_metrics']['total_true_positives'], report_summary_dict['accumulated_metrics']['total_true_negatives'], report_summary_dict['accumulated_metrics']['total_false_positives'], report_summary_dict['accumulated_metrics']['total_false_negatives'], confusion_matrix_filepath)

        report_summary_dict['confusion_matrix_filepath'] = confusion_matrix_filepath

    except Exception as e:
        print(traceback.print_exc())
        failed_job(job_id)
        return
        # breakpoint()

    return report_summary_dict

def accumulate_metrics(report_list):
    total_tp = total_fp = total_tn = total_fn = 0
    total_precision = total_recall = total_accuracy = total_f1_score = 0
    total_false_positive_rate = total_false_negative_rate = 0
    num_samples = len(report_list)

    for report_dict in report_list:
        score_dict = report_dict['scores']
        total_tp += score_dict['true_positives']
        total_fp += score_dict['false_positives']
        total_tn += score_dict['true_negatives']
        total_fn += score_dict['false_negatives']
        total_precision += score_dict['precision']
        total_recall += score_dict['recall']
        total_accuracy += score_dict['accuracy']
        total_f1_score += score_dict['f1_score']
        total_false_positive_rate += score_dict['false_positive_rate']
        total_false_negative_rate += score_dict['false_negative_rate']

    macro_precision = total_precision / num_samples
    macro_recall = total_recall / num_samples
    macro_accuracy = total_accuracy / num_samples
    macro_f1_score = total_f1_score / num_samples
    macro_false_positive_rate = total_false_positive_rate / num_samples
    macro_false_negative_rate = total_false_negative_rate / num_samples

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0
    micro_false_positive_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) != 0 else 0
    micro_false_negative_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_accuracy': macro_accuracy,
        'macro_f1_score': macro_f1_score,
        'macro_false_positive_rate': macro_false_positive_rate,
        'macro_false_negative_rate': macro_false_negative_rate,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_accuracy': micro_accuracy,
        'micro_f1_score': micro_f1_score,
        'micro_false_positive_rate': micro_false_positive_rate,
        'micro_false_negative_rate': micro_false_negative_rate,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_true_negatives': total_tn,
        'total_false_negatives': total_fn
    }


@report_redaction.route("/reportredactionmetrics/<string:job_id>", methods=['GET'])
def report_redaction_metrics(job_id:str):

    # Check if the job exists
    if job_id not in report_redaction_jobs:
        flash(f"Job {job_id} not found!", "danger")
        return redirect(request.url)

    # Get the result from the job
    result = report_redaction_jobs[job_id].result()

    if result is None:
        flash(f"Job {job_id} not found or not finished yet!", "danger")
        return redirect(request.url)
    
    return render_template("report_redaction_metrics.html", total_reports=len(result['report_list']), report_list=result, job_id=job_id, metadata = result['metadata'])

def generate_export_df(result_dict: list):
    # Iterate over every report in result_list['report_list'] and add all scores in ['scores'] as one row to the dataframe, use ['id'] as id column
    # df = pd.DataFrame()

    scores_to_include = ['f1_score', 'accuracy', 'precision', 'recall', 'true_positives', 
                     'false_positives', 'true_negatives', 'false_negatives', 
                     'false_positive_rate', 'false_negative_rate']

    # Initialize a dictionary to store the extracted scores
    data = {'id': []}
    for score in scores_to_include:
        data[score] = []

    # Iterate over the list of dictionaries
    for entry in result_dict['report_list']:
        # Append ID to the 'id' list
        data['id'].append(entry['id'])
        # Iterate over the scores to include
        for score in scores_to_include:
            # If the score exists in the entry, append it to the corresponding list
            if score in entry['scores']:
                data[score].append(entry['scores'][score])
            else:
                data[score].append(None)  # Append None if score doesn't exist

    macro_scores = {}
    micro_scores = {}
    accumulated_metrics = result_dict.get('accumulated_metrics', {})
    for key in accumulated_metrics:
        if key.startswith('macro_'):
            macro_scores[key[len('macro_'):]] = float(accumulated_metrics[key])
        elif key.startswith('micro_'):
            micro_scores[key[len('micro_'):]] = float(accumulated_metrics[key])
        elif key.startswith('total_'):
            macro_scores[key[len('total_'):]] = int(accumulated_metrics[key])
            micro_scores[key[len('total_'):]] = int(accumulated_metrics[key])

    # Append macro and micro scores to the DataFrame
    data['id'].append('macro_scores')
    for score in scores_to_include:
        data[score].append(macro_scores.get(score, None))

    data['id'].append('micro_scores')
    for score in scores_to_include:
        data[score].append(micro_scores.get(score, None))


    # Create a DataFrame using the extracted values
    df = pd.DataFrame(data)

    # breakpoint()

    return df

@report_redaction.route("/downloadall", methods=['GET'])
def download_all():
    job_id = request.args.get('job_id', None)
    if not job_id:
        # Handle the case where the path to the zip file is not found
        flash('No job ID found!', 'danger')

        # Redirect to the generated URL
        return redirect(request.url)


    df = generate_export_df(report_redaction_jobs[job_id].result())

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, float_format='%.2f')
    csv_buffer.seek(0)

    # Send the CSV file as an attachment
    return send_file(
        csv_buffer,
        as_attachment=True,
        download_name=f'report_redaction_job_{job_id}.csv',
        mimetype='text/csv'
    )

@report_redaction.route("/reportredactionviewer/<string:report_id>", methods=['GET'])
def report_redaction_viewer(report_id):
    # Get the DataFrame from the uploaded file
    pdf_file_zip = session.get('pdf_file_zip', None)
    if not pdf_file_zip:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID
    
    df = find_llm_output_csv(session.get('pdf_file_zip', None))
    if df is None or len(df) == 0:
        flash('No CSV file (llm-output***********.csv) found in the uploaded zip file or the csv is empty!', 'danger')
        return redirect(request.url)

    # Check if the current report ID exists in the DataFrame
    if report_id not in df['id'].values:
        flash(f'Report ID {report_id} not found!', 'danger')
        return redirect(request.url)

    # Find the index of the current report ID
    current_index = df[df['id'] == report_id].index[0]

    try:
        personal_info_list = df[df['id'] == report_id]['personal_info_list'].item()
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(request.url)

    personal_info_list = convert_personal_info_list(personal_info_list)

    # Find the previous and next report IDs if they exist
    previous_id = df.at[current_index - 1, 'id'] if current_index > 0 else None
    next_id = df.at[current_index + 1, 'id'] if current_index < len(df) - 1 else None

    orig_pdf_path = os.path.join(pdf_file_zip, f"{report_id}.pdf")
    session['redacted_pdf_filename'], dollartext_redacted, fuzzy_matches = load_redacted_pdf(personal_info_list, orig_pdf_path, df, report_id, enable_fuzzy=session.get('enable_fuzzy', False), threshold=session.get('threshold', 90), exclude_single_chars=session.get('exclude_single_chars', False), scorer=session.get('scorer', None))


    if session.get('annotation_file', None):
        try:
            session['annotation_pdf_filepath'], dollartext_annotated, original_text, colormap = load_annotated_pdf(report_id, pdf_file_zip,session['annotation_file'])
            scores, session["confusion_matrix_filepath"] = generate_score_dict(dollartext_annotated, dollartext_redacted, original_text)
        except FileNotFoundError as e:
            flash(str(e), 'danger')
            scores = None
            colormap = {}

            session['annotation_pdf_filepath'] = None
            session["confusion_matrix_filepath"] = None
    
    else:
        scores = None
        colormap = {}

    import ast

    # Check if metadata key exists in df[df['id'] == report_id]['metadata']

    if 'metadata' not in df[df['id'] == report_id].columns:
        metadata = None
    else:
        metadata = df[df['id'] == report_id]['metadata'].item()
        metadata = ast.literal_eval(metadata)

    return render_template("report_redaction_viewer.html", report_id=report_id, previous_id=previous_id, next_id=next_id, report_number=current_index + 1, total_reports=len(df), personal_info_list=personal_info_list, enable_fuzzy=session.get('enable_fuzzy', False), threshold=session.get('threshold', 90), colormap = colormap, scores=scores, fuzzy_matches = fuzzy_matches, metadata=metadata)

def load_annotated_pdf(report_id, pdf_file, annotation_zip_file):
    print("load_annotated_pdf")
    json_filename = '.'.join(report_id.split('$')[0].split('.')[:-1]) + ".json"

    print("Filename: ", json_filename)
    # Open zip file and get filename file in any level of this zip file
    with zipfile.ZipFile(annotation_zip_file, 'r') as zipf:
        for file_info in zipf.infolist():

            if file_info.filename == json_filename:
                with zipf.open(file_info) as annotation_file:
                    annotation = json.load(annotation_file)
                break
        else:
            raise FileNotFoundError(f"JSON file {json_filename} not found in annotation zip file.")

    annoparser = InceptionAnnotationParser(annotation)

    # Construct the filename based on the session variable and the ID
    if not pdf_file.endswith('.pdf'):
        pdf_file = os.path.join(pdf_file, f"{report_id}.pdf")

    # Check if the file exists
    if not os.path.isfile(pdf_file):
        print("File not found: ", pdf_file)
        raise FileNotFoundError(f"File {pdf_file} not found.")

    print("Apply Annotations to PDF")
    annotation_pdf_filepath, dollartext_annotated, original_text = annoparser.apply_annotations_to_pdf(pdf_file)

    return annotation_pdf_filepath, dollartext_annotated, original_text, annoparser.colormap

@report_redaction.route("/reportredactionfileannotation/<string:id>")
def reportredactionfileannotation(id):

    return send_file(session['annotation_pdf_filepath'], mimetype='application/pdf')

@report_redaction.route("/reportredactionfileoriginal/<string:id>")
def reportredactionfileoriginal(id):
    # Retrieve the path to the zip file from the session
    pdf_file_zip = session.get('pdf_file_zip', None)
    if not pdf_file_zip:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID
    filename = os.path.join(pdf_file_zip, f"{id}.pdf")

    # Check if the file exists
    if not os.path.isfile(filename):
        abort(404)

    # Serve the PDF file
    return send_file(filename, mimetype='application/pdf')

def load_redacted_pdf(personal_info_list, filename, df, id, exclude_single_chars=False, enable_fuzzy=False, threshold=90, scorer="WRatio"):
    print("load_redacted_pdf")

    if exclude_single_chars:
        personal_info_list = [item for item in personal_info_list if len(item) > 1]

    if enable_fuzzy:
        fuzzy_matches = find_fuzzy_matches(df.loc[df['id'] == id, 'report'].iloc[0], personal_info_list, threshold=int(threshold), scorer=scorer)
    else:
        fuzzy_matches = []

    dollartext_redacted = generated_dollartext_stringlist(filename, personal_info_list, fuzzy_matches, ignore_short_sequences=1 if exclude_single_chars else 0)

    anonymize_pdf(filename, personal_info_list, filename.replace(".pdf", "_redacted.pdf"), fuzzy_matches)

    return filename.replace(".pdf", "_redacted.pdf"), dollartext_redacted, fuzzy_matches

    # socketio.emit('reportredaction_done', {'enable_fuzzy': session.get('enable_fuzzy', False), 'threshold': session.get('threshold', 90), 'fuzzy_matches': fuzzy_matches})

@report_redaction.route("/reportredactionfileredacted/<string:id>")
def reportredactionfileredacted(id):

    return send_file(session['redacted_pdf_filename'], mimetype='application/pdf')

def generated_dollartext_stringlist(filename, information_list, fuzzy_matches, ignore_short_sequences:int=0):
    """Replace all occurrences of the strings in information_list with dollar signs in the pdf text"""

    text = get_pymupdf_text_wordwise(filename, add_spaces=True)

    return replace_personal_info(text, information_list, fuzzy_matches, generate_dollarstring=True, ignore_short_sequences=ignore_short_sequences)

@report_redaction.route("/reportredactionconfusionmatrix")
def reportredactionconfusionmatrix():
    job_id = request.args.get('job_id')

    if job_id:
        # If job_id is provided, serve the confusion matrix from the specific job
        result = report_redaction_jobs.get(job_id).result()

        if result is None or 'confusion_matrix_filepath' not in result:
            # Handle the case where the result or confusion matrix filepath is not found
            abort(404)

        # Serve the confusion matrix SVG file from the specific job
        return send_file(result['confusion_matrix_filepath'], mimetype='image/svg+xml')
    else:
        # If no job_id provided, serve the default confusion matrix SVG file
        confusion_matrix_svg_filepath = session.get('confusion_matrix_filepath', None)

        if not confusion_matrix_svg_filepath:
            # Handle the case where the path to the confusion matrix SVG file is not found
            abort(404)

        # Serve the default confusion matrix SVG file
        return send_file(confusion_matrix_svg_filepath, mimetype='image/svg+xml')
