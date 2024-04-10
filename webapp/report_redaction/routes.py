
import base64
import os
import tempfile
import time
from webapp.llm_processing.utils import anonymize_pdf, convert_personal_info_list, find_fuzzy_matches
from webapp.report_redaction.utils import find_llm_output_csv
from . import report_redaction
from flask import abort, render_template, request, redirect, send_file, url_for, current_app, session, flash
from .forms import ReportRedactionForm
from io import BytesIO
from .. import socketio

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

        # First report id

        df = find_llm_output_csv(content_temp_dir)
        if df is None or len(df) == 0:
            flash('No CSV file found in the uploaded file!', 'danger')
            return redirect(request.url)

        report_id = df['id'].iloc[0]

        return redirect(url_for('report_redaction.report_redaction_viewer', report_id=report_id))

    return render_template("report_redaction_form.html", form=form)

import zipfile
import os
from flask import render_template, session

@report_redaction.route("/reportredactionviewer/<string:report_id>", methods=['GET'])
def report_redaction_viewer(report_id):
    # Get the DataFrame from the uploaded file
    df = find_llm_output_csv(session.get('pdf_file_zip', None))
    if df is None or len(df) == 0:
        flash('No CSV file found in the uploaded file!', 'danger')
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

    return render_template("report_redaction_viewer.html", report_id=report_id, previous_id=previous_id, next_id=next_id, report_number=current_index + 1, total_reports=len(df), personal_info_list=personal_info_list, enable_fuzzy=session.get('enable_fuzzy', False), threshold=session.get('threshold', 90))


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

@report_redaction.route("/reportredactionfileredacted/<string:id>")
def reportredactionfileredacted(id):
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

    # if not os.path.isfile(filename.replace(".pdf", "_redacted.pdf")):

    df = find_llm_output_csv(pdf_file_zip)
    if df is None or len(df) == 0:
        flash('No CSV file found in the uploaded file!', 'danger')
        return redirect(request.url)

    # Check if the current report ID exists in the DataFrame
    if id not in df['id'].values:
        flash(f'Report ID {id} not found!', 'danger')
        return redirect(request.url)

    # Get personal_info_list from df where id=id
    personal_info_list = df.loc[df['id'] == id, 'personal_info_list'].iloc[0]

    personal_info_list = convert_personal_info_list(personal_info_list)

    if session.get('exclude_single_chars', False):
        personal_info_list = [item for item in personal_info_list if len(item) > 1]

    if session.get('enable_fuzzy', False):
        fuzzy_matches = find_fuzzy_matches(df.loc[df['id'] == id, 'report'].iloc[0], personal_info_list, threshold=int(session.get('threshold', 90)), scorer=session.get('scorer', 'WRatio'))
    else:
        fuzzy_matches = []

    anonymize_pdf(filename, personal_info_list, filename.replace(".pdf", "_redacted.pdf"), fuzzy_matches)

    time.sleep(1)

    socketio.emit('reportredaction_done', {'enable_fuzzy': session.get('enable_fuzzy', False), 'threshold': session.get('threshold', 90), 'fuzzy_matches': fuzzy_matches})


    # Redaction is yet to be implemented TODO

    # Serve the PDF file
    return send_file(filename.replace(".pdf", "_redacted.pdf"), mimetype='application/pdf')

