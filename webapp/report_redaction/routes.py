import json
import os
import tempfile
from webapp.llm_processing.utils import anonymize_pdf, convert_personal_info_list, find_fuzzy_matches, replace_personal_info
from webapp.report_redaction.utils import InceptionAnnotationParser, find_llm_output_csv
from . import report_redaction
from flask import abort, render_template, request, redirect, send_file, url_for, session, flash
from .forms import ReportRedactionForm
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
    pdf_file_zip = session.get('pdf_file_zip', None)
    if not pdf_file_zip:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID
    
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

    orig_pdf_path = os.path.join(pdf_file_zip, f"{report_id}.pdf")
    session['redacted_pdf_filename'], dollartext_redacted, fuzzy_matches = load_redacted_pdf(personal_info_list, orig_pdf_path, df, report_id)


    if session.get('annotation_file', None):
        try:
            session['annotation_pdf_filepath'], dollartext_annotated, original_text, colormap = load_annotated_pdf(report_id)
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

    return render_template("report_redaction_viewer.html", report_id=report_id, previous_id=previous_id, next_id=next_id, report_number=current_index + 1, total_reports=len(df), personal_info_list=personal_info_list, enable_fuzzy=session.get('enable_fuzzy', False), threshold=session.get('threshold', 90), colormap = colormap, scores=scores, fuzzy_matches = fuzzy_matches)

def generate_score_dict(ground_truth, comparison, original_text, round_digits = 2):
    # check if both dollartext_annotated and dollartext_redacted are set
    print("CHECK SCORES")

    # if not session.get('dollartext_annotated', None) or not session.get('dollartext_redacted', None) or not session.get('original_text', None):
    #     print("Dollartext not yet set")
    #     # breakpoint()
    #     return
    
    print("Ground truth: ", ground_truth)
    print("Comparison: ", comparison)

    precision, recall, accuracy, f1_score, false_positive_rate, false_negative_rate, confusion_matrix_filepath = calculate_metrics(ground_truth, comparison, original_text, '■')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    print("False positive rate: ", false_positive_rate)
    print("False negative rate: ", false_negative_rate)

    score_dict = {
        'precision': round(precision, round_digits),
        'recall': round(recall, round_digits),
        'accuracy': round(accuracy, round_digits),
        'f1_score': round(f1_score, round_digits),
        'false_positive_rate': round(false_positive_rate, round_digits),
        'false_negative_rate': round(false_negative_rate, round_digits)
    }

    return score_dict, confusion_matrix_filepath

def generate_confusion_matrix_from_counts(tp, tn, fp, fn, filename):
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns

    plt.switch_backend('Agg') # otherwise it would not run outside of the main thread
    # Constructing the confusion matrix from counts
    cm = np.array([[tp, fp], [fn, tn]])

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Redacted', 'Not Redacted'], yticklabels=['Redacted', 'Not Redacted'])
    plt.xlabel('Annotation')
    plt.ylabel('LLM Anonymizer')
    plt.title('Confusion Matrix')
    plt.savefig(filename)  # Save the plot to a file
    plt.close()

def calculate_metrics(ground_truth, automatic_redacted, original_text, redacted_char):
    assert len(ground_truth) == len(automatic_redacted) == len(original_text), "All texts must have the same length"

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # comparison_text = ""
    # Build a text 

    for gt_char, auto_char, orig_char in zip(ground_truth, automatic_redacted, original_text):
        # Ignore spaces and other characters for the score calculation!
        if orig_char != ' ' and orig_char != ',' and orig_char != '.' and orig_char != '!' and orig_char != '?' and orig_char != ':' and orig_char != ';' and orig_char != '-' and orig_char != '(' and orig_char != ')' and orig_char != '"' and orig_char != "'" and orig_char != '\n':
            if gt_char == redacted_char and auto_char == redacted_char:
                true_positives += 1
                # comparison_text += "R"
            elif gt_char != redacted_char and auto_char == redacted_char:
                false_positives += 1
                # comparison_text += "+"
            elif gt_char != redacted_char and auto_char != redacted_char:
                true_negatives += 1
                # comparison_text += orig_char # "N"
            elif gt_char == redacted_char and auto_char != redacted_char:
                false_negatives += 1
                # comparison_text += "-"
        else:
            # comparison_text += orig_char # "I"
            pass
            # Optional: count all the spaces in the original text as true negatives
            # true_negatives += 1

    # print("Comparison text: ", comparison_text)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    accuracy = (true_positives + true_negatives) / (len([char for char in original_text if char != ' ']))  # Ignoring spaces in original text
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    false_positive_rate = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    confusion_matrix_filepath = os.path.join(tempfile.mkdtemp(), "confusion_matrix.svg")
    generate_confusion_matrix_from_counts(true_positives, true_negatives, false_positives, false_negatives, confusion_matrix_filepath)

    return precision, recall, accuracy, f1_score, false_positive_rate, false_negative_rate, confusion_matrix_filepath

def load_annotated_pdf(report_id):
    json_filename = '.'.join(report_id.split('$')[0].split('.')[:-1]) + ".json"

    # Open uploaded annotation file
    annotation_zip_file = session.get('annotation_file', None)
    
    if not annotation_zip_file:
        print("Annotation File not set.")
        abort(404)

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
    
    pdf_file_zip = session.get('pdf_file_zip', None)
    if not pdf_file_zip:
        print("PDF File not set.")
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Construct the filename based on the session variable and the ID
    original_pdf_filepath = os.path.join(pdf_file_zip, f"{report_id}.pdf")

    # Check if the file exists
    if not os.path.isfile(original_pdf_filepath):
        print("File not found: ", original_pdf_filepath)
        abort(404)

    annotation_pdf_filepath, dollartext_annotated, original_text = annoparser.apply_annotations_to_pdf(original_pdf_filepath)

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

def load_redacted_pdf(personal_info_list, filename, df, id):

    if session.get('exclude_single_chars', False):
        personal_info_list = [item for item in personal_info_list if len(item) > 1]

    if session.get('enable_fuzzy', False):
        fuzzy_matches = find_fuzzy_matches(df.loc[df['id'] == id, 'report'].iloc[0], personal_info_list, threshold=int(session.get('threshold', 90)), scorer=session.get('scorer', 'WRatio'))
    else:
        fuzzy_matches = []

    dollartext_redacted = generated_dollartext_stringlist(filename, personal_info_list, len(fuzzy_matches) > 0)

    anonymize_pdf(filename, personal_info_list, filename.replace(".pdf", "_redacted.pdf"), fuzzy_matches)

    return filename.replace(".pdf", "_redacted.pdf"), dollartext_redacted, fuzzy_matches

    # socketio.emit('reportredaction_done', {'enable_fuzzy': session.get('enable_fuzzy', False), 'threshold': session.get('threshold', 90), 'fuzzy_matches': fuzzy_matches})

@report_redaction.route("/reportredactionfileredacted/<string:id>")
def reportredactionfileredacted(id):

    return send_file(session['redacted_pdf_filename'], mimetype='application/pdf')

def generated_dollartext_stringlist(filename, information_list, use_fuzzy_matching=False):
        """Replace all occurrences of the strings in information_list with dollar signs in the pdf text"""

        # Load pdf text with pymupdf
        import fitz
        pdf = fitz.open(filename)

        text = ""
        for page in pdf:
            for word_block in page.get_text("dict")["blocks"]:
                for line in word_block['lines']:
                    for span in line['spans']:
                        word = span['text']
                        text += word

        return replace_personal_info(text, information_list, use_fuzzy_matching=use_fuzzy_matching, generate_dollarstring=True)

@report_redaction.route("/reportredactionconfusionmatrix")
def reportredactionconfusionmatrix():
    # 
    confusion_matrix_svg_filepath = session.get('confusion_matrix_filepath', None)

    if not confusion_matrix_svg_filepath:
        # Handle the case where the path to the zip file is not found
        abort(404)

    # Serve the PDF file
    return send_file(confusion_matrix_svg_filepath, mimetype='image/svg+xml')