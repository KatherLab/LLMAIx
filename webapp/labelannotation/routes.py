import io
import json
import os
import re
import tempfile
import uuid
import zipfile
import ast
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import fitz


from flask import (
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
    current_app,
)
import pandas as pd

from webapp.report_redaction.utils import find_llm_output_csv

from . import labelannotation
from .forms import LLMAnnotationResultsForm, LabelSelectorForm
from .. import set_mode
from ..report_redaction.utils import generate_confusion_matrix_from_counts, generate_confusion_matrix_from_matrix

# quick and dirty workaround for session variable. Use object storage and database instead in the future!
file_cache = {}
annotation_file = ""
pdf_file_zip = ""
pdf_filepath = ""
label_type_mapping = {}
report_summary_dict = {}

@labelannotation.before_request
def before_request():
    set_mode(session, current_app.config["MODE"])


@labelannotation.route("/labelannotation", methods=["GET", "POST"])
def main():
    form = LLMAnnotationResultsForm()

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
        
        global annotation_file
        annotation_file = None

        # If annotation_file is sent, save it to a temporary directory
        if "annotation_file" in request.files:
            annotation_file = request.files["annotation_file"]
            if annotation_file.filename != "":
                annotation_file_path = os.path.join(temp_dir, annotation_file.filename)
                annotation_file.save(annotation_file_path)
                annotation_file = annotation_file_path

        # Extract the content from the uploaded file and save it to a new temporary directory
        content_temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(content_temp_dir)


        # Save the path to the extracted content directory to the session variable
        global pdf_file_zip
        pdf_file_zip = content_temp_dir
        global pdf_filepath
        pdf_filepath = None
        global label_type_mapping
        label_type_mapping = {}
        global report_summary_dict
        report_summary_dict = {}

        # First report id
        df = find_llm_output_csv(pdf_file_zip)
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded file!", "danger")
            return redirect(request.url)

        report_id = df["id"].iloc[0]

        if "submit-viewer" in request.form:

            if annotation_file is None:
                form.annotation_file.errors.append(
                    "For the viewer page, please upload an annotation file. Viewer without annotation will be implemented in the future."
                )  # Manually set error message for the 'name' field
                return render_template("labelannotation_form.html", form=form)

            session["current_labelannotation_job"] = None
            return redirect(
                url_for("labelannotation.labelannotationviewer", report_id=report_id)
            )

        elif "submit-metrics" in request.form:
            if annotation_file is None:
                form.annotation_file.errors.append(
                    "For the metrics summary page, please upload an annotation file."
                )  # Manually set error message for the 'name' field
                return render_template("labelannotation_form.html", form=form)

            try:
                # check if annotation file is csv
                if os.path.splitext(annotation_file)[-1] == ".csv":
                    df = pd.read_csv(annotation_file, dtype=str)
                    assert df is not None and len(df) > 0
                elif os.path.splitext(annotation_file)[-1] == ".xlsx":
                    df = pd.read_excel(annotation_file, dtype=str)
                    assert df is not None and len(df) > 0
                else:
                    raise Exception("Invalid annotation file format!")
                assert df is not None and len(df) > 0
            except Exception as e:
                flash(f"Invalid annotation file format: {e}", "danger")
                return redirect(request.url)

            return redirect(
                url_for("labelannotation.labelannotationmetrics", report_id=report_id)
            )

    return render_template("labelannotation_form.html", form=form)


def calculate_metrics_multiclass(annotation_label, llm_output_label, all_classes, label_name):
    # Convert labels to a list
    annotation_labels = [annotation_label]
    llm_output_labels = [llm_output_label]
    
    # Calculate confusion matrix
    cm = confusion_matrix(annotation_labels, llm_output_labels, labels=all_classes)
    
    # Calculate accuracy
    accuracy = accuracy_score(annotation_labels, llm_output_labels)
    
    # Calculate macro precision, recall, and F1 score
    precision = precision_score(annotation_labels, llm_output_labels, labels=all_classes, average='macro', zero_division=0)
    recall = recall_score(annotation_labels, llm_output_labels, labels=all_classes, average='macro', zero_division=0)
    f1 = f1_score(annotation_labels, llm_output_labels, labels=all_classes, average='macro', zero_division=0)
    
    metrics = {
        'confusion_matrix_list': cm.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics



def calculate_metrics_boolean(annotation_label, llm_output_label, label_name):
    # Convert labels to boolean
    annotation_label = annotation_label.lower() in ["true", "1", "yes", "y", "ja"]
    llm_output_label = llm_output_label.lower() in ["true", "1", "yes", "y", "ja"]

    # Convert to lists for sklearn compatibility
    y_true = [annotation_label]
    y_pred = [llm_output_label]

    # Initialize the metrics entry
    metrics_entry = {
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'f1': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'specificity': 0,
        'false_positive_rate': 0,
        'false_negative_rate': 0
    }

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    
    metrics_entry['tp'] = tp
    metrics_entry['tn'] = tn
    metrics_entry['fp'] = fp
    metrics_entry['fn'] = fn
    
    total = tp + tn + fp + fn
    if total > 0:
        metrics_entry['accuracy'] = accuracy_score(y_true, y_pred)
    
    if tp + fp > 0:
        metrics_entry['precision'] = precision_score(y_true, y_pred)
    
    if tp + fn > 0:
        metrics_entry['recall'] = recall_score(y_true, y_pred)
    
    if tn + fp > 0:
        metrics_entry['specificity'] = tn / (tn + fp)
    
    if fp + tn > 0:
        metrics_entry['false_positive_rate'] = fp / (fp + tn)
    
    if fn + tp > 0:
        metrics_entry['false_negative_rate'] = fn / (fn + tp)
    
    if metrics_entry['precision'] + metrics_entry['recall'] > 0:
        metrics_entry['f1'] = f1_score(y_true, y_pred)
    
    metrics_entry['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics_entry

def calculate_metrics_stringmatch(annotation_string, llm_output_string, label_name):
    # Create binary labels based on string matching
    annotation_label = 1 if str(annotation_string).lower() == str(llm_output_string).lower() else 0
    llm_output_label = 1
    
    # Convert to list format for sklearn
    annotation_labels = [annotation_label]
    llm_output_labels = [llm_output_label]
    
    # Calculate accuracy
    accuracy = accuracy_score(annotation_labels, llm_output_labels)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(annotation_labels, llm_output_labels, zero_division=0)
    recall = recall_score(annotation_labels, llm_output_labels, zero_division=0)
    f1 = f1_score(annotation_labels, llm_output_labels, zero_division=0)
    
    metrics = {
        'match': annotation_string == llm_output_string,
        'no_match': annotation_string != llm_output_string,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def calculate_metrics(annotation_labels, llm_output_labels, label_type_mapping:dict):

    # Initialize dictionaries to store label-wise metrics
    label_metrics = {}

    # Calculate metrics for each label
    for label in annotation_labels.keys():
        if label in llm_output_labels:
            if label_type_mapping[label]['label_type'] == "multiclass":
                label_metrics[label] = calculate_metrics_multiclass(str(annotation_labels[label]), str(llm_output_labels[label]), label_type_mapping[label]['label_classes'], label)
            elif label_type_mapping[label]['label_type'] == "boolean":
                label_metrics[label] = calculate_metrics_boolean(str(annotation_labels[label]), str(llm_output_labels[label]), label)
            elif label_type_mapping[label]['label_type'] == "stringmatch":
                label_metrics[label] = calculate_metrics_stringmatch(str(annotation_labels[label]).lower(), str(llm_output_labels[label]).lower(), label)

    overall_accuracy = sum([metric['accuracy'] for metric in label_metrics.values()]) / len(label_metrics)

    overall_metrics = {
        'accuracy': overall_accuracy,
    }

    return {
        'overall': overall_metrics,
        'label_wise': label_metrics
    }

def sum_confusion_matrices(confusion_matrices):

    if not confusion_matrices:
        return []
    
    if not confusion_matrices[0]:
        return confusion_matrices[1]

    # print("Sum: ", confusion_matrices)
    # Convert the first confusion matrix to a NumPy array
    summed_matrix = np.array(confusion_matrices[0])
    
    # Iterate through the rest of the confusion matrices and add them element-wise
    for cm in confusion_matrices[1:]:
        summed_matrix += np.array(cm)
    
    # Convert the summed matrix back to a list
    summed_matrix_list = summed_matrix.tolist()
    
    return summed_matrix_list

def calculate_final_metrics_boolean(tp, tn, fp, fn):
    # Convert to arrays for sklearn compatibility
    y_true = [1] * tp + [0] * tn + [1] * fn + [0] * fp
    y_pred = [1] * tp + [0] * tn + [0] * fn + [1] * fp

    metrics = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    return metrics

def calculate_final_metrics_multiclass(conf_matrix):
    # Assuming conf_matrix is a list of lists
    conf_matrix = np.array(conf_matrix)
    
    # Extract true labels and predicted labels from confusion matrix
    num_classes = conf_matrix.shape[0]
    y_true = []
    y_pred = []

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = conf_matrix[true_class, pred_class]
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)

    # Calculate metrics using sklearn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics

def calculate_final_metrics_stringmatch(match, no_match):
    total = match + no_match

    if total == 0:
        accuracy = 0
    else:
        accuracy = match / total

    metrics = {
        'accuracy': accuracy
    }

    return metrics


def accumulate_metrics(data_list):
    accumulated_metrics = {
        'overall': {'accuracy': 0},
        'label_wise': {}
    }

    for data in data_list:
        # for metric, value in data['metrics']['overall'].items():
        #     accumulated_metrics['overall'][metric] += value


        for label, label_metrics in data['metrics']['label_wise'].items():
            if label not in accumulated_metrics['label_wise']:
                accumulated_metrics['label_wise'][label] = {
                    'tp': 0,
                    'tn': 0,
                    'fp': 0,
                    'fn': 0,
                    'f1': 0,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'specificity': 0,
                    'false_positive_rate': 0,
                    'false_negative_rate': 0,
                    'match': 0,
                    'no_match': 0,
                }
                if label in label_type_mapping and label_type_mapping[label]['label_type'] == 'boolean':
                    accumulated_metrics['label_wise'][label]['confusion_matrix_list'] = [[0,0], [0,0]]
                elif label in label_type_mapping and label_type_mapping[label]['label_type'] == 'multiclass':
                    # initialize confusion matrix list with dimensions according to number of classes
                    number_of_classes = len(label_type_mapping[label]['label_classes'])
                    accumulated_metrics['label_wise'][label]['confusion_matrix_list'] = [[0]*number_of_classes for _ in range(number_of_classes)]

            if label in label_type_mapping and label_type_mapping[label]['label_type'] == 'boolean':
                accumulated_metrics['label_wise'][label]['tp'] += label_metrics['tp']
                accumulated_metrics['label_wise'][label]['tn'] += label_metrics['tn']
                accumulated_metrics['label_wise'][label]['fp'] += label_metrics['fp']
                accumulated_metrics['label_wise'][label]['fn'] += label_metrics['fn']

                accumulated_metrics['label_wise'][label]['confusion_matrix_list'] = sum_confusion_matrices([accumulated_metrics['label_wise'][label]['confusion_matrix_list'], [[label_metrics['tp'], label_metrics['fn']], [label_metrics['fp'], label_metrics['tn']]]])
            elif label in label_type_mapping and label_type_mapping[label]['label_type'] == 'stringmatch':
                accumulated_metrics['label_wise'][label]['match'] += label_metrics['match']
                accumulated_metrics['label_wise'][label]['no_match'] += label_metrics['no_match']
            elif label in label_type_mapping and label_type_mapping[label]['label_type'] == 'multiclass':
                # print("Sum confusion matrices: ", accumulated_metrics['label_wise'][label]['confusion_matrix_list'], label_metrics['confusion_matrix_list'])
                accumulated_metrics['label_wise'][label]['confusion_matrix_list'] = sum_confusion_matrices([accumulated_metrics['label_wise'][label]['confusion_matrix_list'], label_metrics['confusion_matrix_list']])

        
    for label, metrics in accumulated_metrics['label_wise'].items():
        if label in label_type_mapping and label_type_mapping[label]['label_type'] == 'boolean':
            tp = metrics['tp']
            tn = metrics['tn']
            fp = metrics['fp']
            fn = metrics['fn']
            
            final_metrics = calculate_final_metrics_boolean(tp, tn, fp, fn)
            
            # Update accumulated metrics with final calculated metrics
            accumulated_metrics['label_wise'][label].update(final_metrics)
        
        if label in label_type_mapping and label_type_mapping[label]['label_type'] == 'multiclass':
            confusion_matrix = metrics['confusion_matrix_list']
            final_metrics = calculate_final_metrics_multiclass(confusion_matrix)
            
            # Update accumulated metrics with final calculated metrics
            accumulated_metrics['label_wise'][label].update(final_metrics)
        
        if label in label_type_mapping and label_type_mapping[label]['label_type'] == 'stringmatch':
            match = metrics['match']
            no_match = metrics['no_match']
            
            final_metrics = calculate_final_metrics_stringmatch(match, no_match)
            
            # Update accumulated metrics with final calculated metrics
            accumulated_metrics['label_wise'][label].update(final_metrics)

    # Calculate averages for label-wise metrics
    for label, label_metrics in accumulated_metrics['label_wise'].items():
        # for metric in label_metrics:
        #     if metric not in ['tp', 'tn', 'fp', 'fn', 'confusion_matrix', 'match', 'no_match', 'confusion_matrix_list']:
        #         accumulated_metrics['label_wise'][label][metric] /= float(num_entries)
        
        # Calculate confusion matrix filepath
        confusion_matrix_filepath = os.path.join(
                tempfile.mkdtemp(), f"confusion_matrix_{label}.svg"
            )
        
        if label_type_mapping[label]['label_type'] == 'boolean':
            generate_confusion_matrix_from_counts(label_metrics['tp'], label_metrics['tn'], label_metrics['fp'], label_metrics['fn'], confusion_matrix_filepath, ['true', 'false'], title=f"Confusion Matrix for {label}", xlabel='LLM', ylabel='Ground Truth')
        elif label_type_mapping[label]['label_type'] == 'multiclass':
            generate_confusion_matrix_from_matrix(label_metrics['confusion_matrix_list'], confusion_matrix_filepath, title=f"Confusion Matrix for {label}", xlabel='LLM', ylabel='Ground Truth', classes=label_type_mapping[label]['label_classes'])
        else:
            generate_confusion_matrix_from_counts(label_metrics['tp'], label_metrics['tn'], label_metrics['fp'], label_metrics['fn'], confusion_matrix_filepath, ['true', 'false'], title=f"Confusion Matrix for {label}", xlabel='LLM', ylabel='Ground Truth')


        confusion_matrix_id = str(uuid.uuid4())
        accumulated_metrics['label_wise'][label]['confusion_matrix'] = confusion_matrix_id
        file_cache[confusion_matrix_id] = confusion_matrix_filepath

    # calculate overall metrics by adding up the same metrics from the label-wise metrics
    for label, label_metrics in accumulated_metrics['label_wise'].items():
        for metric, value in label_metrics.items():
            if metric in accumulated_metrics['overall'].keys():
                # print("Adding", label, metric, value)
                accumulated_metrics['overall'][metric] += value

    for metric, value in accumulated_metrics['overall'].items():
        accumulated_metrics['overall'][metric] /= float(len(accumulated_metrics['label_wise']))

    return accumulated_metrics

def generate_report_dict(row, df_annotation, label_type_mapping: dict) -> dict:
    report_dict = {}

    report_dict["id"] = row.id
    # report_dict["report"] = row.report
    report_dict['metadata'] = row.metadata
    # for annotation labels, find the corresponding row in the df_annotation (match report(without .pdf) == row.report) and get a list of dict with the other column labels as keys and the corresponding value in the row as value
    report_dict["annotation_labels"] = df_annotation[df_annotation["id"] == row.report_id_short]
    
    if len(report_dict["annotation_labels"]) == 0:
        raise Exception(f"No annotation found for report {row.report_id_short}")
    
    if len(report_dict["annotation_labels"]) > 1:
        raise Exception(f"Multiple annotations found for report {row.report_id_short}")
    
    report_dict["annotation_labels"] = report_dict["annotation_labels"].to_dict("list")

    # if len(df_annotation[df_annotation["id"].str.startswith(row.report_id_short)]) == 0:
    #     raise Exception("No annotation found for report " + row.report_id_short)
    
    # if len(df_annotation[df_annotation["id"].str.startswith(row.report_id_short)]) > 1:
    #     raise Exception("Multiple annotations found for report " + row.report_id_short)

    del report_dict["annotation_labels"]["id"]
    if "report" in report_dict["annotation_labels"]:
        del report_dict["annotation_labels"]["report"]
    # similar with llm output labels from the row, excluding id, report, metadata, matching_report, no_matching_report, report_redacted
    report_dict["llm_output_labels"] = {
        k: v
        for k, v in row._asdict().items()
        if k
        not in [
            "id",
            "report",
            "report_id_short",
            "metadata",
            "matching_report",
            "no_matching_report",
            "report_redacted",
            "Index",
            "personal_info_list",
            "masked_report",
        ]
    }

    annotation_labels = list(report_dict["annotation_labels"].keys())
    llm_output_labels = list(report_dict["llm_output_labels"].keys())

    if not all([key in annotation_labels for key in llm_output_labels]):
        raise Exception("Mismatch in label keys in llm output: " + str(
            llm_output_labels)
            + " vs annotation: "
            + str(annotation_labels)
        )

    # go trough values of annotation labels and use the first list element as value 
    for k, v in report_dict["annotation_labels"].items():
        if len(v) == 0:
            raise Exception("No value in annotation for key: " + k + " for report: " + row.id)
        report_dict["annotation_labels"][k] = str(v[0])

    # the same for llm output labels TODO this might not every time be right
    for k, v in report_dict["llm_output_labels"].items():
        value_list = ast.literal_eval(v)
        # choose the first none-empty value or "" if all values are empty
        report_dict["llm_output_labels"][k] = value_list[0]
        for value in value_list:
            if value != "":
                report_dict["llm_output_labels"][k] = str(value)
                break
            
    report_dict['metrics'] = calculate_metrics(report_dict['annotation_labels'], report_dict['llm_output_labels'], label_type_mapping)

    return report_dict

def generate_report_list(df, df_annotation, label_type_mapping: dict) -> list:
    report_list = []

    for row in df.itertuples():
        report_list.append(generate_report_dict(row, df_annotation, label_type_mapping))
    
    return report_list


def extract_first_non_empty_string(llm_output_values):
    result = []
    for value in llm_output_values:
        if value != "":
            # Convert string representation of list to actual list
            try:
                list_value = ast.literal_eval(value)
            except Exception:
                print("Error processing value: " + value)
                raise("Error processing value, there was probably an error during llm processing. Please check you llm output. The malformed value list: " + value)
            # Find the first non-empty string
            first_non_empty = next((item for item in list_value if item != ""), "")
            result.append(first_non_empty)
    return result


@labelannotation.route("/labelannotationselector", methods=["GET", "POST"])
def labelannotationselector():

    form = LabelSelectorForm()

    if os.path.splitext(annotation_file)[-1] == ".csv":
        df_annotation = pd.read_csv(annotation_file, dtype=str)
    elif os.path.splitext(annotation_file)[-1] == ".xlsx":
        df_annotation = pd.read_excel(annotation_file, dtype=str)
    elif annotation_file == "":
        flash("No annotation file uploaded!", "danger")
        return redirect(url_for("labelannotation.main"))
    else:
        flash("Invalid annotation file format!", "danger")
        return redirect(request.url)
    if df_annotation is None or len(df_annotation) == 0:
        flash(
            "No CSV file found in the annotation file or is the annotation file is empty!",
            "danger",
        )
        return redirect(request.url)

    df = find_llm_output_csv(pdf_file_zip)
    if df is None or len(df) == 0:
        flash("No CSV file found in the uploaded file!", "danger")
        return redirect(request.url)
    
    df = df.rename(columns=lambda x: x.replace(' ', '_'))
    df_annotation = df_annotation.rename(columns=lambda x: x.replace(' ', '_'))

    if form.validate_on_submit():
        global label_type_mapping
        label_type_mapping = {}

        label_data = {}
        for label in form.labels:
            label_data[label.label_name.data] = {'label_type': label.label_type.data, 'label_classes': label.label_classes.data.split(',')}

        label_type_mapping = label_data

        return redirect(url_for("labelannotation.labelannotationmetrics"))
    
    labels = [
        {'label_name': label, 'label_type': ''}
        for label in list(df.keys())
        if label
        not in [
            "id",
            "report",
            "report_id_short",
            "metadata",
            "matching_report",
            "no_matching_report",
            "report_redacted",
            "Index",
            "personal_info_list",
            "masked_report",
            "",
        ]
    ]

    for label in labels:
        llm_output_values = list(df[label["label_name"]])
        # Always choose first label TODO might not be what you want
        # llm_output_values = [ast.literal_eval(value)[0] for value in llm_output_values if value != ""]
        try:
            llm_output_values = extract_first_non_empty_string(llm_output_values)
        except Exception as e:
            flash(f"Error processing label {label['label_name']}: {e}", "danger")
            return redirect(url_for("labelannotation.main"))

        if label["label_name"] not in list(df_annotation.keys()):
            flash(f"Label {label['label_name']} not in the annotation file.", "danger")
            return redirect(url_for("labelannotation.main"))

        annotation_values = [value for value in list(df_annotation[label["label_name"]]) if isinstance(value, str)]
        if len(set(annotation_values)) == 2 and ("True" in annotation_values and "False" in annotation_values or "true" in annotation_values and "false" in annotation_values or "1" in annotation_values and "0" in annotation_values or 1 in annotation_values and 0 in annotation_values or "yes" in annotation_values and "no" in annotation_values):
            label["label_type"] = "boolean"
        elif set(llm_output_values) == set(annotation_values):
            label["label_type"] = "multiclass"
        else:
            label["label_type"] = "stringmatch"
        label["label_classes"] = ",".join(set(annotation_values))

    
        form.labels.append_entry(label)

    return render_template("labelannotation_selector.html", form=form)


def check_labels(df, df_annotation, label_type_mapping):

    for label in label_type_mapping:
        if label_type_mapping[label]['label_type'] == 'multiclass':
            llm_output_values = list(df[label])
            llm_output_values = extract_first_non_empty_string(llm_output_values)
            annotation_values = list(df_annotation[label])
            if set(llm_output_values) != set(annotation_values):
                flash(f"Label {label} is not multiclass. Annotation and LLM output have different classes.", "warning")
            if set(annotation_values) != set(label_type_mapping[label]['label_classes']):
                flash(f"Label {label} is not multiclass. Annotation and selected classes differ.", "warning")

        elif label_type_mapping[label]['label_type'] == 'boolean':
            llm_output_values = list(df[label])
            llm_output_values = extract_first_non_empty_string(llm_output_values)
            annotation_values = list(df_annotation[label])
            if len(set(llm_output_values)) > 2 or len(set(annotation_values)) > 2:
                flash(f"Label {label} is not boolean. Annotation or LLM output has too many values.", "warning")

            invalid_llm_output_values = {value for value in set(llm_output_values) if value not in [True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0', 'yes', 'no']}
            if len(invalid_llm_output_values) > 0:
                flash(f"Label {label} is not boolean. LLM output has invalid values: {invalid_llm_output_values}", "warning")

            invalid_annotation_values = {value for value in set(annotation_values) if value not in [True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0', 'yes', 'no']}
            if len(invalid_annotation_values) > 0:
                flash(f"Label {label} is not boolean. Annotation has invalid values: {invalid_annotation_values}", "warning")

        elif label_type_mapping[label]['label_type'] == 'stringmatch':
            llm_output_values = list(df[label])
            llm_output_values = extract_first_non_empty_string(llm_output_values)
            annotation_values = list(df_annotation[label])
            if "" in llm_output_values:
                flash(f"Label {label}: LLM output has empty values.", "warning")
            if "" in annotation_values:
                flash(f"Label {label}: Annotation has empty values.", "warning")


@labelannotation.route("/labelannotationmetrics", methods=["GET"])
def labelannotationmetrics():

    if label_type_mapping == {}:
        return redirect(url_for("labelannotation.labelannotationselector"))
    # if annotation file is csv file:
    if os.path.splitext(annotation_file)[-1] == ".csv":
        df_annotation = pd.read_csv(annotation_file, dtype=str)
    elif os.path.splitext(annotation_file)[-1] == ".xlsx":
        df_annotation = pd.read_excel(annotation_file, dtype=str)
    else:
        flash("Invalid annotation file format!", "danger")
        return redirect(request.url)
    if df_annotation is None or len(df_annotation) == 0:
        flash(
            "No CSV file found in the annotation file or is the annotation file is empty!",
            "danger",
        )
        return redirect(request.url)

    df = find_llm_output_csv(pdf_file_zip)
    if df is None or len(df) == 0:
        flash("No CSV file found in the uploaded file!", "danger")
        return redirect(request.url)

    # Extract report names from the 'id' column in df1
    df["report_id_short"] = (
        df["id"].str.split(".pdf", expand=True)[0].str.split("$", expand=True)[0]
    )

    # Check if the extracted report names from df1 are present in df2

    df["report_id_short"] = df["report_id_short"].astype(str)
    df_annotation["id"] = df_annotation["id"].astype(str)

    # df["matching_report"] = df["report_id_short"].isin(df_annotation["id"])
    
    merged_df = pd.merge(df, df_annotation, left_on="report_id_short", right_on="id", how="left", indicator=True)
    df["matching_report"] = merged_df["_merge"] == "both"

    # Find IDs with no matching report
    df["no_matching_report"] = ~df["matching_report"]

    # print(df[df["no_matching_report"]][["id", "report_id_short"]])

    if len(df[df["no_matching_report"]][["id", "report_id_short"]]) > 0:
        flash(
            f"Reports not found in the annotation file: {df[df['no_matching_report']][['id', 'report_id_short']]}",
            "danger",
        )
        return redirect(url_for("labelannotation.main"))

    # metrics = calculate_metrics(df, df_annotation)

    global report_summary_dict
    if report_summary_dict == {}:

        try:
            metadata = json.loads(df["metadata"].iloc[0])
        except Exception as e:
            flash(f"Error loading metadata from llm output file: {e}", "danger")
            breakpoint()
            return redirect(url_for("labelannotation.main"))

        report_summary_dict["metadata"] = metadata

        # itertuple does not allow spaces in identifiers
        df = df.rename(columns=lambda x: x.replace(' ', '_'))
        df_annotation = df_annotation.rename(columns=lambda x: x.replace(' ', '_'))
        
        try:
            report_summary_dict['report_list'] = generate_report_list(df, df_annotation, label_type_mapping)
        except Exception as e:
            flash(f"Something went wrong: {e}", "danger")
            report_summary_dict['report_list'] = generate_report_list(df, df_annotation, label_type_mapping)
            return redirect(url_for("labelannotation.main"))

        report_summary_dict["accumulated_metrics"] = accumulate_metrics(report_summary_dict['report_list'])

        session["current_labelannotation_job"] = True

        try: 
            check_labels(df, df_annotation, label_type_mapping)
        except Exception as e:
            flash(f"Something went wrong processing the llm output: {e}", "danger")
            return redirect(url_for("labelannotation.main"))

    return render_template(
        "labelannotation_metrics.html", report_summary_dict=report_summary_dict, label_type_mapping=label_type_mapping
    )

def generate_export_df(result_dict: list):
    # Iterate over every report in result_list['report_list'] and add all scores in ['scores'] as one row to the dataframe, use ['id'] as id column
    # df = pd.DataFrame()

    scores_to_include = []

    for label in result_dict['accumulated_metrics']['label_wise'].keys():
        for metric in result_dict['accumulated_metrics']['label_wise'][label].keys():
            if label != "personal_info_list":
                scores_to_include.append("{}${}".format(label, metric))
            else:
                scores_to_include.append(metric)

    # Initialize a dictionary to store the extracted scores
    data = {"id": []}
    for score in scores_to_include:
        data[score] = []

    macro_scores = {}

    accumulated_metrics = result_dict.get("accumulated_metrics", {})

    data['id'].append('type')
    for score in scores_to_include:
        data[score].append(label_type_mapping[score.split("$")[0]]['label_type'])

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
            if label in entry["metrics"]['label_wise'] and metric in entry["metrics"]['label_wise'][label]:
                data[score].append(entry["metrics"]['label_wise'][label][metric])
            else:
                print("Score {} not found in entry {}".format(score, entry["id"]))
                data[score].append(None)  # Append None if score doesn't exist

            macro_scores[score] = accumulated_metrics['label_wise'][label][metric]

    # Append macro and micro scores to the DataFrame
    data["id"].append("macro_scores")
    for score in scores_to_include:
        data[score].append(macro_scores.get(score, None))

    df = pd.DataFrame(data)

    return df

@labelannotation.route("/labelannotationdownload", methods=["GET"])
def labelannotationdownload():
    '''Enables the download of all metrics including confusion matrices.

    Returns:
        Zip archive with all metrics in csv format and confusion matrices in png format
    '''

    global report_summary_dict
    if report_summary_dict == {}:
        flash("Metrics not yet calculated", "danger")
        return redirect(url_for("labelannotation.labelannotationmetrics"))
    
    df = generate_export_df(report_summary_dict)


    # Find the job ID by looking in pdf_file_zip path for a csv file starting with "llm-output-" and using everything after that without the .csv file extension as job id
    pattern = re.compile(r"llm-output-(.*)\.csv")
    extracted_parts = []
    for filename in os.listdir(pdf_file_zip):
        match = pattern.match(filename)
        if match:
            extracted_parts.append(match.group(1))
    
    if len (extracted_parts) == 0:
        flash("No llm output csv found!", "danger")
        return redirect(url_for("labelannotation.labelannotationmetrics"))
    else:
        job_id = extracted_parts[0]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add the CSV file to the zip
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False, float_format="%.4f")
        csv_buffer.seek(0)
        zip_file.writestr(f"metrics_{job_id}.csv", csv_buffer.getvalue())

        # Loop through redacted PDFs
        for report_dict in report_summary_dict["report_list"]:  # Corrected variable name
            redacted_pdf_filename = os.path.join(pdf_file_zip, f'{report_dict["id"]}.pdf')
            #check if file exists
            if not os.path.exists(redacted_pdf_filename):
                flash("Pdf not found: " + redacted_pdf_filename, "danger")
                return redirect(url_for("labelannotation.labelannotationmetrics"))

            # Apply redactions using PyMuPDF
            with fitz.open(redacted_pdf_filename) as pdf:

                # Store the redacted PDF content in memory
                pdf_buffer = io.BytesIO()
                pdf.save(pdf_buffer)
                pdf_buffer.seek(0)

                # Add redacted PDF to the zip
                zip_file.writestr(
                    os.path.basename(redacted_pdf_filename),
                    pdf_buffer.getvalue(),
                )
                
        macro_scores_row = df[df['id'] == 'macro_scores'].iloc[0]

        for labelmetric in macro_scores_row.index:
            if labelmetric != 'id':
                # label = labelmetric.split('$')[0]
                metric = labelmetric.split('$')[1]

                if metric == 'confusion_matrix':
                    confusion_matrix_filepath = os.path.join(pdf_file_zip, file_cache[macro_scores_row[labelmetric]])
                    # Add confusion matrix to the zip, store it in memory and use the file name of confusion_matrix_filepath
                    with open(confusion_matrix_filepath, 'rb') as f:
                        zip_file.writestr(os.path.basename(confusion_matrix_filepath), f.read())
        
        # include pdf_file_zip/llm-output-JOBID.csv file in zip
        zip_file.writestr(f"llm-output-{job_id}.csv", open(os.path.join(pdf_file_zip, f"llm-output-{job_id}.csv"), 'rb').read())

    # Send the zip file as an attachment
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"metrics_{job_id}.zip",
        mimetype="application/zip",
    )


@labelannotation.route("/labelannotationviewer", methods=["GET", "POST"])
def labelannotationviewer():

    if label_type_mapping == {}:
        return redirect(url_for("labelannotation.labelannotationselector"))
    report_id = request.args.get("report_id")

    df = find_llm_output_csv(pdf_file_zip)
    if df is None or len(df) == 0:
        flash("No llm output CSV file found in the uploaded zip file! Is this a llm output zip file? Note: If you extract and re-zip the llm output file, the content must not be in a directory.", "danger")
        return redirect(request.url)
    
    df["report_id_short"] = (
        df["id"].str.split(".pdf", expand=True)[0].str.split("$", expand=True)[0]
    )

    if report_id not in df["id"].values:
        report_id = df["id"].iloc[0]
    global pdf_filepath
    pdf_filepath = os.path.join(pdf_file_zip, f"{report_id}.pdf")

    current_index = df[df["id"] == report_id].index[0]

    previous_id = df.at[current_index - 1, "id"] if current_index > 0 else None
    next_id = df.at[current_index + 1, "id"] if current_index < len(df) - 1 else None

    if annotation_file.endswith(".csv"):
        df_annotation = pd.read_csv(annotation_file, dtype=str)
    elif annotation_file.endswith(".xlsx"):
        df_annotation = pd.read_excel(annotation_file, dtype=str)
    else:
        flash("Invalid annotation file format! Only CSV and XLSX files are supported.", "danger")
        return redirect(request.url)
    
    if df_annotation is not None and len(df_annotation) == 0:
        flash("Annotation File found but empty!", "danger")
        return redirect(request.url)
    
    report_dict = None

    df = df.rename(columns=lambda x: x.replace(' ', '_'))
    df_annotation = df_annotation.rename(columns=lambda x: x.replace(' ', '_'))
    
    df["report_id_short"] = df["report_id_short"].astype(str)
    df_annotation["id"] = df_annotation["id"].astype(str)

    if df_annotation is not None:
        # row = df[df["id"] == report_id].iloc[0]
        for row in df.itertuples():
            if row.id == report_id:
                try:
                    report_dict = generate_report_dict(row, df_annotation, label_type_mapping)
                except Exception as e:
                    flash(f"Error processing reports and annotations: {e}", "danger")
                    breakpoint()
                    return redirect(url_for("labelannotation.main"))
                break
    
    try:
        check_labels(df, df_annotation, label_type_mapping)
    except Exception as e:
        flash(f"Error processing llm output: {e}", "danger")
        return redirect(url_for("labelannotation.main"))

    return render_template(
        "labelannotation_viewer.html",
        report_id=report_id,
        previous_id=previous_id,
        next_id=next_id,
        report_number=current_index + 1,
        total_reports=len(df),
        report_dict = report_dict,
        label_type_mapping=label_type_mapping
    )


@labelannotation.route("/labelannotationcacheprovider/<file_id>", methods=["GET"])
def labelannotationcacheprovider(file_id):
    # Check if the file ID exists in the session's file cache
    if file_id not in file_cache:
        print("File ID not found in the cache variable")
        breakpoint()
        abort(404)  # Return 404 if the file ID is not in the cache

    file_path = file_cache[file_id]  # Get the file path from the cache

    # print("Send: ", file_path)

    # Send the file if it exists, otherwise return 404
    try:
        return send_file(file_path, mimetype='image/svg+xml')
    except FileNotFoundError:
        abort(404)

@labelannotation.route("/labelannotationpdfprovider/<report_id>", methods=["GET"])
def labelannotationpdfprovider(report_id):
    if report_id:
        df = find_llm_output_csv(pdf_file_zip)
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded file!", "danger")
            return 404

        return send_file(
            os.path.join(pdf_file_zip, f"{report_id}.pdf"),
            mimetype="application/pdf",
        )

    return send_file(pdf_filepath, mimetype="application/pdf")
