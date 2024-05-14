import json
import os
import tempfile
import zipfile
import ast
from sklearn.metrics import accuracy_score, f1_score


from flask import (
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
from .form import LLMAnnotationResultsForm
from .. import set_mode


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

        session["pdf_filepath"] = None

        session["report_list"] = []

        # First report id
        df = find_llm_output_csv(session["pdf_file_zip"])
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded file!", "danger")
            return redirect(request.url)

        report_id = df["id"].iloc[0]

        if "submit-viewer" in request.form:
            return redirect(
                url_for("labelannotation.labelannotationviewer", report_id=report_id)
            )

        elif "submit-metrics" in request.form:
            if session["annotation_file"] is None:
                form.annotation_file.errors.append(
                    "For the metrics summary page, please upload an annotation file."
                )  # Manually set error message for the 'name' field
                return render_template("labelannotation_form.html", form=form)

            try:
                df = pd.read_csv(session["annotation_file"])
                assert df is not None and len(df) > 0
            except Exception as e:
                flash(f"Invalid annotation file format: {e}", "danger")
                return redirect(request.url)

            return redirect(
                url_for("labelannotation.labelannotationmetrics", report_id=report_id)
            )

    return render_template("labelannotation_form.html", form=form)


def calculate_metrics(annotation_labels, llm_output_labels):

    # Initialize dictionaries to store label-wise metrics
    label_metrics = {}

    # Calculate metrics for each label
    for label in annotation_labels.keys():
        if label in llm_output_labels:
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            # Compare annotation and output labels
            annotation_value = annotation_labels[label]
            output_value = llm_output_labels[label]

            # True positives
            if annotation_value == output_value:
                true_positive = 1
            # False negatives
            elif output_value not in annotation_labels.values():
                false_negative = 1
            # False positives
            elif annotation_value != output_value:
                false_positive = 1
            # True negatives (excluding true positives)
            else:
                true_negative = 1

            # Calculate label-wise metrics
            label_accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
            label_f1 = f1_score([annotation_value], [output_value], average="macro")

            label_metrics[label] = {
                'tp': true_positive,
                'tn': true_negative,
                'fp': false_positive,
                'fn': false_negative,
                'f1': label_f1,
                'accuracy': label_accuracy
            }
        else:
            # If the label is not found in llm_output_labels, consider it as false negative
            label_metrics[label] = {
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 1,
                'f1': 0,
                'accuracy': 0
            }

    # Calculate overall metrics
    overall_tp = sum([metrics['tp'] for metrics in label_metrics.values()])
    overall_tn = sum([metrics['tn'] for metrics in label_metrics.values()])
    overall_fp = sum([metrics['fp'] for metrics in label_metrics.values()])
    overall_fn = sum([metrics['fn'] for metrics in label_metrics.values()])

    overall_accuracy = (overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn)
    overall_f1 = f1_score(list(annotation_labels.values()), list(llm_output_labels.values()), average='weighted')

    overall_metrics = {
        'tp': overall_tp,
        'tn': overall_tn,
        'fp': overall_fp,
        'fn': overall_fn,
        'f1': overall_f1,
        'accuracy': overall_accuracy
    }

    return {
        'overall': overall_metrics,
        'label_wise': label_metrics
    }

def accumulate_metrics(data_list):
    accumulated_metrics = {
        'overall': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'f1': 0, 'accuracy': 0},
        'label_wise': {}
    }

    for data in data_list:
        # Accumulate overall metrics
        for metric, value in data['metrics']['overall'].items():
            accumulated_metrics['overall'][metric] += value

        # Accumulate label-wise metrics
        for label, label_metrics in data['metrics']['label_wise'].items():
            if label not in accumulated_metrics['label_wise']:
                accumulated_metrics['label_wise'][label] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'f1': 0, 'accuracy': 0}

            for metric, value in label_metrics.items():
                try:
                    accumulated_metrics['label_wise'][label][metric] += value
                except Exception as e:
                    breakpoint()
                    print(e)

    # Calculate averages for overall metrics
    num_entries = len(data_list)
    for metric in accumulated_metrics['overall']:
        accumulated_metrics['overall'][metric] /= num_entries

    # Calculate averages for label-wise metrics
    for label, label_metrics in accumulated_metrics['label_wise'].items():
        for metric in label_metrics:
            accumulated_metrics['label_wise'][label][metric] /= num_entries

    return accumulated_metrics



@labelannotation.route("/labelannotationmetrics", methods=["GET"])
def labelannotationmetrics():
    df_annotation = pd.read_csv(session["annotation_file"])
    if df_annotation is None or len(df_annotation) == 0:
        flash(
            "No CSV file found in the annotation file or is the annotation file is empty!",
            "danger",
        )
        return redirect(request.url)

    df = find_llm_output_csv(session["pdf_file_zip"])
    if df is None or len(df) == 0:
        flash("No CSV file found in the uploaded file!", "danger")
        return redirect(request.url)

    # Extract report names from the 'id' column in df1
    df["report"] = (
        df["id"].str.split(".pdf", expand=True)[0].str.split("$", expand=True)[0]
    )

    # Check if the extracted report names from df1 are present in df2
    df["matching_report"] = df["report"].isin(df_annotation["report"])

    # Find IDs with no matching report
    df["no_matching_report"] = ~df["matching_report"]

    print(df[df["no_matching_report"]][["id", "report"]])

    if len(df[df["no_matching_report"]][["id", "report"]]) > 0:
        flash(
            f"Reports not found in the annotation file: {df[df['no_matching_report']][['id', 'report']]}",
            "danger",
        )
        return redirect(url_for("labelannotation.main"))

    # metrics = calculate_metrics(df, df_annotation)

    report_summary_dict = {}

    try:
        metadata = json.loads(df["metadata"].iloc[0])
    except Exception as e:
        flash(f"Error loading metadata from llm output file: {e}", "danger")
        breakpoint()
        return redirect(url_for("labelannotation.main"))

    report_summary_dict["metadata"] = metadata

    report_list = []

    for row in df.itertuples():
        report_dict = {}

        report_dict["id"] = row.id
        report_dict["report"] = row.report
        # for annotation labels, find the corresponding row in the df_annotation (match report(without .pdf) == row.report) and get a list of dict with the other column labels as keys and the corresponding value in the row as value
        report_dict["annotation_labels"] = df_annotation[
            df_annotation["report"].str.startswith(row.report)
        ].to_dict("list")
        del report_dict["annotation_labels"]["report"]
        # similar with llm output labels from the row, excluding id, report, metadata, matching_report, no_matching_report, report_redacted
        report_dict["llm_output_labels"] = {
            k: v
            for k, v in row._asdict().items()
            if k
            not in [
                "id",
                "report",
                "metadata",
                "matching_report",
                "no_matching_report",
                "report_redacted",
                "Index",
                "personal_info_list",
                "masked_report",
            ]
        }

        if (
            not report_dict["llm_output_labels"].keys()
            == report_dict["annotation_labels"].keys()
        ):
            breakpoint()
            flash(
                "Mismatch in label keys in annotation and llm output: "
                + str(report_dict["llm_output_labels"].keys())
                + " vs "
                + str(report_dict["annotation_labels"].keys()),
                "danger",
            )
            return redirect(url_for("labelannotation.main"))

        # go trough values of annotation labels and use the first list element as value
        for k, v in report_dict["annotation_labels"].items():
            report_dict["annotation_labels"][k] = v[0]

        # the same for llm output labels
        for k, v in report_dict["llm_output_labels"].items():
            value_list = ast.literal_eval(v)
            # choose the first none-empty value or "" if all values are empty
            report_dict["llm_output_labels"][k] = value_list[0]
            for value in value_list:
                if value != "":
                    report_dict["llm_output_labels"][k] = value
                    break
        
        report_dict['metrics'] = calculate_metrics(report_dict['annotation_labels'], report_dict['llm_output_labels'])

        report_list.append(report_dict)
    
    report_summary_dict['report_list'] = report_list

    report_summary_dict["accumulated_metrics"] = accumulate_metrics(report_list)

    return render_template(
        "labelannotation_metrics.html", report_summary_dict=report_summary_dict
    )


@labelannotation.route("/labelannotationviewer", methods=["GET", "POST"])
def labelannotationviewer():
    report_id = request.args.get("report_id")

    df = find_llm_output_csv(session["pdf_file_zip"])
    if df is None or len(df) == 0:
        flash("No CSV file found in the uploaded file!", "danger")
        return redirect(request.url)

    if report_id not in df["id"].values:
        report_id = df["id"].iloc[0]

    session["pdf_filepath"] = os.path.join(session["pdf_file_zip"], f"{report_id}.pdf")

    current_index = df[df["id"] == report_id].index[0]

    previous_id = df.at[current_index - 1, "id"] if current_index > 0 else None
    next_id = df.at[current_index + 1, "id"] if current_index < len(df) - 1 else None

    return render_template(
        "labelannotation_viewer.html",
        report_id=report_id,
        previous_id=previous_id,
        next_id=next_id,
    )


@labelannotation.route("/labelannotationpdfprovider/<report_id>", methods=["GET"])
def labelannotationpdfprovider(report_id):
    if report_id:
        df = find_llm_output_csv(session["pdf_file_zip"])
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded file!", "danger")
            return 404

        return send_file(
            os.path.join(session["pdf_file_zip"], f"{report_id}.pdf"),
            mimetype="application/pdf",
        )

    return send_file(session["pdf_filepath"], mimetype="application/pdf")
