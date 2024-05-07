import os
import tempfile
import zipfile

from flask import flash, redirect, render_template, request, send_file, session, url_for

from webapp.report_redaction.utils import find_llm_output_csv

from . import labelannotation
from .form import LLMAnnotationResultsForm


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
                flash("No annotation file was sent!", "danger")
                return redirect(request.url)

            print("Metrics Page")

    return render_template("labelannotation_form.html", form=form)


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
