import io
import os
import tempfile
import uuid
import zipfile
from flask import flash, redirect, render_template, request, send_file, url_for
import pandas as pd
import ast

from webapp.report_redaction.utils import find_llm_output_csv
from . import annotationhelper
from .forms import AnnotationHelperForm, LabelSelectorForm, ReAnnotationForm

class AnnotationHelperJob:
    """
    Class representing an annotation helper job.
    """
    def __init__(self, job_id: str, llm_output_file_path: str, llm_output_df: pd.DataFrame) -> None:
        """
        Initialize an AnnotationHelperJob instance.

        Parameters
        ----------
        job_id : str
            Unique identifier for the job.
        llm_output_file_path : str
            Path to the LLM output file that was used to create this job.
        llm_output_df : pandas.DataFrame
            DataFrame containing the LLM output.

        Notes
        -----
        The job creation time is stored as a string in the format "dd.mm.yyyy hh:mm:ss".
        The metadata dictionary is extracted from the first row of the LLM output DataFrame.
        The record list is a list of dictionaries containing the record id, report text, labels and status.
        The labels list is a list of strings containing all unique labels from the LLM output.
        The label type mapping is a dictionary mapping each label to its type (string, boolean, or float).
        """
        self.job_id = job_id
        self.llm_output_df = llm_output_df
        self.llm_output_file_path = llm_output_file_path
        # job creation time in dd.mm.yyyy hh:mm:ss
        self.creation_datetime = pd.Timestamp.now().strftime("%d.%m.%Y %H:%M:%S")
        self.metadata_dict = ast.literal_eval(self.llm_output_df.iloc[0]['metadata'])
        self.record_list = []
        self.labels = self._get_unique_labels()
        self.label_type_mapping = {}

        for index, row in self.llm_output_df.iterrows():
            record_entry = {
                "record_id": row['id'],
                "report": row['report'],
                "labels": [],
                "status": "pending",
            }

            # loop through row columns and use every column label if it is not "id" or "report"
            for column in row.index:
                if column not in ['id', 'report', 'metadata', 'masked_report', 'personal_info_list']:
                    # always use first value, assume there are no reports being split (which is more unlikely in the future with longer context sizes)
                    record_entry['labels'].append({
                        "label": column,
                        "value": ast.literal_eval(row[column])[0] if str(row[column]).startswith('[') else row[column], 
                        "value_annotator": ast.literal_eval(row[column])[0] if str(row[column]).startswith('[') else row[column], 
                    })

            self.record_list.append(record_entry)
            
    def _get_unique_labels(self) -> list[str]:
        """
        Get all unique labels from the LLM output.

        Returns:
            list[str]: A list of all unique labels from the LLM output.
        """
        labels: set[str] = set()
        for index, row in self.llm_output_df.iterrows():
            for column in row.index:
                if column not in ['id', 'report', 'metadata', 'masked_report', 'personal_info_list']:
                    labels.add(column)
        return list(labels)

    def get_number_of_records(self):
        return len(self.llm_output_df)

    def get_completed_records(self):
        return len([record for record in self.record_list if record['status'] == 'completed'])

    def get_job_info(self):
        return {
            "job_id": self.job_id,
            "creation_datetime": self.creation_datetime,
            "number_of_records": self.get_number_of_records(),
            "completed_records": self.get_completed_records(),
            "metadata": self.metadata_dict,
            "labels": self.labels,
            "label_type_mapping": self.label_type_mapping,
            "record_list": self.record_list
        }
    
    def get_record_by_id(self, record_id):
        for record in self.record_list:
            if record['record_id'] == record_id:
                return record
        return None
    
    def update_record_by_id(self, record_id, new_record):
        for record in self.record_list:
            if record['record_id'] == record_id:
                record.update(new_record)
                return True
        return False
    
    def update_record_labels_by_id(self, record_id, label, value):
        # print("Updating record", record_id, "label", label, "value", value)
        for record in self.record_list:
            if record['record_id'] == record_id:
                for record_label in record['labels']:
                    if record_label['label'] == label:
                        record_label['value_annotator'] = value
                        return True
        return False
    
    def update_record_status_by_id(self, record_id, status):
        for record in self.record_list:
            if record['record_id'] == record_id:
                record['status'] = status
                return True
        return False
    
    def get_reannotated_df(self):
        # return a csv file with the id and report columns from the llm_output_df and the self.labels as other columns having the value_annotator
        # you need to match the ids with the record_ids

        annotated_df = self.llm_output_df[['id', 'report']].copy()

        true_equivalents = ["True", "true", "Ja", "ja", "Yes", "yes", "correct", "wahr", "1", "TRUE", True]
        false_equivalents = ["False", "false", "Nein", "nein", "No", "no", "falsch", "0", "FALSE", False]

        # add columns for each label
        for label in self.labels:
            annotated_df.loc[:,label] = ""

        for record in self.record_list:

            for label in record['labels']:

                if self._get_label_type(label['label']) == 'boolean':

                    if label['value_annotator'] in true_equivalents:
                        label['value_annotator'] = '1'
                    elif label['value_annotator'] in false_equivalents:
                        label['value_annotator'] = '0'
                    else:
                        print("WARNING: Boolean value not recognized:", label['value_annotator'])
                
                annotated_df.loc[annotated_df['id'] == record['record_id'], label['label']] = label['value_annotator']

        # edit all ids: split on dollar and only take the first part before the last dollar, then split again and remove everything after the last dot
        annotated_df['id'] = annotated_df['id'].apply(lambda x: '.'.join('$'.join(x.split('$')[:-1]).split('.')[:-1]))

        return annotated_df
    
    def _get_label_type(self, label):
        return self.label_type_mapping[label]['label_type']



annotation_jobs:dict[str, AnnotationHelperJob] = {}


@annotationhelper.route("/annotationhelperdownload", methods=["GET"])
def annotationhelperdownload():

    job_id = request.args.get('job_id')

    # print("Downloading job", job_id)

    if job_id not in annotation_jobs:
        flash("Job not found!", "danger")
        return redirect(url_for("annotationhelper.annotationhelperqueue"))

    annotated_df = annotation_jobs[job_id].get_reannotated_df()

    # build a zip file with all the jobs llm_output_file_path and the annotated_df as a csv file and send the zip file

    zip_buffer = io.BytesIO()

    # include all files in this path in the zip
    llm_output_file_path = annotation_jobs[job_id].llm_output_file_path

    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr(f"annotated-{job_id}.csv", annotated_df.to_csv(index=False).encode('utf-8'))

        for file in os.listdir(llm_output_file_path):
            zip_file.write(os.path.join(llm_output_file_path, file), arcname=file)

    # Send the zip file as an attachment
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"annotated-{job_id}.zip",
        mimetype="application/zip",
    )
    

@annotationhelper.route("/annotationhelperselector", methods=["GET", "POST"])
def annotationhelperselector():

    job_id = request.args.get('job_id')

    form = LabelSelectorForm()

    if job_id not in annotation_jobs:
        flash("Job not found!", "danger")
        return redirect(url_for("annotationhelper.annotationhelperqueue"))

    if form.validate_on_submit():
        
        for label in form.labels.data:
            if label['label_name'] in annotation_jobs[job_id].labels:
                if label['label_name'] not in annotation_jobs[job_id].label_type_mapping:
                    annotation_jobs[job_id].label_type_mapping[label['label_name']] = {}
                annotation_jobs[job_id].label_type_mapping[label['label_name']]['label_type'] = label['label_type']
                annotation_jobs[job_id].label_type_mapping[label['label_name']]['label_classes'] = label['label_classes']
            else:
                flash(f"Label {label['label_name']} not in the job, should not happen.", "danger")
                return redirect(url_for("annotationhelper.annotationhelperselector"))

        return redirect(url_for("annotationhelper.annotationhelperoverview", job_id=job_id))
    
    labels = [{'label_name': label, 'label_type': ''} for label in annotation_jobs[job_id].labels]
    label_type_mapping = annotation_jobs[job_id].label_type_mapping

    for label in labels:
        if label_type_mapping and label["label_name"] in label_type_mapping:
            label["label_type"] = label_type_mapping[label["label_name"]]["label_type"]
            label["label_classes"] = label_type_mapping[label["label_name"]]["label_classes"]
        else:
            llm_output_values = annotation_jobs[job_id].llm_output_df[label["label_name"]]
            llm_output_values = [value for value in llm_output_values if isinstance(value, str)]
            # check if any of the values starts with "["
            if any(str(value).startswith('[') for value in llm_output_values):
                llm_output_values = extract_first_non_empty_string(llm_output_values)

            # check what label type (boolean, multiclass, stringmatch) could be used for a label based on the values (multiclass if at least one value appears more than one time)
            if len(set(llm_output_values)) == 2 and ("True" in llm_output_values and "False" in llm_output_values or "true" in llm_output_values and "false" in llm_output_values or 1 in llm_output_values and 0 in llm_output_values or "yes" in llm_output_values and "no" in llm_output_values or "ja" in llm_output_values and "nein" in llm_output_values):
                label["label_type"] = "boolean"
            elif len(set(llm_output_values)) < len(llm_output_values):
                label["label_type"] = "multiclass"
            else:
                label["label_type"] = "stringmatch"
            label["label_classes"] = ",".join(set(llm_output_values))
        
        form.labels.append_entry(label)

    return render_template("labelannotation_selector.html", form=form)


@annotationhelper.route('/annotationhelperqueue', methods=["GET"])
def annotationhelperqueue():

    annohelper_queue = {}

    for job_id, job in annotation_jobs.items():
        annohelper_queue[job_id] = job.get_job_info()

    return render_template('annotationhelper_queue.html', annohelper_queue=annohelper_queue)


@annotationhelper.route('/annotationhelperoverview', methods=["GET"])
def annotationhelperoverview():

    job_id = request.args.get("job_id")

    if job_id not in annotation_jobs:
        flash("Job not found!", "danger")
        return redirect(url_for("annotationhelper.annotationhelperqueue"))
        

    if not annotation_jobs[job_id].label_type_mapping:
        flash("Please select the corresponding label types.", "success")
        return redirect(url_for("annotationhelper.annotationhelperselector", job_id=job_id))
    
    # job_info = annotation_jobs[job_id].get_job_info()

    return render_template('annotationhelper_overview.html', job_info=annotation_jobs[job_id].get_job_info())


@annotationhelper.route('/annotationhelperviewer', methods=["GET", "POST"])
def annotationhelperviewer():

    job_id = request.args.get("job_id")
    record_id = request.args.get("record_id")

    print("Job id:", job_id, "Record id:", record_id)

    if job_id not in annotation_jobs:
        print("Job not found!")
        flash("Job not found!", "danger")
        return redirect(url_for("annotationhelper.annotationhelperqueue"))
    
    # check if dict with record_id == record_id exists in annnotation_jobs[job_id].record_list and get its index
    for record_index, record in enumerate(annotation_jobs[job_id].record_list):
        if record["record_id"] == record_id:
            print("Record found at index", record_index)
            break
    else:
        print("Record not found!")
        flash("Record not found!", "danger")
        return redirect(url_for("annotationhelper.annotationhelperoverview", job_id=job_id))
    
    # print("Init form for record", record_id)
    # breakpoint()

    form = ReAnnotationForm()

    # print("Form errors", form.errors)
    # print("Request method", request.method)

    if request.method == 'POST':
        # set all choices in case of multiclass

        job_info = annotation_jobs[job_id].get_job_info()
        record = annotation_jobs[job_id].get_record_by_id(record_id)

        for i, label in enumerate(record["labels"]):
            label_type = job_info['label_type_mapping'][label["label"]]["label_type"]

            if label_type == "multiclass":
                classes = job_info['label_type_mapping'][label["label"]]["label_classes"].split(",")

                choices = [(c, c) for c in classes]

                form.labels[i].annotator_categories.choices = choices

    if form.validate_on_submit():
        # print("submit form")

        # record = annotation_jobs[job_id].get_record_by_id(record_id)

        for i, label in enumerate(form.labels):
            # print("Label", i, label.data['label_name'])
            if label.data['label_type'] == 'multiclass':
                # print("Update label", i, label.data['annotator_categories'])
                annotation_jobs[job_id].update_record_labels_by_id(record_id, label.data['label_name'], label.data['annotator_categories'])
            elif label.data['label_type'] == 'boolean':
                # print("Update label", i, label.data['annotator_boolean'])
                annotation_jobs[job_id].update_record_labels_by_id(record_id, label.data['label_name'], label.data['annotator_boolean'])
            elif label.data['label_type'] == 'stringmatch':
                # print("Update label", i, label.data['annotator_string'])
                annotation_jobs[job_id].update_record_labels_by_id(record_id, label.data['label_name'], label.data['annotator_string'])
        
        annotation_jobs[job_id].update_record_status_by_id(record_id, "completed")

        if form.submit_next.data:
            # Handle navigation to the next record
            if record_index + 1 < len(annotation_jobs[job_id].record_list):
                go_to_record = annotation_jobs[job_id].record_list[record_index + 1]['record_id']
            else:
                flash("Next record not found!", "danger")
                go_to_record = 'overview'
        elif form.submit_previous.data:
            # Handle navigation to the previous record
            if record_index - 1 >= 0:
                go_to_record = annotation_jobs[job_id].record_list[record_index - 1]['record_id']
            else:
                flash("Previous record not found!", "danger")
                go_to_record = 'overview'
        
        elif form.submit_save.data:
            go_to_record = record_id

        if not go_to_record or go_to_record in ['overview']:
            return redirect(url_for("annotationhelper.annotationhelperoverview", job_id=job_id))

        return redirect(url_for("annotationhelper.annotationhelperviewer", job_id=job_id, record_id=go_to_record))
    else:
        print(form.errors)
    
    job_info = annotation_jobs[job_id].get_job_info()

    record = annotation_jobs[job_id].get_record_by_id(record_id)

    if not request.method == 'POST':
        # print("Init form")
        for i, label in enumerate(record["labels"]):
            form.labels.append_entry()
            label_type = job_info['label_type_mapping'][label["label"]]["label_type"]
            form.labels[i].label_name.data = label["label"]
            form.labels[i].label_type.data = label_type

            if label_type == "boolean":
                true_equivalents = ["True", "true", "Ja", "ja", "Yes", "yes", "correct", "wahr", "1", True]
                false_equivalents = ["False", "false", "Nein", "nein", "No", "no", "falsch", "0", False]
                if label["value"] not in true_equivalents and label["value"] not in false_equivalents:
                    flash(f"Invalid boolean value for label {label['label']}: {label['value']}", "danger")
                if label["value_annotator"] not in true_equivalents and label["value_annotator"] not in false_equivalents:
                    flash(f"Invalid boolean value for label {label['label']}: {label['value_annotator']}", "danger")
                form.labels[i].llm_boolean.data = True if label["value"] in true_equivalents else False
                form.labels[i].llm_boolean.disabled = True
                form.labels[i].annotator_boolean.data = True if label["value_annotator"] in true_equivalents else False

            elif label_type == "multiclass":
                classes = job_info['label_type_mapping'][label["label"]]["label_classes"].split(",")
                if label["value"] not in classes:
                    flash(f"Invalid multiclass value for label {label['label']}: {label['value']}", "danger")
                if label["value_annotator"] not in classes:
                    flash(f"Invalid multiclass value for label {label['label']}: {label['value_annotator']}", "danger")
                form.labels[i].llm_categories.data = label["value"]
                choices = [(c, c) for c in classes]
                form.labels[i].llm_categories.choices = choices
                form.labels[i].llm_categories.disabled = True
                form.labels[i].annotator_categories.choices = choices
                form.labels[i].annotator_categories.data = label["value_annotator"]

            elif label_type == "stringmatch":
                form.labels[i].llm_string.data = label["value"]
                form.labels[i].llm_string.disabled = True
                form.labels[i].annotator_string.data = label["value_annotator"]
            
            else:
                flash("Invalid label type!", "danger")
                return redirect(url_for("annotationhelper.annotationhelperviewer", job_id=job_id, record_id=record_id))
        
    return render_template('annotationhelper_viewer.html', form=form, job_info=job_info, record_id=record_id, record_status=record['status'], record_index=record_index, next_record = annotation_jobs[job_id].record_list[record_index+1] if record_index+1 < len(annotation_jobs[job_id].record_list) else None, previous_record = annotation_jobs[job_id].record_list[record_index-1] if record_index-1 >= 0 else None)


@annotationhelper.route('/annotationhelperpdfprovider', methods=["GET"])
def annotationhelperpdfprovider():

    job_id = request.args.get("job_id")
    record_id = request.args.get("record_id")

    if job_id not in annotation_jobs:
        print("Job not found!")
        return 404
        # flash("Job not found!", "danger")
        # return redirect(url_for("annotationhelper.annotationhelperqueue"))

    for i, record in enumerate(annotation_jobs[job_id].record_list):
        if record["record_id"] == record_id:
            break
    else:
        print("Record not found!")
        return 404
        # flash("Record not found!", "danger")
        # return redirect(url_for("annotationhelper.annotationhelperoverview", job_id=job_id))
    

    if not os.path.exists(os.path.join(annotation_jobs[job_id].llm_output_file_path, f"{record_id}.pdf")):
        print("pdf not found")
        return 404

    return send_file(
            os.path.join(annotation_jobs[job_id].llm_output_file_path, f"{record_id}.pdf"),
            mimetype="application/pdf",
        )

@annotationhelper.route('/annotationhelper', methods=["GET", "POST"])
def annotationhelperform():
    form = AnnotationHelperForm()

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
        

        # Extract the content from the uploaded file and save it to a new temporary directory
        content_temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(content_temp_dir)

        # First report id
        df = find_llm_output_csv(content_temp_dir)
        if df is None or len(df) == 0:
            flash("No CSV file found in the uploaded file!", "danger")
            return redirect(request.url)
        
        annotation_job_id = uuid.uuid4().hex

        annotation_jobs[annotation_job_id] = AnnotationHelperJob(annotation_job_id, content_temp_dir, df)

        return redirect(
            url_for("annotationhelper.annotationhelperqueue")
        )

    return render_template("annotationhelper_form.html", form=form)

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