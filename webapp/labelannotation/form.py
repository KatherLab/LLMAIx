from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import SubmitField, FileField


class LLMAnnotationResultsForm(FlaskForm):

    file = FileField("File", validators=[
        FileRequired(),
        FileAllowed(['zip'], 'Only .zip file allowed!')
    ])

    annotation_file = FileField("Annotation File (csv)", validators=[
        FileAllowed(['zip'], 'Only .csv file allowed!')
    ])

    submit = SubmitField("LLM Output Viewer")
    submit_metrics = SubmitField("LLM Output Metrics")