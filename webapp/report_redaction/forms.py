from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import SubmitField, BooleanField, FileField, validators, SelectField, IntegerField


class ReportRedactionForm(FlaskForm):

    file = FileField("File", validators=[
        FileRequired(),
        FileAllowed(['zip'], 'Only .zip file allowed!')
    ])

    annotation_file = FileField("Annotation File", validators=[
        FileAllowed(['zip'], 'Only .zip file allowed!')
    ])

    enable_fuzzy = BooleanField("Enable fuzzy matching")
    threshold = IntegerField(
        "Threshold (0-100):", validators=[validators.NumberRange(0, 100)], default=90)

    scorer = SelectField("Fuzzy Matching Method:", choices=[
                         ('QRatio', 'QRatio'), ('WRatio', 'WRatio')])

    exclude_single_chars = BooleanField("Exclude single characters")

    submit = SubmitField("Report Redaction Viewer")
    submit_scores = SubmitField("Report Redaction Metrics")
