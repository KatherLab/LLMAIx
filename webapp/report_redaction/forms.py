from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, BooleanField, TextAreaField, MultipleFileField, FileField, FloatField, validators, SelectField, IntegerField
from wtforms.validators import DataRequired, ValidationError
import os
from flask import current_app


class ReportRedactionForm(FlaskForm):

    file = FileField("File", validators=[
        FileRequired(),  
        FileAllowed(['zip'], 'Only .zip file allowed!')
    ])

    enable_fuzzy = BooleanField("Enable fuzzy matching")
    threshold = IntegerField("Threshold (0-100):", validators=[validators.NumberRange(0,100)], default=90)

    submit = SubmitField("Report Redaction")