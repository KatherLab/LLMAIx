from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, MultipleFileField
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import validators


class PreprocessUploadForm(FlaskForm):

    files = MultipleFileField("Upload Files", validators=[FileRequired(), FileAllowed(
        ['pdf', 'txt', 'csv', 'jpg', 'png', 'jpeg', 'docx', 'xlsx'], 'Only PDF, TXT, CSV, JPG, PNG, XLSX and DOCX files are allowed!')])

    # Add Integer Fields with values between 100 and 128000
    text_split = IntegerField("Split Length", validators=[
                              validators.NumberRange(min=100, max=128000)], default=14000)

    submit = SubmitField("Upload")
