from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField, SelectField, MultipleFileField
from flask_wtf.file import FileAllowed, FileRequired

class PreprocessUploadForm(FlaskForm):

    files = MultipleFileField("Upload Files", validators=[FileRequired(), FileAllowed(['pdf', 'txt', 'csv', 'jpg', 'png', 'jpeg', 'docx', 'odt'], 'Only PDF, TXT, CSV, JPG, PNG, DOCX, and ODT files are allowed!')])

    submit = SubmitField("Upload")