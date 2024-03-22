from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField, SelectField
from wtforms.validators import DataRequired, Length

class PreprocessUploadForm(FlaskForm):
    submit = SubmitField("Upload")