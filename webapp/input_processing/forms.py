from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, MultipleFileField, BooleanField, SelectField
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import validators


class PreprocessUploadForm(FlaskForm):

    def __init__(self, method, *args, **kwargs):
        super(PreprocessUploadForm, self).__init__(*args, **kwargs)
        self.method = method
        self.set_ocr_choices()

    def set_ocr_choices(self):
        if self.method == 'informationextraction':
            self.ocr_method.choices = [('tesseract', 'Tesseract (OCRmyPDF)'), ('phi3vision', 'Phi3Vision'), ('surya', 'Surya')]
        elif self.method == 'anonymizer':
            self.ocr_method.choices = [('tesseract', 'Tesseract (OCRmyPDF)'), ('surya', 'Surya')]
        else:
            self.ocr_method.choices = [('tesseract', 'Tesseract (OCRmyPDF)'), ('phi3vision', 'Phi3Vision'), ('surya', 'Surya')]


    files = MultipleFileField("Upload Files (csv/excel: id and report columns required)", validators=[FileRequired(), FileAllowed(
        ['pdf', 'txt', 'csv', 'jpg', 'png', 'jpeg', 'docx', 'xlsx'], 'Only PDF, TXT, CSV, JPG, PNG, XLSX and DOCX files are allowed!')])

    # Add Integer Fields with values between 100 and 128000
    text_split = IntegerField("Split Length", validators=[
                              validators.NumberRange(min=100, max=128000)], default=14000)


    ocr_method = SelectField("OCR Method", choices=[('tesseract', 'Tesseract (OCRmyPDF)'), ('phi3vision', 'Phi3Vision'), ('surya', 'Surya')], default='tesseract')
    force_ocr = BooleanField("Force OCR", default=False)

    submit = SubmitField("Upload")
