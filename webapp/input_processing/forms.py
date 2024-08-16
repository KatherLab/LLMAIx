from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, MultipleFileField, BooleanField, SelectField, ValidationError
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import validators

def validate_optional_integer(form, field):
    # If field is not empty, attempt to convert and validate it
    if field.data:
        try:
            value = int(field.data)
        except ValueError:
            raise ValidationError('Not a valid integer value.')

        # Validate the integer value
        if not (100 <= value <= 128000):
            raise ValidationError('Value must be between 100 and 128000.')



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

    text_split = StringField(
        "Split Length (after N characters), set for anonymization of very long reports",
        validators=[validate_optional_integer]
    )


    ocr_method = SelectField("OCR Method", choices=[('tesseract', 'Tesseract (OCRmyPDF)'), ('phi3vision', 'Phi3Vision'), ('surya', 'Surya')], default='tesseract')
    force_ocr = BooleanField("Force OCR", default=False)
    remove_previous_ocr = BooleanField("Remove Previous OCR (DANGEROUS)", default=False)

    submit = SubmitField("Upload")
