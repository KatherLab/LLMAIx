from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, MultipleFileField, BooleanField, SelectField, SelectMultipleField, ValidationError
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed, FileRequired

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
    # Language selection fields for different OCR methods
    tesseract_languages = SelectMultipleField('Tesseract Languages', choices=[
        ('afr', 'Afrikaans'),
        ('amh', 'Amharic'),
        ('ara', 'Arabic'),
        ('asm', 'Assamese'),
        ('aze', 'Azerbaijani'),
        ('aze-cyrl', 'Azerbaijani (Cyrillic)'),
        ('bel', 'Belarusian'),
        ('ben', 'Bengali'),
        ('bod', 'Tibetan Standard'),
        ('bos', 'Bosnian'),
        ('bre', 'Breton'),
        ('bul', 'Bulgarian'),
        ('cat', 'Catalan'),
        ('ceb', 'Cebuano'),
        ('ces', 'Czech'),
        ('chi-sim', 'Chinese - Simplified'),
        ('chi-sim-vert', 'Chinese - Simplified (vertical)'),
        ('chi-tra', 'Chinese - Traditional'),
        ('chi-tra-vert', 'Chinese - Traditional (vertical)'),
        ('chr', 'Cherokee'),
        ('cos', 'Corsican'),
        ('cym', 'Welsh'),
        ('dan', 'Danish'),
        ('deu', 'German'),
        ('div', 'Divehi'),
        ('dzo', 'Dzongkha'),
        ('ell', 'Greek'),
        ('eng', 'English'),
        ('enm', 'English, Middle (1100-1500)'),
        ('epo', 'Esperanto'),
        ('est', 'Estonian'),
        ('eus', 'Basque'),
        ('fao', 'Faroese'),
        ('fas', 'Persian'),
        ('fil', 'Filipino'),
        ('fin', 'Finnish'),
        ('fra', 'French'),
        ('frk', 'German (Fraktur)'),
        ('frm', 'French, Middle (ca.1400-1600)'),
        ('fry', 'Frisian (Western)'),
        ('gla', 'Gaelic (Scots)'),
        ('gle', 'Irish'),
        ('glg', 'Galician'),
        ('grc', 'Greek, Ancient (to 1453)'),
        ('guj', 'Gujarati'),
        ('hat', 'Haitian'),
        ('heb', 'Hebrew'),
        ('hin', 'Hindi'),
        ('hrv', 'Croatian'),
        ('hun', 'Hungarian'),
        ('hye', 'Armenian'),
        ('iku', 'Inuktitut'),
        ('ind', 'Indonesian'),
        ('isl', 'Icelandic'),
        ('ita', 'Italian'),
        ('ita-old', 'Italian - Old'),
        ('jav', 'Javanese'),
        ('jpn', 'Japanese'),
        ('jpn-vert', 'Japanese (vertical)'),
        ('kan', 'Kannada'),
        ('kat', 'Georgian'),
        ('kat-old', 'Old Georgian'),
        ('kaz', 'Kazakh'),
        ('khm', 'Khmer'),
        ('kir', 'Kyrgyz'),
        ('kmr', 'Kurmanji (Latin)'),
        ('kor', 'Korean'),
        ('kor-vert', 'Korean (vertical)'),
        ('lao', 'Lao'),
        ('lat', 'Latin'),
        ('lav', 'Latvian'),
        ('lit', 'Lithuanian'),
        ('ltz', 'Luxembourgish'),
        ('mal', 'Malayalam'),
        ('mar', 'Marathi'),
        ('mkd', 'Macedonian'),
        ('mlt', 'Maltese'),
        ('mon', 'Mongolian'),
        ('mri', 'Maori'),
        ('msa', 'Malay'),
        ('mya', 'Burmese'),
        ('nep', 'Nepali'),
        ('nld', 'Dutch'),
        ('nor', 'Norwegian'),
        ('oci', 'Occitan (post 1500)'),
        ('ori', 'Oriya'),
        ('osd', 'script and orientation'),
        ('pan', 'Punjabi'),
        ('pol', 'Polish'),
        ('por', 'Portuguese'),
        ('pus', 'Pashto'),
        ('que', 'Quechua'),
        ('ron', 'Romanian'),
        ('rus', 'Russian'),
        ('san', 'Sanskrit'),
        ('sin', 'Sinhala'),
        ('slk', 'Slovakian'),
        ('slv', 'Slovenian'),
        ('snd', 'Sindhi'),
        ('spa', 'Spanish'),
        ('spa-old', 'Spanish, Castilian - Old'),
        ('sqi', 'Albanian'),
        ('srp', 'Serbian'),
        ('srp-latn', 'Serbian (Latin)'),
        ('sun', 'Sundanese'),
        ('swa', 'Swahili'),
        ('swe', 'Swedish'),
        ('syr', 'Syriac'),
        ('tam', 'Tamil'),
        ('tat', 'Tatar'),
        ('tel', 'Telugu'),
        ('tgk', 'Tajik'),
        ('tha', 'Thai'),
        ('tir', 'Tigrinya'),
        ('ton', 'Tonga'),
        ('tur', 'Turkish'),
        ('uig', 'Uyghur'),
        ('ukr', 'Ukrainian'),
        ('urd', 'Urdu'),
        ('uzb', 'Uzbek'),
        ('uzb-cyrl', 'Uzbek (Cyrillic)'),
        ('vie', 'Vietnamese'),
        ('yid', 'Yiddish'),
        ('yor', 'Yoruba')
    ], default=['eng'])
    
    surya_languages = SelectMultipleField('Surya Languages', choices=[
        ('_math', 'Math'),
        ('af', 'Afrikaans'),
        ('am', 'Amharic'),
        ('ar', 'Arabic'),
        ('as', 'Assamese'),
        ('az', 'Azerbaijani'),
        ('be', 'Belarusian'),
        ('bg', 'Bulgarian'),
        ('bn', 'Bengali'),
        ('br', 'Breton'),
        ('bs', 'Bosnian'),
        ('ca', 'Catalan'),
        ('cs', 'Czech'),
        ('cy', 'Welsh'),
        ('da', 'Danish'),
        ('de', 'German'),
        ('el', 'Greek'),
        ('en', 'English'),
        ('eo', 'Esperanto'),
        ('es', 'Spanish'),
        ('et', 'Estonian'),
        ('eu', 'Basque'),
        ('fa', 'Persian'),
        ('fi', 'Finnish'),
        ('fr', 'French'),
        ('fy', 'Western Frisian'),
        ('ga', 'Irish'),
        ('gd', 'Scottish Gaelic'),
        ('gl', 'Galician'),
        ('gu', 'Gujarati'),
        ('ha', 'Hausa'),
        ('he', 'Hebrew'),
        ('hi', 'Hindi'),
        ('hr', 'Croatian'),
        ('hu', 'Hungarian'),
        ('hy', 'Armenian'),
        ('id', 'Indonesian'),
        ('is', 'Icelandic'),
        ('it', 'Italian'),
        ('ja', 'Japanese'),
        ('jv', 'Javanese'),
        ('ka', 'Georgian'),
        ('kk', 'Kazakh'),
        ('km', 'Khmer'),
        ('kn', 'Kannada'),
        ('ko', 'Korean'),
        ('ku', 'Kurdish'),
        ('ky', 'Kyrgyz'),
        ('la', 'Latin'),
        ('lo', 'Lao'),
        ('lt', 'Lithuanian'),
        ('lv', 'Latvian'),
        ('mg', 'Malagasy'),
        ('mk', 'Macedonian'),
        ('ml', 'Malayalam'),
        ('mn', 'Mongolian'),
        ('mr', 'Marathi'),
        ('ms', 'Malay'),
        ('my', 'Burmese'),
        ('ne', 'Nepali'),
        ('nl', 'Dutch'),
        ('no', 'Norwegian'),
        ('om', 'Oromo'),
        ('or', 'Oriya'),
        ('pa', 'Punjabi'),
        ('pl', 'Polish'),
        ('ps', 'Pashto'),
        ('pt', 'Portuguese'),
        ('ro', 'Romanian'),
        ('ru', 'Russian'),
        ('sa', 'Sanskrit'),
        ('sd', 'Sindhi'),
        ('si', 'Sinhala'),
        ('sk', 'Slovak'),
        ('sl', 'Slovenian'),
        ('so', 'Somali'),
        ('sq', 'Albanian'),
        ('sr', 'Serbian'),
        ('su', 'Sundanese'),
        ('sv', 'Swedish'),
        ('sw', 'Swahili'),
        ('ta', 'Tamil'),
        ('te', 'Telugu'),
        ('th', 'Thai'),
        ('tl', 'Tagalog'),
        ('tr', 'Turkish'),
        ('ug', 'Uyghur'),
        ('uk', 'Ukrainian'),
        ('ur', 'Urdu'),
        ('uz', 'Uzbek'),
        ('vi', 'Vietnamese'),
        ('xh', 'Xhosa'),
        ('yi', 'Yiddish'),
        ('zh', 'Chinese')
    ], default=['en'])
    
    force_ocr = BooleanField("Force OCR", default=False)
    remove_previous_ocr = BooleanField("Remove Previous OCR (DANGEROUS)", default=False)
    submit = SubmitField("Upload")
    
    def validate(self, extra_validators=None):
        # Call the parent validate method with the extra_validators parameter
        if not super(PreprocessUploadForm, self).validate(extra_validators=extra_validators):
            return False
            
        # Custom validation to check that at least one language is selected based on OCR method
        if self.ocr_method.data == 'tesseract' and not self.tesseract_languages.data:
            self.tesseract_languages.errors = ["At least one language must be selected for Tesseract OCR"]
            return False
        elif self.ocr_method.data == 'surya' and not self.surya_languages.data:
            self.surya_languages.errors = ["At least one language must be selected for Surya OCR"]
            return False
            
        return True