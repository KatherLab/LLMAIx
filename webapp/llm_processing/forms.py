from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import BooleanField, StringField, SubmitField, TextAreaField, FileField, FloatField, validators, SelectField, FormField, FieldList, IntegerField
from wtforms.validators import ValidationError
import wtforms
import os

default_system_prompt = r"""You are a helpful assistant that helps extract information from medical reports."""

default_prompt = r"""From the following medical report, extract the following information and return it in JSON format:

    patientname: The full name of the patient.
    patientsex: The sex of the patient. Use "m" for male, "w" for female, and "d" for diverse.

This is the medical report:
{report}

The JSON:"""


default_grammar = r"""root ::= allrecords

allrecords ::= (
  "{"
ws "\"patientname\":" ws "\"" char{2,60} "\"" ","
ws "\"patientsex\":" ws "\"" ( "m" | "w" | "d" ) "\"" ","

  ws "}"
  ws
)

ws ::= ([ \t\n])?

char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
"""


class FileExistsValidator:
    def __init__(self, message=None, path=""):
        self.message = message or 'File does not exist.'
        self.path = path

    def __call__(self, form, field):
        filename = os.path.join(self.path, field.data)
        if not os.path.exists(filename):
            raise ValidationError(self.message)


class GrammarValidator:
    def __call__(self, form, field):
        enable_grammar = form.enable_grammar.data
        grammar = field.data
        if enable_grammar:
            # print("Checking grammar placeholder")
            # TODO: Add grammar validation
            pass
        if enable_grammar and not grammar:
            raise ValidationError(
                'Grammar field is required when "Enable Grammar" is checked.')

class GrammarField(wtforms.Form):
  field_name = StringField('Field Name', validators=[validators.Optional()])
  field_type = SelectField('Field Type', choices=[('string', 'String'), ('number', 'Number'), ('fp-number', 'Floating Point Number'), ('boolean', 'Boolean'), ('options', 'Categories'), ('custom', 'Custom Rule')])
  string_length = IntegerField('Max Length', [validators.Optional()])
  string_min_length = IntegerField('Min Length', [validators.Optional()], default=1)
  number_length = IntegerField('Max Length', [validators.Optional()])
  number_min_length = IntegerField('Min Length', [validators.Optional()], default=1)
  fp_number_length = IntegerField('FP Number Length', [validators.Optional()])
  options = StringField('Categories (comma-separated)', [validators.Optional()])
  custom_rule = StringField('Custom Rule (RULENAME ::= <YOURCUSTOMRULE>)', [validators.Optional()])


class LLMPipelineForm(FlaskForm):
    def __init__(self, config_file_path, model_path, *args, **kwargs):
        super(LLMPipelineForm, self).__init__(*args, **kwargs)
        import yaml

        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Extract model choices from config data
        model_choices = [(model["file_name"], model["display_name"])
                         for model in config_data["models"]]

        # Set choices for the model field
        self.model.choices = model_choices
        if model_path:
            self.model.validators = [FileExistsValidator(
                message='Model path does not exist.', path=model_path)]
            # self.model.validators.append(FileExistsValidator(message='File does not exist.', path=model_path))
        else:
            raise ValueError("Model path is required")

    job_name = StringField("Job Name:")

    file = FileField("File", validators=[
        FileRequired(),
        FileAllowed(['zip'], # for now remove csv and xlsx as they are not (longer and yet) supported
                    'Only .zip files allowed!')
    ])
    grammar = TextAreaField("Grammar:", validators=[], default=default_grammar)

    grammarbuilder = FieldList(FormField(GrammarField), min_entries=1, max_entries=100)

    extra_grammar_rules = TextAreaField("Extra Grammar Rules:", validators=[], default="")

    chat_endpoint = BooleanField("Chat Endpoint:", default=False)

    system_prompt = TextAreaField("System Prompt:", validators=[], default=default_system_prompt)
    prompt = TextAreaField("Prompt:", validators=[], default=default_prompt)
    variables = StringField(
        "Variables (separated by commas):", validators=[], default="Patienteninfos")
    temperature = FloatField("Temperature:", validators=[
                             validators.NumberRange(0, 1)], default=0)
    model = SelectField("Model:", validators=[])

    n_predict = IntegerField("n_predict:", validators=[validators.NumberRange(1, 96000)], default=1024)

    top_k = IntegerField("Top K:", validators=[validators.NumberRange(1, 100)], default=1)
    top_p = FloatField("Top P:", validators=[validators.NumberRange(0, 1)], default=0)

    submit = SubmitField("Run Pipeline")
