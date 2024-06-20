from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed, DataRequired
from wtforms import StringField, SubmitField, TextAreaField, FileField, FloatField, validators, SelectField, FormField, FieldList, IntegerField
from wtforms.validators import ValidationError
import wtforms
import os

default_prompt = r"""Du bist ein hilfreicher medizinischer Assistent. Im Folgenden findest du Berichte. Bitte extrahiere die gesuchte Information aus dem Bericht. Wenn du die Information nicht findest, antworte null. Bitte generiere die selbe Form wie im Text.

Das ist der Bericht:
{report}"""


default_grammar = r"""root   ::= allrecords
value  ::= object | array | string | number | ("true" | "false" | "null") ws

allrecords ::= (
  "{"
  ws "\"patientennachname\":" ws string ","
  ws "\"patientenvorname\":" ws string ","
  ws "\"patientenname\":" ws string ","
  ws "\"patientengeschlecht\":" ws string ","
  ws "\"patientengeburtsdatum\":" ws string ","
  ws "\"patientenid\":" ws idartiges ","
  ws "\"patientenstrasse\":" ws string ","
  ws "\"patientenhausnummer\":" ws string ","
  ws "\"patientenpostleitzahl\":" ws plz ","
  ws "\"patientenstadt\":" ws string ","
  ws "\"patientengeburtsname\":" ws string ","
  ws "}"
  ws
)

record ::= (
    "{"
    ws "\"excerpt\":" ws ( string | "null" ) ","
    ws "\"present\":" ws ("true" | "false") ws 
    ws "}"
    ws
)

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
string ::=
  "\"" (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)? "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

plz ::= ("\"" [0-9][0-9][0-9][0-9][0-9] "\"" | "\"\"") ws
idartiges ::= ("\"" [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] "\"" | "\"\"") ws
tel ::= ("\"" [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]?[0-9]?[0-9]?[0-9]?[0-9]? "\"" | "\"\"") ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n])?"""


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
            print("Check grammar")
        if enable_grammar and not grammar:
            raise ValidationError(
                'Grammar field is required when "Enable Grammar" is checked.')

class GrammarField(wtforms.Form):
  field_name = StringField('Label Name', validators=[validators.Optional()])
  field_type = SelectField('Select Type', choices=[('string', 'String'), ('number', 'Number'), ('fp-number', 'Floating Point Number'), ('boolean', 'Boolean'), ('options', 'Categories'), ('custom', 'Custom Rule')])
  string_length = IntegerField('String Max Length (empty=no limit)', [validators.Optional()])
  string_min_length = IntegerField('String Min Length', [validators.Optional()], default=1)
  number_length = IntegerField('Number Max Length (empty=no limit)', [validators.Optional()])
  number_min_length = IntegerField('Number Min Length', [validators.Optional()], default=1)
  fp_number_length = IntegerField('FP Number Length', [validators.Optional()])
  options = StringField('Categories (comma-separated)', [validators.Optional()])
  custom_rule = StringField('Custom Rule (rulename ::= <YOURCUSTOMRULE>)', [validators.Optional()])


class LLMPipelineForm(FlaskForm):
    def __init__(self, config_file_path, model_path, *args, **kwargs):
        super(LLMPipelineForm, self).__init__(*args, **kwargs)
        import yaml

        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Extract model choices from config data
        model_choices = [(model["path_to_gguf"], model["name"])
                         for model in config_data["models"]]

        # Set choices for the model field
        self.model.choices = model_choices
        if model_path:
            self.model.validators = [FileExistsValidator(
                message='File does not exist.', path=model_path)]
            # self.model.validators.append(FileExistsValidator(message='File does not exist.', path=model_path))
        else:
            raise ValueError("Model path is required")

    file = FileField("File", validators=[
        FileRequired(),
        FileAllowed(['zip'], # for now remove csv and xlsx as they are not (longer and yet) supported
                    'Only .zip files allowed!')
    ])
    grammar = TextAreaField("Grammar:", validators=[], default=default_grammar)

    grammarbuilder = FieldList(FormField(GrammarField), min_entries=1, max_entries=10)

    extra_grammar_rules = TextAreaField("Extra Grammar Rules:", validators=[], default="")

    prompt = TextAreaField("Prompt:", validators=[], default=default_prompt)
    variables = StringField(
        "Variables (separated by commas):", validators=[], default="Patienteninfos")
    temperature = FloatField("Temperature:", validators=[
                             validators.NumberRange(0, 1)], default=0)
    model = SelectField("Model:", validators=[])

    n_predict = IntegerField("n_predict:", validators=[validators.NumberRange(1, 96000)], default=1024)

    submit = SubmitField("Run Pipeline")
