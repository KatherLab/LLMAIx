from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, BooleanField, TextAreaField, MultipleFileField, FileField, FloatField, validators, SelectField
from wtforms.validators import DataRequired, ValidationError
import os
from flask import current_app

default_prompt = r"""[INST] <<SYS>>
Du bist ein hilfreicher medizinischer Assistent. Im Folgenden findest du Berichte. Bitte finde die gesuchte Information aus dem Bericht. Wenn du die Information nicht findest, antworte null. 
<</SYS>>
[/INST]

[INST]
Das ist der Bericht:
{report}

Extrahiere diese Elemente aus dem Text: {symptom}? 
[/INST]"""


default_grammer = r"""root   ::= allrecords
value  ::= object | array | string | number | ("true" | "false" | "null") ws

allrecords ::= (
  "{"
  ws "\"patientennachname\":" ws string ","
  ws "\"patientenvorname\":" ws string ","
  ws "\"patientengeschlecht\":" ws string ","
  ws "\"patientengeburtsdatum\":" ws string ","
  ws "\"patientenid\":" ws string ","
  ws "\"patientenstrasse\":" ws string ","
  ws "\"patientenhausnummer\":" ws string ","
  ws "\"patientenpostleitzahl\":" ws string ","
  ws "\"patientenstadt\":" ws string ","
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

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

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
        grammar = field.data#
        if enable_grammar:
            print("Check grammar")
        if enable_grammar and not grammar:
            raise ValidationError('Grammar field is required when "Enable Grammar" is checked.')

class LLMPipelineForm(FlaskForm):
    def __init__(self, model_path, *args, **kwargs):
        super(LLMPipelineForm, self).__init__(*args, **kwargs)
        
        self.model.choices = [
            ("sauerkrautlm-7b-v1.Q5_K_M.gguf", "LLaMA 2 7b chat sauerkraut"),
            ("sauerkrautlm-70b-v1.Q5_K_M.gguf", "LLaMA 2 70b chat sauerkraut"),
            ("sauerkrautlm-13b-v1.Q5_K_M.gguf", "LLaMA 2 13b chat sauerkraut"),
            ("mistral-7b-instruct-v0.1.Q5_K_M.gguf", "Mistral 7b"),
            ("mixtral-8x7b-v0.1.Q5_K_M.gguf", "Mistral 8x7b"),
        ]
        if model_path:
            self.model.validators = [FileExistsValidator(message='File does not exist.', path=model_path)]
            # self.model.validators.append(FileExistsValidator(message='File does not exist.', path=model_path))
        else:
            raise ValueError("Model path is required")


    file = FileField("File", validators=[
        FileRequired(),  
        FileAllowed(['zip', 'csv', 'xlms'], 'Only .zip, .csv and .xlms files allowed!')
    ])
    grammar = TextAreaField("Grammar:", validators=[], default=default_grammer)
    prompt = TextAreaField("Prompt:", validators=[], default=default_prompt)
    variables = StringField("Variables (separated by commas):", validators=[], default="Patienteninfos")
    temperature = FloatField("Temperature:", validators=[validators.NumberRange(0,1)], default=0.7)
    model = SelectField("Model:", choices=[("sauerkrautlm-7b-v1.Q5_K_M.gguf", "LLaMA 2 7b chat sauerkraut"), ("sauerkrautlm-70b-v1.Q5_K_M.gguf", "LLaMA 2 70b chat sauerkraut"), ("sauerkrautlm-13b-v1.Q5_K_M.gguf", "LLaMA 2 13b chat sauerkraut")], validators=[])

    submit = SubmitField("Run Pipeline")