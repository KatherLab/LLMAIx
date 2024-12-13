from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import SubmitField, FileField, Form, StringField, SelectField, FieldList, FormField


class LabelField(Form):
    label_name = StringField("Label Name")
    label_type = SelectField("Label Type", choices=[("multiclass", "Multiclass"), ("boolean", "Boolean"), ("stringmatch", "String Match"), ("ignore", "Ignore Label")])
    label_classes = StringField("Classes")

class LabelSelectorForm(FlaskForm):
    labels = FieldList(FormField(LabelField))
    submit = SubmitField("Continue")

class LLMAnnotationResultsForm(FlaskForm):

    file = FileField("File", validators=[
        FileRequired(),
        FileAllowed(['zip'], 'Only .zip file allowed!')
    ])

    annotation_file = FileField("Annotation File (csv)", validators=[
        FileAllowed(['csv', 'xlsx'], 'Only .csv and .xlsx files allowed!')
    ])

    submit = SubmitField("LLM Output Viewer")
    submit_metrics = SubmitField("LLM Output Metrics")