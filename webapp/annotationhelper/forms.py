from flask_wtf import FlaskForm
from wtforms import FieldList, Form, FormField, SelectField, StringField, SubmitField, FileField, BooleanField
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import validators

class ReAnnotationField(Form):
    label_name = StringField("Label Name")
    label_type = StringField("Label Type")
    llm_string = StringField("LLM String")
    llm_categories = SelectField("LLM Categories", choices=[])
    llm_boolean = BooleanField("LLM Boolean")

    annotator_string = StringField("Annotator String")
    annotator_categories = SelectField("Annotator Categories", choices=[])
    annotator_boolean = BooleanField("Annotator Boolean")
    
class ReAnnotationForm(FlaskForm):
    # Form for reannotation
    labels = FieldList(FormField(ReAnnotationField))

    submit_previous = SubmitField("❮ Save & Previous Record")
    submit_next = SubmitField("Save & Next Record ❯")
    submit_save = SubmitField("Save")

    def validate(self, extra_validators=None):

        print("validate")
        # Perform default validation
        # if not super().validate():
        #     return False

        # Custom validation logic
        for label_entry in self.labels:
            label_type = label_entry.label_type.data
            # Validate LLM fields based on the label type
            # DO NOT VALIDATE hidden fields
            # if label_type == 'multiclass':
            #     if label_entry.llm_categories.data not in dict(label_entry.llm_categories.choices):
            #         breakpoint()
            #         label_entry.llm_categories.errors.append("Invalid choice for LLM Categories")
            #         return False
            # elif label_type == 'boolean':
            #     if label_entry.llm_boolean.data not in [True, False]:
            #         breakpoint()
            #         label_entry.llm_boolean.errors.append("Invalid value for LLM Boolean")
            #         return False
            # elif label_type == 'stringmatch':
            #     if not label_entry.llm_string.data:
            #         breakpoint()
            #         label_entry.llm_string.errors.append("LLM String is required for String Match")
            #         return False

            # Validate Annotator fields based on the label type
            if label_type == 'multiclass':
                if label_entry.annotator_categories.data not in dict(label_entry.annotator_categories.choices):
                    breakpoint()
                    label_entry.annotator_categories.errors.append("Invalid choice for Annotator Categories")
                    return False
            elif label_type == 'boolean':
                if label_entry.annotator_boolean.data not in [True, False]:
                    breakpoint()
                    label_entry.annotator_boolean.errors.append("Invalid value for Annotator Boolean")
                    return False
            elif label_type == 'stringmatch':
                if not label_entry.annotator_string.data:
                    breakpoint()
                    label_entry.annotator_string.errors.append("Annotator String is required for String Match")
                    return False

        return True


class LabelField(Form):
    label_name = StringField("Label Name")
    label_type = SelectField("Label Type", choices=[("multiclass", "Multiclass"), ("boolean", "Boolean"), ("stringmatch", "String Match")])
    label_classes = StringField("Classes")

class LabelSelectorForm(FlaskForm):
    labels = FieldList(FormField(LabelField))
    submit = SubmitField("Continue")
class AnnotationHelperForm(FlaskForm):

    file = FileField("Upload File (LLM Output File)", validators=[FileRequired(), FileAllowed(
        ['zip'], 'Only .zip llm output files are allowed!')])

    submit = SubmitField("Upload")