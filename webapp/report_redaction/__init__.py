from flask import Blueprint

report_redaction = Blueprint('report_redaction', __name__)

from . import routes