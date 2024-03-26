from flask import Blueprint

input_processing = Blueprint('input_processing', __name__)

from . import routes
