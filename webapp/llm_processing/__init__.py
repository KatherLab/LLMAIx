from flask import Blueprint

llm_processing = Blueprint('llm_processing', __name__)

from . import routes
