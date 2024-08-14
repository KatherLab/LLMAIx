from flask import Blueprint

annotationhelper = Blueprint('annotationhelper', __name__)

from . import routes
