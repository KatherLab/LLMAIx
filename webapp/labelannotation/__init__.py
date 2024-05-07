from flask import Blueprint

labelannotation = Blueprint('labelannotation', __name__)

from . import routes
