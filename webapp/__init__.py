from flask import Flask
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy(session_options={
    'expire_on_commit': False
})


def create_app():
    app = Flask(__name__)

    db.init_app(app)

    from webapp.input_processing.routes import input_processing
    app.register_blueprint(input_processing)

    return app