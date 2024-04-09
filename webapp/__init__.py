from flask import Flask
from flask_socketio import SocketIO


socketio = SocketIO()

def create_app():
    app = Flask(__name__)

    app.secret_key = 'jf894puwt8ahg9piofdmhv78943oewmhrtfsud98pmhor3e8r9pi'

    from .input_processing import input_processing
    from .llm_processing import llm_processing
    from .report_redaction import report_redaction
    app.register_blueprint(input_processing)
    app.register_blueprint(llm_processing)
    app.register_blueprint(report_redaction)

    socketio.init_app(app)
    return app