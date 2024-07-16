from flask import Flask, Response, request
from flask_socketio import SocketIO
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash


socketio = SocketIO()

def create_app(auth_required:bool = False):
    app = Flask(__name__)

    app.secret_key = 'jf894puwt8ahg9piofdmhv78943oewmhrtfsud98pmhor3e8r9pi'

    auth = HTTPBasicAuth()

    # Dictionary to store users and their hashed passwords (in a real application, use a database)
    users = {
        #"katherlab": generate_password_hash("kinetic-thesaurus-repurpose"),
        "katherlab": "scrypt:32768:8:1$GERvnWuqIlJqAiFS$e57df57d7d6b31c05ddc9771d71a8685fcb25ebe8fd6d8253ae5b14052a764f78a5a92b007d5dc90507535383c634f7fed58fe8389759b3c7d3cbfcccf6b3e93"
    }

    @auth.verify_password
    def verify_password(username, password):
        if username in users:
            return check_password_hash(users.get(username), password)
        return False
    # Apply authentication to all routes
    @app.before_request
    def before_request():
        # Call the login_required method
        if auth_required:
            return auth.login_required()(lambda: None)()

    from .input_processing import input_processing
    from .llm_processing import llm_processing
    from .report_redaction import report_redaction
    from .labelannotation import labelannotation
    from .main import main
    app.register_blueprint(input_processing)
    app.register_blueprint(llm_processing)
    app.register_blueprint(report_redaction)
    app.register_blueprint(labelannotation)
    app.register_blueprint(main)

    socketio.init_app(app)

    return app
    

    

def set_mode(session, mode):
    pass
    # print("set mode to ", mode)
    # session['mode'] = mode