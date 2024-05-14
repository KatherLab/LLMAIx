from . import main
from .. import set_mode
from flask import current_app, redirect, request, session

@main.before_request
def before_request():
    set_mode(session, current_app.config['MODE'])

@main.route("/set_mode", methods=["GET"])
def set_mode_route():
    mode = request.args.get("mode")
    if mode is None or not mode:
        return redirect(request.referrer)
    current_app.config['MODE'] = mode
    session['mode'] = mode

    # redirect to the page where the request came from
    return redirect(request.referrer)