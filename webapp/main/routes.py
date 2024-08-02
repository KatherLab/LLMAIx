from . import main
from .. import set_mode
from flask import current_app, redirect, request, session, flash, url_for

@main.before_request
def before_request():
    set_mode(session, current_app.config['MODE'])

@main.route("/set_mode", methods=["GET"])
def set_mode_route():
    mode = request.args.get("mode")
    if mode is None or not mode:
        flash("Error changing application mode.", "danger")
        return redirect(request.referrer)
    allowed_modes = ["anonymizer", "informationextraction"]
    if mode not in allowed_modes:
        flash(f"Cannot set mode to {mode}, allowed are: {','.join(allowed_modes)}", "danger")
        return redirect(request.referrer)
    if current_app.config['MODE'] == 'choice':
        # print("set mode: ", mode)
        session['mode'] = mode
    else:
        flash(f"Cannot change model: The application was launched with --mode {current_app.config['MODE']} option.", "danger")

    if session['mode'] == 'anonymizer' and "labelannotation" in request.referrer:
        flash("Switched to Anonymizer", "info")
        return redirect(url_for("report_redaction.main"))
    
    if session['mode'] == 'informationextraction' and "reportredaction" in request.referrer:
        flash("Switched to Information Extraction", "info")
        return redirect(url_for("labelannotation.main"))
    # breakpoint()
    # redirect to the page where the request came from
    # print("Mode is: ", session['mode'])
    return redirect(request.referrer)