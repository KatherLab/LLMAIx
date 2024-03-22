from flask import Blueprint, render_template, request, redirect, url_for, flash

input_processing = Blueprint('input_processing', __name__)

@input_processing.route("/", methods=['GET', 'POST'])
def main():


    return render_template("index.html", title="LLM Anonymizer", uploadForm=uploadForm)