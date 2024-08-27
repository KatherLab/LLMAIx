from setuptools import setup, find_packages

setup(
    name="LLM-AIx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyMuPDF",
        "flask",
        "flask_socketio",
        "pandas",
        "openpyxl",
        "lxml", 
    ],
    tests_require=[
        "unittest",
        "os"
    ],
    test_suite="tests",
    author="KatherLab",
    author_email="jakob-nikolas.kather@alumni.dkfz.de",
    description="LLM-AIx - A pipeline for information extraction and anonymization of medical documents",
    url="https://github.com/KatherLab/LLM-AIx",
)
