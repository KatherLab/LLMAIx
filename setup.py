from setuptools import setup, find_packages

setup(
    name="LLMAnonymizer",
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
    description="Anonymize medical reports using LLMs.",
    url="https://github.com/KatherLab/LLMAnonymizer",
)
