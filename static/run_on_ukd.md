# How to run on UKD computer

1. Install Python: https://www.python.org/downloads/

2. Go to the install location and find python.exe (e.g. `C:\Users\YOURUSERNAME\AppData\Local\Programs\Python\Python312\python.exe`)

3. Open command line, cd to your project, enter the path to the python.exe with `-m venv YOURVENVNAME python=3.12` to create a virtual environment.

4. Run `YOURVENVNAME\Scripts\activate`

6. - Before you install any requirements, run these: `set HTTP_PROXY=http://ukd-proxy:80` and `set HTTPS_PROXY=http://ukd-proxy:80`
5. Install uv and set up the environment:
   - `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - `uv venv && source .venv/bin/activate`
   - `uv sync`

6. Every time you install anything with pip (or any other package), use the UKD Proxy!

7. Add proxy to git: `git config --global http.proxy http://ukd.proxy:80`

8. Clone Repository: `git clone https://github.com/KatherLab/LLMAnonymizer.git`

9. Install OCRmyPDF: 
    - Refert to this guide: https://ocrmypdf.readthedocs.io/en/latest/installation.html#installing-on-windows
    - install chocolatey and install tesseract
    - Install Ghostscript on another computer in a custom location and copy this directory to the UKD computer. Set the PATH to include tesseract and ghostscript

    - Before you install OCRmyPDF with pip, run these: `set HTTP_PROXY=http://ukd-proxy:80` and `set HTTPS_PROXY=http://ukd-proxy:80`

    - `pip install ocrmypdf --proxy=http://ukd-proxy:80`

    - set PATH to include both tesseract and ghostscript: `set PATH=D:\Path\to\tesseract\;D:\Path\to\gs\bin;%PATH%`
    You need to do this every time you logout / restart the computer! Only run this command once!

10. `cd LLMAnonymizer`

11. Open in VSCode / refer to the README on how to run.