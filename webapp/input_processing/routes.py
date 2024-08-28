from datetime import datetime
import shutil
from flask import render_template, request, redirect, url_for, flash, send_file, session, current_app
import os
import tempfile
from werkzeug.utils import secure_filename
from .forms import PreprocessUploadForm
import secrets
from concurrent import futures
import pandas as pd
import time
import subprocess
from PIL import Image
from docx import Document
from odf import teletype
from odf.opendocument import load
import uuid
import zipfile
from io import BytesIO
from docx2pdf import convert as convert_docx
import fitz
import traceback
from . import input_processing
from .. import socketio
from .. import set_mode
from ..llm_processing.utils import is_empty_string_nan_or_none


JobID = str
jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

job_progress = {}


@socketio.on('connect')
def handle_connect():
    # print("Client Connected")
    pass


@socketio.on('disconnect')
def handle_disconnect():
    pass


def update_progress(job_id, progress: tuple[int, int, bool]):
    global job_progress
    job_progress[job_id] = progress
    
    socketio.emit('progress_update', {
                  'job_id': job_id, 'progress': progress[0], 'total': progress[1]})


def failed_job(job_id):
    
    print("Preprocessing Job Failed: ", job_id)

    time.sleep(0.5)

    global job_progress
    socketio.emit('progress_failed', {'job_id': job_id})


def complete_job(job_id):
    print("Preprocessing Job Complete: ", job_id)

    # Sometimes when the preprocessing is too fast, the socketio client did not reconnect in time, so the progress bar is never updated
    time.sleep(0.5)

    global job_progress
    socketio.emit('progress_complete', {'job_id': job_id})

def create_pdf(text, filename):
    # Create a new PDF with A4 format
    doc = fitz.open()
    
    # Set up the font and margins
    font = "helv"
    fontsize = 11
    line_height = fontsize * 1.2
    margin_left = 50
    margin_top = 50
    margin_right = 50
    margin_bottom = 50
    
    # Calculate available text area
    page_width = 595  # A4 width in points
    page_height = 842  # A4 height in points
    text_width = page_width - margin_left - margin_right
    text_height = page_height - margin_top - margin_bottom
    
    # Function to add a new page
    def add_page():
        return doc.new_page(width=page_width, height=page_height)
    
    # Initialize the first page
    page = add_page()
    current_y = margin_top
    
    # Split text into words
    words = text.split()
    
    # Process words
    line = []
    for word in words:
        line.append(word)
        # Check if current line fits within text width
        line_width = fitz.get_text_length(" ".join(line), fontname=font, fontsize=fontsize)
        if line_width > text_width:
            # Remove last word and write current line
            line.pop()
            line_text = " ".join(line)
            page.insert_text(fitz.Point(margin_left, current_y), line_text, fontname=font, fontsize=fontsize)
            current_y += line_height
            
            # Check if we need a new page
            if current_y + line_height > page_height - margin_bottom:
                page = add_page()
                current_y = margin_top
            
            # Start new line with the word that didn't fit
            line = [word]
        
    # Write the last line if there's any content left
    if line:
        line_text = " ".join(line)
        page.insert_text(fitz.Point(margin_left, current_y), line_text, fontname=font, fontsize=fontsize)
    
    # Save the PDF
    doc.save(filename)
    doc.close()

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Resize image to fit within 1344x1344 while maintaining aspect ratio
        img.thumbnail((1344, 1344), Image.LANCZOS)
        images.append(img)
    return images

def extract_text_from_images(images, model, model_id, device):
    text_responses = []
    from transformers import AutoProcessor
    for image in images:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        messages = [
            {"role": "user", "content": "<|image_1|>\nYou are a OCR engine. Extract all text from the image. Do not modify or summarize the text."},
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt").to(device)
        generation_args = {
            "max_new_tokens": 10000, # increase this if necessary
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text_responses.append(response)
    return '\n'.join(text_responses)

def ocr_phi3vision(file_path):
    from PIL import Image
    from transformers import AutoModelForCausalLM
    import torch

    # Define model and processor
    model_id = "microsoft/Phi-3-vision-128k-instruct"

    use_flash_attn = False

    # Load the model and processor
    if torch.cuda.is_available():
        device = 'cuda'
        try: 
            import flash_attn
            use_flash_attn = True
        except:
            print("Flash Attention not available, using Eager Mode")

    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("Running Phi3Vision OCR on: ", device)
    if not device == 'cuda':
        print("Disable Flash Attention, no CUDA device!")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16 if not device == 'cuda' else "auto", _attn_implementation='flash_attention_2' if use_flash_attn else 'eager')

    if file_path.endswith('.pdf'):
        images = pdf_to_images(file_path)
        text = extract_text_from_images(images, model, model_id, device)
        
    elif file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
        text = extract_text_from_images([Image.open(file_path)], model, model_id, device)

    del model

    return text

def scale_bbox(bbox, src_dpi=96, dst_dpi=72):
    scale_factor = dst_dpi / src_dpi
    return [coord * scale_factor for coord in bbox]

def estimate_font_size(bbox_width, text_length, char_width_to_height_ratio=0.5):
    if text_length == 0:  # Prevent division by zero
        return 12  # Default font size if no text is present
    avg_char_width = bbox_width / text_length
    font_size = avg_char_width / char_width_to_height_ratio
    return font_size


def add_text_layer_to_pdf(pdf_path, ocr_results, output_path, src_dpi=96, dst_dpi=72):
    pdf_document = fitz.open(pdf_path)
    full_text = ""
    for page_num, page_ocr in enumerate(ocr_results):
        page = pdf_document[page_num]
        for line in page_ocr[0].text_lines:
            bbox = scale_bbox(line.bbox, src_dpi, dst_dpi)
            text = line.text
            # print("Add text: ", text)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # print("Rect: ", rect)
            # Estimate font size from bounding box height
            font_size = estimate_font_size(rect.width, len(text))
            # print("Estimated font size: ", font_size)
            # Insert invisible but selectable text
            page.insert_text(rect.bottom_left, text, fontsize=font_size+1, fontname="helv", render_mode=3)
            # page.draw_rect(rect, color=(1, 0, 0), width=1) # for debugging

            full_text += text
    pdf_document.save(output_path)
    print("Text added to PDF, saved to: ", output_path)
    pdf_document.close()

    return full_text


def images_to_pdf(images, output_path):
    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=100.0)

def extract_text_from_images_surya(images, det_model, rec_model, langs):
    from surya.ocr import run_ocr
    from surya.model.recognition.processor import load_processor
    from surya.model.detection.model import load_processor as load_det_processor

    predictions = []

    for image in images:
        det_processor = load_det_processor()
        rec_processor = load_processor()

        predictions.append(run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor))

    return predictions

def ocr_surya(file_path, ocr_file_output_path, det_model, rec_model):
    from PIL import Image

    from surya.input.load import load_pdf


    langs = ["de", "en"] 

    if file_path.endswith('.pdf'):
        pass
        
    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        
        # convert image to pdf
        from PIL import Image  
        image = Image.open(file_path)

        # save as file_path but with .pdf extension
        file_path = file_path.replace(".jpg", ".pdf").replace(".jpeg", ".pdf").replace(".png", ".pdf")
            
        image.save(
            file_path, "PDF", resolution=100.0, save_all=True
        )
    

    else:
        raise ValueError("Unsupported file type")
    
    images, names = load_pdf(file_path)
    ocr_output = extract_text_from_images_surya(images, det_model, rec_model, langs)
    text = add_text_layer_to_pdf(file_path, ocr_output, ocr_file_output_path)

    del rec_model
    del det_model

    return text


def preprocess_file(file_path, force_ocr=False, ocr_method='tesseract', remove_previous_ocr=False, det_model=None, rec_model=None):
    merged_data = []
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            for index, row in enumerate(df.itertuples()):
                text = str(getattr(row, 'report'))
                id = str(getattr(row, 'id'))
                pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"{os.path.splitext(os.path.basename(file_path))[0]}-{index}.pdf")
                create_pdf(text, pdf_file_save_path)
                merged_data.append(pd.DataFrame({'report': [text], 'filepath': pdf_file_save_path, 'id': id}))

        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            for index, row in enumerate(df.itertuples()):
                text = str(getattr(row, 'report'))
                id = str(getattr(row, 'id'))
                pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"{os.path.splitext(os.path.basename(file_path))[0]}-{index}.pdf")
                create_pdf(text, pdf_file_save_path)
                merged_data.append(pd.DataFrame({'report': [text], 'filepath': pdf_file_save_path, "id": id}))

        elif file_path.endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            if not file_path.endswith('.pdf'):
                pdf_output_path = os.path.join(tempfile.mkdtemp(), f"{os.path.splitext(os.path.basename(file_path))[0]}.pdf")
                image = Image.open(file_path)
                image.save(pdf_output_path)
                file_path = pdf_output_path
                

            contains_text = False
            print("Opening PDF: ", file_path)
            with fitz.open(file_path) as pdf_document:
                contains_text = False
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    if len(page.get_text()) > 0 and not is_empty_string_nan_or_none(page.get_text()):
                        contains_text = True
                        break


            print("Contains text: ", contains_text)

            if not contains_text or force_ocr:
                if contains_text and force_ocr and remove_previous_ocr:
                    remove_selectable_text_from_pdf(file_path)
                if ocr_method == 'tesseract':
                    ocr_output_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path)}")
                    if shutil.which("tesseract") is not None:
                        if shutil.which("ocrmypdf") is not None:
                            subprocess.run(['ocrmypdf', '-l', 'deu', '--force-ocr', file_path, ocr_output_path])
                        else:
                            return "OCRMyPDF not found but required for OCR."
                    else:
                        return "Tesseract not found but required for OCR."

                elif ocr_method == 'phi3vision':
                    # ocr_output_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path)}")
                    
                    try: 
                        import transformers
                    except Exception as e:
                        print(str(e))
                        return "Python transformers library not found but required for phi3vision OCR."
                    
                    try:
                        import torch
                    except Exception as e:
                        print(str(e))
                        return "Python torch library not found but required for phi3vision OCR."
                    
                    # import torch
                    # if not torch.cuda.is_available() or not torch.backends.mps.is_available():
                    #     return "GPU / CUDA device or MPS not found but required for phi3vision OCR."

                    ocr_text = ocr_phi3vision(file_path)
                    ocr_output_path = file_path
                
                elif ocr_method == 'surya':
                    ocr_output_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path)}")

                    try:
                        import torch
                        import surya
                    except Exception as e:
                        print(str(e))
                        return "surya or torch pyhton library not found but required for surya OCR."

                    ocr_text = ocr_surya(file_path, ocr_output_path, det_model, rec_model)
                    # ocr_output_path = file_path

                
                else:
                    return "No valid OCR method selected."

            else:
                ocr_output_path = file_path

            # for phi3vision, take the text directly from the pdf output
            if not ocr_method == 'phi3vision':
                with fitz.open(ocr_output_path) as ocr_pdf:
                    ocr_text = ''
                    for page_num in range(len(ocr_pdf)):
                        page = ocr_pdf.load_page(page_num)
                        ocr_text += page.get_text()

            ocr_text = ocr_text.replace("'", "").replace('"', '')

            merged_data.append(pd.DataFrame({'report': [ocr_text], 'filepath': ocr_output_path, 'id': ''}))

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path).split('.txt')[0]}.pdf")
            create_pdf(text, pdf_file_save_path)
            merged_data.append(pd.DataFrame({'report': [text], 'filepath': pdf_file_save_path, 'id': ''}))

        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            doc_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path).split('.docx')[0]}.pdf")
            convert_docx(file_path, pdf_file_save_path)
            merged_data.append(pd.DataFrame({'report': [doc_text], 'filepath': pdf_file_save_path, 'id': ''}))

        elif file_path.endswith('.odt'):
            doc = load(file_path)
            doc_text = ''
            for element in doc.getElementsByType(text.P):
                doc_text += teletype.extractText(element)
            merged_data.append(pd.DataFrame({'report': [doc_text], 'filepath': '', 'id': ''}))

        else:
            return f"Unsupported file format: {file_path}"

    except Exception as e:
        traceback.print_exc()
        return f"Error processing file {file_path}: {e}"

    return merged_data

def remove_selectable_text_from_pdf(pdf_path):
    """
    Remove the selectable text from a PDF document while preserving the actual text graphics.
    The input PDF file will be overwritten with the modified version.
    
    Args:
        pdf_path (str): The path to the input PDF file.
    """
    # Open the PDF file
    pdf_doc = fitz.open(pdf_path)

    # Iterate through each page
    for page in pdf_doc:
        # Add a redaction annotation covering the entire page
        page.add_redact_annot(page.rect)
        
        # Apply the redaction
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Create a temporary file path
    temp_path = pdf_path + ".temp"

    # Save the modified PDF to the temporary file
    pdf_doc.save(temp_path)

    # Close the document
    pdf_doc.close()

    # Remove the original file and rename the temporary file
    os.remove(pdf_path)
    os.rename(temp_path, pdf_path)


def preprocess_input(job_id, file_paths, parallel_preprocessing=False, force_ocr=False, ocr_method='tesseract', remove_previous_ocr=False):
    merged_data = []

    if ocr_method == 'tesseract':
        max_workers = 20 if parallel_preprocessing else 1
    elif ocr_method == 'surya':
        max_workers = 2 if parallel_preprocessing else 1
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.recognition.model import load_model

        det_model = load_det_model()
        rec_model = load_model()
    else:
        print("Set max workers to 1 for OCR method: ", ocr_method)
        max_workers = 1

    with futures.ThreadPoolExecutor(max_workers=max_workers) as inner_executor:
        if ocr_method == 'surya':
            future_to_file = {inner_executor.submit(preprocess_file, file_path, force_ocr, ocr_method, remove_previous_ocr, det_model, rec_model): file_path for file_path in file_paths}
        else:
            future_to_file = {inner_executor.submit(preprocess_file, file_path, force_ocr, ocr_method, remove_previous_ocr): file_path for file_path in file_paths}
        
        for i, future in enumerate(futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if isinstance(result, list):
                    merged_data.extend(result)
                else:
                    print(result)  # Print any error messages
                update_progress(job_id=job_id, progress=(i+1, len(file_paths), True))
            except Exception as e:
                print(f"Error in future result for {file_path}: {e}")

    merged_df = pd.concat(merged_data)
    complete_job(job_id)
    return merged_df


@input_processing.before_request
def before_request():
    set_mode(session, current_app.config['MODE'])

@input_processing.route("/download", methods=['GET'])
def download():
    job_id = request.args.get("job")
    global jobs

    if job_id not in jobs:
        flash(f"Input processing job {job_id} not found!", "danger")
        return redirect(url_for('input_processing.main'))
    job = jobs[job_id]

    if job.cancelled():
        flash(f"Job {job} was cancelled", "danger")
        return redirect(url_for('input_processing.main'))
    elif job.running():
        flash(f"Job {job} is still running", "warning")
        return redirect(url_for('input_processing.main'))
    elif job.done():
        try:
            df = job.result()
        except Exception:
            flash("Preprocessing failed / did not output anything useful!", "danger")
            return redirect(url_for('input_processing.main'))

        if isinstance(df, str):
            flash(df, "danger")
            return redirect(url_for('input_processing.main'))

        # split the text in chunks
        try:
            max_length = int(session['text_split'])
        except Exception as e:
            max_length = None

        # Add an 'id' column and generate unique IDs for every row
        # df['id'] = df.apply(lambda x: str(uuid.uuid4()), axis=1)

        def remove_ocr_prefix(filename):
            if filename.startswith('ocr_'):
                return filename[len('ocr_'):]
            else:
                return filename

        df['filename'] = df['filepath'].apply(lambda x: remove_ocr_prefix(os.path.basename(x)))

        def generate_id(row):
            # check if id is NaN or empty string
            if pd.isna(row['id']) or row['id'] == '':
                return row['filename'] + '$' + str(uuid.uuid4())[:8]
            else:
                return row['id'] + '$' + str(uuid.uuid4())[:8]

        # Apply the function to the DataFrame
        df['id'] = df.apply(generate_id, axis=1)

        # add metadata column with json structure. Add the current date and time as preprocessing key in the json structure
        df['metadata'] = df.apply(lambda x: {'preprocessing': {
                                  'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}, axis=1)

        # Optionally, you can drop the 'filename' column if you don't need it anymore
        df.drop(columns=['filename'], inplace=True)

        # Function to add files to a zip file
        def add_files_to_zip(zipf, files, ids):
            for file, file_id in zip(files, ids):
                zipf.write(file, f"{file_id}.{os.path.basename(file).split('.')[-1]}")
                # os.remove(file)sss

        # Add dataframe as CSV to zip
        def add_dataframe_to_zip(zipf, df):
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_filename = f'preprocessed_{job_id}.csv'
                csv_filepath = os.path.join(temp_dir, csv_filename)

                # Drop unnecessary columns and save the dataframe to a CSV file
                df.drop(columns=['filepath'], inplace=True)
                df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
                df['report'] = df['report'].str.replace('\n', '\\n')
                df.to_csv(csv_filepath, index=False)

                # Write the CSV file to the zip archive
                zipf.write(csv_filepath, arcname=csv_filename)

        files_to_zip = df['filepath'].tolist()
        ids = df['id'].tolist()

        # Function to split text without breaking words
        def split_text(text, max_length):
            words = text.split()
            split_texts = []
            current_text = ""
            
            for word in words:
                if len(current_text) + len(word) + 1 <= max_length:  # +1 for space
                    current_text += " " + word if current_text else word
                else:
                    split_texts.append(current_text)
                    current_text = word
            
            if current_text:
                split_texts.append(current_text)
            
            return split_texts

        # Split rows containing more than max_length letters
        split_rows = []
        for _, row in df.iterrows():
            if max_length and len(row['report']) > max_length:
                split_texts = split_text(row['report'], max_length)
                for i, text in enumerate(split_texts):
                    split_row = row.copy()
                    split_row['report'] = text
                    split_row['id'] = f'{row["id"]}_{i}'
                    split_rows.append(split_row)
            else:
                split_rows.append(row)


        # Create a new DataFrame with the split rows
        df_split = pd.DataFrame(split_rows)

        print("Files to zip:", files_to_zip)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            add_files_to_zip(zipf, files_to_zip, ids)
            add_dataframe_to_zip(zipf, df_split)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"preprocessed-{job_id}.zip",
        )
    else:
        flash(f"Job {job}: An unknown error occurred!", "danger")
        return redirect(url_for('input_processing.main'))


@input_processing.route("/", methods=['GET', 'POST'])
def main():

    form = PreprocessUploadForm(method=session.get('mode', 'informationextraction'))

    if form.validate_on_submit():

        current_datetime = datetime.now()
        prefix = current_datetime.strftime("%Y%m%d%H%M")

        job_id = f"{form.text_split.data}-{prefix}-" + secrets.token_urlsafe(8)

        temp_dir = tempfile.mkdtemp()

        session['text_split'] = form.text_split.data

        # Save each uploaded file to the temporary directory
        file_paths = []
        for file in form.files.data:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                # print("File saved:", file_path)
                file_paths.append(file_path)

                # read excel and csv files only and check if they have id and report column
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if 'id' not in df.columns or 'report' not in df.columns:
                        flash(f"CSV file {filename}: Missing 'id' or 'report' column!", "danger")
                        return redirect(url_for('input_processing.main'))
                elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                    df = pd.read_excel(file_path)
                    if 'id' not in df.columns or 'report' not in df.columns:
                        flash(f"ExcelFile {filename}: Missing 'id' or 'report' column! Are they in the first sheet? Are they without leading or trailing whitespace?", "danger")
                        return redirect(url_for('input_processing.main'))

        update_progress(job_id=job_id, progress=(
            0, len(form.files.data), True))

        # For debugging purposes, run outside of executor
        # preprocess_input(job_id=job_id, file_paths=file_paths)

        # print("Start Executor")
        global jobs
        jobs[job_id] = executor.submit(
            preprocess_input,
            job_id=job_id,
            file_paths=file_paths,
            parallel_preprocessing=current_app.config['PARALLEL_PREPROCESSING'],
            force_ocr=form.force_ocr.data,
            ocr_method=form.ocr_method.data,
            remove_previous_ocr=form.remove_previous_ocr.data,
        )

        flash('Upload Successful!', "success")
        return redirect(url_for('input_processing.main'))

    global job_progress

    return render_template("index.html", title="LLM Anonymizer", form=form, progress=job_progress)
