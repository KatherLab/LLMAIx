from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
import fitz  # PyMuPDF
import torch

# Define model and processor
model_id = "microsoft/Phi-3-vision-128k-instruct"

# Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Function to convert PDF pages to images
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Resize image to fit within 1344x1344 while maintaining aspect ratio
        img.thumbnail((1344, 1344), Image.ANTIALIAS)
        images.append(img)
    return images

# Function to process each image and extract text
def extract_text_from_images(images, model, processor, device):
    text_responses = []
    for image in images:
        messages = [
            {"role": "user", "content": "<|image_1|>\You are a OCR engine. Extract all text from the image. Do not modify or summarize the text."},
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt").to(device)
        generation_args = {
            "max_new_tokens": 5000,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text_responses.append(response)
    return text_responses

# Main function to read PDF, convert to images, and extract text
def main(pdf_path, device):
    images = pdf_to_images(pdf_path)
    text_responses = extract_text_from_images(images, model, processor, device)
    for i, text in enumerate(text_responses):
        print(f"Text from page {i + 1}:\n{text}\n")

# Example usage
pdf_path = "arztbericht-bild.pdf"
main(pdf_path, device)
