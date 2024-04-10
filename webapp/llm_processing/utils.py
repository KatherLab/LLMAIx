import zipfile
import fitz
import io

import pandas as pd

def convert_personal_info_list(personal_info_list) -> None:
    import ast
    personal_info_list = ast.literal_eval(personal_info_list)
    personal_info_list = list(set(personal_info_list))
    personal_info_list = [item for item in personal_info_list if item != ""]

    personal_info_list_output = []
    for info in personal_info_list:
        if is_empty_string_nan_or_none(info):
            continue

        personal_info_list_output.append(info)
    
    return personal_info_list_output

def anonymize_pdf(input_pdf: str | io.BytesIO, text_to_anonymize: list[str], output_pdf_path: str | None = None, fuzzy_matches: list[tuple[str, int]] = []) -> io.BytesIO | None:
    """
    Anonymizes the specified text in a PDF by covering it with black rectangles and removes the underlying text.

    Args:
        input_pdf (str | io.BytesIO): Path to the input PDF file or a BytesIO object containing the PDF content.
        text_to_anonymize (list[str]): List of strings to anonymize in the PDF.
        output_pdf_path (str | None, optional): Path to save the anonymized PDF file. If None, returns the modified PDF as a BytesIO object. Defaults to None.

    Returns:
        io.BytesIO | None: If output_pdf_path is None, returns the modified PDF as a BytesIO object. Otherwise, returns None.

    Raises:
        ValueError: If the input_pdf is neither a file path nor a BytesIO object.
    """
    # Open the PDF
    if isinstance(input_pdf, str):
        pdf_document = fitz.open(input_pdf)
    elif isinstance(input_pdf, io.BytesIO):
        pdf_document = fitz.open(stream=input_pdf.read(), filetype="pdf")
    else:
        raise ValueError("Input PDF must be either a file path or a BytesIO object.")

    # Iterate through each page of the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Search for the text to anonymize on the page
        if text_to_anonymize:
            for text in text_to_anonymize:
                text_instances = page.search_for(text)

                # Redact each instance of the text
                for inst in text_instances:
                    page.add_redact_annot(inst, fill=(0, 0, 0))
                    # page.apply_redactions()

        # Add the fuzzy matches
        if fuzzy_matches:
            print("Add fuzzy matches: ", fuzzy_matches)
            for text, score in fuzzy_matches:
                text_instances = page.search_for(text)

                # Redact each instance of the text
                for inst in text_instances:
                    page.add_redact_annot(inst, fill=(0, 0, 0))
                    # page.apply_redactions()

        # Save the modified page
        # try:
        #     pdf_document[page_number] = page
        # except Exception as e:
        #     print(e)

    # Save the modified PDF or return as BytesIO
    if output_pdf_path is None:
        modified_pdf_bytes = io.BytesIO()
        pdf_document.save(modified_pdf_bytes)
        pdf_document.close()
        modified_pdf_bytes.seek(0)
        return modified_pdf_bytes
    else:
        pdf_document.save(output_pdf_path)
        pdf_document.close()

import math
def is_empty_string_nan_or_none(variable) -> bool:
        """
        Check if the input variable is None, an empty string, or a string containing only whitespace or '?', or a NaN float value.
        
        :param variable: The input variable to check.
        :return: True if the variable is None, an empty string, a string with only whitespace or '?', or a NaN float value, False otherwise.
        :rtype: bool
        """
        if variable is None:
            return True
        elif isinstance(variable, str) and ( variable.strip() == "" or variable.isspace() or variable == "?"):
            return True
        elif isinstance(variable, float) and math.isnan(variable):
            return True
        else:
            return False
        

import re
from thefuzz import process
from thefuzz.fuzz import QRatio, WRatio

def replace_personal_info(text: str, personal_info_list: dict[str, str], use_fuzzy_matching: bool = False, fuzzy_matching_threshold: int = 90) -> str:
    """
    Replace personal information in the given text with asterisks.

    Args:
        text (str): The text containing personal information.
        personal_info_list (dict[str, str]): A list of personal information to be replaced.
        use_fuzzy_matching (bool, optional): Whether to use fuzzy matching for replacement. Defaults to False.
        fuzzy_matching_threshold (int, optional): The threshold for fuzzy matching (how good should the match be to be accepted). Defaults to 90.

    Returns:
        str: The text with personal information masked with asterisks (***).
    """
    # remove redundant items
    personal_info_list = list(set(personal_info_list))
    personal_info_list = [item for item in personal_info_list if item != ""]
    masked_text = text

    # Replace remaining personal information with asterisks (*)
    for info in personal_info_list:
        if is_empty_string_nan_or_none(info):
            continue
        masked_text = re.sub(f"\\b{re.escape(info)}\\b", "***", masked_text)


    if use_fuzzy_matching:
        for info in personal_info_list:
            if is_empty_string_nan_or_none(info):
                continue
            # Get a list of best matches for the current personal information from the text
            best_matches = process.extract(info, text.split())
            best_score = best_matches[0][1]
            for match, score in best_matches:
                if score == best_score and score >= fuzzy_matching_threshold:
                    print(f"match: {match}, score: {score}")
                    # Replace best matches with asterisks (*)
                    masked_text = re.sub(f"\\b{re.escape(match)}\\b", "***", masked_text) #TODO #masked_text.replace(match, '*' * len(match))

                
    return masked_text

def read_preprocessed_csv_from_zip(zip_file: str) -> pd.DataFrame | None:
    """
    A function that reads a preprocessed CSV file from a zip file and returns it as a Pandas DataFrame.
    
    Parameters:
    zip_file (str): The path to the zip file containing the preprocessed CSV file.
    
    Returns:
    pandas.DataFrame or None: The preprocessed CSV data as a DataFrame if found, else None.
    """
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        for file_info in zipf.infolist():
            if file_info.filename.startswith('preprocessed_') and file_info.filename.endswith('.csv'):
                with zipf.open(file_info.filename) as csvfile:
                    df = pd.read_csv(csvfile)
                return df
    return None

def find_fuzzy_matches(text: str, personal_info_list: list[str], threshold: int = 90, scorer = "WRatio") -> list[str]:
    fuzzy_matches = []
    def meets_split_criteria(substring):
        # Split if substring has at least 3 characters or at least 4 digits
        return len(substring) >= 3 or (len(re.findall(r'\d', substring)) >= 4)
    
    if scorer == "QRatio":
        scorer = QRatio
    elif scorer == "WRatio":
        scorer = WRatio
    else:
        raise ValueError("Invalid scorer. Must be 'QRatio' or 'WRatio'")

    for info in personal_info_list:
        if is_empty_string_nan_or_none(info):
            continue

        for substring in re.findall(r'\b\w+\b', info):  # Using regex to split by word boundaries
            if meets_split_criteria(substring):
                # Get a list of best matches for the current personal information from the text
                best_matches = process.extract(substring, text.split(), scorer=scorer)
                best_score = best_matches[0][1]
                for match, score in best_matches:
                    if score == best_score and score >= threshold:
                        print(f"match: {match}, score: {score}")
                        fuzzy_matches.append((match, score))

            
        
    return list(set(fuzzy_matches))  # remove duplicates