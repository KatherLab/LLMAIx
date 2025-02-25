import math
from thefuzz.fuzz import QRatio, WRatio
from thefuzz import process
import re
import zipfile
import fitz
import io

import pandas as pd


def replace_umlauts(text):
    umlaut_replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
        'ß': 'ss'
    }
    
    for umlaut, replacement in umlaut_replacements.items():
        text = text.replace(umlaut, replacement)
    
    return text

def add_strings_with_no_umlauts(string_list):
    new_list = []
    
    for string in string_list:
        new_list.append(string)
        if any(umlaut in string for umlaut in "äöüÄÖÜß"):
            new_list.append(replace_umlauts(string))
    
    return new_list


def convert_personal_info_list(personal_info_list: str) -> list:
    import ast
    from collections import OrderedDict
    
    # Clean the input string
    personal_info_list = personal_info_list.replace("nan,", "").replace("'',", "").replace("nan", '')
    
    try:
        # Try to convert to list using ast.literal_eval
        personal_info_list = ast.literal_eval(personal_info_list)
        
        # If it's a single item (not a list), wrap it in a list
        if not isinstance(personal_info_list, list):
            personal_info_list = [personal_info_list]
            
    except (ValueError, SyntaxError):
        # If ast.literal_eval fails, assume it's a single value
        # Strip any extra quotes and put it in a list
        personal_info_list = personal_info_list.strip("'\"")
        personal_info_list = [personal_info_list]
    
    # Use OrderedDict to remove duplicates while preserving order
    personal_info_list = list(OrderedDict.fromkeys(personal_info_list))
    
    # Use list comprehension to filter out empty strings, "nan", and None
    personal_info_list_output = [item for item in personal_info_list if not is_empty_string_nan_or_none(item)]
    
    # Add another version without umlauts
    personal_info_list_output = add_strings_with_no_umlauts(personal_info_list_output)
    
    return personal_info_list_output



def anonymize_pdf(input_pdf: str | io.BytesIO, text_to_anonymize: list[str], output_pdf_path: str | None = None, fuzzy_matches: list[tuple[str, int]] = [], apply_redaction: bool = False) -> io.BytesIO | None:
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
        raise ValueError(
            "Input PDF must be either a file path or a BytesIO object.")

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
                    if apply_redaction:
                        page.apply_redactions()

        # Add the fuzzy matches
        if fuzzy_matches:
            # print("Add fuzzy matches: ", fuzzy_matches)
            for text, score in fuzzy_matches:
                text_instances = page.search_for(text)

                # Redact each instance of the text
                for inst in text_instances:
                    page.add_redact_annot(inst, fill=(0, 0, 0))
                    if apply_redaction:
                        page.apply_redactions()

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


def is_empty_string_nan_or_none(variable) -> bool:
    """
    Check if the input variable is None, an empty string, a string containing only whitespace or '?', or a NaN float value.

    :param variable: The input variable to check.
    :return: True if the variable is None, an empty string, a string with only whitespace or '?', or a NaN float value, False otherwise.
    :rtype: bool
    """
    if variable is None:
        return True
    if isinstance(variable, str):
        stripped = variable.strip()
        if stripped == "" or stripped == "?" or variable.isspace():
            return True
        return False
    if isinstance(variable, float) and math.isnan(variable):
        return True
    if isinstance(variable, int) or isinstance(variable, bool) or isinstance(variable, float):
        return False
    
    # If variable is not a recognized type, we assume it's invalid and return True.
    # print(f"WARNING: Removed {variable} from list.")
    return True

def replace_text_with_placeholder(text, personal_info_list, replacement_char='*'):
    """
    Replace text in the given string with a placeholder character.

    Args:
        text (str): The string to be processed.
        personal_info_list (list): List of personal information to be replaced.
        replacement_char (str): The character to use for replacement. Default is '*'.

    Returns:
        str: The processed string with personal information replaced with the placeholder character.
    """
    # Create a list to store tuples of match positions
    match_positions = []

    # Find all matches and store their positions as tuples (start, end)
    for info in personal_info_list:
        if is_empty_string_nan_or_none(info):
            continue
        matches = re.finditer(re.escape(info.lower()), text.lower())
        for match in matches:
            match_positions.append((match.start(), match.end()))

    # Replace characters within the match positions with the placeholder character
    for start, end in match_positions:
        text = text[:start] + replacement_char * (end - start) + text[end:]

    return text



def replace_personal_info(text: str, personal_info_list: dict[str, str], fuzzy_matches: list[str, int], fuzzy_matching_threshold: int = 90, generate_dollarstring: bool = False, replacement_char: str = "■", ignore_short_sequences: int = 0, debug: bool = False) -> str:
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

    assert len(
        replacement_char) == 1, "replacement_char must be a single character"

    fuzzy_list = []

    for match_text, score in fuzzy_matches:
        if score >= fuzzy_matching_threshold:
            fuzzy_list.append(match_text)

    personal_info_list = personal_info_list + fuzzy_list

    if debug:
        print("PERSONAL INFORMATION LIST: " + ', '.join(personal_info_list))
        print("FUZZY LIST: " + ', '.join(fuzzy_list))

    if ignore_short_sequences > 0:
        if debug:
            print("IGNORE SEQUENCES SHORTER THAN ", ignore_short_sequences)
        personal_info_list = [item for item in personal_info_list if len(
            item) > ignore_short_sequences]

    masked_text = replace_text_with_placeholder(
        masked_text, personal_info_list, replacement_char=replacement_char)

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


def find_fuzzy_matches_old(text: str, personal_info_list: list[str], threshold: int = 90, scorer="WRatio") -> list[str]:
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

        # Using regex to split by word boundaries
        for substring in re.findall(r'\b\w+\b', info):
            if meets_split_criteria(substring):
                # Get a list of best matches for the current personal information from the text
                best_matches = process.extract(
                    substring, text.split(), scorer=scorer)
                best_score = best_matches[0][1]
                for match, score in best_matches:
                    if score == best_score and score >= threshold:
                        # print(f"match: {match}, score: {score}")
                        fuzzy_matches.append((match, score))

    return list(set(fuzzy_matches))  # remove duplicates



