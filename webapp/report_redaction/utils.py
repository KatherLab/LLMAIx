import os
import random
import tempfile
from matplotlib import pyplot as plt
import pandas as pd
import fitz
import numpy as np
import seaborn as sns
from thefuzz.fuzz import QRatio, WRatio
from thefuzz import process
import re
from ..llm_processing.utils import is_empty_string_nan_or_none


def find_llm_output_csv(directory: str) -> pd.DataFrame | None:
    """
    Find the first CSV file in the given directory that starts with 'llm-output' and ends with '.csv'.

    Args:
        directory (str): The directory to search for the CSV file.

    Returns:
        Optional[pd.DataFrame]: The DataFrame containing the CSV file, if found. Otherwise, None.
    """
    csv_files = [file for file in os.listdir(directory) if file.startswith('llm-output') and file.endswith('.csv')]
    if csv_files:
        csv_file_path = os.path.join(directory, csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df
    return None


class InceptionAnnotationParser:
    """Parser for Inception's annotation format."""
    def __init__(self, json_file, cas, debug:bool = False):
        # self.annotations = self._load_json(json_file)
        self.debug = debug
        self.cas = cas
        self.annotations = self.get_annotations()
        self.unique_labels = self._get_unique_labels()
        self.colormap = self.generate_colormap(self.unique_labels)

    def get_sofastring(self) -> str:
        """
        Get the 'sofastring' aka the text of the annotated document.

        Returns:
            str: The text of the annotated document.
        """
        return self.cas.sofa_string

    def _get_unique_labels(self) -> set[str]:
        """
        Get all unique labels from the annotations.

        Returns:
            set[str]: A set of all unique labels from the annotations.
        """
        labels: set[str] = set()
        for annotation in self.annotations:  
            try:
                labels.add(annotation["label"])
            except Exception as e:
                print(str(e))
        return labels

    def get_annotations(self) -> list[dict[str, int, int]]:
        annotations = []
        pdf_pages = []
        for pdf_page in self.cas.select('org.dkpro.core.api.pdf.type.PdfPage'):
            pdf_pages.append({'begin': pdf_page.begin, 'end': pdf_page.end, 'width': pdf_page.width,
                             'height': pdf_page.height, 'pageNumber': pdf_page.pageNumber})

        for custom_span in self.cas.select('custom.Span'):
            if custom_span.label is None:
                if self.debug:
                    print("Annotation has no label. Skip. Annotation: " +
                        str(custom_span))
                continue
            span_begin = custom_span.begin
            span_end = custom_span.end
            if self.debug:
                print("Annotation Start: ", span_begin)
                print("Annotation End: ", span_end)

            boundingboxes = []

            # Iterate over PdfChunk annotations in the CAS
            for pdf_chunk in self.cas.select('org.dkpro.core.api.pdf.type.PdfChunk'):
                chunk_begin = pdf_chunk.begin
                chunk_end = pdf_chunk.end

                # Check if the PdfChunk overlaps with the custom_Span
                if span_begin <= chunk_end and span_end >= chunk_begin:
                    if self.debug:
                        print("Found fitting chunk")
                        print("Chunk Begin: ", chunk_begin)
                        print("Chunk End: ", chunk_end)
                        print("Length: ", len(pdf_chunk.g.elements))
                    # Calculate the indices within the PdfChunk
                    start_index = max(span_begin - chunk_begin, 0)
                    end_index = min(span_end - chunk_begin,
                                    len(pdf_chunk.g.elements) - 1)

                    if self.debug:
                        print("Start Index: ", start_index)
                        print("End Index: ", end_index)

                    x_start = pdf_chunk.g.elements[start_index]
                    x_end = pdf_chunk.g.elements[end_index]

                    page_number = None

                    for pdf_page in pdf_pages:
                        # breakpoint()
                        if pdf_page['begin'] <= chunk_end and pdf_page['end'] >= chunk_begin:
                            page_number = pdf_page['pageNumber']

                    if page_number is None:
                        print("Page not found for chunk")
                        if self.debug:
                            breakpoint()
                        assert("Page not found for chunk. The annotations do not seem to match the provided pdf.")
                        
                    bounding_box = (
                        page_number, (x_start, pdf_chunk.y, x_end, pdf_chunk.y + pdf_chunk.h))
                    boundingboxes.append(bounding_box)

                    if self.debug:
                        print(f"Custom Span '{custom_span.label}' Bounding Box: x={x_start}, y={pdf_chunk.y}, width={x_end-x_start}, height={pdf_chunk.h}, page={page_number}")
                        print(f"Resulting BB: {bounding_box}")
                    
            annotations.append({"label": custom_span.label, "begin": span_begin,
                               "end": span_end, "bounding_boxes": boundingboxes})

        return annotations


    def generate_colormap(self, labels: list[str]) -> dict[str, tuple[float, float, float]]:
        """
        Generate a colormap for the given labels.

        Args:
            labels (List[str]): The labels for which to generate the colormap.

        Returns:
            Dict[str, Tuple[float, float, float]]: A dictionary mapping each label to its corresponding color.
        """
        unique_labels: list[str] = list(set(labels))
        color_map: dict[str, tuple[float, float, float]] = {}
        colors: list[tuple[float, float, float]] = plt.cm.tab10.colors  # Using the 'tab10' colormap from matplotlib
        for i, label in enumerate(unique_labels):
            # Ensure cycling through colors if num_labels > 10
            color_map[label] = colors[i % 10]
        return color_map

    def overlay_annotations(self, pdf_path, pdf_save_path, annotations, colormap, offset=0, random_factor=0.005):
        doc = fitz.open(pdf_path)
        for annotation in annotations:
            label = annotation['label']
            color = colormap[label]
            for page_num, bbox in annotation['bounding_boxes']:
                page = doc[page_num]
                x0, y0, x1, y1 = bbox
                offset += random.randint(-100, 100) * random_factor
                rect = fitz.Rect(x0, y0 + offset, x1, y1 + offset)

                page.draw_rect(rect, color=color, fill=None)
        doc.save(pdf_save_path)
        doc.close()

    def generate_dollartext(self, text, annotations, replacement_character="$"):
        """Replace all text characters within each annotations begin and end range with dollar signs"""

        assert len(
            replacement_character) == 1, "Replacement character must be a single character"

        for annotation in annotations:
            begin = annotation['begin']
            end = annotation['end']

            # replace text characters within begin and end range with dollar signs
            text = text[:begin] + replacement_character * \
                (end - begin) + text[end:]

        return text

    def generate_classwise_dollartext(
        self,
        text: str,
        annotations: list[dict[str, str | int | list[int]]],
        replacement_character: str = "$"
    ) -> dict[str, str]:
        """
        Replace all text characters within each annotations begin and end range with dollar signs.

        Args:
            text (str): The text to be modified.
            annotations (List[Dict[str, Union[str, int, List[int]]]]): The list of annotations, each containing the label, begin, and end indices.
            replacement_character (str, optional): The character to replace the text with. Defaults to "$".

        Returns:
            Dict[str, str]: A dictionary mapping each label to its modified text.
        """
        assert len(replacement_character) == 1, "Replacement character must be a single character"

        dollartext_labelwise: dict[str, str] = {}
        labels: list[str] = self._get_unique_labels()
        for label in labels:
            dollartext_labelwise[label] = text
            for annotation in annotations:
                if annotation['label'] == label:
                    begin: int = annotation['begin']
                    end: int = annotation['end']

                    # replace text characters within begin and end range with dollar signs
                    dollartext_labelwise[label] = dollartext_labelwise[label][:begin] + replacement_character * (end - begin) + dollartext_labelwise[label][end:]

        return dollartext_labelwise

    def apply_annotations_to_pdf(self, pdf_input_path):

        dirpath = tempfile.mkdtemp()
        overlay_output_file = os.path.join(
            dirpath, pdf_input_path.replace(".pdf", "_redacted_bboxes.pdf"))

        self.overlay_annotations(
            pdf_input_path, overlay_output_file, self.annotations, self.colormap)

        sofastring = self.get_sofastring()

        dollartext_annotated = self.generate_dollartext(
            sofastring, self.annotations, "■")

        dollartext_annotated_labelwise = self.generate_classwise_dollartext(
            sofastring, self.annotations, "■")

        return overlay_output_file, dollartext_annotated, sofastring, dollartext_annotated_labelwise


def generate_score_dict(ground_truth, comparison, original_text, round_digits=4):

    # print("Ground truth: ", ground_truth)
    # print("Comparison: ", comparison)

    precision, recall, accuracy, f1_score, specificity, false_positive_rate, false_negative_rate, confusion_matrix_filepath, tp, fp, tn, fn = calculate_metrics(
        ground_truth, comparison, original_text, '■')
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 score: ", f1_score)
    # print("Specificity: ", specificity)
    # print("False positive rate: ", false_positive_rate)
    # print("False negative rate: ", false_negative_rate)

    score_dict = {
        'precision': round(precision, round_digits),
        'recall': round(recall, round_digits),
        'accuracy': round(accuracy, round_digits),
        'f1_score': round(f1_score, round_digits),
        'specificity': round(specificity, round_digits),
        'false_positive_rate': round(false_positive_rate, round_digits),
        'false_negative_rate': round(false_negative_rate, round_digits),
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

    return score_dict, confusion_matrix_filepath


def generate_confusion_matrix_from_matrix(confusion_matrix_list, filename, title='Confusion Matrix for Report Redaction (char-wise)', xlabel='LLM Anonymizer', ylabel='Annotation', classes=[]):
    plt.switch_backend('Agg')

    cm = np.array(confusion_matrix_list)

    # Normalize the confusion matrix
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    try:
        row_sums = cm.sum(axis=1, keepdims=True)
    except Exception as e:
        print(e)
        breakpoint()

    # Avoid division by zero by setting zero sums to one
    row_sums[row_sums == 0] = 1

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / row_sums

    # Convert to DataFrame for Seaborn heatmap
    cm_df = pd.DataFrame(cm_normalized)

    # Generate annotations with both absolute counts and normalized values
    annotations = [["{0:d}\n({1:.2f})".format(abs_num, frac) for abs_num, frac in zip(row_abs, row_frac)]
                   for row_abs, row_frac in zip(cm, cm_normalized)]

    # Plotting the confusion matrix using Seaborn with increased font sizes
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm_df, annot=annotations, fmt="", cmap='Blues', vmin=0, vmax=1, annot_kws={
                     "size": 16}, xticklabels=classes, yticklabels=classes)

    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

    # Adjust font size for x-axis tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)  #
    plt.title(title, fontsize=18)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()

def generate_confusion_matrix_from_counts(tp, tn, fp, fn, filename, labels=['Redacted', 'Not Redacted'], title='Confusion Matrix for Report Redaction (char-wise)', xlabel='LLM Anonymizer', ylabel='Annotation'):

    plt.switch_backend('Agg')

    # Constructing the confusion matrix from counts
    cm = np.array([[tp, fn], [fp, tn]])

    # Normalize the confusion matrix
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    row_sums = cm.sum(axis=1, keepdims=True)

    # Avoid division by zero by setting zero sums to one
    row_sums[row_sums == 0] = 1

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / row_sums

    # Convert to DataFrame for Seaborn heatmap
    cm_df = pd.DataFrame(cm_normalized)

    # Generate annotations with both absolute counts and normalized values
    annotations = [["{0:d}\n({1:.2f})".format(abs_num, frac) for abs_num, frac in zip(row_abs, row_frac)]
                   for row_abs, row_frac in zip(cm, cm_normalized)]

    # Plotting the confusion matrix using Seaborn with increased font sizes
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm_df, annot=annotations, fmt="", cmap='Blues', vmin=0, vmax=1, annot_kws={
                     "size": 16}, xticklabels=labels, yticklabels=labels)

    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

    # Adjust font size for x-axis tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)  #
    plt.title(title, fontsize=18)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()

def calculate_metrics(ground_truth, automatic_redacted, original_text, redacted_char):
    assert len(ground_truth) == len(automatic_redacted) == len(
        original_text), "All texts must have the same length"

    tp, fp, tn, fn = 0, 0, 0, 0

    non_special_characters = set(
        [' ', ',', '.', '!', '?', ':', ';', '-', '(', ')', '"', "'", '\n'])

    for gt_char, auto_char, orig_char in zip(ground_truth, automatic_redacted, original_text):
        if orig_char not in non_special_characters:
            if gt_char == redacted_char and auto_char == redacted_char:
                tp += 1
            elif gt_char != redacted_char and auto_char == redacted_char:
                fp += 1
            elif gt_char != redacted_char and auto_char != redacted_char:
                tn += 1
            elif gt_char == redacted_char and auto_char != redacted_char:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    false_positive_rate = fp / (tn + fp) if (tn + fp) != 0 else 0
    false_negative_rate = fn / (tp + fn) if (tp + fn) != 0 else 0

    confusion_matrix_filepath = os.path.join(
        tempfile.mkdtemp(), "confusion_matrix.svg")
    generate_confusion_matrix_from_counts(
        tp, tn, fp, fn, confusion_matrix_filepath)

    return precision, recall, accuracy, f1_score, specificity, false_positive_rate, false_negative_rate, confusion_matrix_filepath, tp, fp, tn, fn


def get_pymupdf_text_wordwise(input_file, add_spaces=False):
    pdf = fitz.open(input_file)

    char_count = 0

    text = ""
    for page in pdf:
        bboxes = []
        for word_block in page.get_text("dict")["blocks"]:
            if 'lines' in word_block:
                for line in word_block['lines']:
                    for span in line['spans']:
                        if span['bbox'] in bboxes:
                            # print("DUPLICATE BBOX, SKIP")
                            continue
                        bboxes.append(span['bbox'])
                        word = span['text']
                        # print("W: '" + word + "'")
                        text += word  # + " "
                        if add_spaces:
                            text += " "
                            char_count += 1

                        char_count += len(word)
            else:
                pass
                # print("No text in word block - ignore")

    return text

def find_fuzzy_matches(text: str, personal_info_list: list[str], threshold: int = 90, scorer="WRatio") -> list[str]:
    if scorer == "QRatio":
        scorer = QRatio
    elif scorer == "WRatio":
        scorer = WRatio
    else:
        raise ValueError("Invalid scorer. Must be 'QRatio' or 'WRatio'")

    def meets_split_criteria(substring):
        return len(substring) >= 3 or len(re.findall(r'\d', substring)) >= 4

    def process_info(info):
        if is_empty_string_nan_or_none(info):
            return []
        return [
            (match, score)
            for substring in re.findall(r'\b\w+\b', info)
            if meets_split_criteria(substring)
            for match, score in process.extract(substring, text.split(), scorer=scorer)
            if score >= threshold
        ]

    # Flatten the list of lists and remove duplicates using a set
    fuzzy_matches = list(set(match for info in personal_info_list for match in process_info(info)))

    return fuzzy_matches