import json
import os
import random
import tempfile
from matplotlib import pyplot as plt
import pandas as pd
import pdfplumber
import fitz
import numpy as np
import seaborn as sns 

def find_llm_output_csv(directory: str) -> pd.DataFrame | None:
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate over the files to find the first one starting with 'llm-output'
    for file in files:
        if file.startswith('llm-output') and file.endswith('.csv'):
            # Construct the full path to the CSV file
            csv_file_path = os.path.join(directory, file)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Return the DataFrame
            return df

    # If no file is found, return None
    return None

class InceptionAnnotationParser:
    def __init__(self, json_file, cas):
        self.annotations = self._load_json(json_file)
        self.unique_labels = self._get_unique_labels()
        self.colormap = self.generate_colormap(self.unique_labels)
        self.cas = cas

    def _load_json(self, json_file):
        if isinstance(json_file, dict):
            return json_file["%FEATURE_STRUCTURES"]
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data["%FEATURE_STRUCTURES"]

    def get_sofastring(self):
        # get 'sofastring' aka text from last data entry

        return self.annotations[-1]["sofaString"]

    def _get_unique_labels(self):
        labels = set()
        for annotation in self.annotations:
            anno_type = annotation["%TYPE"]
            if anno_type != "custom.Span":
                break
            if not "label" in annotation:
                print("Annotation has no label. Skip. Annotation: " + str(annotation))
                continue
            try:
                labels.add(annotation["label"])
            except Exception as e:
                print(str(e))
        return labels
    
    def get_annotations(self) -> list[dict[str, int, int]]:
        annotations = []
        pdf_pages = []
        for pdf_page in self.cas.select('org.dkpro.core.api.pdf.type.PdfPage'):
            pdf_pages.append({'begin': pdf_page.begin, 'end': pdf_page.end, 'width': pdf_page.width, 'height': pdf_page.height, 'pageNumber': pdf_page.pageNumber})

        for custom_span in self.cas.select('custom.Span'):
            if custom_span.label is None:
                print("Annotation has no label. Skip. Annotation: " + str(custom_span))
                continue
            span_begin = custom_span.begin
            span_end = custom_span.end
            print("Annotation Start: ", span_begin)
            print("Annotation End: ", span_end)

            boundingboxes = []

            # Iterate over PdfChunk annotations in the CAS
            for pdf_chunk in self.cas.select('org.dkpro.core.api.pdf.type.PdfChunk'):
                chunk_begin = pdf_chunk.begin
                chunk_end = pdf_chunk.end

                # Check if the PdfChunk overlaps with the custom_Span
                if span_begin <= chunk_end and span_end >= chunk_begin:
                    print("Found fitting chunk")
                    print("Chunk Begin: ", chunk_begin)
                    print("Chunk End: ", chunk_end)
                    print("Length: ", len(pdf_chunk.g.elements))
                    # Calculate the indices within the PdfChunk
                    start_index = max(span_begin - chunk_begin, 0)
                    end_index = min(span_end - chunk_begin, len(pdf_chunk.g.elements) - 1)

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
                        breakpoint()

                    print(f"Custom Span '{custom_span.label}' Bounding Box: x={x_start}, y={pdf_chunk.y}, width={x_end-x_start}, height={pdf_chunk.h}, page={page_number}")
                    bounding_box = (page_number, (x_start, pdf_chunk.y, x_end, pdf_chunk.y + pdf_chunk.h))
                    print(f"Resulting BB: {bounding_box}")
                    boundingboxes.append(bounding_box)

            annotations.append({"label": custom_span.label, "begin": span_begin, "end": span_end, "bounding_boxes": boundingboxes})

        return annotations

    def get_annotations_old(self) -> list[dict[str, int, int]]:
        # Return a list of annotations. Each annotation contains a label, begin and end
        annotations = []
        for annotation in self.annotations:
            anno_type = annotation["%TYPE"]
            if anno_type != "custom.Span":
                break
            if not "label" in annotation:
                print("Annotation has no label. Skip. Annotation: " + str(annotation))
                continue
            label = annotation["label"]
            begin = annotation["begin"]
            end = annotation["end"]
            annotations.append({"label": label, "begin": begin, "end": end})
        return annotations

    def extract_parts(self, text):
        labeled_parts = {}
        for annotation in self.annotations:
            anno_type = annotation["%TYPE"]
            if anno_type != "custom.Span":
                break
            if not "label" in annotation:
                print("Annotation has no label. Skip. Annotation: " + str(annotation))
                continue
            label = annotation["label"]
            begin = annotation["begin"]
            end = annotation["end"]
            anno_type = annotation["%TYPE"]

            labeled_text = text[begin:end]
            if label in labeled_parts:
                labeled_parts[label].append(labeled_text)
            else:
                labeled_parts[label] = [labeled_text]
        return labeled_parts
    
    def _count_newlines_until_position(self, text, begin):
        newline_count = 0
        for char_index, char in enumerate(text):
            if char_index >= begin:
                break
            if char == '\n':
                newline_count += 1
        return newline_count
    
    def compare_text_and_count_removals(self, diff:str, position=None):
        """
        Compare two texts and mark the differences to remove from text1 to match text2.
        Optionally, return the count of removed characters up to a specified position in text2.
        """
        
        # Workaround, as sometimes when there are a lot of + entries at the end, they are substracted from the removed_count even if they are occurring after the position in the first text
        if position and position > 0:
            position -= 1

        position_text_current = 0

        removed_count = 0
        for item in diff:
            if position is not None and position_text_current >= position:
                break
            if item.startswith('- '):
                removed_count += 1
                position_text_current += 1
                # print("Removed count++")
            elif item.startswith('  '):
                position_text_current += 1
            elif item.startswith('+ '):
                removed_count -= 1
                # print("Removed count--")
            elif not item.startswith('? '):
                print("?: " + item[-1])
            else:
                # print("No change")
                pass

        # self.save_data_to_file(text_with_extra, "text_with_extra.pkl")
        # self.save_data_to_file(clean_text, "clean_text.pkl")
        # self.save_data_to_file(position, "position.pkl")

        # breakpoint()
        
        if position is not None:
            return removed_count
        else:
            return 0
    
    def get_pymupdf_text_pageswise(self, input_file):
        print("get_pymupdf_text_pageswise")
        pdf = fitz.open(input_file)

        text = ""
        for page in pdf:
            text += page.get_text("text") + " "

        return text
    
    def save_data_to_file(self, data, filename):

        import pickle

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def convert_annotations(self, text_with_extra, clean_text, annotations):
        """
        Convert a list of annotations according to the text with extra and the clean text.
        """

        import difflib
        differ = difflib.Differ()
        diff = list(differ.compare(text_with_extra, clean_text))
        
        converted_annotations = []
        for annotation in annotations:
            begin = annotation['begin']
            end = annotation['end']
            
            clean_begin = begin - self.compare_text_and_count_removals(diff, begin)
            clean_end = end - self.compare_text_and_count_removals(diff, end)
            
            # TODO: Very dirty workaround
            if clean_end-clean_begin == end-begin-1 and clean_text[clean_begin-1:clean_begin] != " ":
                clean_begin -= 1
                print("WORKAROUND1")
            if clean_end-clean_begin == end-begin+1:
                print("WORKAROUND2")
                clean_begin += 1

            try:
                # assert(clean_end-clean_begin == end-begin)
                # Comment this for more performance
                text_1 = text_with_extra[begin:end]
                text_1 = text_1.replace("\r\n", " ")
                text_1 = text_1.replace("\n", " ")
                text_2 = clean_text[clean_begin:clean_end]
                assert(text_1 == text_2)
            except AssertionError as e:
                print("Error: Length of converted text positions does not look right! ", e)
                breakpoint()

            converted_annotations.append({'begin': clean_begin, 'end': clean_end, 'label': annotation['label'], 'coveredText': annotation['coveredText']})
        
        return converted_annotations

    def extract_positions_text(self, annotations, text):
    
        for annotation in annotations:
            begin = annotation['begin']
            end = annotation['end']
            
            annotation['extracted_text'] = text[begin:end]
            
        return annotations
    
    def combine_bboxes(self, list_of_bboxes): 
        x0 = min(bbox[0] for bbox in list_of_bboxes)
        y0 = min(bbox[1] for bbox in list_of_bboxes)
        x1 = max(bbox[2] for bbox in list_of_bboxes)
        y1 = max(bbox[3] for bbox in list_of_bboxes)
        return [x0, y0, x1, y1]
    
    def merge_bounding_boxes_within_range(self, doc, start_pos, end_pos, annotation_text):
        print("Run for annotation text: " + annotation_text + " with start pos: " + str(start_pos) + " and end pos: " + str(end_pos))
        merged_bboxes = []

        # Initialize variables for tracking character count and page number
        char_count = 0
        page_num = 0

        # Iterate over each page
        while char_count <= end_pos and page_num < doc.page_count:
            page = doc.load_page(page_num)
            bboxes = []

            # Get the words and their bounding boxes on the page
            words = page.get_text("dict")["blocks"]

            # Iterate over each word
            for word_block in words:
                if 'lines' not in word_block:
                    print("No text in word block - ignore")
                    continue
                for l, line in enumerate(word_block['lines']):
                    for span in line['spans']:
                        word_line = span['text']
                        # print("W: '" + word + "'")
                        bbox = span['bbox']  # Bounding box: [x0, y0, x1, y1]
                        # print("READ WL: ", word_line)
                        if span['bbox'] in bboxes:
                            print("DUPLICATE BBOX, SKIP")
                            continue
                        bboxes.append(bbox)

                        # Get the start and end positions of the word
                        word_line_start_pos = char_count
                        word_line_end_pos = char_count + len(word_line)

                        # Calculate average character width for this word
                        avg_char_width = (bbox[2] - bbox[0]) / len(word_line)

                        # Check if the word falls within the range
                        # if the end position of the annotation is after the start of the line and the start position of the annotation is before the end of the line
                        if word_line_end_pos > start_pos and word_line_start_pos <= end_pos:
                            print("Found match for annotation text: " + annotation_text + " in line " + str(l) + " of page " + str(page_num))
                            # Calculate the adjusted start and end positions for this word
                            adjusted_start_pos = max(start_pos - char_count, 0)
                            adjusted_end_pos = min(end_pos - char_count, len(word_line)) # TODO: removed word_line - 1

                            cut_from_text = word_line[adjusted_start_pos:adjusted_end_pos]
                            
                            if len(cut_from_text) == 0:
                                print("Empty text match, skipping")
                                continue
                            else:
                                print("Found non-empty text match: '" + cut_from_text + "'")

                            bboxes = []
                            # Iterate over each character
                            for i in range(adjusted_start_pos, adjusted_end_pos + 1):
                                char_x0 = bbox[0] + i * avg_char_width
                                char_y0 = bbox[1]
                                char_x1 = char_x0 + avg_char_width
                                char_y1 = bbox[3]

                                # Check if the character falls within the range
                                bboxes.append([char_x0, char_y0, char_x1, char_y1])
                            
                            if len(bboxes) > 0:
                                

                                search_mask = self.combine_bboxes(bboxes)

                                def expand_search_mask(search_mask, width=5, height=5):
                                    x0, y0, x1, y1 = search_mask
                                    return [x0 - width, y0 - height, x1 + width, y1 + height]

                                # search_mask_expanded = [search_mask[0] -5, search_mask[1] - 5, search_mask[2] +5, search_mask[3] + 5]

                                search_mask_expanded = fitz.Rect(expand_search_mask(search_mask))
                                
                                if cut_from_text in annotation_text:
                                    annotation_in_text_match = page.search_for(cut_from_text, clip=search_mask_expanded)
                                else:
                                    breakpoint()
                                    raise Exception("Annotation text does not match the text gathered from the PDF.")


                                if not annotation_in_text_match:
                                    print("Did not find text in the bounding box: ", cut_from_text)
                                    print("Iteratively enlargeing the bounding box.")
                                    for i in range(10):
                                        search_mask_expanded = fitz.Rect(expand_search_mask(search_mask_expanded, i*3, i))
                                        annotation_in_text_match = page.search_for(cut_from_text, clip=search_mask_expanded)

                                        if annotation_in_text_match:
                                            break
                                    
                                    if not annotation_in_text_match:
                                        print("Did not find text in the bounding box: ", cut_from_text)
                                        raise Exception("Did not find text in the bounding box, even after enlarging it. Text: ", cut_from_text)

                                elif len(annotation_in_text_match) == 1:
                                    merged_bboxes.append((page_num,annotation_in_text_match[0]))
                                elif len(annotation_in_text_match) > 1:
                                    print("More than one match in bounding box. Ambiguous. Choose first.")
                                    merged_bboxes.append((page_num,annotation_in_text_match[0]))
                                    # breakpoint()
                                    # raise Exception("More than one match in bounding box. Ambiguous.")
                                else:
                                    breakpoint()
                                    print("No match, use inaccurrate bounding boxes.")
                                    merged_bboxes.append((page_num, search_mask))


                        # Update the character count
                        char_count += len(word_line) + 1 # TODO: +1 only if additional spaces per line are used!

            # Move to the next page
            page_num += 1

        return merged_bboxes

    def generate_colormap(self, labels: list[str]) -> dict[str, tuple[float, float, float]]:
        """
        Generate a colormap for the given labels.
        """
        unique_labels = list(set(labels))
        num_labels = len(unique_labels)
        color_map = {}
        colors = plt.cm.tab10.colors  # Using the 'tab10' colormap from matplotlib
        for i, label in enumerate(unique_labels):
            color_map[label] = colors[i % 10]  # Ensure cycling through colors if num_labels > 10
        return color_map

    def overlay_annotations(self, pdf_path, pdf_save_path,annotations, colormap, offset=0, random_factor=0.005):
        doc = fitz.open(pdf_path)
        for annotation in annotations:
            label = annotation['label']
            color = colormap[label]
            for page_num, bbox in annotation['bounding_boxes']:
                page = doc[page_num]
                x0, y0, x1, y1 = bbox
                offset += random.randint(-100, 100) * random_factor
                rect = fitz.Rect(x0, y0 + offset, x1, y1 + offset)
                print("Draw Rect: ", rect)
                # rect = fitz.Rect(x0, y0, x1, y1)
                # rect.y1 += offset
                # print("Color: ", color)
                page.draw_rect(rect, color=color, fill=None)
        doc.save(pdf_save_path)
        doc.close()

    def redact_pdf_with_bboxes(self, pdf_filename, output_filename, merged_bboxes):
        # Open the PDF file
        doc = fitz.open(pdf_filename)

        # Flatten the list if it's a list of list of bounding box tuples
        if any(isinstance(item, list) for item in merged_bboxes):
            print("Flattening list of lists")
            merged_bboxes = [bbox for page_bboxes in merged_bboxes for bbox in page_bboxes]

        # Iterate over each bounding box tuple
        for page_num, bbox in merged_bboxes:
            # Load the page
            page = doc.load_page(page_num)

            # Create a redaction annotation for the bounding box
            rect = fitz.Rect(bbox)
            annot = page.add_redact_annot(rect)

            # Apply redaction to the page
            # page.apply_redactions()

        # Save the redacted PDF
        doc.save(output_filename)

        # Close the PDF document
        doc.close()

    def generate_dollartext(self, text, annotations, replacement_character="$"):
        """Replace all text characters within each annotations begin and end range with dollar signs"""

        assert len(replacement_character) == 1, "Replacement character must be a single character"

        for annotation in annotations:
            begin = annotation['begin']
            end = annotation['end']
            
            # replace text characters within begin and end range with dollar signs
            text = text[:begin] + replacement_character*(end - begin) + text[end:]

        return text
    
    def generate_classwise_dollartext(self, text, annotations, replacement_character="$"):
        """Replace all text characters within each annotations begin and end range with dollar signs"""

        assert len(replacement_character) == 1, "Replacement character must be a single character"

        dollartext_labelwise = {}


        for label in self._get_unique_labels():
            dollartext_labelwise[label] = text
            print("Generate Dollar text for label: ", label)
            for annotation in annotations:
                if annotation['label'] == label:
                    begin = annotation['begin']
                    end = annotation['end']
                    
                    # replace text characters within begin and end range with dollar signs
                    dollartext_labelwise[label] = dollartext_labelwise[label][:begin] + replacement_character*(end - begin) + dollartext_labelwise[label][end:]

        return dollartext_labelwise
    
    def apply_annotations_to_pdf(self, pdf_input_path):
        
        dirpath = tempfile.mkdtemp()
        overlay_output_file = os.path.join(dirpath, pdf_input_path.replace(".pdf", "_redacted_bboxes.pdf"))

        annotations = self.get_annotations()

        self.overlay_annotations(pdf_input_path, overlay_output_file, annotations, self.colormap)

        sofastring = self.get_sofastring()

        dollartext_annotated = self.generate_dollartext(sofastring, annotations, "■")

        dollartext_annotated_labelwise = self.generate_classwise_dollartext(sofastring, annotations, "■")

        return overlay_output_file, dollartext_annotated, sofastring, dollartext_annotated_labelwise


    def apply_annotations_to_pdf_old(self, pdf_input_path):
        # with pdfplumber.open(pdf_input_path) as pdf:
        #     text_pdfplumber = pdf.pages[0].extract_text()

        # labels = self.unique_labels
        sofastring = self.get_sofastring()

        annotations = self.get_annotations()

        for annotation in annotations:
            annotation['coveredText'] = sofastring[annotation['begin']:annotation['end']]

        # total_text_len_diff = len(sofastring) - len(text_pdfplumber)
        # measured_text_len_diff = self.compare_text_and_count_removals(sofastring, text_pdfplumber)[1]
        # assert total_text_len_diff == measured_text_len_diff


        # t1 = self.get_pymupdf_text_pageswise(pdf_input_path)
        t2 = get_pymupdf_text_wordwise(pdf_input_path, add_spaces=True)

        print("Convert Annotations")
        anconv = self.convert_annotations(sofastring, t2, annotations)
            

        # DO NOT COMMENT
        print("Extract new Text")
        anex = self.extract_positions_text(anconv, t2)

        print("pymupdf text: ", t2)

        print("\n*************************************************************************************\n")

        print("annotation_text: ", sofastring)

        print("Check if the converted annotation positions extract the same text.")
        for anno in anex:
            # try:
            assert anno['coveredText'].replace("\r\n", " ").replace("\n", " ") == anno['extracted_text'], f"The original text '{anno['coveredText']}' does not match the extracted text with the updated positions '{anno['extracted_text']}'"
            # except AssertionError as e:
            #     breakpoint()

        dollartext_annotated = self.generate_dollartext(t2, anex, "■")

        # Generate the PDF with the redacted bboxes

        pdf = fitz.open(pdf_input_path)

        all_bboxes = []

        # breakpoint()


        print("Merge Bounding Boxes")
        from tqdm import tqdm
        for annotation in tqdm(anconv):
            bboxes = self.merge_bounding_boxes_within_range(pdf, annotation['begin'], annotation['end'], annotation_text=annotation['extracted_text'])
            all_bboxes.append(bboxes)
            annotation['bounding_boxes'] = bboxes

        
        colormap = self.colormap

        dirpath = tempfile.mkdtemp()

        overlay_output_file = os.path.join(dirpath, pdf_input_path.replace(".pdf", "_redacted_bboxes.pdf"))

        print("Overlay Annotations")
        self.overlay_annotations(pdf_input_path, overlay_output_file, anconv, colormap)

        # self.redact_pdf_with_bboxes(pdf_input_path, 'arztbericht_redacted_bboxes.pdf', all_bboxes)

        return overlay_output_file, dollartext_annotated, t2
        

        


def generate_score_dict(ground_truth, comparison, original_text, round_digits = 4):
    # check if both dollartext_annotated and dollartext_redacted are set
    print("CHECK SCORES")

    # if not session.get('dollartext_annotated', None) or not session.get('dollartext_redacted', None) or not session.get('original_text', None):
    #     print("Dollartext not yet set")
    #     # breakpoint()
    #     return
    
    print("Ground truth: ", ground_truth)
    print("Comparison: ", comparison)

    precision, recall, accuracy, f1_score, specificity, false_positive_rate, false_negative_rate, confusion_matrix_filepath, tp, fp, tn, fn = calculate_metrics(ground_truth, comparison, original_text, '■')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    print("Specificity: ", specificity)
    print("False positive rate: ", false_positive_rate)
    print("False negative rate: ", false_negative_rate)

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


def generate_confusion_matrix_from_counts(tp, tn, fp, fn, filename):
    
    plt.switch_backend('Agg')
    
    # Constructing the confusion matrix from counts
    cm = np.array([[tp, fn], [fp, tn]])

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to DataFrame for Seaborn heatmap
    cm_df = pd.DataFrame(cm_normalized)

    # Generate annotations with both absolute counts and normalized values
    annotations = [["{0:d}\n({1:.2f})".format(abs_num, frac) for abs_num, frac in zip(row_abs, row_frac)] 
                    for row_abs, row_frac in zip(cm, cm_normalized)]

    # Plotting the confusion matrix using Seaborn with increased font sizes
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm_df, annot=annotations, fmt="", cmap='Blues', vmin=0, vmax=1, annot_kws={"size": 16}, xticklabels=['Redacted', 'Not Redacted'], yticklabels=['Redacted', 'Not Redacted'])

    plt.ylabel('Annotation', fontsize=16)
    plt.xlabel('LLM Anonymizer', fontsize=16)

    ax.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis tick labels
    ax.tick_params(axis='y', labelsize=16)  #
    plt.title('Confusion Matrix for Report Redaction (char-wise)', fontsize=18)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()


# def generate_confusion_matrix_from_counts_old(tp, tn, fp, fn, filename):
#     import numpy as np
#     from matplotlib import pyplot as plt
#     import seaborn as sns

#     plt.switch_backend('Agg') # otherwise it would not run outside of the main thread
#     # Constructing the confusion matrix from counts
#     cm = np.array([[tp, fp], [fn, tn]])

#     # Plotting the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Redacted', 'Not Redacted'], yticklabels=['Redacted', 'Not Redacted'])
#     plt.ylabel('Annotation')
#     plt.xlabel('LLM Anonymizer')
#     plt.title('Confusion Matrix')
#     plt.savefig(filename)  # Save the plot to a file
#     plt.close()

def calculate_metrics(ground_truth, automatic_redacted, original_text, redacted_char):
    assert len(ground_truth) == len(automatic_redacted) == len(original_text), "All texts must have the same length"

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # comparison_text = ""
    # Build a text 

    for i, (gt_char, auto_char, orig_char) in enumerate(zip(ground_truth, automatic_redacted, original_text)):
        # Ignore spaces and other characters for the score calculation!
        if orig_char != ' ' and orig_char != ',' and orig_char != '.' and orig_char != '!' and orig_char != '?' and orig_char != ':' and orig_char != ';' and orig_char != '-' and orig_char != '(' and orig_char != ')' and orig_char != '"' and orig_char != "'" and orig_char != '\n':
            if gt_char == redacted_char and auto_char == redacted_char:
                true_positives += 1
                # comparison_text += "R"
            elif gt_char != redacted_char and auto_char == redacted_char:
                false_positives += 1
                # comparison_text += "+"
            elif gt_char != redacted_char and auto_char != redacted_char:
                true_negatives += 1
                # comparison_text += orig_char # "N"
            elif gt_char == redacted_char and auto_char != redacted_char:
                false_negatives += 1
                print("False Negative: GT Char: ", gt_char, " auto_char ", auto_char)
                # breakpoint()
                # comparison_text += "-"
        else:
            # comparison_text += orig_char # "I"
            pass
            # Optional: count all the spaces in the original text as true negatives
            # true_negatives += 1

    # print("Comparison text: ", comparison_text)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)  # Ignoring spaces in original text
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
    false_positive_rate = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    confusion_matrix_filepath = os.path.join(tempfile.mkdtemp(), "confusion_matrix.svg")
    generate_confusion_matrix_from_counts(true_positives, true_negatives, false_positives, false_negatives, confusion_matrix_filepath)

    return precision, recall, accuracy, f1_score, specificity, false_positive_rate, false_negative_rate, confusion_matrix_filepath, true_positives, false_positives, true_negatives, false_negatives


def get_pymupdf_text_wordwise(input_file, add_spaces=False):
    print("get_pymupdf_text_wordwise")
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
                            print("DUPLICATE BBOX, SKIP")
                            continue
                        bboxes.append(span['bbox'])
                        word = span['text']
                        # print("W: '" + word + "'")
                        text += word # + " "
                        if add_spaces:
                            text += " "
                            char_count += 1

                        char_count += len(word)

                        # print("W: '" + word + "'" + " char_count: " + str(char_count))
            else:
                print("No text in word block - ignore")

    return text