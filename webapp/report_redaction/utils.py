import json
import os
import random
import tempfile
from matplotlib import pyplot as plt
import pandas as pd
import pdfplumber
import fitz

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
    def __init__(self, json_file):
        self.annotations = self._load_json(json_file)
        self.unique_labels = self._get_unique_labels()
        self.colormap = self.generate_colormap(self.unique_labels)

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
        # Return a list of annotations. Each annotation contains a label, begin and end
        annotations = []
        for annotation in self.annotations:
            anno_type = annotation["%TYPE"]
            if anno_type != "custom.Span":
                break
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
    
    def compare_text_and_count_removals(self, text_with_extra:str, clean_text:str, position=None):
        """
        Compare two texts and mark the differences to remove from text1 to match text2.
        Optionally, return the count of removed characters up to a specified position in text2.
        """
        import difflib
        differ = difflib.Differ()
        diff = list(differ.compare(text_with_extra, clean_text))
        
        cleaned_text = []
        removed_count = 0
        for idx, item in enumerate(diff):
            if position is not None and idx >= position:
                break
            if item.startswith('-'):
                removed_count += 1
                # print("Removed count++")
            elif item.startswith('+'):
                removed_count -= 1
                # print("Removed count--")
            elif not item.startswith('?'):
                cleaned_text.append(item[-1])
            else:
                # print("No change")
                pass
        
        return ''.join(cleaned_text), removed_count
    
    def get_pymupdf_text_wordwise(self, input_file):
        print("get_pymupdf_text_wordwise")
        pdf = fitz.open(input_file)

        char_count = 0

        text = ""
        for page in pdf:
            for word_block in page.get_text("dict")["blocks"]:
                if 'lines' in word_block:
                    for line in word_block['lines']:
                        for span in line['spans']:
                            word = span['text']
                            # print("W: '" + word + "'")
                            text += word

                            char_count += len(word) 
                            # print("W: '" + word + "'" + " char_count: " + str(char_count))
                else:
                    print("No text in word block - ignore")

        return text
    
    def convert_annotations(self, text_with_extra, clean_text, annotations):
        """
        Convert a list of annotations according to the text with extra and the clean text.
        """

        
        converted_annotations = []
        for annotation in annotations:
            begin = annotation['begin']
            end = annotation['end']
            clean_begin = begin - self.compare_text_and_count_removals(text_with_extra[:begin], clean_text, begin)[1]
            clean_end = end - self.compare_text_and_count_removals(text_with_extra[:end], clean_text, end)[1]
            converted_annotations.append({'begin': clean_begin, 'end': clean_end, 'label': annotation['label']})
        
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
        print("Run for annotation text: " + annotation_text)
        merged_bboxes = []

        # Initialize variables for tracking character count and page number
        char_count = 0
        page_num = 0

        # Iterate over each page
        while char_count <= end_pos and page_num < doc.page_count:
            page = doc.load_page(page_num)

            # Get the words and their bounding boxes on the page
            words = page.get_text("dict")["blocks"]

            # Iterate over each word
            for word_block in words:
                for l, line in enumerate(word_block['lines']):
                    for span in line['spans']:
                        word_line = span['text']
                        # print("W: '" + word + "'")
                        bbox = span['bbox']  # Bounding box: [x0, y0, x1, y1]

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
                            adjusted_end_pos = min(end_pos - char_count, len(word_line) - 1)

                            cut_from_text = word_line[adjusted_start_pos:adjusted_end_pos]

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

                                search_mask_expanded = [search_mask[0] -5, search_mask[1] - 5, search_mask[2] +5, search_mask[3] + 5]

                                search_mask_expanded = fitz.Rect(search_mask_expanded)

                                if cut_from_text in annotation_text:
                                    annotation_in_text_match = page.search_for(cut_from_text, clip=search_mask_expanded)
                                else:
                                    raise Exception("Annotation text does not match the text gathered from the PDF.")

                                if len(annotation_in_text_match) == 1:
                                    merged_bboxes.append((page_num,annotation_in_text_match[0]))
                                elif len(annotation_in_text_match) > 1:
                                    print("More than one match")
                                    raise Exception("More than one match")
                                else:
                                    breakpoint()
                                    print("No match, use inaccurrate bounding boxes.")
                                    merged_bboxes.append((page_num, search_mask))


                        # Update the character count
                        char_count += len(word_line) 

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

    def overlay_annotations(self, pdf_path, pdf_save_path,annotations, colormap, offset=0):
        doc = fitz.open(pdf_path)
        for annotation in annotations:
            label = annotation['label']
            color = colormap[label]
            for page_num, bbox in annotation['bounding_boxes']:
                page = doc[page_num]
                x0, y0, x1, y1 = bbox
                offset += random.randint(-100, 100) / 200.0
                rect = fitz.Rect(x0, y0 + offset, x1, y1 + offset)
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
    
    


    def apply_annotations_to_pdf(self, pdf_input_path):
        with pdfplumber.open(pdf_input_path) as pdf:
            text_pdfplumber = pdf.pages[0].extract_text()

        labels = self.unique_labels
        sofastring = self.get_sofastring()

        annotations = self.get_annotations()

        for annotation in annotations:
            annotation['coveredText'] = sofastring[annotation['begin']:annotation['end']]

        total_text_len_diff = len(sofastring) - len(text_pdfplumber)
        measured_text_len_diff = self.compare_text_and_count_removals(sofastring, text_pdfplumber)[1]
        assert total_text_len_diff == measured_text_len_diff

        t2 = self.get_pymupdf_text_wordwise(pdf_input_path)

        anconv = self.convert_annotations(sofastring, t2, annotations)
        # DO NOT COMMENT
        anex = self.extract_positions_text(anconv, t2)

        dollartext_annotated = self.generate_dollartext(t2, anex, "■")

        # Generate the PDF with the redacted bboxes

        pdf = fitz.open(pdf_input_path)

        all_bboxes = []

        for annotation in anconv:
            bboxes = self.merge_bounding_boxes_within_range(pdf, annotation['begin'], annotation['end'], annotation_text=annotation['extracted_text'])
            all_bboxes.append(bboxes)
            annotation['bounding_boxes'] = bboxes

        
        colormap = self.colormap

        dirpath = tempfile.mkdtemp()

        overlay_output_file = os.path.join(dirpath, pdf_input_path.replace(".pdf", "_redacted_bboxes.pdf"))

        self.overlay_annotations(pdf_input_path, overlay_output_file, anconv, colormap)

        # self.redact_pdf_with_bboxes(pdf_input_path, 'arztbericht_redacted_bboxes.pdf', all_bboxes)

        return overlay_output_file, dollartext_annotated, t2
        

        


