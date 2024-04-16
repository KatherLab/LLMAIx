import unittest
import os
from webapp.llm_processing.utils import anonymize_pdf

class TestAnonymizePDF(unittest.TestCase):
    def test_anonymize_pdf(self):
        # Define paths
        input_pdf_path = 'examples/arztbericht.pdf'
        output_pdf_path = 'test_redaction.pdf'

        # Strings to anonymize
        strings_to_anonymize = ["Frau", "Sophie", "Berger", "10. März 1975", "Rosenweg 5", "63446", "Mannheim", "134523"]

        # Anonymize the PDF
        anonymize_pdf(input_pdf_path, strings_to_anonymize, output_pdf_path)

        # Check if the output PDF exists
        self.assertTrue(os.path.exists(output_pdf_path))

        # Clean up

        # os.remove(output_pdf_path)

if __name__ == '__main__':
    unittest.main()
