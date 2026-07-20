import os
import tempfile
import unittest

import pymupdf

from webapp.llm_processing.utils import anonymize_pdf


class TestAnonymizePDF(unittest.TestCase):
    # Existing example report; "Andrew Smith" is the patient name in the text.
    INPUT_PDF = "examples/documents/0090075.pdf"
    STRINGS_TO_ANONYMIZE = ["Andrew", "Smith"]

    def test_anonymize_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_pdf_path = os.path.join(tmp, "test_redaction.pdf")

            anonymize_pdf(
                self.INPUT_PDF,
                self.STRINGS_TO_ANONYMIZE,
                output_pdf_path,
                apply_redaction=True,
            )

            # The output PDF must have been written.
            self.assertTrue(os.path.exists(output_pdf_path))

            # And the redacted strings must no longer be present in its text.
            doc = pymupdf.open(output_pdf_path)
            try:
                text = "".join(page.get_text() for page in doc)
            finally:
                doc.close()
            for redacted in self.STRINGS_TO_ANONYMIZE:
                self.assertNotIn(redacted, text)


if __name__ == "__main__":
    unittest.main()
