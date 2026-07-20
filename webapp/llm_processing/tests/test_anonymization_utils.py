"""Unit tests for the anonymization / de-identification core.

These functions decide what personal information is stripped from medical
documents, so a silent bug here means PII leaks. They are pure and hermetic -
no server, model or network required.
"""

import io
import unittest
import zipfile

import pandas as pd

from webapp.llm_processing.utils import (
    add_strings_with_no_umlauts,
    convert_personal_info_list,
    find_fuzzy_matches_old,
    is_empty_string_nan_or_none,
    read_preprocessed_csv_from_zip,
    replace_personal_info,
    replace_text_with_placeholder,
    replace_umlauts,
)


class TestIsEmptyStringNanOrNone(unittest.TestCase):
    def test_empty_like_values_are_true(self):
        for value in (None, "", "   ", "\t\n", "?", float("nan"), [], {}):
            self.assertTrue(is_empty_string_nan_or_none(value), repr(value))

    def test_meaningful_values_are_false(self):
        for value in ("Smith", "0", 0, 0.0, 3.5, -1, True, False):
            self.assertFalse(is_empty_string_nan_or_none(value), repr(value))


class TestUmlauts(unittest.TestCase):
    def test_replace_umlauts(self):
        self.assertEqual(replace_umlauts("Müller"), "Mueller")
        self.assertEqual(replace_umlauts("Straße"), "Strasse")
        self.assertEqual(replace_umlauts("Öz Ähre Über"), "Oez Aehre Ueber")

    def test_add_strings_with_no_umlauts_appends_ascii_variant(self):
        result = add_strings_with_no_umlauts(["Müller", "Smith"])
        self.assertEqual(result, ["Müller", "Mueller", "Smith"])

    def test_no_umlaut_string_is_not_duplicated(self):
        self.assertEqual(add_strings_with_no_umlauts(["Smith"]), ["Smith"])


class TestConvertPersonalInfoList(unittest.TestCase):
    def test_parses_list_literal(self):
        self.assertEqual(
            convert_personal_info_list("['Smith', 'John']"), ["Smith", "John"]
        )

    def test_deduplicates_preserving_order(self):
        self.assertEqual(
            convert_personal_info_list("['Smith', 'Smith', 'John']"), ["Smith", "John"]
        )

    def test_strips_nan_and_empty_entries(self):
        self.assertEqual(convert_personal_info_list("['Smith', nan]"), ["Smith"])
        self.assertEqual(convert_personal_info_list("['Smith', '']"), ["Smith"])

    def test_single_bare_value_becomes_singleton_list(self):
        self.assertEqual(convert_personal_info_list("John"), ["John"])

    def test_appends_umlaut_free_variants(self):
        self.assertEqual(
            convert_personal_info_list("['Müller']"), ["Müller", "Mueller"]
        )


class TestReplaceTextWithPlaceholder(unittest.TestCase):
    def test_case_insensitive_masking_preserves_length(self):
        out = replace_text_with_placeholder("Patient John Smith", ["john", "smith"])
        self.assertEqual(out, "Patient **** *****")
        self.assertEqual(len(out), len("Patient John Smith"))

    def test_custom_replacement_char(self):
        self.assertEqual(
            replace_text_with_placeholder("abc", ["abc"], replacement_char="#"), "###"
        )

    def test_skips_empty_entries(self):
        self.assertEqual(replace_text_with_placeholder("abc", ["", "  "]), "abc")


class TestReplacePersonalInfo(unittest.TestCase):
    def test_basic_masking(self):
        out = replace_personal_info("Patient John Smith", ["John", "Smith"], [])
        self.assertNotIn("John", out)
        self.assertNotIn("Smith", out)
        self.assertIn("Patient", out)
        self.assertEqual(out.count("■"), len("John") + len("Smith"))

    def test_fuzzy_matches_respect_threshold(self):
        text = "Patient Jon Smith"
        # score 95 >= default threshold 90 -> "Jon" gets masked
        masked = replace_personal_info(text, ["Smith"], [("Jon", 95)])
        self.assertNotIn("Jon", masked)
        # below threshold -> "Jon" stays
        kept = replace_personal_info(text, ["Smith"], [("Jon", 80)])
        self.assertIn("Jon", kept)

    def test_ignore_short_sequences(self):
        # "Li" (len 2) is dropped when ignore_short_sequences=2 (keeps len > 2).
        out = replace_personal_info(
            "Dr Li met John", ["Li", "John"], [], ignore_short_sequences=2
        )
        self.assertIn("Li", out)
        self.assertNotIn("John", out)

    def test_replacement_char_must_be_single_char(self):
        with self.assertRaises(AssertionError):
            replace_personal_info("abc", ["abc"], [], replacement_char="XY")


class TestReadPreprocessedCsvFromZip(unittest.TestCase):
    def _make_zip(self, names):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name in names:
                zf.writestr(name, "report\nhello world\n")
        buf.seek(0)
        return buf

    def test_returns_dataframe_for_preprocessed_csv(self):
        buf = self._make_zip(["preprocessed_run1.csv", "other.txt"])
        df = read_preprocessed_csv_from_zip(buf)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df["report"]), ["hello world"])

    def test_returns_none_when_absent(self):
        buf = self._make_zip(["results.csv", "notes.txt"])
        self.assertIsNone(read_preprocessed_csv_from_zip(buf))


class TestFindFuzzyMatchesOld(unittest.TestCase):
    def test_exact_word_is_matched(self):
        matches = find_fuzzy_matches_old("Patient Johnson admitted", ["Johnson"])
        self.assertTrue(any(m == "Johnson" and s >= 90 for m, s in matches))

    def test_invalid_scorer_raises(self):
        with self.assertRaises(ValueError):
            find_fuzzy_matches_old("text", ["info"], scorer="nope")

    def test_short_substrings_are_skipped(self):
        # Single/2-char, non-numeric tokens don't meet the split criteria.
        self.assertEqual(find_fuzzy_matches_old("a b c", ["a"]), [])


if __name__ == "__main__":
    unittest.main()
