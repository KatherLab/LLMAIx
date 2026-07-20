"""Tests for LLM-output post-processing and small route helpers.

``postprocess_grammar`` turns raw llama.cpp output strings into the structured
DataFrame the rest of LLMAIx consumes. It has to tolerate the messy things LLMs
emit - chat end-of-turn markers, trailing prose after the JSON object, trailing
commas, and outright malformed JSON - without crashing the whole batch. These
tests pin that behaviour.
"""

import unittest

import pandas as pd

from webapp.llm_processing.routes import (
    format_time,
    is_path,
    parse_metrics,
    postprocess_grammar,
)


def run(content, metadata="{'source': 'unit-test'}"):
    """Run postprocess_grammar on a single report with the given LLM output."""
    result = {"rep1": {"content": content, "report": "the report text"}}
    df = pd.DataFrame([{"id": "rep1", "metadata": metadata}])
    out_df, error_count = postprocess_grammar(
        result, df, llm_metadata={"model": "test"}, mode="informationextraction"
    )
    return out_df, error_count


class TestPostprocessGrammar(unittest.TestCase):
    def test_clean_json(self):
        out, errors = run('{"name": "John", "age": "30"}')
        self.assertEqual(errors, 0)
        row = out.iloc[0]
        self.assertEqual(row["name"], "John")
        self.assertEqual(row["age"], "30")
        self.assertEqual(row["id"], "rep1")
        self.assertEqual(row["report"], "the report text")

    def test_strips_chat_end_markers(self):
        for suffix in ("<|eot_id|>", "</s>"):
            out, errors = run('{"name": "John"}' + suffix)
            self.assertEqual(errors, 0, suffix)
            self.assertEqual(out.iloc[0]["name"], "John")

    def test_ignores_trailing_prose_after_object(self):
        out, errors = run('{"name": "John"} Here is your answer, hope it helps!')
        self.assertEqual(errors, 0)
        self.assertEqual(out.iloc[0]["name"], "John")

    def test_removes_trailing_comma(self):
        out, errors = run('{"name": "John", "age": "30",}')
        self.assertEqual(errors, 0)
        self.assertEqual(out.iloc[0]["age"], "30")

    def test_empty_and_null_values_become_empty_string(self):
        out, errors = run('{"name": "John", "age": null}')
        self.assertEqual(errors, 0)
        self.assertEqual(out.iloc[0]["age"], "")

    def test_malformed_json_counts_error_and_does_not_crash(self):
        out, errors = run('{"name": "John", "age": ')
        self.assertEqual(errors, 1)
        # The report row is still present even though parsing failed.
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["id"], "rep1")

    def test_non_json_output_is_isolated_per_report(self):
        result = {
            "good": {"content": '{"x": "1"}', "report": "r1"},
            "bad": {"content": "totally not json", "report": "r2"},
        }
        df = pd.DataFrame(
            [
                {"id": "good", "metadata": "{'a': 1}"},
                {"id": "bad", "metadata": "{'a': 2}"},
            ]
        )
        out, errors = postprocess_grammar(
            result, df, llm_metadata={}, mode="informationextraction"
        )
        self.assertEqual(errors, 1)
        self.assertEqual(len(out), 2)


class TestFormatTime(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(format_time(5), "5s")
        self.assertEqual(format_time(119), "119s")
        self.assertEqual(format_time(120), "2.0min")
        self.assertEqual(format_time(3600), "1.0h")
        self.assertEqual(format_time(86400), "1.0d")


class TestIsPath(unittest.TestCase):
    def test_separators_and_absolute(self):
        self.assertTrue(is_path("models/config.yml"))
        self.assertTrue(is_path("dir\\file.yml"))
        self.assertTrue(is_path("/etc/hosts"))

    def test_bare_filename_is_not_a_path(self):
        self.assertFalse(is_path("config.yml"))


class TestParseMetrics(unittest.TestCase):
    def test_parses_prometheus_gauge(self):
        text = (
            "# HELP llamacpp_slots_idle idle slots\n"
            "# TYPE llamacpp_slots_idle gauge\n"
            "llamacpp_slots_idle 3.0\n"
        )
        self.assertEqual(parse_metrics(text)["llamacpp_slots_idle"], 3.0)


if __name__ == "__main__":
    unittest.main()
