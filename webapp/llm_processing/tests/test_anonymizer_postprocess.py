"""Regression tests for anonymizer-mode postprocessing.

These guard three bugs fixed in ``postprocess_grammar``:

1. Anonymizer mode was never applied because ``mode`` was not forwarded, so no
   ``masked_report`` column was produced.
2. Every report was masked with the *first* row's PII list (hard-coded ``[0]``),
   leaking every other patient's PII.
3. A single report whose LLM request failed (no ``content`` key) aborted
   postprocessing for the whole batch.
"""

import pandas as pd

from webapp.llm_processing.routes import postprocess_grammar


def test_anonymizer_masks_each_report_with_its_own_pii():
    result = {
        "pA.pdf$aaa": {
            "report": "Patient Alice Anderson, MRN 111.",
            "symptom": "x",
            "content": '{"name": "Alice Anderson", "mrn": "111"}',
        },
        "pB.pdf$bbb": {
            "report": "Patient Bob Brown, MRN 222.",
            "symptom": "x",
            "content": '{"name": "Bob Brown", "mrn": "222"}',
        },
    }
    df = pd.DataFrame(
        [
            {"id": "pA.pdf$aaa", "metadata": "{'source':'a'}"},
            {"id": "pB.pdf$bbb", "metadata": "{'source':'b'}"},
        ]
    )

    out, errors = postprocess_grammar(result, df, {"model": "t"}, mode="anonymizer")

    assert errors == 0
    assert "masked_report" in out.columns  # bug #1: branch actually ran
    row_a = out[out["report"].str.contains("Anderson", na=False)].iloc[0]
    row_b = out[out["report"].str.contains("Brown", na=False)].iloc[0]
    assert "Alice" not in row_a["masked_report"]
    assert "111" not in row_a["masked_report"]
    # bug #2: B must be masked with B's own PII, not row 0's.
    assert "Bob" not in row_b["masked_report"]
    assert "222" not in row_b["masked_report"]


def test_failed_report_does_not_discard_the_batch():
    result = {
        "pA.pdf$aaa": {"report": "Alice", "symptom": "x", "content": '{"name":"Alice"}'},
        # Simulates a report whose request failed: no "content" key.
        "pC.pdf$ccc": {"report": "Carol", "symptom": "x"},
    }
    df = pd.DataFrame(
        [
            {"id": "pA.pdf$aaa", "metadata": "{'s':'a'}"},
            {"id": "pC.pdf$ccc", "metadata": "{'s':'c'}"},
        ]
    )

    out, errors = postprocess_grammar(result, df, {"model": "t"}, mode="informationextraction")

    assert len(out) == 2  # bug #3: nothing was discarded
    assert errors == 1  # the failed report is counted as an error
