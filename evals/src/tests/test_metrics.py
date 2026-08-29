"""Unit tests for rat_evals.metrics."""

from __future__ import annotations

from rat_evals.metrics import citation_accuracy, recall_at_k, refusal_calibration


def test_recall_at_k_no_expected_citations() -> None:
    assert recall_at_k([], [{"citation": "§151.02"}], k=5) == 1.0


def test_recall_at_k_partial_hit() -> None:
    chunks = [{"citation": "§151.02(a)"}, {"citation": "§3-110"}]
    assert recall_at_k(["§151.02", "§27-2018"], chunks, k=5) == 0.5


def test_recall_at_k_respects_k() -> None:
    chunks = [{"citation": "§3-110"}, {"citation": "§151.02(a)"}]
    assert recall_at_k(["§151.02"], chunks, k=1) == 0.0


def test_citation_accuracy_matches_variant_formatting() -> None:
    answer = "The penalty is set by §151.02(a) of the Health Code."
    assert citation_accuracy(["§151.02"], answer) == 1.0


def test_citation_accuracy_no_match() -> None:
    answer = "The retrieved sources do not address this section."
    assert citation_accuracy(["§151.02"], answer) == 0.0


def test_refusal_calibration_no_unanswerable_items_defaults_pass() -> None:
    items = [{"failure_mode": "citation_accuracy"}]
    assert refusal_calibration(items, ["some answer"]) == 1.0


def test_refusal_calibration_detects_do_not_contain_phrasing() -> None:
    # Regression test: gpt-oss's default refusal phrasing ("do not contain",
    # "do not mention") was previously missed by a too-narrow regex that only
    # matched "not answered/in the/covered/specified/supported/provided/found",
    # producing a flat 0.0 score even when every refusal was correct.
    items = [{"failure_mode": "refusal_calibration", "must_not_say": []}] * 4
    responses = [
        "The excerpts do not contain any information about federal oversight.",
        "The retrieved documents do not contain the text of that section.",
        "The supplied sources do not mention any criminal penalty.",
        "None of the retrieved excerpts mention a criminal sanction for repeat offenses.",
    ]
    assert refusal_calibration(items, responses) == 1.0


def test_refusal_calibration_penalizes_hallucinated_answer() -> None:
    items = [{"failure_mode": "refusal_calibration", "must_not_say": ["$5,000"]}]
    responses = ["The maximum criminal penalty is $5,000 under §151.09."]
    assert refusal_calibration(items, responses) == 0.0


def test_refusal_calibration_detects_no_source_indicates_phrasing() -> None:
    # Regression test: the 2026-08-24 nightly run scored this a miss (2/5
    # refusal_calibration items failed detection, refusal_calibration=0.6)
    # because the regex required a specific noun ("sources/excerpts/...")
    # directly after "none/nothing", missing "No source ... indicates" and
    # "none of the ... excerpts contain information about" (no "mention").
    items = [{"failure_mode": "refusal_calibration", "must_not_say": []}] * 2
    responses = [
        "No source in the provided material indicates that a federal agency "
        "oversees or can override the DOHMH's enforcement of rat-control rules.",
        "I'm sorry, but none of the provided excerpts contain information "
        "about the maximum criminal penalty for a repeat rodent violation.",
    ]
    assert refusal_calibration(items, responses) == 1.0


def test_refusal_calibration_mixed_items_only_scores_unanswerable() -> None:
    items = [
        {"failure_mode": "citation_accuracy"},
        {"failure_mode": "refusal_calibration", "must_not_say": []},
    ]
    responses = [
        "The penalty is $300 per §151.02(a).",
        "The sources do not contain that information.",
    ]
    assert refusal_calibration(items, responses) == 1.0
