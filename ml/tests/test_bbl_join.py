"""Unit tests for rat_ml.data.bbl_join.

All tests are pure-Python with no DB dependency.
"""

from __future__ import annotations

import pytest

from rat_ml.data.bbl_join import (
    emit_unmatched_report,
    get_unmatched_report,
    normalize_bbl,
    reset_unmatched_report,
    resolve_bbl,
)


# ===========================================================================
# normalize_bbl
# ===========================================================================

class TestNormalizeBbl:
    def test_already_10_digits(self) -> None:
        assert normalize_bbl("3007390001") == "3007390001"

    def test_integer_input(self) -> None:
        assert normalize_bbl(3007390001) == "3007390001"

    def test_pads_short_string(self) -> None:
        # 9 digits → zero-pad to 10
        assert normalize_bbl("300739001") == "0300739001"

    def test_pads_very_short_string(self) -> None:
        assert normalize_bbl("1") == "0000000001"

    def test_strips_non_digits(self) -> None:
        # Some sources include hyphens or spaces
        assert normalize_bbl("3-00739-0001") == "3007390001"

    def test_none_returns_none(self) -> None:
        assert normalize_bbl(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_bbl("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_bbl("   ") is None

    def test_non_numeric_string_returns_none(self) -> None:
        assert normalize_bbl("N/A") is None

    def test_alpha_only_returns_none(self) -> None:
        assert normalize_bbl("UNKNOWN") is None

    def test_zero_integer_returns_none(self) -> None:
        # BBL 0 is not a valid lot; treat as unmatched
        assert normalize_bbl(0) is None

    def test_11_digit_truncates_to_10(self) -> None:
        # Prepended leading zero — some sources do this
        assert normalize_bbl("03007390001") == "3007390001"

    def test_leading_zeros_preserved_after_pad(self) -> None:
        # Borough 1 (Manhattan), low block/lot numbers
        assert normalize_bbl("1000010001") == "1000010001"

    def test_string_with_leading_whitespace(self) -> None:
        assert normalize_bbl("  3007390001  ") == "3007390001"


# ===========================================================================
# resolve_bbl
# ===========================================================================

class TestResolveBbl:
    def test_same_bbl_and_appbbl_returns_bbl(self) -> None:
        assert resolve_bbl("3007390001", "3007390001") == "3007390001"

    def test_different_appbbl_returns_appbbl(self) -> None:
        # Condo unit → use billing BBL
        assert resolve_bbl("3007390002", "3007390001") == "3007390001"

    def test_none_appbbl_returns_bbl(self) -> None:
        assert resolve_bbl("3007390001", None) == "3007390001"

    def test_zero_appbbl_returns_bbl(self) -> None:
        # PLUTO stores '0000000000' to mean "no billing BBL"
        assert resolve_bbl("3007390001", "0000000000") == "3007390001"

    def test_none_bbl_returns_none(self) -> None:
        assert resolve_bbl(None, "3007390001") is None

    def test_none_bbl_none_appbbl_returns_none(self) -> None:
        assert resolve_bbl(None, None) is None

    def test_condo_resolution_is_not_symmetric(self) -> None:
        # resolve_bbl(unit, billing) != resolve_bbl(billing, unit)
        unit = "3007390002"
        billing = "3007390001"
        assert resolve_bbl(unit, billing) == billing
        assert resolve_bbl(billing, unit) == unit  # billing has different appbbl → returns unit


# ===========================================================================
# emit_unmatched_report / get_unmatched_report
# ===========================================================================

class TestUnmatchedReport:
    def setup_method(self) -> None:
        reset_unmatched_report()

    def test_single_source(self) -> None:
        emit_unmatched_report("rodent_inspections", total=1000, unmatched=42)
        report = get_unmatched_report()
        assert report["rodent_inspections"]["total"] == 1000
        assert report["rodent_inspections"]["unmatched"] == 42
        assert report["rodent_inspections"]["unmatched_pct"] == 4.2

    def test_multiple_sources(self) -> None:
        emit_unmatched_report("rodent_inspections", total=1000, unmatched=10)
        emit_unmatched_report("dob_permits", total=500, unmatched=5)
        report = get_unmatched_report()
        assert "rodent_inspections" in report
        assert "dob_permits" in report

    def test_accumulates_across_batches(self) -> None:
        emit_unmatched_report("rodent_inspections", total=500, unmatched=10)
        emit_unmatched_report("rodent_inspections", total=500, unmatched=5)
        report = get_unmatched_report()
        assert report["rodent_inspections"]["total"] == 1000
        assert report["rodent_inspections"]["unmatched"] == 15

    def test_zero_unmatched(self) -> None:
        emit_unmatched_report("weather_daily", total=365, unmatched=0)
        report = get_unmatched_report()
        assert report["weather_daily"]["unmatched_pct"] == 0.0

    def test_zero_total_does_not_divide_by_zero(self) -> None:
        emit_unmatched_report("empty_source", total=0, unmatched=0)
        report = get_unmatched_report()
        assert report["empty_source"]["unmatched_pct"] == 0.0

    def test_reset_clears_state(self) -> None:
        emit_unmatched_report("rodent_inspections", total=100, unmatched=5)
        reset_unmatched_report()
        assert get_unmatched_report() == {}
