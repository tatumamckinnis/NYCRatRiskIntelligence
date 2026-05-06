"""BBL (Borough-Block-Lot) normalization and condo billing-BBL resolution.

BBL is a 10-character identifier used across NYC datasets (DOHMH inspections,
DOB permits, PLUTO, restaurant inspections).  Raw values arrive in inconsistent
formats — numeric integers, strings of varying length, None — and must be
normalized before any cross-dataset join.

Condo billing-BBL note (PLUTO §5.3):
    In NYC condo buildings, individual units have their own BBL but taxes are
    billed against a single "billing BBL" (APPBBL in PLUTO).  When joining a
    non-PLUTO source to PLUTO, use APPBBL as the join key whenever it differs
    from the unit BBL, so that condo units map to the same PLUTO row as the
    rest of the building.

Data-quality reporting:
    emit_unmatched_report() accumulates per-source unmatched counts in memory.
    build_panel.py calls it after each ingest script and write_dq_report()
    serialises the totals to the markdown data-quality report (T-11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_DIGITS_RE = re.compile(r"\D")


def normalize_bbl(raw: str | int | None) -> str | None:
    """Normalize a raw BBL value to a 10-character zero-padded string.

    Returns None for NULL, empty, or non-numeric inputs so callers can emit
    an unmatched count rather than producing a silently wrong join key.

    Args:
        raw: BBL as it arrives from a source dataset — may be an integer,
             a string of 1–10 digits, or None.

    Returns:
        10-character zero-padded string, e.g. '3007390001', or None.

    Examples:
        >>> normalize_bbl("3007390001")
        '3007390001'
        >>> normalize_bbl(3007390001)
        '3007390001'
        >>> normalize_bbl("300739001")   # 9 digits → pad to 10
        '0300739001'
        >>> normalize_bbl(None)
        >>> normalize_bbl("N/A")
    """
    if raw is None:
        return None

    digits = _DIGITS_RE.sub("", str(raw).strip())

    if not digits or set(digits) == {"0"}:
        # All-zero BBL (e.g. integer 0, or "0000000000") is not a valid NYC lot.
        return None

    if len(digits) > 10:
        # Truncate to rightmost 10 digits — some sources prepend a leading 0.
        digits = digits[-10:]

    return digits.zfill(10)


# ---------------------------------------------------------------------------
# Condo billing-BBL resolution
# ---------------------------------------------------------------------------

def resolve_bbl(bbl: str | None, appbbl: str | None) -> str | None:
    """Return the join-key BBL for a PLUTO row, applying condo billing logic.

    When a condo unit's BBL differs from its billing BBL (APPBBL), the billing
    BBL is the correct key for joining to building-level features.  If APPBBL
    is absent or identical to BBL, BBL is returned unchanged.

    Args:
        bbl:    Normalized 10-char BBL for the individual unit/lot.
        appbbl: Normalized 10-char APPBBL from PLUTO; may be None or '0000000000'.

    Returns:
        The effective join-key BBL (string or None).

    Examples:
        >>> resolve_bbl("3007390001", "3007390001")  # same → return bbl
        '3007390001'
        >>> resolve_bbl("3007390002", "3007390001")  # condo → return appbbl
        '3007390001'
        >>> resolve_bbl("3007390001", None)           # no appbbl → return bbl
        '3007390001'
    """
    if bbl is None:
        return None

    # APPBBL is sometimes stored as '0000000000' to mean "no billing BBL".
    if appbbl and appbbl != "0000000000" and appbbl != bbl:
        return appbbl

    return bbl


# ---------------------------------------------------------------------------
# Data-quality unmatched reporting
# ---------------------------------------------------------------------------

@dataclass
class _UnmatchedRecord:
    total: int = 0
    unmatched: int = 0


_unmatched_counts: dict[str, _UnmatchedRecord] = field(default_factory=dict)  # type: ignore[assignment]
_unmatched_counts = {}


def emit_unmatched_report(source: str, total: int, unmatched: int) -> None:
    """Accumulate unmatched BBL counts for a source dataset.

    Called by each ingest script after processing a batch.  The totals are
    aggregated across calls so a script can call this once per batch rather
    than once per run.

    Args:
        source:    Dataset identifier, e.g. 'rodent_inspections', 'dob_permits'.
        total:     Total rows processed in this batch.
        unmatched: Rows where normalize_bbl() returned None.
    """
    if source not in _unmatched_counts:
        _unmatched_counts[source] = _UnmatchedRecord()
    _unmatched_counts[source].total += total
    _unmatched_counts[source].unmatched += unmatched


def get_unmatched_report() -> dict[str, dict[str, int | float]]:
    """Return the accumulated unmatched BBL report as a plain dict.

    Returns:
        {source: {total, unmatched, unmatched_pct}} for each source that
        called emit_unmatched_report().
    """
    report: dict[str, dict[str, int | float]] = {}
    for source, rec in _unmatched_counts.items():
        pct = (rec.unmatched / rec.total * 100) if rec.total else 0.0
        report[source] = {
            "total": rec.total,
            "unmatched": rec.unmatched,
            "unmatched_pct": round(pct, 2),
        }
    return report


def reset_unmatched_report() -> None:
    """Clear accumulated counts. Used in tests to isolate state between cases."""
    _unmatched_counts.clear()
