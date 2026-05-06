"""Socrata pagination helper shared across ingest scripts.

All NYC Open Data datasets use the same Socrata API. This module wraps
sodapy.Socrata to provide a consistent paginated iterator and to pick up
NYC_SOCRATA_APP_TOKEN from the environment automatically.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sodapy import Socrata

DOMAIN = "data.cityofnewyork.us"
PAGE_SIZE = 50_000


def get_client() -> Socrata:
    """Return a Socrata client authenticated with the env token (if set)."""
    token = os.environ.get("NYC_SOCRATA_APP_TOKEN") or None
    return Socrata(DOMAIN, token, timeout=120)


def paginate(
    client: Socrata,
    dataset_id: str,
    where: str | None = None,
    order: str | None = None,
    select: str | None = None,
) -> Iterator[tuple[list[dict], int]]:
    """Yield (batch, offset) tuples from a Socrata dataset, paginating automatically.

    Args:
        client:     Authenticated Socrata client.
        dataset_id: Socrata dataset identifier (e.g. 'p937-wjvj').
        where:      SoQL $where clause, e.g. "inspection_date >= '2023-01-01'".
        order:      SoQL $order clause for stable pagination, e.g. ":id".
        select:     SoQL $select clause to limit columns.

    Yields:
        (rows, offset) — rows is a list of dicts; offset is the current page offset.
    """
    kwargs: dict = {}
    if where:
        kwargs["where"] = where
    if order:
        kwargs["order"] = order
    if select:
        kwargs["select"] = select

    offset = 0
    while True:
        rows = client.get(dataset_id, limit=PAGE_SIZE, offset=offset, **kwargs)
        if not rows:
            break
        yield rows, offset
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
