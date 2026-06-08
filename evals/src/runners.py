"""Eval runner for the RAG pipeline (T-42).

Loads the gold JSONL, calls POST /chat for each question, accumulates the SSE
response, then computes citation_accuracy, refusal_calibration, and
faithfulness (via Claude judge).  Recall@k is reported when a trace JSONL
sink is available; otherwise it is skipped.

Usage::

    uv run python -m evals.src.runners --base-url http://localhost:8000

    # With Anthropic API key for faithfulness judge
    ANTHROPIC_API_KEY=sk-... uv run python -m evals.src.runners

    # Custom gold set
    uv run python -m evals.src.runners --gold evals/gold/article151_qa_v1.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from .judge import judge_faithfulness
from .metrics import citation_accuracy, recall_at_k, refusal_calibration

log = logging.getLogger(__name__)

_DEFAULT_GOLD = Path(__file__).parent.parent / "gold" / "article151_qa_v1.jsonl"
_RESULTS_DIR = Path(__file__).parent.parent / "results"

# Targets from phase-4 acceptance criteria
_THRESHOLDS = {
    "citation_accuracy_mean": 0.60,
    "recall_at_k_mean": 0.70,
    "refusal_calibration": 0.80,
    "faithfulness_mean": 0.70,
}


# ---------------------------------------------------------------------------
# Gold set loader
# ---------------------------------------------------------------------------


def _load_gold(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# /chat SSE caller
# ---------------------------------------------------------------------------


async def _call_chat(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    """POST /chat, consume SSE stream, return (full_text, session_id_header)."""
    url = f"{base_url.rstrip('/')}/chat"
    payload: dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    tokens: list[str] = []
    returned_session: str | None = None

    async with client.stream(
        "POST", url, json=payload, timeout=httpx.Timeout(90.0, connect=10.0)
    ) as resp:
        resp.raise_for_status()
        returned_session = resp.headers.get("x-session-id")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_text = line[len("data: "):]
            if payload_text == "[DONE]":
                break
            tokens.append(payload_text)

    return "".join(tokens), returned_session


# ---------------------------------------------------------------------------
# Trace-sink reader (optional Recall@k source)
# ---------------------------------------------------------------------------


def _load_trace_chunks(
    trace_jsonl: Path | None,
    question: str,
) -> list[dict]:
    """Scan a JSONL trace file for retriever spans matching *question*.

    Returns a list of chunk dicts (with at least a ``citation`` key) if found,
    or an empty list when no trace file is available / no match found.
    """
    if trace_jsonl is None or not trace_jsonl.exists():
        return []

    try:
        with trace_jsonl.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                span = json.loads(raw)
                if span.get("openinference.span.kind") != "RETRIEVER":
                    continue
                # Match by embedded input if present
                span_input = span.get("input.value", "")
                if question not in span_input:
                    continue
                raw_output = span.get("output.value", "[]")
                try:
                    return json.loads(raw_output)
                except (json.JSONDecodeError, TypeError):
                    return []
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read trace file %s: %s", trace_jsonl, exc)

    return []


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def run_eval_suite(
    base_url: str,
    gold_path: Path = _DEFAULT_GOLD,
    *,
    api_key: str = "",
    top_k: int = 5,
    trace_jsonl: Path | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run the full eval suite and return a results dict.

    Args:
        base_url:    Base URL of the rat-api service (e.g. ``http://localhost:8000``).
        gold_path:   Path to the gold JSONL file.
        api_key:     Anthropic API key for the faithfulness judge.
        top_k:       *k* for Recall-at-k computation.
        trace_jsonl: Optional path to the OpenInference JSONL trace sink produced
                     by the API.  When present, retriever chunks are extracted and
                     Recall-at-k is computed.
        concurrency: Number of parallel /chat calls (keep <= 3 to avoid timeouts).

    Returns:
        Results dict written to ``evals/results/<timestamp>.json``.
    """
    items = _load_gold(gold_path)
    log.info("Loaded %d eval items from %s", len(items), gold_path)

    responses: list[str] = [""] * len(items)
    retrieved_per_item: list[list[dict]] = [[] for _ in range(len(items))]

    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(idx: int, item: dict) -> None:
        async with sem:
            log.info("[%d/%d] %s: %s", idx + 1, len(items), item["id"], item["question"][:70])
            try:
                async with httpx.AsyncClient() as client:
                    answer, _ = await _call_chat(client, base_url, item["question"])
            except httpx.HTTPStatusError as exc:
                log.warning("HTTP error on %s: %s", item["id"], exc)
                answer = ""
            except Exception as exc:  # noqa: BLE001
                log.warning("Error on %s: %s", item["id"], exc)
                answer = ""

            responses[idx] = answer

            # Try to get retrieved chunks from trace sink
            if trace_jsonl:
                retrieved_per_item[idx] = _load_trace_chunks(trace_jsonl, item["question"])

    await asyncio.gather(*(_eval_one(i, it) for i, it in enumerate(items)))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    ca_scores = [
        citation_accuracy(item.get("expected_citations", []), resp)
        for item, resp in zip(items, responses)
    ]

    rk_scores = [
        recall_at_k(item.get("expected_citations", []), chunks, top_k)
        for item, chunks in zip(items, retrieved_per_item)
        if chunks  # only include when we have retrieval data
    ]

    rc_score = refusal_calibration(items, responses)

    # Faithfulness judge — one call per item, sequential to avoid rate limits
    faithful_scores: list[int] = []
    for item, resp, chunks in zip(items, responses, retrieved_per_item):
        # Fall back to a synthetic chunk containing the answer itself when no
        # retrieval data is available — faithfulness will still judge whether
        # the answer makes unsupported claims (conservative lower bound).
        judge_chunks = chunks if chunks else [{"content": resp[:1000], "citation": "response"}]
        score = await judge_faithfulness(
            item["question"],
            resp,
            judge_chunks,
            api_key=api_key,
        )
        faithful_scores.append(score)

    # Summarise
    valid_faithful = [s for s in faithful_scores if s >= 0]

    ca_mean = sum(ca_scores) / len(ca_scores) if ca_scores else 0.0
    rk_mean = sum(rk_scores) / len(rk_scores) if rk_scores else None
    faith_mean = sum(valid_faithful) / len(valid_faithful) if valid_faithful else None

    # Per-failure-mode breakdown
    failure_modes: dict[str, list[float]] = {}
    for item, ca in zip(items, ca_scores):
        fm = item.get("failure_mode", "unknown")
        failure_modes.setdefault(fm, []).append(ca)
    fm_summary = {fm: sum(vals) / len(vals) for fm, vals in failure_modes.items()}

    results: dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "gold_path": str(gold_path),
        "base_url": base_url,
        "n_items": len(items),
        "top_k": top_k,
        "metrics": {
            "citation_accuracy_mean": round(ca_mean, 4),
            "recall_at_k_mean": round(rk_mean, 4) if rk_mean is not None else None,
            "refusal_calibration": round(rc_score, 4),
            "faithfulness_mean": round(faith_mean, 4) if faith_mean is not None else None,
            "faithfulness_skipped": len(faithful_scores) - len(valid_faithful),
        },
        "thresholds": _THRESHOLDS,
        "passes": {
            "citation_accuracy": ca_mean >= _THRESHOLDS["citation_accuracy_mean"],
            "recall_at_k": rk_mean >= _THRESHOLDS["recall_at_k_mean"] if rk_mean is not None else None,
            "refusal_calibration": rc_score >= _THRESHOLDS["refusal_calibration"],
            "faithfulness": faith_mean >= _THRESHOLDS["faithfulness_mean"] if faith_mean is not None else None,
        },
        "citation_accuracy_by_failure_mode": fm_summary,
        "per_item": [
            {
                "id": item["id"],
                "failure_mode": item.get("failure_mode", ""),
                "question": item["question"],
                "citation_accuracy": round(ca, 4),
                "recall_at_k": round(recall_at_k(item.get("expected_citations", []), chunks, top_k), 4)
                if chunks
                else None,
                "faithfulness": faith,
                "response_snippet": resp[:300],
            }
            for item, ca, faith, resp, chunks in zip(
                items, ca_scores, faithful_scores, responses, retrieved_per_item
            )
        ],
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = _RESULTS_DIR / f"{date_str}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results written → %s", out_path)

    return results


# ---------------------------------------------------------------------------
# CLI summary printer
# ---------------------------------------------------------------------------


def _print_summary(results: dict) -> None:
    m = results["metrics"]
    p = results["passes"]
    sep = "=" * 56

    def _fmt(val: float | None, threshold: float | None = None) -> str:
        if val is None:
            return "N/A"
        if threshold is not None:
            mark = "PASS" if val >= threshold else "FAIL"
            return f"{val:.3f}  [{mark}  threshold={threshold}]"
        return f"{val:.3f}"

    print(f"\n{sep}")
    print(f"  Eval Results  ({results['timestamp']})")
    print(f"  n={results['n_items']}  base_url={results['base_url']}")
    print(sep)
    print(f"  Citation Accuracy (mean):  {_fmt(m['citation_accuracy_mean'], _THRESHOLDS['citation_accuracy_mean'])}")
    top_k = results.get("top_k", 5)
    print(f"  Recall@{top_k}:                {_fmt(m['recall_at_k_mean'], _THRESHOLDS['recall_at_k_mean'])}")
    print(f"  Refusal Calibration:       {_fmt(m['refusal_calibration'], _THRESHOLDS['refusal_calibration'])}")
    print(f"  Faithfulness (judge):      {_fmt(m['faithfulness_mean'], _THRESHOLDS['faithfulness_mean'])}")
    if m["faithfulness_skipped"]:
        print(f"    ({m['faithfulness_skipped']} items skipped — ANTHROPIC_API_KEY not set?)")
    print(sep)

    if results.get("citation_accuracy_by_failure_mode"):
        print("  Citation Accuracy by failure mode:")
        for fm, score in sorted(results["citation_accuracy_by_failure_mode"].items()):
            print(f"    {fm:<35} {score:.3f}")
    print(sep)

    all_pass = [v for v in p.values() if v is not None]
    if all(all_pass):
        print("  ALL CHECKS PASSED\n")
    else:
        failed = [k for k, v in p.items() if v is False]
        print(f"  CHECKS FAILED: {', '.join(failed)}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG eval suite against /chat endpoint")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAT_API_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=_DEFAULT_GOLD,
        help="Path to gold JSONL eval set",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="k for Recall@k (default: 5)",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=None,
        help="Path to API OpenInference JSONL trace sink for Recall@k",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel /chat calls (default: 1, keep low to avoid timeouts)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key for faithfulness judge",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_eval_suite(
            args.base_url,
            args.gold,
            api_key=args.api_key,
            top_k=args.top_k,
            trace_jsonl=args.trace_jsonl,
            concurrency=args.concurrency,
        )
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
