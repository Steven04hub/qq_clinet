"""Replay synthetic decision cases without connecting to QQ or generating replies."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "fixtures" / "jev_cases.json"
SAFE_ERROR_KINDS = {
    "auth",
    "balance",
    "rate_limit",
    "timeout",
    "connection",
    "api",
    "malformed",
    "schema",
    "empty",
    "unknown",
}


def read_cases(path: Path) -> list[dict[str, Any]]:
    """Reject incomplete fixtures before making billable requests."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a non-empty JSON array.")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Case IDs must be unique non-empty strings.")
        seen.add(case_id)
        if case.get("expected") not in {"reply", "silent", "optional"}:
            raise ValueError("Expected route must be reply, silent, or optional.")
        if not isinstance(case.get("title"), str) or not case["title"]:
            raise ValueError("Every case requires a title.")
        state = case.get("state")
        if not isinstance(state, dict) or not isinstance(state.get("bot_id"), str):
            raise ValueError("Every case requires a state and bot_id.")
        for field in (
            "recent_context",
            "new_messages",
            "recent_bot_replies",
            "relevant_aliases",
            "relevant_memories",
        ):
            if not isinstance(state.get(field), list):
                raise ValueError(f"State requires a list field: {field}.")
        if not state["new_messages"]:
            raise ValueError("Every case requires new messages.")
        if not isinstance(state.get("unknown_media_present"), bool):
            raise ValueError("State requires unknown_media_present as a boolean.")
    return cases


def summarize(results: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    """Keep model disagreement, fallbacks, and request failures separate."""
    decisive = [r for r in results if r["expected"] != "optional"]
    resolved = [r for r in decisive if r["decision"] in {"reply", "silent"}]
    matched = [r for r in resolved if r["decision"] == r["expected"]]
    elapsed = sorted(r["latency_ms"] for r in results)
    return {
        "total_cases": total_cases,
        "attempted": len(results),
        "not_attempted": total_cases - len(results),
        "error_count": sum(r["error_kind"] is not None for r in results),
        "fallback_count": sum(r["decision"] == "fallback" for r in results),
        "optional_count": sum(r["expected"] == "optional" for r in results),
        "decisive_expected_count": len(decisive),
        "decisive_resolved_count": len(resolved),
        "matched_count": len(matched),
        "mismatch_count": len(resolved) - len(matched),
        "coverage": round(len(resolved) / len(decisive), 4) if decisive else None,
        "agreement_when_decided": round(len(matched) / len(resolved), 4) if resolved else None,
        "correct_over_all_decisive": round(len(matched) / len(decisive), 4) if decisive else None,
        "unwanted_reply_count": sum(
            r["expected"] == "silent" and r["decision"] == "reply" for r in results
        ),
        "missed_reply_count": sum(
            r["expected"] == "reply" and r["decision"] == "silent" for r in results
        ),
        "latency_p50_ms": round(statistics.median(elapsed), 1) if elapsed else None,
        "latency_p95_ms": elapsed[math.ceil(len(elapsed) * 0.95) - 1] if elapsed else None,
        "input_tokens": sum(r["input_tokens"] or 0 for r in results),
        "input_token_count_known_for": sum(r["input_tokens"] is not None for r in results),
    }


async def evaluate(args: argparse.Namespace, cases: list[dict[str, Any]]) -> int:
    # Import only after --validate-only, which never opens .env or a network client.
    from dotenv import dotenv_values

    from app.services.jev import DEFAULT_MODEL, JevClient, JevError

    settings = {**dotenv_values(ROOT / ".env"), **os.environ}
    api_key = settings.get("TYPESAFE_API_KEY")
    if not api_key:
        print("Missing TYPESAFE_API_KEY; no requests made.")
        return 2
    model = settings.get("JEV_MODEL") or DEFAULT_MODEL
    results = []
    client = JevClient(api_key, model=model, timeout_seconds=args.timeout)
    try:
        for case in cases:
            started = time.monotonic()
            row = {
                "id": case["id"],
                "title": case["title"],
                "expected": case["expected"],
                "decision": "error",
                "reason": None,
                "scores": {},
                "latency_ms": None,
                "input_tokens": None,
                "error_kind": None,
            }
            try:
                # An outer limit also bounds any future client queueing behavior.
                async with asyncio.timeout(args.timeout):
                    decision = await client.evaluate(case["state"])
                row.update(
                    decision=decision.route,
                    reason=decision.reason,
                    scores=decision.scores,
                    latency_ms=decision.elapsed_ms,
                    input_tokens=decision.input_tokens,
                )
            except JevError as exc:
                row["error_kind"] = exc.kind if exc.kind in SAFE_ERROR_KINDS else "unknown"
            except TimeoutError:
                row["error_kind"] = "timeout"
            except Exception:
                # Never print provider bodies, exception text, request headers or keys.
                row["error_kind"] = "unknown"
            if row["latency_ms"] is None:
                row["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
            results.append(row)
            print(
                f"{case['id']}: expected={case['expected']} "
                f"decision={row['decision']} error={row['error_kind']} "
                f"elapsed_ms={row['latency_ms']}"
            )
            if row["error_kind"] == "auth":
                break
    finally:
        await client.close()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "dataset": "synthetic Chinese group-chat windows; author labels, not user validation",
        "timeout_seconds": args.timeout,
        "summary": summarize(results, len(cases)),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")
    return 1 if report["summary"]["error_count"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / ".state/evals/jev-latest.json")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--limit", type=int, help="Run only the first N synthetic cases.")
    parser.add_argument("--validate-only", action="store_true", help="Validate fixtures offline.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        cases = read_cases(args.cases)
    except (OSError, ValueError, TypeError):
        print("Invalid or unreadable case fixture; no requests made.")
        return 2
    if args.limit:
        cases = cases[: args.limit]
    if args.validate_only:
        print(f"Validated {len(cases)} synthetic cases; no credentials read or requests made.")
        return 0
    try:
        return asyncio.run(evaluate(args, cases))
    except Exception:
        print("Evaluation could not complete (sanitized error; no credentials shown).")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
