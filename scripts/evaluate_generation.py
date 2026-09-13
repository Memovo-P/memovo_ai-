"""Run the generation evaluation against the real, approved provider.

    MEMOVO_GENERATION_ENABLED=true MEMOVO_OPENROUTER_API_KEY=... \\
        uv run python scripts/evaluate_generation.py

Sends the synthetic evaluation cases -- never user data -- to OpenRouter, so
it is a manual, opt-in tool rather than part of the suite. It refuses to run
unless generation is enabled and a key is configured; it never prints the
key. Writes a JSON report with every answer to ``eval/results/`` for human
review against each case's rubric, and prints the structural summary.

The summary reports; it decides nothing. Quality targets are set with the
owner from these reviews (plan P7).
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from memovo_ai.api.dependencies import build_generation_provider
from memovo_ai.core.config import GenerationSettings, OpenRouterSettings
from memovo_ai.evaluation.generation import (
    evaluate_chat,
    evaluate_notes,
    load_chat_cases,
    load_note_cases,
    summarize_chat,
    summarize_notes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "eval" / "datasets"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, default=DATASETS / "generation_chat.json")
    parser.add_argument("--notes", type=Path, default=DATASETS / "note_preparation.json")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-notes", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = GenerationSettings()
    credentials = OpenRouterSettings()

    if not settings.enabled or not credentials.is_configured:
        print(
            "refusing to run: set MEMOVO_GENERATION_ENABLED=true and "
            "MEMOVO_OPENROUTER_API_KEY (the key is never printed)"
        )
        return 2

    provider = build_generation_provider(settings, credentials)
    if provider is None:
        print("refusing to run: generation is disabled")
        return 2
    print(f"provider: {settings.provider}  model: {settings.model}")

    started = time.time()
    report: dict[str, object] = {
        "provider": settings.provider,
        "model": settings.model,
        "chat_max_output_tokens": settings.chat_max_output_tokens,
        "note_max_output_tokens": settings.note_max_output_tokens,
    }

    try:
        if not args.skip_chat:
            chat_cases = load_chat_cases(args.chat)
            chat_outcomes = await evaluate_chat(chat_cases, generation=provider, settings=settings)
            chat_summary = summarize_chat(chat_outcomes)
            report["chat"] = {
                "dataset": args.chat.name,
                "summary": asdict(chat_summary),
                "outcomes": [asdict(o) for o in chat_outcomes],
            }
            print("\nchat:", json.dumps(asdict(chat_summary), indent=2))
            for outcome in chat_outcomes:
                flag = "ERR " if outcome.error else ("    " if not outcome.checks else "CHK ")
                print(f"  {flag}{outcome.case_id:<24} {outcome.error or ', '.join(outcome.checks)}")

        if not args.skip_notes:
            note_cases = load_note_cases(args.notes)
            note_outcomes = await evaluate_notes(note_cases, generation=provider, settings=settings)
            note_summary = summarize_notes(note_outcomes)
            report["notes"] = {
                "dataset": args.notes.name,
                "summary": asdict(note_summary),
                "outcomes": [asdict(o) for o in note_outcomes],
            }
            print("\nnotes:", json.dumps(asdict(note_summary), indent=2))
            for outcome in note_outcomes:
                detail = outcome.error or (
                    f"missing={list(outcome.missing)} added={list(outcome.added)}"
                )
                print(f"  {outcome.case_id:<24} {detail}")
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()

    report["elapsed_seconds"] = round(time.time() - started, 1)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"generation_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO_ROOT)} -- review every answer against its rubric")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
