"""Record what each question costs the language model: calls, tokens and time.

Runs every question of ``questions.txt`` through the real pipeline (fresh, no
cache) and writes one JSON line per question with every chat-model call it
made: prompt and completion tokens (reasoning included) and the call's wall
time. A question that triggers a clarification is answered again with the
first option, as a user would. ``estimate.py`` turns the result into energy and
CO2.

    python -m scripts.energy.measure --out energy_runs.jsonl --workers 3

Needs the same ``.env`` as the app. Questions are sent to the configured
gateway and land in the request log like any other question.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
_local = threading.local()


def _questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _recording(create):
    """Wrap ``chat.completions.create`` so every call's usage and time is kept."""

    def wrapper(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        response = create(*args, **kwargs)
        seconds = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = (kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking")
        _local.calls.append(
            {
                "seconds": round(seconds, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "thinking": reasoning is not False,
            }
        )
        return response

    return wrapper


def _pipeline():
    from oekg.cache import QueryCache
    from oekg.config import get_config
    from oekg.pipeline import RagPipeline
    from oekg.resources import load_resources

    config = get_config()
    pipeline = RagPipeline(load_resources(config), QueryCache(config.cache_path, enabled=False), config)
    client = pipeline._llm._client  # noqa: SLF001 - measurement hook
    client.chat.completions.create = _recording(client.chat.completions.create)
    return pipeline


def _run(question: str) -> dict[str, Any]:
    if not hasattr(_local, "pipeline"):
        _local.pipeline = _pipeline()
    _local.calls = []
    started = time.perf_counter()
    result = _local.pipeline.answer(question, prefer_cache=False)
    clarified = False
    if result.clarification:
        clarified = True
        option = result.clarification.options[0]
        result = _local.pipeline.answer(question, prefer_cache=False, clarification=option)
    return {
        "question": question,
        "seconds": round(time.perf_counter() - started, 3),
        "clarified": clarified,
        "rows": result.row_count,
        "error": bool(getattr(result, "error", False)),
        "bot_message": result.is_bot_message,
        "calls": _local.calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=HERE / "questions.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--env", type=Path, default=HERE.parents[1] / ".env")
    args = parser.parse_args()

    load_dotenv(args.env)
    # Keep the measurement out of the app's own logs and cache.
    scratch = Path(tempfile.mkdtemp(prefix="oekg-energy-"))
    os.environ.setdefault("OEKG_LOG_DIR", str(scratch / "logs"))
    os.environ.setdefault("OEKG_CACHE_PATH", str(scratch / "cache.json"))

    questions = _questions(args.questions)
    lock = threading.Lock()
    with args.out.open("w", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for done, record in enumerate(pool.map(_run, questions), 1):
            with lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            tokens = sum((c["completion_tokens"] or 0) for c in record["calls"])
            print(f"{done}/{len(questions)} {record['seconds']:6.1f}s {tokens:6d} out-tokens  {record['question']}", flush=True)


if __name__ == "__main__":
    main()
