"""Health check for the running chatbot, with e-mail alerts.

Run as ``python -m oekg.healthcheck`` (the systemd timer in ``deploy/`` does so
every five minutes). It exercises what a question actually needs -- the web app,
the chat model, the embedding model and the OEKG SPARQL endpoint -- instead of
only asking whether the page loads. A page that answers HTTP 200 while every
question fails (an expired key, an exhausted account) is exactly the outage a
plain uptime probe cannot see.

Alerting is edge-triggered, so a long outage costs two mails rather than
hundreds: one once ``OEKG_HEALTH_FAILURES`` consecutive runs have failed, one
when the checks pass again. The state is a small JSON file beside the logs. A
mail that could not be sent never counts as delivered, so the next run retries.

The chat check calls the fast (summary) agent only. The reasoning agent can need
minutes for one answer, which is no basis for a probe that runs every five
minutes; key, quota and endpoint failures hit both agents alike.

This process cannot report that its own host is down. An external probe on the
public URL covers that case; this one covers everything behind it.

``python -m oekg.healthcheck --test-mail`` sends one message to verify the SMTP
settings without running any check. Exit codes: 0 all checks passed, 1 a check
failed, 2 an alert could not be sent.
"""

from __future__ import annotations

import argparse
import json
import re
import smtplib
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Sequence

import faiss
import requests
from dotenv import load_dotenv
from openai import OpenAI

from oekg.config import ROOT_DIR, AppConfig, get_config
from oekg.oep import OEPClient

# Per request. A cold gateway was observed to need ~20 s for its first call.
_REQUEST_TIMEOUT = 60.0
_SMTP_TIMEOUT = 30.0
_KEY_FRAGMENT = re.compile(r"sk-[A-Za-z0-9_\-*.]+")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check."""

    name: str
    ok: bool
    detail: str
    seconds: float


def _redact(text: str, config: AppConfig) -> str:
    """Strip secrets from error text before it lands in a log or a mail.

    Provider error bodies often echo a masked key, so both the configured values
    and anything shaped like one are removed.
    """
    secrets = (
        config.llm_api_key,
        config.embedding_api_key,
        config.oep_token,
        config.health_smtp_password,
    )
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return _KEY_FRAGMENT.sub("sk-***", text)


def _brief(exc: BaseException, config: AppConfig) -> str:
    """One redacted line describing an exception."""
    text = " ".join(str(exc).split())
    line = f"{type(exc).__name__}: {text[:240]}" if text else type(exc).__name__
    return _redact(line, config)


# -- checks -----------------------------------------------------------------
# Each check returns a short success note or raises. They never fall back to a
# different key: with api_key=None the OpenAI SDK would quietly read
# OPENAI_API_KEY and send it to a host it does not belong to.


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _check_app(config: AppConfig) -> str:
    response = requests.get(config.health_app_url, timeout=10)
    response.raise_for_status()
    return f"HTTP {response.status_code}"


def _check_chat(config: AppConfig) -> str:
    client = OpenAI(
        api_key=_require(config.llm_api_key, "LLM_API_KEY"),
        base_url=config.llm_base_url,
        timeout=_REQUEST_TIMEOUT,
        max_retries=0,
    )
    try:
        reply = client.chat.completions.create(
            model=config.summary_model,
            messages=[
                {"role": "system", "content": "Reply with one word."},
                {"role": "user", "content": "Say OK."},
            ],
            max_tokens=8,
        )
    finally:
        client.close()
    if not (reply.choices[0].message.content or "").strip():
        raise RuntimeError("the model returned an empty completion")
    return "answered"


def _check_embeddings(config: AppConfig) -> str:
    client = OpenAI(
        api_key=_require(config.embedding_api_key, "EMBEDDING_API_KEY"),
        base_url=config.embedding_base_url,
        timeout=_REQUEST_TIMEOUT,
        max_retries=0,
    )
    try:
        response = client.embeddings.create(model=config.embedding_model, input="health check")
    finally:
        client.close()
    dimension = len(response.data[0].embedding)
    expected = faiss.read_index(str(config.faiss_index_path)).d
    if dimension != expected:
        raise RuntimeError(
            f"embedding dimension {dimension} does not match the vector store ({expected}); "
            "was the model changed without rebuilding the index?"
        )
    return f"{config.embedding_model}, {dimension} dimensions"


def _check_sparql(config: AppConfig) -> str:
    result = OEPClient(config.oep_token, config).run_sparql(
        "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", strict=True
    )
    if not result.get("results", {}).get("bindings"):
        raise RuntimeError("the endpoint answered but returned no triples")
    return "answered"


CHECKS: tuple[tuple[str, Callable[[AppConfig], str]], ...] = (
    ("app", _check_app),
    ("chat", _check_chat),
    ("embeddings", _check_embeddings),
    ("sparql", _check_sparql),
)


def run_checks(
    config: AppConfig, checks: Sequence[tuple[str, Callable[[AppConfig], str]]] = CHECKS
) -> list[CheckResult]:
    """Run every check; one failing never prevents the others from running."""
    results: list[CheckResult] = []
    for name, check in checks:
        started = time.perf_counter()
        try:
            ok, detail = True, _redact(check(config), config)
        except Exception as exc:  # noqa: BLE001 - any failure is the finding
            ok, detail = False, _brief(exc, config)
        results.append(CheckResult(name, ok, detail, time.perf_counter() - started))
    return results


# -- alert state ------------------------------------------------------------


def _empty_state() -> dict[str, Any]:
    return {"failures": 0, "alerted": False, "since": None, "failed": []}


def load_state(path: Path) -> dict[str, Any]:
    """Read the saved state; a missing or corrupt file means "all was well"."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_state()
    return {**_empty_state(), **data} if isinstance(data, dict) else _empty_state()


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write the state atomically so a crash cannot leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def advance(
    state: dict[str, Any], results: Sequence[CheckResult], now: datetime, threshold: int
) -> tuple[dict[str, Any], str | None]:
    """Fold one run into the state and say whether to mail.

    The event is ``"down"`` when the failure streak reaches ``threshold`` (once
    per outage), ``"recovered"`` on the first clean run after such an alert, and
    ``None`` otherwise. The returned state assumes the mail goes out; the caller
    restores ``alerted`` if it does not, so the next run tries again.
    """
    failed = [result.name for result in results if not result.ok]
    if not failed:
        return _empty_state(), ("recovered" if state["alerted"] else None)

    failures = state["failures"] + 1
    since = state["since"] if state["failures"] else now.isoformat(timespec="seconds")
    new = {"failures": failures, "alerted": state["alerted"], "since": since, "failed": failed}
    if failures >= threshold and not state["alerted"]:
        new["alerted"] = True
        return new, "down"
    return new, None


# -- mail -------------------------------------------------------------------


def _sender(config: AppConfig) -> str:
    return (
        config.health_mail_from
        or config.health_smtp_user
        or f"oekg-healthcheck@{socket.gethostname()}"
    )


def _report(results: Sequence[CheckResult]) -> list[str]:
    return [
        f"  {'ok  ' if r.ok else 'FAIL'}  {r.name:<11} {r.detail}  ({r.seconds:.1f}s)"
        for r in results
    ]


def build_message(
    config: AppConfig, event: str, results: Sequence[CheckResult], outage: dict[str, Any]
) -> EmailMessage:
    """Compose the alert. ``outage`` is the state describing the failure streak."""
    failed = [r for r in results if not r.ok]
    if event == "down":
        subject = f"[Ask OEKG] DOWN: {', '.join(r.name for r in failed)}"
        lead = (
            f"{len(failed)} of {len(results)} checks failed in {outage['failures']} "
            f"consecutive runs (first failure {outage['since']} UTC)."
        )
    else:
        subject = "[Ask OEKG] recovered"
        lead = f"All checks pass again. The outage began {outage['since']} UTC."
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = [lead, "", *_report(results), "", f"Host: {socket.gethostname()}", f"Time: {stamp} UTC"]

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _sender(config)
    message["To"] = config.health_mail_to
    message.set_content("\n".join(body))
    return message


def send_mail(config: AppConfig, message: EmailMessage) -> None:
    """Send ``message`` through the configured SMTP relay."""
    if not (config.health_smtp_host and config.health_mail_to):
        raise RuntimeError(
            "SMTP is not configured (set OEKG_HEALTH_SMTP_HOST and OEKG_HEALTH_MAIL_TO)"
        )
    implicit_tls = config.health_smtp_port == 465
    use_starttls = config.health_smtp_starttls and not implicit_tls
    if config.health_smtp_user and not (implicit_tls or use_starttls):
        raise RuntimeError("refusing to send SMTP credentials without TLS")

    factory = smtplib.SMTP_SSL if implicit_tls else smtplib.SMTP
    with factory(config.health_smtp_host, config.health_smtp_port, timeout=_SMTP_TIMEOUT) as smtp:
        smtp.ehlo()
        if use_starttls:
            smtp.starttls()
            smtp.ehlo()
        if config.health_smtp_user:
            smtp.login(config.health_smtp_user, config.health_smtp_password or "")
        smtp.send_message(message)


# -- entry point ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m oekg.healthcheck", description="Check the chatbot and mail on failure."
    )
    parser.add_argument(
        "--test-mail",
        action="store_true",
        help="send one test message to verify the SMTP settings, then exit",
    )
    args = parser.parse_args(argv)

    # Fields that read the environment lazily pick this up; the package has
    # already been imported, so anything read at import time has not.
    load_dotenv(ROOT_DIR / ".env")
    config = get_config()

    if args.test_mail:
        message = EmailMessage()
        message["Subject"] = "[Ask OEKG] test mail"
        message["From"] = _sender(config)
        message["To"] = config.health_mail_to
        message.set_content("The health check can send mail. Nothing is wrong.")
        try:
            send_mail(config, message)
        except Exception as exc:  # noqa: BLE001 - report, do not trace
            print(f"test mail failed: {_brief(exc, config)}", file=sys.stderr)
            return 2
        print(f"test mail sent to {config.health_mail_to}")
        return 0

    results = run_checks(config)
    print("\n".join(line.lstrip() for line in _report(results)))

    state = load_state(config.health_state_path)
    new, event = advance(
        state, results, datetime.now(timezone.utc), config.health_failure_threshold
    )
    code = 0 if all(result.ok for result in results) else 1

    if event:
        outage = new if event == "down" else state
        try:
            send_mail(config, build_message(config, event, results, outage))
            print(f"alert sent: {event}")
        except Exception as exc:  # noqa: BLE001 - keep the state honest, then report
            new["alerted"] = state["alerted"]
            print(f"alert NOT sent ({event}): {_brief(exc, config)}", file=sys.stderr)
            code = 2

    save_state(config.health_state_path, new)
    return code


if __name__ == "__main__":
    sys.exit(main())
