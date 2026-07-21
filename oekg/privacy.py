"""Privacy policy loading and consent bookkeeping.

The chatbot must not process a question before the user has been shown the
privacy policy and has actively agreed to it. Rendering that gate is the UI
layer's job (see ``streamlit_app.py``); everything that can be decided without
Streamlit lives here so it stays testable.

Consent is bound to a *version* derived from the policy text itself. Editing
``PRIVACY.md`` therefore invalidates consent given for the previous wording
automatically -- there is no separate version number that could be forgotten.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

CONSENT_STATE_KEY = "privacy_consent"


def load_policy(path: Path) -> str:
    """Return the privacy policy text, or ``""`` if it cannot be read.

    A missing or unreadable policy is deliberately *not* fatal here: the caller
    decides what to do with it. The UI treats an empty policy as a hard stop --
    consent to a document nobody can see would be worthless.
    """
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def policy_version(text: str) -> str:
    """Return a short, stable fingerprint of a policy text.

    Whitespace is normalised first so that reflowing a paragraph or changing
    line endings (LF vs CRLF) does not needlessly invalidate consent, while any
    change to the actual wording does.
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def build_consent_record(version: str) -> dict:
    """Return the record stored in the session once consent has been given."""
    return {
        "accepted": True,
        "version": version,
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def consent_is_current(record: dict | None, version: str) -> bool:
    """Whether ``record`` is an acceptance of exactly this policy version."""
    if not isinstance(record, dict) or not version:
        return False
    return bool(record.get("accepted")) and record.get("version") == version
