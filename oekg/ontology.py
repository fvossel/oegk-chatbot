"""Link OEO ontology terms to their authoritative OEP definition pages.

When a result contains OEO term IRIs (e.g. a study-descriptor tag or a sector
class), turn the opaque ``OEO_########`` ids into clickable links to the OEP
ontology browser, using the human label from the loaded ontology documents.
"""

from __future__ import annotations

import re
from typing import Any

_OEO_IRI_RE = re.compile(r"https://openenergyplatform\.org/ontology/oeo/(OEO_\d+)")
_DETAIL_URL = "https://openenergyplatform.org/ontology/oeo/{oeo_id}/"


def term_links(values: Any, documents: dict | None = None) -> list[dict[str, str]]:
    """Return ``[{id, label, url}]`` for each distinct OEO term IRI in ``values``."""
    found: dict[str, dict[str, str]] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        match = _OEO_IRI_RE.search(value)
        if not match:
            continue
        oeo_id = match.group(1)
        if oeo_id in found:
            continue
        doc = documents.get("oeo:" + oeo_id) if documents else None
        label = doc.get("label") if isinstance(doc, dict) else None
        found[oeo_id] = {
            "id": oeo_id,
            "label": label or oeo_id,
            "url": _DETAIL_URL.format(oeo_id=oeo_id),
        }
    return list(found.values())
