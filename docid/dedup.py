"""Document deduplication helpers shared by URI connectors and apps.

This module keeps the document identity heuristics in the ``docid`` package
instead of duplicating them in every connector. It combines OCR-stable
transaction tokens with visual fingerprints from :mod:`docid.visual_fingerprint`.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
import unicodedata
from pathlib import Path
from typing import Any

from .visual_fingerprint import FieldSource, compute_fingerprint, hamming_distance, merge_records

# Distinctive, transaction-unique fields. Terminal-constant tokens (POS ID / MID /
# AID) are deliberately excluded: they are identical for every transaction at a
# terminal and so cannot tell two receipts apart.
FINGERPRINT_DISTINCT_FIELDS = ("number", "auth", "time", "card")

# Max dHash Hamming distance at which two images are treated as near-identical.
VISUAL_NEAR_DISTANCE = 10

# Max distance on BOTH perceptual channels (dHash AND pHash) at which two scans
# are the same document even without agreeing OCR tokens.
VISUAL_STRONG_DISTANCE = 6

# Distinctive fields that must agree for a confident fingerprint-only match.
FINGERPRINT_MIN_MATCH = 2

# Business key for receipts that carry no transaction token (cash receipts): the
# merchant + calendar date + grand total + currency. It is the only stable identity
# signal for such documents, so it is used -- corroborated, never alone -- to collapse
# re-scans that differ in framing (visual hashes too far apart) and OCR text.
BUSINESS_KEY_FIELDS = ("contractor", "date", "amount", "currency")

# Corroboration for a business-key match: monetary tokens (line-item prices + totals)
# are far more OCR-stable than words, so two scans of one receipt share most amounts.
# Require both a minimum count of shared amounts and a minimum Jaccard so that two
# distinct receipts that merely share a grand total do not collapse together.
MONEY_OVERLAP_MIN_SHARED = 3
MONEY_OVERLAP_MIN_JACCARD = 0.4

# Visual corroboration distance for the business-key path. Looser than the standalone
# VISUAL_NEAR_DISTANCE because the exact merchant + date + total + currency match is
# already a strong gate, so the images only need to be roughly the same layout (a hand
# re-scan that drifted just past the strict threshold, e.g. the real ORLEN pair at 11).
BUSINESS_KEY_VISUAL_NEAR = 14

METADATA_UNKNOWN = {"", "nieznana", "kwota-nieznana", "unknown", "kontrahent-nieznany", "ina-gruba"}


def _path(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve()


def normalize_text(text: str) -> str:
    """Normalize OCR text for stable fallback hashing."""
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"[^a-zA-Z0-9.,:/@+\- ]+", " ", folded.lower())
    return re.sub(r"\s+", " ", folded).strip()


def transaction_fingerprint(text: str) -> dict[str, str]:
    """Extract OCR-stable, transaction-distinctive tokens from receipt/invoice text."""
    raw = text or ""
    low = raw.lower()
    fp = {"number": "", "auth": "", "time": "", "card": ""}

    number = re.search(r"(?:rachunek|paragon|faktura)\s*nr\b[^0-9]{0,6}([0-9]{3,})", low)
    if not number:
        number = re.search(r"\bnr\b[^0-9]{0,4}([0-9]{4,})", low)
    if number:
        fp["number"] = number.group(1)

    auth = re.search(r"(?:autoryzacji|ryzacji|authoriz\w*|\bcji)\b[^0-9]{0,6}([0-9]{4,})", low)
    if not auth:
        auth = re.search(r"([0-9]{5,7})\s*\(\s*1\s*\)", low)
    if auth:
        fp["auth"] = auth.group(1)

    tmatch = re.search(r"\b([0-2]?\d):([0-5]\d):([0-5]\d)\b", raw)
    if tmatch:
        fp["time"] = f"{int(tmatch.group(1)):02d}{tmatch.group(2)}{tmatch.group(3)}"

    card = re.search(r"\b(\d{4})\b\s*wa\w?zna\s*do", low)
    if card:
        fp["card"] = card.group(1)

    # Full transaction timestamp (date + time). Distinctive for non-terminal fiscal
    # receipts that lack RACHUNEK NR / AUTORYZACJI. Encoded as YYYYMMDDHHMM[SS]; the
    # trailing SS marks second resolution, which `document_matches` treats as a
    # high-precision standalone match (a full-second timestamp collision across two
    # different receipts is effectively impossible).
    ymd = ""
    iso = re.search(r"\b(20\d{2})[-./](\d{2})[-./](\d{2})\b", raw)
    if iso:
        ymd = f"{iso.group(1)}{iso.group(2)}{iso.group(3)}"
    else:
        dmy = re.search(r"\b(\d{2})[-./](\d{2})[-./](20\d{2})\b", raw)
        if dmy:
            ymd = f"{dmy.group(3)}{dmy.group(2)}{dmy.group(1)}"
    hm = re.search(r"\b([0-2]?\d):([0-5]\d)(?::([0-5]\d))?\b", raw)
    if ymd and hm:
        stamp = f"{ymd}{int(hm.group(1)):02d}{hm.group(2)}"
        if hm.group(3):
            stamp += hm.group(3)
        fp["datetime"] = stamp
    return fp


def fingerprint_match_count(a: dict[str, Any] | None, b: dict[str, Any] | None) -> int:
    """How many distinctive fields agree (both present and equal)."""
    if not a or not b:
        return 0
    count = 0
    for key in FINGERPRINT_DISTINCT_FIELDS:
        va, vb = str(a.get(key) or ""), str(b.get(key) or "")
        if va and va == vb:
            count += 1
    return count


def image_dhash(path: str | Path) -> str:
    """64-bit difference hash of an image as a hex string, or ``""`` on failure."""
    try:
        fp = compute_fingerprint(_path(path))
        return fp.dhash if fp else ""
    except Exception:  # noqa: BLE001
        return ""


def image_phash(path: str | Path) -> str:
    """64-bit DCT perceptual hash of an image as a hex string, or ``""`` on failure."""
    try:
        fp = compute_fingerprint(_path(path))
        return fp.phash if fp else ""
    except Exception:  # noqa: BLE001
        return ""


def dhash_distance(a: str, b: str) -> int:
    """Hamming distance between equal-length hex perceptual hashes.

    Returns a large sentinel if the values are unusable. The historical name is
    kept for connector compatibility, but this works for dHash and pHash.
    """
    distance = hamming_distance(a, b)
    return 999 if distance is None else distance


def metadata_completeness(meta: dict[str, Any] | None) -> int:
    """Score how complete extracted metadata is. Higher means a better scan."""
    if not meta:
        return 0
    score = 0
    amount = str(meta.get("amount") or "").strip().lower()
    if amount and amount not in METADATA_UNKNOWN:
        score += 2
    contractor = str(meta.get("contractor") or "").strip().lower()
    if contractor and contractor not in METADATA_UNKNOWN:
        score += 1
    if str(meta.get("date") or "").strip():
        score += 1
    doc_type = str(meta.get("type") or "").strip().lower()
    if doc_type and doc_type != "dokument":
        score += 1
    return score


def document_id(path: str | Path, ocr_text: str, *, normalized_text: str | None = None) -> dict[str, Any]:
    """Stable document id via the docid pipeline, with a local sha fallback."""
    docid_error = ""
    docid_log = ""
    try:
        from .pipeline import get_document_id as pipeline_get_document_id

        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            value = str(pipeline_get_document_id(str(_path(path))) or "").strip()
        docid_log = log_buffer.getvalue().strip()
        if value:
            result = {"id": value, "provider": "docid", "source": "get_document_id"}
            if docid_log:
                result["docidLog"] = docid_log[:240]
            return result
    except Exception as exc:  # noqa: BLE001
        docid_error = str(exc)

    normalized = normalized_text if normalized_text is not None else normalize_text(ocr_text)
    if len(normalized) >= 24:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        source = "ocr-text"
    else:
        target = _path(path)
        source_bytes = target.read_bytes() if target.is_file() else normalized.encode("utf-8")
        digest = hashlib.sha256(source_bytes).hexdigest()
        source = "file-sha256"
    result = {"id": f"LOCAL-DOC-{digest[:16].upper()}", "provider": "local-fallback", "source": source}
    if docid_error:
        result["docidError"] = docid_error[:240]
    if docid_log:
        result["docidLog"] = docid_log[:240]
    return result


def _meta_value(doc: dict[str, Any] | None, field: str) -> str:
    """Read a metadata field from a document, top-level or under ``metadata``."""
    if not isinstance(doc, dict):
        return ""
    value = doc.get(field)
    if (value is None or str(value).strip() == "") and isinstance(doc.get("metadata"), dict):
        value = doc["metadata"].get(field)
    return str(value or "").strip().lower()


def business_key(doc: dict[str, Any] | None) -> tuple[str, str, str, str] | None:
    """Return ``(contractor, date, amount, currency)`` when all are present and
    meaningful, else ``None``.

    Reads either top-level fields (archive records) or a nested ``metadata`` dict
    (candidate objects). Returns ``None`` if the merchant or total amount is missing or
    a known placeholder, since the key would not then identify a specific receipt.
    """
    contractor = _meta_value(doc, "contractor")
    if not contractor or contractor in METADATA_UNKNOWN:
        return None
    date = _meta_value(doc, "date")
    if not date:
        return None
    amount = _meta_value(doc, "amount")
    if not amount or amount in METADATA_UNKNOWN:
        return None
    return (contractor, date, amount, _meta_value(doc, "currency"))


def money_tokens(text: str) -> set[str]:
    """Monetary amounts (``n.dd``) found in OCR text, comma normalized to a dot.

    More OCR-stable than words: a shared set of line-item prices and totals corroborates
    that two scans are the same physical receipt.
    """
    return {m.replace(",", ".") for m in re.findall(r"\d{1,4}[.,]\d{2}", text or "")}


def money_overlap(a_text: str, b_text: str) -> tuple[int, float]:
    """Return ``(shared_count, jaccard)`` of the monetary tokens in two OCR texts."""
    a, b = money_tokens(a_text), money_tokens(b_text)
    if not a or not b:
        return 0, 0.0
    shared = a & b
    union = a | b
    return len(shared), (len(shared) / len(union) if union else 0.0)


def document_matches(
    existing: dict[str, Any],
    *,
    doc_id: str = "",
    source_sha256: str = "",
    text_sha256: str = "",
    fingerprint: dict[str, Any] | None = None,
    dhash: str = "",
    phash: str = "",
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> str:
    """Return a non-empty reason if ``existing`` is the same document."""
    fingerprint = fingerprint or {}
    if doc_id and existing.get("docId") == doc_id:
        return "docId"
    if source_sha256 and existing.get("sourceSha256") == source_sha256:
        return "sourceSha256"
    if text_sha256 and existing.get("textSha256") == text_sha256:
        return "textSha256"
    # A full second-resolution transaction timestamp (date + HH:MM:SS) uniquely
    # identifies a transaction, so it stands alone even with no other token.
    cand_dt = str(fingerprint.get("datetime") or "")
    exist_dt = str((existing.get("fingerprint") or {}).get("datetime") or "")
    if cand_dt and cand_dt == exist_dt and len(cand_dt) >= 14:
        return "datetime"
    matches = fingerprint_match_count(fingerprint, existing.get("fingerprint"))
    if matches >= FINGERPRINT_MIN_MATCH:
        return f"fingerprint:{matches}"
    if matches >= 1 and dhash and dhash_distance(dhash, str(existing.get("dhash") or "")) <= VISUAL_NEAR_DISTANCE:
        return "fingerprint+visual"

    ephash = str(existing.get("phash") or "")
    if phash and ephash:
        edhash = str(existing.get("dhash") or "")
        d_phash = dhash_distance(phash, ephash)
        d_dhash = dhash_distance(dhash, edhash) if (dhash and edhash) else 0
        if d_phash <= VISUAL_STRONG_DISTANCE and d_dhash <= VISUAL_STRONG_DISTANCE:
            return "visual-strong"

    # Cash receipts carry no transaction token and a hand re-scan rarely lands within the
    # strict visual threshold, so none of the signals above can collapse them. Fall back to
    # the merchant + date + total + currency business key, but only when it is corroborated
    # -- by a strongly overlapping set of monetary tokens (same line items) or by visually
    # near images -- so two distinct receipts sharing only a grand total stay separate.
    cand_key = business_key(metadata)
    if cand_key is not None and cand_key == business_key(existing):
        shared, jaccard = money_overlap(text, str(existing.get("text") or ""))
        money_corroborated = shared >= MONEY_OVERLAP_MIN_SHARED and jaccard >= MONEY_OVERLAP_MIN_JACCARD
        edhash = str(existing.get("dhash") or "")
        visual_corroborated = bool(dhash and edhash and dhash_distance(dhash, edhash) <= BUSINESS_KEY_VISUAL_NEAR)
        if money_corroborated or visual_corroborated:
            return "business-key"
    return ""


def find_duplicate(
    documents: list[Any],
    *,
    doc_id: str = "",
    source_sha256: str = "",
    text_sha256: str = "",
    fingerprint: dict[str, Any] | None = None,
    dhash: str = "",
    phash: str = "",
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> dict[str, Any] | None:
    """Find an already-known document that is the same as the incoming scan."""
    match: dict[str, Any] | None = None
    for item in documents or []:
        if not isinstance(item, dict):
            continue
        reason = document_matches(
            item,
            doc_id=doc_id,
            source_sha256=source_sha256,
            text_sha256=text_sha256,
            fingerprint=fingerprint,
            dhash=dhash,
            phash=phash,
            metadata=metadata,
            text=text,
        )
        if reason:
            match = {**item, "matchReason": reason}
    return match


def evaluate(candidate: dict[str, Any], documents: list[Any]) -> dict[str, Any]:
    """Decide whether an incoming scan is new, duplicate, or supersedes an archive doc."""
    candidate = candidate or {}
    match = find_duplicate(
        documents,
        doc_id=str(candidate.get("docId") or ""),
        source_sha256=str(candidate.get("sourceSha256") or ""),
        text_sha256=str(candidate.get("textSha256") or ""),
        fingerprint=candidate.get("fingerprint") or {},
        dhash=str(candidate.get("dhash") or ""),
        phash=str(candidate.get("phash") or ""),
        metadata=candidate,
        text=str(candidate.get("text") or ""),
    )
    if not match:
        return {"action": "new", "reason": "", "match": None}
    # Read completeness from a nested metadata dict when present, else the record itself,
    # symmetrically for both sides -- otherwise a top-level-field archive record (no nested
    # "metadata") always scores 0 on the candidate side and a better re-scan could never
    # supersede the worse archived one.
    new_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else candidate
    new_score = metadata_completeness(new_meta)
    old_meta = match.get("metadata") if isinstance(match.get("metadata"), dict) else match
    old_score = metadata_completeness(old_meta)
    action = "supersede" if new_score > old_score else "duplicate"
    return {
        "action": action,
        "reason": match.get("matchReason") or "exact",
        "match": match,
        "newCompleteness": new_score,
        "existingCompleteness": old_score,
    }


def reconcile(documents: list[Any]) -> list[dict[str, Any]]:
    """Group archived documents that are the same physical document."""
    docs = [d for d in (documents or []) if isinstance(d, dict)]
    n = len(docs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    reasons: dict[tuple[int, int], str] = {}
    for i in range(n):
        for j in range(i + 1, n):
            reason = document_matches(
                docs[j],
                doc_id=str(docs[i].get("docId") or ""),
                fingerprint=docs[i].get("fingerprint") or {},
                dhash=str(docs[i].get("dhash") or ""),
                phash=str(docs[i].get("phash") or ""),
                metadata=docs[i],
                text=str(docs[i].get("text") or ""),
            )
            # docId equality alone is not physical duplicate evidence.
            if reason and reason != "docId":
                parent[find(i)] = find(j)
                reasons[(i, j)] = reason

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    groups: list[dict[str, Any]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        ranked = sorted(
            members,
            key=lambda m: (metadata_completeness(_meta_of(docs[m])), -m),
            reverse=True,
        )
        keep = docs[ranked[0]]
        drop = [docs[m] for m in ranked[1:]]
        merged = _fuse_metadata(
            [_meta_of(docs[m]) for m in members],
            weights=[metadata_completeness(_meta_of(docs[m])) for m in members],
        )
        a, b = members[0], members[1]
        reason = reasons.get((min(a, b), max(a, b))) or "visual"
        groups.append(
            {
                "keep": keep.get("docId"),
                "drop": [d.get("docId") for d in drop],
                "reason": reason,
                "mergedMetadata": merged,
            }
        )
    return groups


def _meta_of(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        return meta
    return {k: doc.get(k) for k in ("type", "date", "contractor", "amount", "currency")}


def _fuse_metadata(metas: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    fields = ("type", "date", "contractor", "amount", "currency")
    sources = [
        FieldSource(fields={k: m.get(k) for k in fields}, weight=max(float(w), 0.0001))
        for m, w in zip(metas, weights)
    ]
    return merge_records(sources, fields=list(fields))["fields"]


def document_signature(
    text: str = "",
    image: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle identity signals for a scan: fingerprint, visual hashes, completeness."""
    fingerprint = transaction_fingerprint(text)
    dhash = image_dhash(image) if image else ""
    phash = image_phash(image) if image else ""
    return {
        "fingerprint": fingerprint,
        "dhash": dhash,
        "phash": phash,
        "completeness": metadata_completeness(metadata),
    }
