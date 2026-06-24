from __future__ import annotations

from pathlib import Path

import pytest

from docid.dedup import (
    business_key,
    dhash_distance,
    document_matches,
    document_signature,
    evaluate,
    find_duplicate,
    fingerprint_match_count,
    image_dhash,
    image_phash,
    metadata_completeness,
    money_overlap,
    reconcile,
    transaction_fingerprint,
)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


RECEIPT_TOKENS = "\n".join(
    [
        "Polskie ePlatnosci",
        "POS ID: 00522425 RACHUNEK NR: 181149",
        "1671 WAZNA DO: KK/KK",
        "KOD AUTORYZACJI: 784683 (1)",
        "DATA: 19.06.2026 GODZINA: 09:52:51",
    ]
)


def _doc_like_image(path: Path, seed: int, noise: int = 0) -> None:
    img = Image.new("RGB", (300, 440), (245, 244, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 15, 280, 55], fill=(40, 40, 40))
    rng = seed
    y = 80
    for _ in range(14):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        draw.rectangle([30, y, 30 + 90 + rng % 150, y + 10], fill=(30, 30, 30))
        y += 24
    if noise:
        px = img.load()
        for _ in range(noise):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            px[rng % 300, (rng >> 8) % 440] = (200, 200, 200)
    img.save(path)


def _doc(meta: dict, *, doc_id: str, text: str = RECEIPT_TOKENS, dhash: str = "", phash: str = "") -> dict:
    return {
        "docId": doc_id,
        "fingerprint": transaction_fingerprint(text),
        "dhash": dhash,
        "phash": phash,
        "metadata": meta,
    }


def test_transaction_fingerprint_is_stable_across_ocr_noise() -> None:
    good = "DUO CAFE HANNA GRUBA\nKWOTA: 30,26 zl\n" + RECEIPT_TOKENS
    noisy = "INA GRUBA\n2425 RACHUNEK NR: 181149\nih 1671 WAZNA DO: KX/KX\nCJI: 784663 (1)\nGODZINA: 09:52:51"
    fp_good = transaction_fingerprint(good)
    fp_noisy = transaction_fingerprint(noisy)

    assert fp_good == {
        "number": "181149", "auth": "784683", "time": "095251", "card": "1671",
        "datetime": "20260619095251",
    }
    assert fingerprint_match_count(fp_good, fp_noisy) == 3

    other = transaction_fingerprint(
        "RACHUNEK NR: 999000\n4242 WAZNA DO: KK/KK\nKOD AUTORYZACJI: 111222 (1)\nGODZINA: 17:00:00"
    )
    assert fingerprint_match_count(fp_good, other) == 0


def test_visual_hashes_match_rescans_and_reject_other_documents(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    other = tmp_path / "other.jpg"
    _doc_like_image(first, seed=12345)
    _doc_like_image(second, seed=12345, noise=120)
    _doc_like_image(other, seed=54321)

    first_dhash, second_dhash = image_dhash(first), image_dhash(second)
    first_phash, second_phash = image_phash(first), image_phash(second)

    assert first_dhash and second_dhash and dhash_distance(first_dhash, second_dhash) <= 2
    other_dhash, other_phash = image_dhash(other), image_phash(other)
    assert first_phash and second_phash and dhash_distance(first_phash, second_phash) <= 2

    assert (
        document_matches(
            {"docId": "DOC-A", "fingerprint": {}, "dhash": first_dhash, "phash": first_phash},
            fingerprint={},
            dhash=second_dhash,
            phash=second_phash,
        )
        == "visual-strong"
    )
    assert (
        document_matches(
            {"docId": "DOC-A", "fingerprint": {}, "dhash": first_dhash, "phash": first_phash},
            fingerprint={},
            dhash=other_dhash,
            phash=other_phash,
        )
        == ""
    )


def test_metadata_completeness_and_evaluate() -> None:
    poor_meta = {"type": "rachunek", "date": "2026-06-19", "contractor": "duo cafe", "amount": ""}
    good_meta = {"type": "rachunek", "date": "2026-06-19", "contractor": "duo cafe", "amount": "30.26"}
    assert metadata_completeness(good_meta) > metadata_completeness(poor_meta)

    cand_poor = {"docId": "DOC-1", "fingerprint": transaction_fingerprint(RECEIPT_TOKENS), "metadata": poor_meta}
    assert evaluate(cand_poor, [])["action"] == "new"

    archived = [_doc(poor_meta, doc_id="DOC-1")]
    cand_same = {
        "docId": "DOC-2",
        "fingerprint": transaction_fingerprint("INA GRUBA\n" + RECEIPT_TOKENS),
        "metadata": poor_meta,
    }
    dup = evaluate(cand_same, archived)
    assert dup["action"] == "duplicate"
    assert dup["reason"].startswith("fingerprint") or dup["reason"] == "datetime"

    cand_good = {"docId": "DOC-3", "fingerprint": transaction_fingerprint(RECEIPT_TOKENS), "metadata": good_meta}
    sup = evaluate(cand_good, archived)
    assert sup["action"] == "supersede"
    assert sup["match"]["docId"] == "DOC-1"


def test_find_duplicate_ignores_unrelated_documents() -> None:
    archived = [
        _doc(
            {"amount": "5.00"},
            doc_id="OTHER",
            text="RACHUNEK NR: 999000\n4242 WAZNA DO: KK\nCJI: 111222 (1)\nGODZINA: 17:00:00",
        )
    ]
    assert find_duplicate(archived, fingerprint=transaction_fingerprint(RECEIPT_TOKENS)) is None


def test_reconcile_groups_visual_duplicates_and_fuses_metadata() -> None:
    empty_fp = transaction_fingerprint("garbled")
    docs = [
        {
            "docId": "DOC-A",
            "fingerprint": empty_fp,
            "dhash": "949cacece898bcdc",
            "phash": "aa55ae55ab55aa05",
            "metadata": {"type": "rachunek", "contractor": "DUO CAFE", "amount": ""},
        },
        {
            "docId": "DOC-B",
            "fingerprint": empty_fp,
            "dhash": "949cdcecec98bcdc",
            "phash": "aa55ae55ab55aa05",
            "metadata": {"type": "rachunek", "contractor": "", "amount": "30.26"},
        },
        {
            "docId": "DOC-C",
            "fingerprint": empty_fp,
            "dhash": "0011223344556677",
            "phash": "1122334455667788",
            "metadata": {"type": "faktura", "amount": "99.00"},
        },
    ]
    groups = reconcile(docs)
    assert len(groups) == 1
    group = groups[0]
    assert {group["keep"], *group["drop"]} == {"DOC-A", "DOC-B"}
    assert group["reason"] == "visual-strong"
    assert group["mergedMetadata"]["amount"] == "30.26"
    assert group["mergedMetadata"]["contractor"] == "DUO CAFE"


def test_document_signature_contains_all_identity_signals(tmp_path: Path) -> None:
    image = tmp_path / "scan.jpg"
    _doc_like_image(image, seed=7)

    signature = document_signature(
        text=RECEIPT_TOKENS,
        image=image,
        metadata={"type": "rachunek", "amount": "30.26", "contractor": "DUO CAFE"},
    )
    assert signature["fingerprint"]["number"] == "181149"
    assert signature["dhash"]
    assert signature["phash"]
    assert signature["completeness"] > 0


def test_datetime_fingerprint_dedups_full_timestamp_receipts():
    """A full date+HH:MM:SS timestamp identifies a transaction with no other token."""
    # Same non-terminal receipt, drifting OCR, but the same full timestamp.
    a = {"docId": "DOC-A",
         "fingerprint": transaction_fingerprint("CYFRONIKA\nSUMA 54.61\n06-03-2025 11:43:21\nEAH 1901348533")}
    b = transaction_fingerprint("garbled cyfronlka\nF62995 06-03-2025 11:43:21")
    assert a["fingerprint"]["datetime"] == "20250306114321"
    assert document_matches(a, fingerprint=b) == "datetime"

    # Minute-only timestamp must NOT stand alone (too coarse to be unique).
    am = {"docId": "DOC-M", "fingerprint": transaction_fingerprint("06-03-2025 11:43")}
    assert am["fingerprint"]["datetime"] == "202503061143"
    assert document_matches(am, fingerprint=transaction_fingerprint("06-03-2025 11:43")) == ""

    # A different transaction (one second apart) must not match.
    assert document_matches(a, fingerprint=transaction_fingerprint("06-03-2025 11:43:59")) == ""


# A cash receipt: no RACHUNEK NR / KOD AUTORYZACJI / card / HH:MM:SS, so its transaction
# fingerprint is empty. Two phone re-scans drift in OCR words but share the monetary tokens.
_CASH_META = {"type": "paragon", "contractor": "CYFRONIKA", "date": "2026-06-24",
              "amount": "54.61", "currency": "PLN"}
_CASH_TEXT_A = "\n".join([
    "CYFRONIKA Zaklad Elektroniki", "30-385 Krakow ul.Sasiedzka 43", "PARAGON FISKALNY",
    "Plytka uniwersalna 4.68", "Plytka uniwersalna 6.59", "Plytka uniwersalna 7.32",
    "Plytka uniwersalna 12.50", "Sprzedaz A 54.61", "SUMA PLN 54.61",
])
_CASH_TEXT_B = "\n".join([  # same receipt, different OCR noise, same amounts
    "CYFRONIKA Zaktad Elektronlki", "30-385 Krakow ul.Sasledzka 43", "PARAGON F ISKALNY",
    "Plytka uniuersalna 4.68", "Frytka uniwersaina 6.59", "Plytka uniwersalna 7.32",
    "Plytka uniwersalna 12.50", "Sprzedaz opodatkowana A 54.61", "DO ZAPLATY 54.61 Gotowka",
])


def _cash_doc(doc_id: str, text: str, meta: dict, *, dhash: str = "", phash: str = "") -> dict:
    return {"docId": doc_id, "fingerprint": transaction_fingerprint(text),
            "dhash": dhash, "phash": phash, "metadata": meta, "text": text}


def test_business_key_helper_and_money_overlap() -> None:
    assert business_key(_CASH_META) == ("cyfronika", "2026-06-24", "54.61", "pln")
    # Top-level fields (archive records) are read too.
    assert business_key({"contractor": "X", "date": "2026-06-24", "amount": "9.99"}) == ("x", "2026-06-24", "9.99", "")
    # Missing/placeholder merchant or total -> no usable key.
    assert business_key({"contractor": "", "date": "d", "amount": "1"}) is None
    assert business_key({"contractor": "x", "date": "d", "amount": "nieznana"}) is None

    shared, jaccard = money_overlap(_CASH_TEXT_A, _CASH_TEXT_B)
    assert shared == 5 and jaccard == 1.0


def test_business_key_collapses_cash_receipt_rescans() -> None:
    """Empty fingerprints + far-apart visual hashes: only the corroborated business key
    can collapse two scans of the same cash receipt (the real CYFRONIKA duplicate)."""
    archived = [_cash_doc("DOC-1", _CASH_TEXT_A, _CASH_META, dhash="1111111111111111", phash="2222222222222222")]
    candidate = {"docId": "DOC-2", "fingerprint": transaction_fingerprint(_CASH_TEXT_B),
                 "dhash": "eeeeeeeeeeeeeeee", "phash": "dddddddddddddddd",
                 "metadata": _CASH_META, "text": _CASH_TEXT_B}

    # Guard the premise: no fingerprint agreement and the images are far apart.
    assert fingerprint_match_count(candidate["fingerprint"], archived[0]["fingerprint"]) == 0
    assert dhash_distance(candidate["dhash"], archived[0]["dhash"]) > 10

    result = evaluate(candidate, archived)
    assert result["action"] in {"duplicate", "supersede"}
    assert result["reason"] == "business-key"


def test_business_key_collapses_when_visually_near_even_with_low_text_overlap() -> None:
    """Business key + near-identical images is enough even when OCR text barely overlaps."""
    a_text = "PARAGON\nfoo 1.00\nbar 2.00\nSUMA 54.61"
    b_text = "PARAGON\nqux 9.99\nbaz 8.88\nSUMA 54.61"  # shares only the total -> weak money overlap
    archived = [_cash_doc("DOC-1", a_text, _CASH_META, dhash="949cacece898bcdc")]
    candidate = {"docId": "DOC-2", "fingerprint": transaction_fingerprint(b_text),
                 "dhash": "949cdcecec98bcdc", "metadata": _CASH_META, "text": b_text}

    assert money_overlap(a_text, b_text)[0] < 3  # money path would NOT fire
    assert evaluate(candidate, archived)["reason"] == "business-key"


def test_business_key_collapses_real_orlen_pair_just_past_strict_visual() -> None:
    """The real ORLEN duplicate: identical merchant/date/total, no usable text on one side,
    dHash 11 apart -- one bit past the strict near threshold but within the looser
    business-key visual bound, so the exact key + rough visual match collapses them."""
    meta = {"type": "faktura", "contractor": "ORLEN S.A.", "date": "2026-03-16",
            "amount": "150.05", "currency": "PLN"}
    archived = [{"docId": "DOC-FV-1", "fingerprint": transaction_fingerprint("garbled"),
                 "dhash": "61616151233b0178", "metadata": meta, "text": ""}]
    candidate = {"docId": "DOC-FV-2", "fingerprint": transaction_fingerprint("garbled"),
                 "dhash": "416060502a0a4078", "metadata": meta, "text": ""}

    assert 10 < dhash_distance(candidate["dhash"], archived[0]["dhash"]) <= 14
    assert evaluate(candidate, archived)["reason"] == "business-key"


def test_business_key_keeps_distinct_receipts_sharing_only_total() -> None:
    """Two different receipts, same merchant/date/total but different line items and
    different framing, must stay separate (no false merge)."""
    a_text = "PARAGON\nPozycja 3.00\nPozycja 51.61\nSUMA 54.61"
    b_text = "PARAGON\nPozycja 20.00\nPozycja 34.61\nSUMA 54.61"
    archived = [_cash_doc("DOC-1", a_text, _CASH_META)]  # no visual hashes
    candidate = {"docId": "DOC-2", "fingerprint": transaction_fingerprint(b_text),
                 "dhash": "", "phash": "", "metadata": _CASH_META, "text": b_text}

    assert evaluate(candidate, archived)["action"] == "new"


def test_business_key_supersede_uses_top_level_fields() -> None:
    """A more complete re-scan supersedes the archived one even when both carry their
    fields at the top level (no nested ``metadata``) -- the symmetric-completeness fix."""
    poor = {"docId": "DOC-1", "fingerprint": transaction_fingerprint(_CASH_TEXT_A),
            "dhash": "1111111111111111", "text": _CASH_TEXT_A,
            "contractor": "CYFRONIKA", "date": "2026-06-24", "amount": "54.61", "currency": "PLN", "type": "dokument"}
    rich = {"docId": "DOC-2", "fingerprint": transaction_fingerprint(_CASH_TEXT_B),
            "dhash": "eeeeeeeeeeeeeeee", "text": _CASH_TEXT_B,
            "contractor": "CYFRONIKA", "date": "2026-06-24", "amount": "54.61", "currency": "PLN", "type": "paragon"}

    result = evaluate(rich, [poor])
    assert result["reason"] == "business-key"
    assert result["action"] == "supersede"


def test_reconcile_groups_business_key_cash_duplicates() -> None:
    docs = [
        _cash_doc("DOC-1", _CASH_TEXT_A, _CASH_META, dhash="1111111111111111"),
        _cash_doc("DOC-2", _CASH_TEXT_B, _CASH_META, dhash="eeeeeeeeeeeeeeee"),
    ]
    groups = reconcile(docs)
    assert len(groups) == 1
    assert {groups[0]["keep"], *groups[0]["drop"]} == {"DOC-1", "DOC-2"}
    assert groups[0]["reason"] == "business-key"
