from __future__ import annotations

from pathlib import Path

import pytest

from docid.dedup import (
    dhash_distance,
    document_matches,
    document_signature,
    evaluate,
    find_duplicate,
    fingerprint_match_count,
    image_dhash,
    image_phash,
    metadata_completeness,
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

    assert fp_good == {"number": "181149", "auth": "784683", "time": "095251", "card": "1671"}
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
    assert dup["reason"].startswith("fingerprint")

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
