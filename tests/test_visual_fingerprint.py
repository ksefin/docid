"""Testy wizualnego odcisku dokumentu i scalania pól."""

import pytest

from docid.visual_fingerprint import (
    FieldSource,
    VisualFingerprint,
    compute_fingerprint,
    find_best_match,
    hamming_distance,
    merge_records,
    similarity,
)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _doc_image(seed: int, size=(420, 600), noise=0):
    """Generuje deterministyczny obraz przypominający dokument."""
    img = Image.new("L", size, color=245)
    draw = ImageDraw.Draw(img)
    rng = seed
    # nagłówek
    draw.rectangle([30, 20, size[0] - 30, 70], fill=40 + (seed * 7) % 60)
    # "linie tekstu"
    y = 100
    for i in range(18):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        width = 120 + (rng % (size[0] - 180))
        shade = 30 + ((rng >> 5) % 40)
        draw.rectangle([40, y, 40 + width, y + 12], fill=shade)
        y += 26
    if noise:
        # delikatny szum symulujący inny skan tego samego dokumentu
        px = img.load()
        for n in range(noise):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            x = rng % size[0]
            yy = (rng >> 8) % size[1]
            px[x, yy] = (px[x, yy] + 25) % 256
    return img


def test_hamming_and_similarity():
    assert hamming_distance("ff00", "ff01") == 1
    assert hamming_distance("0000", "ffff") == 16
    assert hamming_distance("abc", None) is None
    assert hamming_distance("ab", "abcd") is None  # różne długości
    assert similarity("ffff", "ffff") == 1.0
    assert similarity("0000", "ffff") == 0.0


def test_fingerprint_stable_for_same_image():
    img = _doc_image(1)
    fp1 = compute_fingerprint(img)
    fp2 = compute_fingerprint(img.copy())
    assert fp1 is not None and fp2 is not None
    assert fp1.dhash == fp2.dhash
    assert fp1.phash == fp2.phash
    assert fp1.distance(fp2) == 0


def test_same_document_with_noise_matches():
    base = _doc_image(7)
    rescan = _doc_image(7, noise=400)  # ten sam dokument, inny "skan"
    fp_a = compute_fingerprint(base)
    fp_b = compute_fingerprint(rescan)
    assert fp_a.matches(fp_b, max_distance=6)


def test_different_documents_do_not_match():
    fp_a = compute_fingerprint(_doc_image(11))
    fp_b = compute_fingerprint(_doc_image(99))
    assert not fp_a.matches(fp_b, max_distance=6)


def test_find_best_match_picks_nearest():
    target = compute_fingerprint(_doc_image(3))
    candidates = [
        ("other-1", compute_fingerprint(_doc_image(50))),
        ("same", compute_fingerprint(_doc_image(3, noise=300))),
        ("other-2", compute_fingerprint(_doc_image(77))),
    ]
    match = find_best_match(target, candidates, max_distance=6)
    assert match is not None
    assert match.key == "same"


def test_find_best_match_returns_none_when_no_match():
    target = compute_fingerprint(_doc_image(3))
    candidates = [("x", compute_fingerprint(_doc_image(500)))]
    assert find_best_match(target, candidates, max_distance=6) is None


def test_serialization_roundtrip():
    fp = compute_fingerprint(_doc_image(5))
    restored = VisualFingerprint.from_dict(fp.to_dict())
    assert restored.dhash == fp.dhash
    assert restored.phash == fp.phash
    assert restored.distance(fp) == 0


# --- field fusion ---------------------------------------------------------

def test_merge_fills_gaps_from_other_scan():
    sources = [
        FieldSource(fields={"date": "2026-06-19", "amount": "", "contractor": "DUO CAFE"}, weight=0.8),
        FieldSource(fields={"date": "", "amount": "30.26", "contractor": "DUO CAFE"}, weight=0.6),
    ]
    result = merge_records(sources)
    assert result["fields"]["date"] == "2026-06-19"
    assert result["fields"]["amount"] == "30.26"   # luka uzupełniona z drugiego skanu
    assert result["fields"]["contractor"] == "DUO CAFE"
    assert "amount" in result["filledGaps"]


def test_merge_consensus_majority_wins():
    sources = [
        FieldSource(fields={"amount": "30.26"}, weight=0.5),
        FieldSource(fields={"amount": "30.26"}, weight=0.5),
        FieldSource(fields={"amount": "38.26"}, weight=0.6),  # odczyt OCR-owy, ale przegłosowany
    ]
    result = merge_records(sources)
    assert result["fields"]["amount"] == "30.26"
    assert result["provenance"]["amount"]["support"] == 2


def test_merge_ignores_unknown_markers():
    sources = [
        FieldSource(fields={"amount": "kwota-nieznana"}, weight=0.9),
        FieldSource(fields={"amount": "12.00"}, weight=0.3),
    ]
    result = merge_records(sources)
    assert result["fields"]["amount"] == "12.00"


def test_merge_empty_sources():
    assert merge_records([])["fields"] == {}
