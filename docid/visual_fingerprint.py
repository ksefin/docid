"""
Wizualny odcisk dokumentu (perceptual fingerprint).

Cel: rozpoznać, że dwa skany to *ten sam fizyczny dokument*, ZANIM uruchomimy
OCR. OCR bywa zawodny i ten sam paragon bywa odczytany różnie (inna data, inna
kwota, inny kontrahent) -> deterministyczne ID liczone z pól OCR rozjeżdża się
i powstają duplikaty. Odcisk graficzny jest odporny na szum OCR, bo liczony jest
z samego obrazu.

Kluczowa różnica względem ``document_id_universal._calculate_visual_hash``:
ten moduł NIE haszuje kryptograficznie bitów percepcyjnych. Hash kryptograficzny
(SHA-256) niszczy "zgrubne" podobieństwo - zmiana jednego piksela daje zupełnie
inny wynik. Tutaj przechowujemy surowe bity jako hex i porównujemy je
odległością Hamminga, dzięki czemu "prawie identyczne" obrazy mają "prawie
identyczny" odcisk.

Trzy uzupełniające się odciski 64-bitowe:
  * aHash  - średnia jasność (gruba struktura, układ jasnych/ciemnych stref)
  * dHash  - gradient poziomy (krawędzie, układ tekstu) - bardzo odporny na
             jasność/ekspozycję
  * pHash  - DCT niskich częstotliwości (najbardziej odporny percepcyjnie)

Dodatkowo liczymy ``quality_score`` (ostrość, kontrast, rozdzielczość), żeby
przy dwóch skanach tego samego dokumentu móc wybrać ten lepszej jakości.

Wymaga Pillow. ``pHash`` korzysta z numpy jeśli jest dostępne, w przeciwnym
razie używa czystego Pythona (wolniej, ale działa).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMPY_AVAILABLE = False


# Liczba bitów każdego odcisku (8x8).
HASH_BITS = 64

# Domyślne progi (konserwatywne). Odległość Hamminga <= próg => "to samo".
# 64-bitowy odcisk: 6/64 ~= 90% podobieństwa.
DEFAULT_MAX_DISTANCE = 6


def _bits_to_hex(bits: List[int]) -> str:
    """Zamienia listę bitów (0/1) na hex o stałej długości."""
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    width = (len(bits) + 3) // 4
    return format(value, "0{}x".format(width))


def hamming_distance(hex_a: Optional[str], hex_b: Optional[str]) -> Optional[int]:
    """Odległość Hamminga między dwoma odciskami hex (liczba różnych bitów).

    Zwraca ``None`` jeśli któryś odcisk jest pusty albo mają różne długości
    (nieporównywalne).

    >>> hamming_distance("ff00", "ff01")
    1
    >>> hamming_distance("0000", "ffff")
    16
    """
    if not hex_a or not hex_b:
        return None
    if len(hex_a) != len(hex_b):
        return None
    try:
        return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")
    except ValueError:
        return None


def similarity(hex_a: Optional[str], hex_b: Optional[str]) -> Optional[float]:
    """Podobieństwo 0..1 (1 = identyczne). ``None`` jeśli nieporównywalne.

    >>> similarity("ffff", "ffff")
    1.0
    >>> similarity("0000", "ffff")
    0.0
    """
    dist = hamming_distance(hex_a, hex_b)
    if dist is None:
        return None
    bits = len(hex_a) * 4
    if bits == 0:
        return None
    return 1.0 - (dist / bits)


@dataclass
class VisualFingerprint:
    """Wizualny odcisk dokumentu + metryki jakości obrazu."""

    ahash: str
    dhash: str
    phash: str
    width: int
    height: int
    quality_score: float  # 0..1, wyższy = lepszy skan
    sharpness: float = 0.0
    contrast: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ahash": self.ahash,
            "dhash": self.dhash,
            "phash": self.phash,
            "width": self.width,
            "height": self.height,
            "qualityScore": round(self.quality_score, 4),
            "sharpness": round(self.sharpness, 2),
            "contrast": round(self.contrast, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualFingerprint":
        return cls(
            ahash=str(data.get("ahash") or ""),
            dhash=str(data.get("dhash") or ""),
            phash=str(data.get("phash") or ""),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            quality_score=float(data.get("qualityScore") or 0.0),
            sharpness=float(data.get("sharpness") or 0.0),
            contrast=float(data.get("contrast") or 0.0),
        )

    def distance(self, other: "VisualFingerprint") -> Optional[int]:
        """Łączna (najlepsza dostępna) odległość względem innego odcisku.

        Bierzemy minimum z dHash i pHash - dwa najbardziej percepcyjnie odporne
        kanały. aHash jest tylko pomocniczy.
        """
        candidates = [
            hamming_distance(self.dhash, other.dhash),
            hamming_distance(self.phash, other.phash),
        ]
        vals = [c for c in candidates if c is not None]
        return min(vals) if vals else None

    def matches(self, other: "VisualFingerprint", max_distance: int = DEFAULT_MAX_DISTANCE) -> bool:
        """Czy to ten sam dokument?

        Konserwatywnie: wymagamy aby ZARÓWNO dHash JAK I pHash były blisko
        (oba <= max_distance). Dwa różne paragony z tej samej kasy mają podobny
        szablon (dHash), ale różne pHash - wymóg obu kanałów chroni przed
        fałszywym scaleniem.
        """
        d_dhash = hamming_distance(self.dhash, other.dhash)
        d_phash = hamming_distance(self.phash, other.phash)
        checks = [d for d in (d_dhash, d_phash) if d is not None]
        if not checks:
            return False
        return all(d <= max_distance for d in checks)


def _load_grayscale(source: Union[str, Path, "Image.Image"]) -> "Image.Image":
    """Wczytuje obraz i zwraca go w skali szarości (z autokontrastem)."""
    if hasattr(source, "convert"):  # już obraz PIL
        img = source
    else:
        img = Image.open(str(source))
    if getattr(img, "mode", "L") != "L":
        img = img.convert("L")
    # Normalizacja ekspozycji - skany telefonem mają różną jasność.
    img = ImageOps.autocontrast(img, cutoff=2)
    return img


def _ahash(gray: "Image.Image") -> str:
    small = gray.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    avg = sum(pixels) / len(pixels)
    return _bits_to_hex([1 if p >= avg else 0 for p in pixels])


def _dhash(gray: "Image.Image") -> str:
    # 9x8 -> porównanie sąsiadów w poziomie -> 8x8 bitów
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    bits: List[int] = []
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits.append(1 if left > right else 0)
    return _bits_to_hex(bits)


def _phash(gray: "Image.Image") -> str:
    """pHash przez DCT niskich częstotliwości (32x32 -> górne 8x8)."""
    size = 32
    small = gray.resize((size, size), Image.Resampling.LANCZOS)

    if NUMPY_AVAILABLE:
        pix = np.asarray(small, dtype=np.float64)
        dct = _dct2(pix)
        block = dct[:8, :8]
        # pomijamy składową DC (0,0) przy liczeniu mediany
        flat = block.flatten()
        med = np.median(flat[1:])
        bits = [1 if v > med else 0 for v in flat]
        return _bits_to_hex(bits)

    # Fallback czysto-Pythonowy (wolny, ale poprawny)
    pixels = list(small.tobytes())
    matrix = [pixels[r * size:(r + 1) * size] for r in range(size)]
    dct = _dct2_py(matrix, size)
    block = [dct[r][c] for r in range(8) for c in range(8)]
    rest = sorted(block[1:])
    med = rest[len(rest) // 2]
    return _bits_to_hex([1 if v > med else 0 for v in block])


def _dct2(matrix: "np.ndarray") -> "np.ndarray":
    """Dwuwymiarowe DCT-II przez macierze bazowe (bez scipy)."""
    n = matrix.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    return basis @ matrix @ basis.T


def _dct2_py(matrix: List[List[float]], n: int) -> List[List[float]]:  # pragma: no cover
    import math
    cos = [[math.cos(math.pi * (2 * x + 1) * u / (2 * n)) for x in range(n)] for u in range(n)]
    # wiersze
    tmp = [[sum(matrix[y][x] * cos[u][x] for x in range(n)) for u in range(n)] for y in range(n)]
    # kolumny
    out = [[sum(tmp[y][u] * cos[v][y] for y in range(n)) for u in range(n)] for v in range(n)]
    return out


def _quality_metrics(gray: "Image.Image") -> Tuple[float, float, float]:
    """Zwraca (quality_score 0..1, sharpness, contrast).

    * sharpness - energia gradientu (im ostrzej, tym wyższa). Rozmyte/poruszone
      zdjęcia mają niski gradient.
    * contrast  - odchylenie standardowe jasności.
    * rozdzielczość wchodzi do score osobno (większy skan = więcej detali).
    """
    width, height = gray.size
    work = gray
    # Ujednolicamy skalę liczenia ostrości, by nie zależała od rozdzielczości.
    target = 256
    if max(width, height) > target:
        ratio = target / max(width, height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        work = gray.resize(new_size, Image.Resampling.LANCZOS)

    if NUMPY_AVAILABLE:
        arr = np.asarray(work, dtype=np.float64)
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)
        sharpness = float(gx.var() + gy.var())
        contrast = float(arr.std())
    else:  # pragma: no cover
        pixels = list(work.tobytes())
        w, h = work.size
        mean = sum(pixels) / len(pixels)
        contrast = (sum((p - mean) ** 2 for p in pixels) / len(pixels)) ** 0.5
        diffs = []
        for y in range(h):
            row = pixels[y * w:(y + 1) * w]
            diffs.extend((row[i + 1] - row[i]) for i in range(w - 1))
        gmean = sum(diffs) / len(diffs) if diffs else 0
        sharpness = sum((d - gmean) ** 2 for d in diffs) / len(diffs) if diffs else 0.0

    # Normalizacja do 0..1 (progi dobrane empirycznie dla skanów telefonem).
    sharp_n = min(1.0, sharpness / 500.0)
    contrast_n = min(1.0, contrast / 80.0)
    res_n = min(1.0, (width * height) / (1200.0 * 1600.0))
    quality = 0.5 * sharp_n + 0.3 * contrast_n + 0.2 * res_n
    return quality, sharpness, contrast


def compute_fingerprint(source: Union[str, Path, "Image.Image"]) -> Optional[VisualFingerprint]:
    """Liczy wizualny odcisk dla obrazu (ścieżka lub obiekt PIL).

    Zwraca ``None`` jeśli Pillow nie jest dostępne lub obraz nie da się wczytać.
    """
    if not PIL_AVAILABLE:
        return None
    try:
        if hasattr(source, "convert"):
            base = source
            width, height = base.size
        else:
            base = Image.open(str(source))
            base.load()
            width, height = base.size
        gray = _load_grayscale(base)
        quality, sharpness, contrast = _quality_metrics(gray)
        return VisualFingerprint(
            ahash=_ahash(gray),
            dhash=_dhash(gray),
            phash=_phash(gray),
            width=width,
            height=height,
            quality_score=quality,
            sharpness=sharpness,
            contrast=contrast,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Visual fingerprint failed for %s: %s", source, exc)
        return None


@dataclass
class VisualMatch:
    """Wynik wyszukiwania podobnego dokumentu."""

    key: Any
    distance: int
    similarity: float
    record: Any = None


def find_best_match(
    fingerprint: VisualFingerprint,
    candidates: List[Tuple[Any, VisualFingerprint]],
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> Optional[VisualMatch]:
    """Znajduje najbliższy pasujący odcisk wśród kandydatów.

    Args:
        fingerprint: odcisk nowego skanu.
        candidates: lista (klucz, odcisk) już zarchiwizowanych dokumentów.
        max_distance: maksymalna dopuszczalna odległość Hamminga.

    Returns:
        ``VisualMatch`` z najmniejszą odległością spełniającą próg, albo ``None``.
    """
    best: Optional[VisualMatch] = None
    for key, cand in candidates:
        if cand is None:
            continue
        if not fingerprint.matches(cand, max_distance=max_distance):
            continue
        dist = fingerprint.distance(cand)
        if dist is None:
            continue
        sim = 1.0 - (dist / HASH_BITS)
        if best is None or dist < best.distance:
            best = VisualMatch(key=key, distance=dist, similarity=sim, record=cand)
    return best


# ---------------------------------------------------------------------------
# Łączenie pól z wielu skanów tego samego dokumentu (field fusion).
# ---------------------------------------------------------------------------

@dataclass
class FieldSource:
    """Pojedyncze źródło danych do scalenia (jeden skan)."""

    fields: Dict[str, Any]
    weight: float = 1.0  # waga zaufania (np. jakość obrazu * liczba znaków OCR)
    label: str = ""


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    # Markery "nie rozpoznano" używane przez scanner.
    return text.lower() in {"kwota-nieznana", "nieznana", "unknown", "n/a", "-"}


def merge_records(
    sources: List[FieldSource],
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Buduje jeden spójny rekord z wielu skanów tego samego dokumentu.

    Dla każdego pola wybiera wartość przez ważone głosowanie:
      * puste/"nieznane" wartości są ignorowane (luka w jednym skanie wypełniana
        z drugiego),
      * spośród niepustych wygrywa wartość o największej sumie wag (konsensus);
        remis rozstrzyga najwyższa pojedyncza waga (najlepszej jakości skan).

    Zwraca słownik:
      {
        "fields":     {pole: wybrana_wartość, ...},
        "provenance": {pole: {"value", "weight", "support", "candidates"}},
        "filledGaps": [pola uzupełnione z innego skanu niż dominujący],
      }
    """
    if not sources:
        return {"fields": {}, "provenance": {}, "filledGaps": []}

    if fields is None:
        keys: List[str] = []
        for src in sources:
            for key in src.fields.keys():
                if key not in keys:
                    keys.append(key)
        fields = keys

    merged: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    filled_gaps: List[str] = []

    # Skan o najwyższej wadze - "dominujący" punkt odniesienia.
    dominant = max(range(len(sources)), key=lambda i: sources[i].weight)

    for fieldname in fields:
        # Zbierz głosy: wartość -> [suma_wag, max_waga, liczność]
        votes: Dict[str, List[float]] = {}
        raw_by_norm: Dict[str, Any] = {}
        for src in sources:
            value = src.fields.get(fieldname)
            if _is_empty(value):
                continue
            norm = str(value).strip().upper()
            entry = votes.setdefault(norm, [0.0, 0.0, 0.0])
            entry[0] += max(src.weight, 0.0001)
            entry[1] = max(entry[1], src.weight)
            entry[2] += 1
            raw_by_norm.setdefault(norm, value)

        if not votes:
            merged[fieldname] = sources[dominant].fields.get(fieldname)
            provenance[fieldname] = {
                "value": merged[fieldname], "weight": 0.0, "support": 0, "candidates": 0,
            }
            continue

        # Najlepszy: suma wag, potem max waga.
        best_norm = max(votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        merged[fieldname] = raw_by_norm[best_norm]
        provenance[fieldname] = {
            "value": raw_by_norm[best_norm],
            "weight": round(votes[best_norm][0], 4),
            "support": int(votes[best_norm][2]),
            "candidates": len(votes),
        }

        # Czy uzupełniliśmy lukę / poprawiliśmy dominujący skan?
        dom_value = sources[dominant].fields.get(fieldname)
        if _is_empty(dom_value) or str(dom_value).strip().upper() != best_norm:
            filled_gaps.append(fieldname)

    return {"fields": merged, "provenance": provenance, "filledGaps": filled_gaps}
