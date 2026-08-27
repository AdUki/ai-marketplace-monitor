#!/usr/bin/env python3
"""Bulk benchmark database, downloaded once and looked up offline.

PassMark publishes complete, current lists as plain HTML tables:

    cpu_list.php      ~2,800 desktop/laptop CPUs   (CPU Mark)
    gpu_list.php      ~2,800 GPUs                  (G3D Mark)
    device_list.php   ~5,200 Android phones        (PassMark device score)

That is the whole population -- CPUs and phone models are finite and
already published, so there is no reason to ask a language model about
them one at a time. One 300KB download answers every lookup instantly,
offline, for free, and deterministically.

Refresh with:  ai-marketplace-monitor --update-benchmarks
"""
from __future__ import annotations

import difflib
import html
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .utils import amm_home as CACHE_DIR
except ImportError:  # running standalone
    CACHE_DIR = Path.home() / ".ai-marketplace-monitor"

DB_PATH = CACHE_DIR / "benchmarks.json"
MAX_AGE = 30 * 24 * 3600  # PassMark updates continuously; monthly is plenty
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

SOURCES = {
    "cpu": ("https://www.cpubenchmark.net/cpu_list.php",
            r'<tr id="cpu\d+"><td><a href="[^"]*">([^<]+)</a></td><td>([\d,]+)</td>'),
    "gpu": ("https://www.videocardbenchmark.net/gpu_list.php",
            r'<TR id="gpu\d+"><TD><A HREF="[^"]*">([^<]+)</A></TD><TD>([\d,]+)</TD>'),
    "device": ("https://www.androidbenchmark.net/device_list.php",
               r'<tr><td><a href="/phone\.php\?phone=[^"]*">([^<]+)</a></td>'
               r'<td><a href="passmark_lookup\.php[^"]*">([\d,]+)</a></td>'),
}


def _fetch(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def download(verbose: bool = True) -> Dict[str, Dict[str, int]]:
    """Fetch and parse every list. ~300KB total."""
    tables: Dict[str, Dict[str, int]] = {}
    for kind, (url, pattern) in SOURCES.items():
        try:
            page = _fetch(url)
        except Exception as e:  # noqa: BLE001 - keep whatever else succeeded
            if verbose:
                print(f"  {kind}: download failed ({e})")
            tables[kind] = {}
            continue
        rows = {
            html.unescape(n).strip(): int(v.replace(",", ""))
            for n, v in re.findall(pattern, page, re.I)
        }
        tables[kind] = rows
        if verbose:
            print(f"  {kind}: {len(rows)} entries")
    return tables


def save(tables: Dict[str, Dict[str, int]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fetched_at": time.time(), "tables": tables}))
    tmp.replace(DB_PATH)


def load(auto_download: bool = True, verbose: bool = False) -> Dict[str, Dict[str, int]]:
    try:
        blob = json.loads(DB_PATH.read_text())
        if blob.get("fetched_at", 0) + MAX_AGE > time.time():
            return blob.get("tables", {})
    except (OSError, ValueError):
        blob = None
    if not auto_download:
        return (blob or {}).get("tables", {})
    tables = download(verbose=verbose)
    if any(tables.values()):
        save(tables)
        return tables
    return (blob or {}).get("tables", {})


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Lowercase, fold accents, reduce to alphanumerics.

    Folding matters: stripping non-ASCII turns "Predám" into "pred m", two
    tokens that match nothing and drag every similarity score down.
    """
    folded = unicodedata.normalize("NFKD", s.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


# Capacity ("128GB") is not part of a model name; a model number ("S21",
# "Note 12") is, and confusing the two is how "Redmi Note 12 Pro" matched
# the entirely different "Redmi Pro".
_CAPACITY = re.compile(r"^\d+(gb|tb|mb)$")
_VENDOR_FILLER = {
    "intel", "amd", "apple", "core", "processor", "cpu", "gpu", "with",
    "graphics", "mobile", "series", "nvidia", "geforce", "radeon",
}


# Sales words and colours that appear in listings but never in a model name.
_LISTING_NOISE = {
    "predam", "predavam", "novy", "nova", "nove", "uplne", "super", "stav",
    "zanovny", "top", "lacno", "mobil", "telefon", "smartfon", "cierny",
    "biely", "modry", "zeleny", "cierna", "biela", "black", "white", "blue",
    "green", "grey", "gray", "silver", "gold", "red", "pink", "purple",
    "gb", "tb", "ram", "ssd", "dual", "sim", "5g", "4g", "lte",
}


def _tokens(s: str) -> list:
    # Sellers write the configuration as "8/256", "8 GB / 256 GB" or
    # "6GB/128GB". Those digits are RAM and storage, not model numbers, and
    # reading them as identity made "Samsung A54 8/256" fail to match the
    # plain "Samsung Galaxy A54". Drop the pair outright.
    s = re.sub(r"\b\d{1,4}\s*(?:gb|tb)?\s*/\s*\d{1,4}\s*(?:gb|tb)?\b", " ", s, flags=re.I)
    # "128 GB" is likewise one fact; join it so it reads as capacity.
    s = re.sub(r"(\d+)\s*(gb|tb|mb)\b", r"\1\2", s, flags=re.I)
    return _norm(s).split()


def _query_model_tokens(tokens) -> list:
    return [t for t in tokens if t not in _LISTING_NOISE and not _CAPACITY.match(t)]


def _entry_tokens(name: str) -> list:
    """Tokens of a table entry, minus the bits a listing never repeats.

    PassMark qualifies entries in ways sellers do not: a clock suffix
    ("@ 1.70GHz") and a variant in brackets ("(Exynos)"). Requiring those
    would reject the correct row for almost every real listing.
    """
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"@.*$", " ", name)
    return _tokens(name)


def _model_numbers(tokens) -> set:
    """Tokens carrying a digit that are not a storage size."""
    return {t for t in tokens if any(c.isdigit() for c in t) and not _CAPACITY.match(t)}


# Words a seller may or may not type. Their absence must not sink a match
# ("Samsung S21" is the same phone as "Samsung Galaxy S21"), and their
# presence must not carry one either.
_FILLER = {
    "galaxy", "phone", "mobile", "smartphone", "5g", "4g", "lte", "dual",
    "sim", "various", "models", "edition", "version",
}


# Words that name a DIFFERENT product in the same line. If one side has one
# and the other does not, they are not the same device -- an S21 Ultra is
# not an S21, and a Note is not a plain model.
_VARIANTS = {
    "ultra", "plus", "pro", "max", "mini", "lite", "fe", "neo", "prime",
    "note", "edge", "fold", "flip", "air", "se", "xl",
}


def _signature(tokens) -> set:
    return {t for t in tokens if t not in _FILLER and not _CAPACITY.match(t)}


def match_device(query: str, table: Dict[str, int]) -> Optional[Tuple[str, int]]:
    """Fuzzy-match a product name, tolerating how sellers actually write them.

    Strict subset matching was too brittle: it needed every word of the
    table entry to appear in the listing, so "Samsung S21 128GB" missed
    "Samsung Galaxy S21" over the dropped word "Galaxy".

    Instead:
      * model numbers must agree EXACTLY -- "s21" is not "s22", and an
        "s21 ultra" is not an "s21". This is the discriminating signal and
        is never fuzzy-matched.
      * remaining words are compared as sets, with filler ignored, and
        scored by how much each side leaves unexplained.
      * close spellings still count, via difflib, so "thinkpad"/"think pad"
        or a stray plural does not break a match.
    """
    q = _tokens(query)
    q_sig = _signature(_query_model_tokens(q))
    q_nums = _model_numbers(q_sig)
    if not q_sig:
        return None

    best, best_score = None, 0.0
    for name, score in table.items():
        e_sig = _signature(_entry_tokens(name))
        if not e_sig:
            continue
        # Model numbers are the identity of the device: require exact agreement.
        if _model_numbers(e_sig) != q_nums:
            continue
        # A variant word on one side only means a different product.
        if (e_sig & _VARIANTS) != (q_sig & _VARIANTS):
            continue
        missing = e_sig - q_sig      # entry claims words the listing lacks
        extra = q_sig - e_sig        # listing has words the entry lacks
        # Allow a near-spelling to count as present before penalising it.
        missing = {m for m in missing
                   if not _close_to_any(m, q_sig)}
        extra = {x for x in extra if not _close_to_any(x, e_sig)}
        denom = max(len(e_sig | q_sig), 1)
        sim = 1.0 - (len(missing) + len(extra)) / denom
        if sim > best_score:
            best, best_score = (name, score), sim
    return best if best_score >= 0.6 else None


def _close_to_any(word: str, others, threshold: float = 0.86) -> bool:
    """True if `word` is a near-spelling of any token in `others`."""
    return any(difflib.SequenceMatcher(None, word, o).ratio() >= threshold
               for o in others)


def match_component(query: str, table: Dict[str, int]) -> Optional[Tuple[str, int]]:
    """Match a part named inside a longer listing, e.g. a CPU in a laptop ad.

    The listing carries model numbers the part name does not (the laptop's
    own), so only one direction can be required: the part's distinctive
    words must appear in the listing.
    """
    qset = set(_tokens(query))
    best, best_len = None, 0
    for name, score in table.items():
        et = [t for t in _entry_tokens(name) if t not in _VENDOR_FILLER]
        if not et or len(et) <= best_len:
            continue
        if set(et) <= qset:
            best, best_len = (name, score), len(et)
    return best


def lookup_device(title: str) -> Optional[Tuple[str, int]]:
    """PassMark score for an Android phone, by listing title."""
    return match_device(title, load(auto_download=False).get("device", {}))


def lookup_cpu(text: str) -> Optional[Tuple[str, int]]:
    return match_component(text, load(auto_download=False).get("cpu", {}))


def lookup_gpu(text: str) -> Optional[Tuple[str, int]]:
    return match_component(text, load(auto_download=False).get("gpu", {}))


def main() -> int:
    print("Downloading benchmark tables from PassMark...")
    tables = download()
    if not any(tables.values()):
        print("Nothing downloaded; keeping any existing database.")
        return 1
    save(tables)
    total = sum(len(t) for t in tables.values())
    print(f"Saved {total} entries to {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
