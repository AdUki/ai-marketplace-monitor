#!/usr/bin/env python3
"""Look up hard facts about a phone model once, then cache them forever.

Why this exists
---------------
The LLM was guessing specs and getting them wrong (it rejected a Galaxy S21
for "failing the 600k AnTuTu floor" and for lacking "custom ROM
significance" -- both false). Asking the model to web-search on every
listing fixes accuracy but is expensive: search results are token-heavy and
the same handful of phone models repeat across hundreds of listings.

So: resolve facts per MODEL, cache them on disk, and let the per-listing
evaluation stay on a cheap model that is handed the facts rather than
recalling them.

Sources
-------
LineageOS support : the project's own build API. NOTE this covers OFFICIAL
                    builds only. Verified against the full wiki archive (735
                    device files, discontinued ones included): a device with
                    only unofficial/XDA builds -- e.g. the Galaxy S21 -- does
                    not appear at all. "not official" therefore does NOT mean
                    "cannot run LineageOS".
AnTuTu / RAM      : Groq's agentic `groq/compound` model, which performs real
                    web searches server-side. Spec sites (kimovil, nanoreview)
                    return HTTP 403 to scripts, so scraping them directly is
                    not an option.

Usage
-----
    ./device_facts.py "Samsung Galaxy S21 5G"
    ./device_facts.py --refresh "Xiaomi Redmi Note 12"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # inside the package
    from .utils import amm_home as CACHE_DIR
except ImportError:  # running the file directly
    CACHE_DIR = Path.home() / ".ai-marketplace-monitor"
FACTS_CACHE = CACHE_DIR / "device_facts.json"
# Hand-maintained truth that beats anything looked up. For facts you know and
# the automated sources get wrong or cannot see (unofficial ROM support,
# benchmark scores the model misremembers). Same shape as a cache entry.
OVERRIDES = CACHE_DIR / "device_facts_overrides.json"
# Models seen during evaluation that have no cached facts yet. Evaluation is
# never allowed to block on a slow web lookup, so misses are recorded here and
# resolved later by `ai-marketplace-monitor --warm-facts`.
PENDING = CACHE_DIR / "device_facts_pending.json"
LINEAGE_CACHE = CACHE_DIR / "lineageos_devices.json"
LINEAGE_TTL = 7 * 24 * 3600  # the supported-device list changes slowly
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Agentic models with server-side web search. compound-mini is tried first:
# full compound pulls in far more search context and gets rejected outright
# (HTTP 413) whenever the per-minute token budget is partly used, which on a
# free tier is most of the time.
GROQ_SEARCH_MODELS = ("groq/compound-mini", "groq/compound")
# Plain (non-agentic) models, tried only after the searching ones are busy.
# They answer from training data instead of the live web, so the result is
# marked unverified -- but a known RAM figure and a roughly right benchmark
# beat no spec line at all, and these have their own separate rate limits.
GROQ_FALLBACK_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")
UA = "ai-marketplace-monitor device-facts"


class RateLimited(RuntimeError):
    """The provider is busy (HTTP 413/429). Retry later; nothing is wrong."""


def _get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)  # atomic, so a crash mid-write cannot corrupt the cache


# --------------------------------------------------------------------------
# LineageOS
# --------------------------------------------------------------------------
def lineage_devices(refresh: bool = False) -> List[Dict[str, str]]:
    """Return [{codename, vendor, name}] for every LineageOS-supported device.

    Uses the official build API, which returns vendor + marketing name (e.g.
    "Samsung" / "Galaxy S21 5G"). The wiki's _data/devices/*.yml files carry
    the same info but only as ~800 separate files, and the wiki has no JSON
    index (both documented index URLs 404), so this endpoint is the only
    cheap source of names that can be matched against a listing title.
    """
    cached = _load(LINEAGE_CACHE)
    if not refresh and cached.get("fetched_at", 0) + LINEAGE_TTL > time.time():
        return cached.get("devices", [])

    try:
        oems = _get_json("https://download.lineageos.org/api/v2/oems")
    except (urllib.error.URLError, ValueError) as e:
        print(f"warn: LineageOS device list unavailable ({e}); using cache", file=sys.stderr)
        return cached.get("devices", [])

    devices = []
    for oem in oems or []:
        vendor = oem.get("name") or oem.get("oem") or ""
        for d in oem.get("devices", []) or []:
            devices.append(
                {
                    "codename": d.get("model", ""),
                    "vendor": vendor,
                    "name": d.get("name", ""),
                }
            )
    if devices:
        _save(LINEAGE_CACHE, {"fetched_at": time.time(), "devices": devices})
        return devices
    return cached.get("devices", [])


# Words that appear in listing titles but say nothing about which model it is.
# Without stripping these, "Predam Samsung Galaxy S21 5G 128GB" and "Samsung
# Galaxy S21" become two different cache keys for one device -- which meant a
# web lookup per LISTING instead of per MODEL, exhausting the daily request
# quota. Slovak/Czech included, since that is what the listings are written in.
_NOISE = {
    "predam", "predám", "predaj", "novy", "nový", "nova", "nová", "nove", "nové",
    "uplne", "úplne", "super", "stav", "zanovny", "zánovný", "top", "lacno",
    "ako", "novy!", "cierny", "čierny", "biely", "modry", "modrý", "zeleny",
    "cierna", "biela", "sivy", "sivá", "zlty", "zlatý", "gold", "black", "white",
    "blue", "green", "grey", "gray", "silver", "red", "pink", "purple",
    "dual", "sim", "dualsim", "esim", "nabijacka", "zaruka", "záruka",
    "faktura", "faktúra", "used", "new", "mint", "like", "condition",
    "phone", "telefon", "telefón", "mobil", "smartfon", "smartfón", "handy",
    "laptop", "notebook", "pc", "eur", "e", "s", "v", "so", "za",
}
# Capacity/spec tokens: variant detail, not model identity.
_SPEC_RE = re.compile(r"^\d+(gb|tb|mb|mah|hz|ram|w)$|^\d+/\d+$|^\d+gb\d+gb$")


def model_key(text: str) -> str:
    """Reduce a listing title to a stable per-model cache key.

    "Predám Samsung Galaxy S21 5G 128GB, super stav" -> "samsung galaxy s21 5g"

    Keeps brand and model tokens, drops sales patter, colours, capacities and
    condition words. Imperfect by nature -- sellers write whatever they like --
    but it collapses the many ways one device is advertised into a single
    lookup, which is the whole point of the cache.
    """
    tokens = []
    for tok in _norm(text).split():
        if tok in _NOISE or _SPEC_RE.match(tok):
            continue
        tokens.append(tok)
        if len(tokens) >= 5:  # brand + model is short; the rest is description
            break
    return " ".join(tokens) or _norm(text)


def _norm(s: str) -> str:
    """Lowercase, strip accents, collapse to alphanumerics.

    Accents are folded rather than dropped: naively removing non-ASCII turned
    "Predám" into "pred m" (two meaningless tokens) instead of "predam", so
    the Slovak sales words never matched the noise list.
    """
    folded = unicodedata.normalize("NFKD", s.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def lineage_supported(model: str) -> Optional[Dict[str, str]]:
    """Best-effort match of a free-text model name to a LineageOS device."""
    want = _norm(model)
    if not want:
        return None
    best = None
    for d in lineage_devices():
        full = _norm(f"{d.get('vendor','')} {d.get('name','')}")
        if not full.strip():
            continue
        # Require the device's full marketing name to appear in the query, so
        # "Galaxy S21" does not spuriously match "Galaxy S21 Ultra".
        if full and full in want:
            if best is None or len(full) > len(_norm(f"{best['vendor']} {best['name']}")):
                best = d
    return best


# --------------------------------------------------------------------------
# Specs via web search (cached per model)
# --------------------------------------------------------------------------
SPEC_PROMPT = """Look up this device and report ONLY facts you verify on the web.

Device kind: {kind}
Listing text: {model}

Reply as strict JSON, no prose, exactly these keys:
{{"model": "canonical product name",
 "kind": "phone" or "laptop",
 "chip": "exact SoC or CPU, e.g. Snapdragon 888 / Intel Core i5-1135G7",
 "benchmark_name": "antutu_v10" for phones, "passmark_cpu" for laptops,
 "benchmark_score": integer or null,
 "ram_gb": [RAM variants in GB as integers],
 "storage_gb": [storage variants in GB as integers],
 "release_year": integer or null,
 "used_price_eur": integer or null,
 "source": "where the numbers came from"}}

benchmark_score: for a phone use its AnTuTu v10 total score; for a laptop use
the PassMark CPU Mark of its processor. Use null if you cannot verify it --
never estimate. used_price_eur = typical second-hand price in Europe, EUR,
good working condition."""


# Rough class-typical ceilings used to normalise a raw benchmark to 0..1.
# Chosen as "a genuinely high-end current device", so ~1.0 means flagship and
# ~0.3 means entry level. They only need to be consistent, not exact.
BENCHMARK_CEILING = {
    "antutu_v10": 2_000_000,   # current flagship phones land around here
    "antutu_v9": 1_000_000,    # v9 scores run roughly half of v10
    "antutu": 1_500_000,       # version unstated - assume something between
    "passmark_cpu": 30_000,    # strong current laptop CPU
    "geekbench6_multi": 15_000,
}


def quality_score(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic good/bad verdict from chip + RAM + storage.

    The LLM supplies the raw numbers (cached per model); the judgement is made
    here in plain arithmetic so it is reproducible and cannot be talked into a
    different answer by a persuasive listing.

    Two rules learned from live data:

    * The chip is mandatory for a confident verdict. Without a benchmark the
      score is capped and marked low-confidence -- otherwise a device with no
      known CPU but generous RAM/storage options scored "great", which is
      how a ThinkPad with no benchmark once came out at 125/100.
    * RAM/storage use the SMALLEST advertised variant. These lists describe
      configurations the model was sold in, not the unit in the listing, so
      the smallest is the only figure that cannot overstate what is on offer.
    """
    bench = facts.get("benchmark_score")
    name = (facts.get("benchmark_name") or "").lower()
    ceiling = BENCHMARK_CEILING.get(name)
    if ceiling is None:
        ceiling = BENCHMARK_CEILING["passmark_cpu" if facts.get("kind") == "laptop" else "antutu_v10"]

    parts: Dict[str, Any] = {}
    if isinstance(bench, (int, float)) and not isinstance(bench, bool) and bench > 0:
        parts["chip"] = min(bench / ceiling, 1.0)

    def _smallest(key: str) -> Optional[float]:
        vals = [
            v for v in (facts.get(key) or [])
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
        ]
        return min(vals) if vals else None

    ram = _smallest("ram_gb")
    if ram is not None:
        parts["ram"] = min(ram / 8.0, 1.0)      # 8GB is "fine"
    store = _smallest("storage_gb")
    if store is not None:
        parts["storage"] = min(store / 256.0, 1.0)

    if not parts:
        return {"score": None, "verdict": "unknown", "confidence": "none", "parts": {}}

    weights = {"chip": 0.70, "ram": 0.25, "storage": 0.05}
    used = {k: v for k, v in parts.items() if k in weights}
    total_w = sum(weights[k] for k in used) or 1.0
    score = 100 * sum(parts[k] * weights[k] for k in used) / total_w

    if "chip" in parts:
        confidence = "high"
    else:
        # No benchmark: RAM/storage alone cannot justify calling it good.
        confidence = "low"
        score = min(score, 45)

    score = int(round(max(0.0, min(100.0, score))))
    verdict = "great" if score >= 75 else "good" if score >= 55 else "ok" if score >= 35 else "weak"
    if confidence == "low":
        verdict += " (no benchmark)"
    return {
        "score": score,
        "verdict": verdict,
        "confidence": confidence,
        "parts": {k: round(v, 2) for k, v in parts.items()},
    }


def fetch_specs(model: str, api_key: str, kind: str = "auto", timeout: int = 45) -> Dict[str, Any]:
    """Ask a web-searching model for this device's specs, as JSON.

    Failures split into two kinds:

    * busy (HTTP 413/429) -- Groq answers 413, not 429, when a request would
      exceed the remaining tokens-per-minute allowance, and this helper shares
      that budget with the monitor service. Worth one short wait, then the
      next (lighter) model. Surfaced as RateLimited so callers can tell
      "try later" from "broken".
    * fatal (bad request, auth, server error, DNS) -- another model would fail
      identically, so stop immediately rather than burning further requests.
    """
    last_err: Optional[Exception] = None

    for model_name in GROQ_SEARCH_MODELS + GROQ_FALLBACK_MODELS:
        for attempt in range(2):
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": SPEC_PROMPT.format(model=model, kind=kind)}
                ],
            }
            req = urllib.request.Request(
                GROQ_URL,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": UA,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (413, 429):
                    last_err = RateLimited(f"{model_name}: {e}")
                    if attempt == 0:
                        time.sleep(20)
                        continue
                    break  # move on to the next model
                raise
            except urllib.error.URLError as e:
                raise RuntimeError(f"network error contacting {GROQ_URL}: {e}") from e

            content = body["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in response from {model_name}: {content[:150]}")
            facts = json.loads(m.group(0))
            facts["looked_up_by"] = model_name
            if model_name in GROQ_FALLBACK_MODELS:
                # No web access: recalled, not verified. Recorded so the
                # figures can be told apart from searched ones later.
                facts["unverified"] = True
            return facts

    raise last_err or RuntimeError("spec lookup failed")


def has_real_facts(facts: Dict[str, Any]) -> bool:
    """True when an entry holds actual looked-up data, not just a failed try.

    An offline call can only report LineageOS support, so caching its result
    as if the model were resolved makes every later call a cache hit and the
    real lookup never happens.
    """
    return bool(
        facts.get("benchmark_score")
        or facts.get("chip")
        or facts.get("ram_gb")
        or facts.get("overridden")
    )


def device_facts(model: str, refresh: bool = False, offline: bool = False, kind: str = "auto") -> Dict[str, Any]:
    """Facts for one model. Cached, so each model costs one lookup ever.

    offline=True skips the (slow, rate-limited) web spec lookup and returns
    whatever is cached plus LineageOS support, which is a fast local check.
    Use that on any hot path; warm the cache separately.
    """
    cache = _load(FACTS_CACHE)
    key = model_key(model)
    if not refresh and key in cache:
        return cache[key]

    if kind == "auto":
        kind = guess_kind(model)
    facts: Dict[str, Any] = {"query": model, "kind": kind}
    api_key = "" if offline else os.environ.get("GROQ_API_KEY", "")
    if api_key:
        try:
            facts.update(fetch_specs(model, api_key, kind=kind))
        except Exception as e:  # network/quota/parse - degrade, never crash
            facts["specs_error"] = str(e)[:200]
    else:
        facts["specs_error"] = "offline" if offline else "GROQ_API_KEY not set"

    lo = lineage_supported(model)
    facts["lineageos_official"] = bool(lo)
    if lo:
        facts["lineageos_device"] = f"{lo['vendor']} {lo['name']} ({lo['codename']})".strip()

    # Manual overrides win over everything above.
    for ov_key, ov in _load(OVERRIDES).items():
        # Normalize the override key too, so an entry written naturally
        # ("iPhone 11 phone") still matches a key that has had filler words
        # stripped. Match either direction: an override may be broader
        # ("galaxy s21") or narrower than the key derived from the title.
        ov_norm = model_key(ov_key)
        if ov_norm and (ov_norm in key or key in ov_norm):
            facts.update(ov)
            facts["overridden"] = True

    facts.update(quality_score(facts))
    facts["cached_at"] = int(time.time())
    # Only persist a resolved entry. Storing an offline/failed attempt would
    # mask the model as "known" and stop it ever being looked up properly.
    if has_real_facts(facts):
        cache[key] = facts
        _save(FACTS_CACHE, cache)
    return facts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
# Only phones, tablets and laptops have a benchmark this tool can use. The
# monitor also surfaces furniture, clothing, consoles, TVs and accessories,
# and every one of those queued for a lookup wastes a rate-limited web search
# on a device that has no SoC or CPU score at all.
#
# Order matters: accessories are checked FIRST, because they almost always
# name the device they attach to ("Podstavec pre notebook", "Asus Notebook
# SSD disk", "Sieťový adaptér Asus ROG"), and a brand match alone would
# happily queue a laptop stand.

# Things sold FOR a device, not the device. Slovak and English.
ACCESSORY_MARKERS = (
    "podstavec", "stojan", "stand", "adapter", "adaptér", "nabijacka", "nabíjačka",
    "charger", "kabel", "kábel", "cable", "puzdro", "obal", "kryt", "case", "cover",
    "sklo", "folia", "fólia", "screen protector", "dokovacia", "dokovaci",
    "docking", "dock", "klavesnica", "klávesnica", "keyboard", "mys", "myš",
    "mouse", "pencil", "stylus", "pero", "remienok", "strap", "sluchadla",
    "slúchadlá", "headphones", "earbuds", "buds", "taska", "taška", "batoh",
    "backpack", "sleeve", "ssd disk", "hdd", "ram modul", "pamet", "pamäť",
    "chladic", "chladič", "cooler", "webcam", "hub", "redukcia", "napajaci",
    "napájací", "power supply", "diely", "nahradne", "náhradné",
)

# Devices with no comparable CPU/SoC benchmark in this scheme.
NON_COMPUTE_MARKERS = (
    "playstation", "ps3", "ps4", "ps5", "xbox", "nintendo", "switch", "konzola",
    "console", "televizor", "televízor", " tv ", "tv ", "smart tv", "monitor",
    "projektor", "reproduktor", "speaker", "pracka", "práčka", "susicka",
    "sušička", "chladnicka", "chladnička", "mraznicka", "mraznička", "bicykel",
    "gitara", "hodinky", "watch", "band", "kamera", "fotoaparat", "dron",
    # Displays are often named only by product line and refresh rate
    # ("MSI Optix G27C6 - 27\u201d, 165 Hz"), never the word "monitor".
    " hz", "optix", "odyssey g", "ultragear", "nitro xv", "viewsonic",
    # A graphics tablet is not a compute tablet.
    "wacom", "graficky tablet", "grafický tablet", "intuos", "huion",
)

LAPTOP_HINTS = (
    "laptop", "notebook", "macbook", "thinkpad", "ideapad", "vivobook", "zenbook",
    "latitude", "elitebook", "probook", "inspiron", "pavilion", "chromebook",
    "ultrabook", "aspire", "nitro", "legion", "omen", "predator", "yoga",
    "surface laptop", "zephyrus", "alienware", "loq", "tuf", "rog",
)

# Phones and tablets alike run a mobile SoC with an AnTuTu score.
MOBILE_HINTS = (
    "iphone", "ipad", "galaxy", "samsung a", "samsung s", "redmi", "poco",
    "xiaomi", "huawei", "honor", "realme", "oneplus", "oppo", "vivo", "pixel",
    "motorola", "moto g", "nokia", "xperia", "nothing phone", "telefon",
    "telefón", "mobil", "smartfon", "smartfón", "tablet", "tab s",
)

COMPUTE_BRANDS = (
    "samsung", "apple", "xiaomi", "huawei", "honor", "realme", "oneplus", "oppo",
    "vivo", "google", "motorola", "nokia", "sony", "nothing", "asus", "lenovo",
    "hp", "dell", "acer", "msi", "microsoft", "alienware", "gigabyte", "razer",
)


def _has(text: str, markers) -> bool:
    return any(m in text for m in markers)


def classify(title: str) -> str:
    """Return 'phone', 'laptop' or 'other' for a listing title.

    'other' means: do not spend a spec lookup on this.
    """
    t = " " + _norm(title) + " "

    # Accessories are judged on the part BEFORE a "+", because a bundle lists
    # the device first and the extras after it: "MacBook Air + Magic Mouse" is
    # a laptop, while "Asus Notebook SSD disk" and "Podstavec pre notebook"
    # are an SSD and a stand that merely name what they attach to.
    # Split the RAW title: _norm strips punctuation, so splitting afterwards
    # would never find the separator.
    primary = " " + _norm(title.split("+")[0]) + " "
    if _has(primary, ACCESSORY_MARKERS):
        return "other"

    # Laptops first: gaming models legitimately mention refresh rates ("144Hz")
    # that would otherwise read as a monitor.
    if _has(t, LAPTOP_HINTS):
        return "laptop"
    if _has(t, NON_COMPUTE_MARKERS):
        return "other"
    if _has(t, MOBILE_HINTS):
        return "phone"
    # Deliberately no "known brand + a digit" fallback: it classified an
    # "MSI Optix G27C6" monitor as a phone. Skipping an unrecognised device
    # costs one missing spec line; guessing wastes a rate-limited lookup and
    # produces a nonsense score.
    return "other"


def guess_kind(text: str) -> str:
    """Device kind for spec lookup; defaults to phone for unknown devices."""
    kind = classify(text)
    return "laptop" if kind == "laptop" else "phone"


def looks_benchmarkable(title: str) -> bool:
    """Whether a listing is a phone/tablet/laptop worth resolving specs for."""
    return classify(title) != "other"


def facts_for_listing(title: str, kind: str = "auto") -> Dict[str, Any]:
    """Cache-only facts for a listing title, queueing a lookup on a miss.

    Called on the evaluation hot path, so it must never touch the network:
    a spec lookup takes tens of seconds and shares a rate-limited budget with
    the monitor itself. A miss records the title for `--warm-facts` and
    returns whatever is locally knowable (LineageOS support, overrides).
    """
    cache = _load(FACTS_CACHE)
    key = model_key(title)
    hit = cache.get(key)
    if hit is None:
        # Containment match, so a longer cached key still serves a shorter
        # title (and vice versa) rather than triggering a fresh lookup.
        for cached_key, facts in cache.items():
            if cached_key and (cached_key in key or key in cached_key):
                hit = facts
                break
    if hit is not None and has_real_facts(hit):
        return hit
    try:
        pending = _load(PENDING)
        if key and key not in pending and looks_benchmarkable(title):
            pending[key] = {"title": title, "kind": kind, "seen_at": int(time.time())}
            _save(PENDING, pending)
    except OSError:
        pass
    return device_facts(title, offline=True, kind=kind)


def warm_cache(limit: int = 5, verbose: bool = True) -> int:
    """Resolve queued models. Run from cron/systemd, never inline.

    The batch is small on purpose: a single agentic web lookup takes tens of
    seconds, so a large batch would still be running when the next timer
    fires, and two overlapping runs would compete for the same rate limit.
    """
    pending = _load(PENDING)
    if not pending:
        return 0
    done = 0
    for idx, (key, info) in enumerate(list(pending.items())[:limit]):
        title = info.get("title", key)
        # Pace the batch. An agentic search costs roughly a third of the
        # per-minute token allowance, and that allowance is shared with the
        # monitor's own evaluations, so firing a batch back-to-back exhausts
        # it within seconds and the rest fails on 429 -- which is what left
        # the cache empty while the daily quota still looked healthy.
        if idx:
            time.sleep(25)
        try:
            facts = device_facts(title, refresh=True, kind=info.get("kind", "auto"))
            if not has_real_facts(facts):
                # device_facts swallows lookup errors into specs_error, so a
                # rate-limited attempt returns "successfully" with nothing in
                # it. Popping that would silently retire the model from the
                # queue forever -- which is why the queue drained while no
                # benchmark was ever cached. Leave it queued and stop: the
                # limit is per-minute and shared, so the rest of the batch
                # would fail the same way.
                if verbose:
                    print(f"  {title}: no data ({facts.get('specs_error', '?')}), left queued")
                break
            if verbose:
                print(f"  {title}: {summarize(facts)}")
            pending.pop(key, None)
            done += 1
        except RateLimited as e:
            if verbose:
                print(f"  {title}: provider busy, leaving queued ({e})")
            break  # budget is exhausted; stop rather than hammer it
        except Exception as e:
            if verbose:
                print(f"  {title}: lookup failed, dropping from queue ({e})")
            pending.pop(key, None)
    _save(PENDING, pending)
    return done


def summarize(facts: Dict[str, Any]) -> str:
    """One compact line to paste into an LLM prompt."""
    bits = [facts.get("kind", "device")]
    if facts.get("chip"):
        bits.append(str(facts["chip"]))
    if facts.get("benchmark_score"):
        bits.append(f"{facts.get('benchmark_name','benchmark')} {facts['benchmark_score']:,}")
    if facts.get("ram_gb"):
        bits.append("RAM " + "/".join(f"{r}GB" for r in facts["ram_gb"]))
    if facts.get("storage_gb"):
        bits.append("storage " + "/".join(f"{g}GB" for g in facts["storage_gb"]))
    if facts.get("used_price_eur"):
        bits.append(f"typical used ~EUR {facts['used_price_eur']}")
    if facts.get("release_year"):
        bits.append(f"released {facts['release_year']}")
    if facts.get("kind") != "laptop":
        if facts.get("lineageos_unofficial"):
            bits.append("LineageOS: unofficial build available")
        elif facts.get("lineageos_official"):
            bits.append("LineageOS: OFFICIAL support")
        else:
            bits.append("LineageOS: no official build")
    if facts.get("score") is not None:
        bits.append(f"QUALITY {facts['score']}/100 ({facts['verdict']})")
    elif facts.get("specs_error"):
        bits.append(f"(no specs: {str(facts['specs_error'])[:40]})")
    return "; ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description="Look up cached facts for a phone model.")
    ap.add_argument("model", nargs="+")
    ap.add_argument("--refresh", action="store_true", help="ignore cached value")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--offline", action="store_true", help="cache + LineageOS only, no web lookup")
    ap.add_argument("--kind", default="auto", choices=("auto", "phone", "laptop"))
    a = ap.parse_args()
    facts = device_facts(" ".join(a.model), refresh=a.refresh, offline=a.offline, kind=a.kind)
    print(json.dumps(facts, indent=1, ensure_ascii=False) if a.json else summarize(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
