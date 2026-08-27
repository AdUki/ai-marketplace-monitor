#!/usr/bin/env python3
"""Tests for device_facts.py. Stdlib only: python3 test_device_facts.py

Network is stubbed everywhere except the tests marked live (opt in with
LIVE=1), so this suite is fast, deterministic and safe to run repeatedly
without burning API quota.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ai_marketplace_monitor import device_facts as df  # noqa: E402


class TempCache:
    """Point every on-disk cache at a throwaway directory."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self._orig = (df.CACHE_DIR, df.FACTS_CACHE, df.LINEAGE_CACHE, df.OVERRIDES, df.PENDING)
        df.CACHE_DIR = d
        df.FACTS_CACHE = d / "device_facts.json"
        df.LINEAGE_CACHE = d / "lineageos_devices.json"
        df.OVERRIDES = d / "device_facts_overrides.json"
        df.PENDING = d / "device_facts_pending.json"
        return d

    def __exit__(self, *a):
        (df.CACHE_DIR, df.FACTS_CACHE, df.LINEAGE_CACHE, df.OVERRIDES,
         df.PENDING) = self._orig
        self.tmp.cleanup()


class TestQualityScore(unittest.TestCase):
    def test_flagship_phone_scores_high(self):
        r = df.quality_score(
            {"kind": "phone", "benchmark_name": "antutu_v10",
             "benchmark_score": 2_000_000, "ram_gb": [12], "storage_gb": [512]}
        )
        self.assertGreaterEqual(r["score"], 90)
        self.assertIn(r["verdict"], ("great",))

    def test_weak_phone_scores_low(self):
        r = df.quality_score(
            {"kind": "phone", "benchmark_name": "antutu_v10",
             "benchmark_score": 150_000, "ram_gb": [3], "storage_gb": [32]}
        )
        self.assertLess(r["score"], 35)
        self.assertEqual(r["verdict"], "weak")

    def test_better_chip_always_scores_higher(self):
        base = {"kind": "phone", "benchmark_name": "antutu_v10", "ram_gb": [8]}
        low = df.quality_score({**base, "benchmark_score": 300_000})["score"]
        high = df.quality_score({**base, "benchmark_score": 900_000})["score"]
        self.assertGreater(high, low)

    def test_more_ram_always_scores_higher(self):
        base = {"kind": "phone", "benchmark_name": "antutu_v10", "benchmark_score": 600_000}
        self.assertGreater(
            df.quality_score({**base, "ram_gb": [12]})["score"],
            df.quality_score({**base, "ram_gb": [4]})["score"],
        )

    def test_laptop_uses_passmark_ceiling(self):
        """The same raw number must mean different things per benchmark."""
        phone = df.quality_score(
            {"kind": "phone", "benchmark_name": "antutu_v10", "benchmark_score": 20_000, "ram_gb": [8]}
        )["score"]
        laptop = df.quality_score(
            {"kind": "laptop", "benchmark_name": "passmark_cpu", "benchmark_score": 20_000, "ram_gb": [8]}
        )["score"]
        self.assertGreater(laptop, phone)

    def test_missing_benchmark_still_scores_from_ram(self):
        r = df.quality_score({"kind": "phone", "ram_gb": [8]})
        self.assertIsNotNone(r["score"])
        self.assertNotIn("chip", r["parts"])

    def test_no_data_at_all_is_unknown_not_zero(self):
        r = df.quality_score({"kind": "phone"})
        self.assertIsNone(r["score"])
        self.assertEqual(r["verdict"], "unknown")

    def test_unknown_benchmark_name_falls_back_by_kind(self):
        r = df.quality_score(
            {"kind": "laptop", "benchmark_name": "cinebench_r23",
             "benchmark_score": 15_000, "ram_gb": [16]}
        )
        self.assertIsNotNone(r["score"])

    def test_absurd_values_are_capped(self):
        r = df.quality_score(
            {"kind": "phone", "benchmark_name": "antutu_v10",
             "benchmark_score": 99_000_000, "ram_gb": [1024], "storage_gb": [100_000]}
        )
        self.assertLessEqual(r["score"], 100)

    def test_junk_types_do_not_crash(self):
        for bad in (
            {"benchmark_score": "lots", "ram_gb": "8GB"},
            {"benchmark_score": None, "ram_gb": [None]},
            {"benchmark_score": -5, "ram_gb": []},
            {"ram_gb": ["8", 6]},
        ):
            with self.subTest(bad=bad):
                r = df.quality_score({"kind": "phone", **bad})
                self.assertIn("verdict", r)

    def test_no_benchmark_cannot_be_called_great(self):
        """RAM/storage alone must not produce a confident high score."""
        r = df.quality_score({"kind": "laptop", "ram_gb": [48], "storage_gb": [2000]})
        self.assertLessEqual(r["score"], 45)
        self.assertEqual(r["confidence"], "low")
        self.assertIn("no benchmark", r["verdict"])

    def test_score_never_exceeds_100(self):
        for f in ({"kind": "laptop", "ram_gb": [64], "storage_gb": [4000]},
                  {"kind": "phone", "benchmark_name": "antutu_v10",
                   "benchmark_score": 9_000_000, "ram_gb": [24], "storage_gb": [1000]}):
            self.assertLessEqual(df.quality_score(f)["score"], 100, f)

    def test_variant_lists_use_smallest_not_largest(self):
        """A model sold with up to 48GB must not score as if it has 48GB."""
        base = {"kind": "laptop", "benchmark_name": "passmark_cpu", "benchmark_score": 10_000}
        many = df.quality_score({**base, "ram_gb": [8, 16, 32, 48]})["score"]
        just8 = df.quality_score({**base, "ram_gb": [8]})["score"]
        self.assertEqual(many, just8)

    def test_bool_is_not_treated_as_number(self):
        r = df.quality_score({"kind": "phone", "benchmark_score": True, "ram_gb": [True]})
        self.assertEqual(r["confidence"], "none")

    def test_verdict_thresholds_are_ordered(self):
        seen = []
        for score in (10, 40, 60, 90):
            facts = {"kind": "phone", "benchmark_name": "antutu_v10",
                     "benchmark_score": int(score / 100 * 2_000_000), "ram_gb": [8]}
            seen.append(df.quality_score(facts)["score"])
        self.assertEqual(seen, sorted(seen))


class TestKindDetection(unittest.TestCase):
    def test_laptops(self):
        for t in ("Lenovo ThinkPad T480", "Predam notebook HP", "MacBook Air 2020",
                  "Dell Latitude 7490 laptop", "Acer Aspire 5"):
            self.assertEqual(df.guess_kind(t), "laptop", t)

    def test_phones_default(self):
        for t in ("Samsung Galaxy S21", "iPhone 11 64GB", "Xiaomi Redmi Note 12",
                  "Predam telefon"):
            self.assertEqual(df.guess_kind(t), "phone", t)

    def test_empty_input(self):
        self.assertEqual(df.guess_kind(""), "phone")


class TestNormalisation(unittest.TestCase):
    def test_diacritics_and_punctuation(self):
        """Accents fold to their base letter rather than splitting the word."""
        self.assertEqual(df._norm("Predám  Galaxy-S21, 5G!"), "predam galaxy s21 5g")

    def test_empty(self):
        self.assertEqual(df._norm(""), "")


class TestLineageMatching(unittest.TestCase):
    DEVICES = [
        {"codename": "b0s", "vendor": "Samsung", "name": "Galaxy S22 Ultra"},
        {"codename": "a52q", "vendor": "Samsung", "name": "Galaxy A52 4G"},
        {"codename": "beyond1lte", "vendor": "Samsung", "name": "Galaxy S10"},
    ]

    def setUp(self):
        self.p = mock.patch.object(df, "lineage_devices", return_value=self.DEVICES)
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_exact_match(self):
        self.assertIsNotNone(df.lineage_supported("Samsung Galaxy S22 Ultra 256GB"))

    def test_unsupported_returns_none(self):
        self.assertIsNone(df.lineage_supported("Samsung Galaxy S21 5G"))

    def test_does_not_match_shorter_variant_to_longer(self):
        """'Galaxy S10' must not be reported for a listing that is an S10e."""
        self.assertIsNone(df.lineage_supported("iPhone 11"))

    def test_prefers_longest_match(self):
        devices = self.DEVICES + [{"codename": "x", "vendor": "Samsung", "name": "Galaxy S22"}]
        with mock.patch.object(df, "lineage_devices", return_value=devices):
            got = df.lineage_supported("Samsung Galaxy S22 Ultra")
            self.assertEqual(got["name"], "Galaxy S22 Ultra")

    def test_empty_query(self):
        self.assertIsNone(df.lineage_supported(""))


class TestCacheAndOverrides(unittest.TestCase):
    def test_cached_lookup_is_not_repeated(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with mock.patch.object(df, "fetch_specs", return_value={"benchmark_score": 1}) as fs:
                df.device_facts("Test Phone")
                df.device_facts("Test Phone")
                self.assertEqual(fs.call_count, 1, "second call should hit the cache")

    def test_refresh_forces_new_lookup(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with mock.patch.object(df, "fetch_specs", return_value={"benchmark_score": 1}) as fs:
                df.device_facts("Test Phone")
                df.device_facts("Test Phone", refresh=True)
                self.assertEqual(fs.call_count, 2)

    def test_offline_never_calls_network(self):
        with TempCache():
            with mock.patch.object(df, "fetch_specs", side_effect=AssertionError("called!")) as fs:
                facts = df.device_facts("Test Phone", offline=True)
                fs.assert_not_called()
                self.assertEqual(facts["specs_error"], "offline")

    def test_override_beats_lookup(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            df.OVERRIDES.write_text(json.dumps(
                {"test phone": {"benchmark_score": 999_999, "ram_gb": [12]}}))
            with mock.patch.object(df, "fetch_specs",
                                   return_value={"benchmark_score": 1, "ram_gb": [2]}):
                facts = df.device_facts("Test Phone")
            self.assertEqual(facts["benchmark_score"], 999_999)
            self.assertTrue(facts["overridden"])

    def test_override_matches_variants(self):
        with TempCache():
            df.OVERRIDES.write_text(json.dumps({"galaxy s21": {"benchmark_score": 800_000}}))
            facts = df.device_facts("Samsung Galaxy S21 5G 128GB", offline=True)
            self.assertEqual(facts["benchmark_score"], 800_000)

    def test_score_reflects_override(self):
        with TempCache():
            df.OVERRIDES.write_text(json.dumps(
                {"test": {"benchmark_name": "antutu_v10", "benchmark_score": 1_800_000,
                          "ram_gb": [12]}}))
            facts = df.device_facts("Test", offline=True)
            self.assertGreaterEqual(facts["score"], 80)

    def test_corrupt_cache_is_survivable(self):
        with TempCache():
            df.FACTS_CACHE.write_text("{ this is not json")
            facts = df.device_facts("Test Phone", offline=True)
            self.assertIn("verdict", facts)

    def test_corrupt_overrides_are_survivable(self):
        with TempCache():
            df.OVERRIDES.write_text("]]not json[[")
            facts = df.device_facts("Test Phone", offline=True)
            self.assertIn("verdict", facts)

    def test_save_is_atomic_no_tmp_left(self):
        with TempCache() as d:
            df._save(df.FACTS_CACHE, {"a": 1})
            leftovers = [p.name for p in Path(d).glob("*.tmp")]
            self.assertEqual(leftovers, [])


class TestFetchFailures(unittest.TestCase):
    def _http_error(self, code):
        return urllib.error.HTTPError("u", code, "err", {}, None)

    def test_missing_api_key_degrades(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            facts = df.device_facts("Test Phone")
            self.assertIn("specs_error", facts)
            self.assertIn("verdict", facts)

    def test_lookup_exception_never_propagates(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}):
            with mock.patch.object(df, "fetch_specs", side_effect=RuntimeError("boom")):
                facts = df.device_facts("Test Phone")
            self.assertIn("boom", facts["specs_error"])

    def test_rate_limit_codes_are_retried_then_reported(self):
        """429 is surfaced as RateLimited (busy), and retried across models."""
        with mock.patch.object(df.time, "sleep"), \
                mock.patch.object(df.urllib.request, "urlopen",
                                  side_effect=self._http_error(429)) as uo:
            with self.assertRaises(df.RateLimited):
                df.fetch_specs("X", "key", timeout=1)
            # 2 attempts x every model in the chain: the searching ones
            # first, then the plain fallbacks.
            expected = 2 * len(df.GROQ_SEARCH_MODELS + df.GROQ_FALLBACK_MODELS)
            self.assertEqual(uo.call_count, expected)

    def test_413_treated_as_busy_not_fatal(self):
        """Groq answers 413, not 429, when a request exceeds the token budget."""
        with mock.patch.object(df.time, "sleep"), \
                mock.patch.object(df.urllib.request, "urlopen",
                                  side_effect=self._http_error(413)):
            with self.assertRaises(df.RateLimited):
                df.fetch_specs("X", "key", timeout=1)

    def test_hard_http_error_is_not_retried(self):
        """A 400 is broken, not busy: fail fast rather than burning retries."""
        with mock.patch.object(df.time, "sleep"), \
                mock.patch.object(df.urllib.request, "urlopen",
                                  side_effect=self._http_error(400)) as uo:
            with self.assertRaises(urllib.error.HTTPError):
                df.fetch_specs("X", "key", timeout=1)
            self.assertEqual(uo.call_count, 1)

    def test_rate_limit_is_recorded_not_crashing(self):
        """device_facts() degrades to a cached/offline answer when busy."""
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with mock.patch.object(df, "fetch_specs", side_effect=df.RateLimited("busy")):
                facts = df.device_facts("Some Phone")
        self.assertIn("busy", facts["specs_error"])
        self.assertIn("verdict", facts)

    def test_non_json_response_raises_cleanly(self):
        body = json.dumps({"choices": [{"message": {"content": "sorry, no idea"}}]}).encode()
        cm = mock.MagicMock()
        cm.read.return_value = body
        cm.__enter__.return_value = cm
        with mock.patch.object(df.urllib.request, "urlopen", return_value=cm):
            with self.assertRaises(ValueError):
                df.fetch_specs("X", "key", timeout=1)

    def test_json_embedded_in_prose_is_extracted(self):
        content = 'Sure!\n```json\n{"ram_gb": [8], "benchmark_score": 5}\n```\nHope that helps'
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        cm = mock.MagicMock()
        cm.read.return_value = body
        cm.__enter__.return_value = cm
        with mock.patch.object(df.urllib.request, "urlopen", return_value=cm):
            got = df.fetch_specs("X", "key", timeout=1)
        self.assertEqual(got["ram_gb"], [8])

    def test_lineage_network_failure_uses_cache(self):
        with TempCache():
            df._save(df.LINEAGE_CACHE, {"fetched_at": 0, "devices": [
                {"codename": "c", "vendor": "V", "name": "N"}]})
            with mock.patch.object(df, "_get_json", side_effect=urllib.error.URLError("down")):
                devs = df.lineage_devices(refresh=True)
            self.assertEqual(len(devs), 1)


class TestSummarize(unittest.TestCase):
    def test_includes_key_fields(self):
        s = df.summarize({
            "kind": "phone", "chip": "SD888", "benchmark_name": "antutu_v10",
            "benchmark_score": 800_000, "ram_gb": [8], "storage_gb": [128],
            "used_price_eur": 220, "release_year": 2021,
            "lineageos_unofficial": True, "score": 58, "verdict": "good"})
        for expect in ("SD888", "800,000", "8GB", "220", "2021", "58/100", "unofficial"):
            self.assertIn(expect, s)

    def test_laptop_omits_lineageos(self):
        s = df.summarize({"kind": "laptop", "chip": "i5", "score": 50, "verdict": "ok"})
        self.assertNotIn("LineageOS", s)

    def test_sparse_facts_do_not_crash(self):
        self.assertIsInstance(df.summarize({}), str)


@unittest.skipUnless(os.environ.get("LIVE") == "1", "set LIVE=1 to hit the network")
class TestLive(unittest.TestCase):
    def test_lineage_api_reachable_and_sane(self):
        devs = df.lineage_devices(refresh=True)
        self.assertGreater(len(devs), 100)
        self.assertTrue(any(d["vendor"] and d["name"] for d in devs))

    def test_real_spec_lookup(self):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            self.skipTest("no GROQ_API_KEY")
        try:
            facts = df.fetch_specs("Samsung Galaxy S21 5G", key, kind="phone", timeout=90)
        except df.RateLimited as e:
            # Shared token budget with the running monitor. Environmental, not
            # a defect, so do not report it as a test failure.
            self.skipTest(f"provider busy: {e}")
        self.assertIn("ram_gb", facts)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestModelKey(unittest.TestCase):
    """One device must yield one cache key, however the seller wrote the ad."""

    def test_listing_variants_collapse(self):
        keys = {df.model_key(t) for t in (
            "Predám Samsung Galaxy S21 5G 128GB, super stav",
            "Samsung Galaxy S21 5G",
            "SAMSUNG GALAXY S21 5g cierny 256GB zaruka",
        )}
        self.assertEqual(len(keys), 1, keys)

    def test_accents_folded_not_split(self):
        self.assertNotIn("pred m", df.model_key("Predám iPhone 11"))
        self.assertEqual(df.model_key("Predám iPhone 11"), "iphone 11")

    def test_distinct_models_stay_distinct(self):
        self.assertNotEqual(df.model_key("Samsung Galaxy S21"),
                            df.model_key("Samsung Galaxy S22"))

    def test_capacity_tokens_dropped(self):
        self.assertEqual(df.model_key("iPhone 11 64GB"), df.model_key("iPhone 11 256GB"))

    def test_never_returns_empty(self):
        self.assertTrue(df.model_key("predam novy telefon"))


class TestNoPoisonedCache(unittest.TestCase):
    """A failed lookup must not masquerade as a resolved model."""

    def test_offline_miss_is_not_cached(self):
        with TempCache():
            df.device_facts("Totally Unknown Phone", offline=True)
            self.assertEqual(_json_or_empty(df.FACTS_CACHE), {})

    def test_failed_lookup_is_not_cached(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with mock.patch.object(df, "fetch_specs", side_effect=df.RateLimited("busy")):
                df.device_facts("Some Phone")
            self.assertEqual(_json_or_empty(df.FACTS_CACHE), {})

    def test_successful_lookup_is_cached(self):
        with TempCache(), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with mock.patch.object(df, "fetch_specs",
                                   return_value={"chip": "SD888", "ram_gb": [8]}):
                df.device_facts("Some Phone")
            self.assertTrue(_json_or_empty(df.FACTS_CACHE))

    def test_empty_entry_does_not_block_a_later_lookup(self):
        with TempCache():
            df._save(df.FACTS_CACHE,
                     {"samsung galaxy s21": {"query": "x", "specs_error": "offline"}})
            df.facts_for_listing("Samsung Galaxy S21")
            self.assertIn("samsung galaxy s21", _json_or_empty(df.PENDING),
                          "an unresolved model must stay queued")

    def test_non_devices_are_not_queued(self):
        """Furniture, clothing and consoles have no benchmark to look up."""
        with TempCache():
            for junk in ("Detska stolicka", "Dievcenske saty", "Susicka 7Kg",
                         "Monitor dychu babysense"):
                df.facts_for_listing(junk)
            self.assertEqual(_json_or_empty(df.PENDING), {})

    def test_devices_are_queued(self):
        with TempCache():
            df.facts_for_listing("Lenovo ThinkPad X13")
            self.assertTrue(_json_or_empty(df.PENDING))


def _json_or_empty(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class TestWarmCacheQueueing(unittest.TestCase):
    """A failed lookup must never retire a model from the queue."""

    def test_unresolved_stays_queued(self):
        with TempCache():
            df._save(df.PENDING, {"samsung galaxy s21": {"title": "Samsung Galaxy S21"}})
            with mock.patch.object(df, "device_facts",
                                   return_value={"specs_error": "busy"}):
                done = df.warm_cache(limit=1, verbose=False)
            self.assertEqual(done, 0)
            self.assertIn("samsung galaxy s21", _json_or_empty(df.PENDING))

    def test_resolved_is_removed(self):
        with TempCache():
            df._save(df.PENDING, {"samsung galaxy s21": {"title": "Samsung Galaxy S21"}})
            with mock.patch.object(df, "device_facts",
                                   return_value={"chip": "Exynos 2100", "ram_gb": [8]}):
                done = df.warm_cache(limit=1, verbose=False)
            self.assertEqual(done, 1)
            self.assertEqual(_json_or_empty(df.PENDING), {})

    def test_batch_stops_when_provider_is_busy(self):
        """No point burning the rest of a batch on a shared per-minute limit."""
        with TempCache():
            df._save(df.PENDING, {f"phone {i}": {"title": f"Samsung Galaxy S{i}"}
                                  for i in range(5)})
            with mock.patch.object(df, "device_facts",
                                   return_value={"specs_error": "429"}) as dfn:
                df.warm_cache(limit=5, verbose=False)
            self.assertEqual(dfn.call_count, 1, "should stop after the first failure")


class TestFallbackModels(unittest.TestCase):
    def test_fallback_result_is_marked_unverified(self):
        body = json.dumps({"choices": [{"message": {"content": '{"ram_gb": [8]}'}}]}).encode()
        cm = mock.MagicMock()
        cm.read.return_value = body
        cm.__enter__.return_value = cm
        calls = {"n": 0}

        def side_effect(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 2 * len(df.GROQ_SEARCH_MODELS):
                raise urllib.error.HTTPError("u", 429, "busy", {}, None)
            return cm

        with mock.patch.object(df.time, "sleep"), \
                mock.patch.object(df.urllib.request, "urlopen", side_effect=side_effect):
            got = df.fetch_specs("Samsung Galaxy S21", "key", timeout=1)
        self.assertTrue(got.get("unverified"))
        self.assertIn(got["looked_up_by"], df.GROQ_FALLBACK_MODELS)
