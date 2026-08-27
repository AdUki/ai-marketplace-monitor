#!/usr/bin/env python3
"""Matching tests for benchmark_db (no network; uses a fixed mini-table)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ai_marketplace_monitor import benchmark_db as b  # noqa: E402

DEVICES = {
    "Samsung Galaxy S21 5G (Exynos)": 11972,
    "Samsung Galaxy S21 Ultra 5G": 13500,
    "Samsung Galaxy A54": 12026,
    "Huawei P30": 8680,
    "Huawei P30 Pro": 9587,
    "Xiaomi 12": 16114,
    "Xiaomi Redmi Note 11": 4000,
}
CPUS = {
    "Intel Core i5-8350U @ 1.70GHz": 6085,
    "Intel Core i5-7300U @ 2.60GHz": 3620,
    "Intel Core i5-1135G7 @ 2.40GHz": 9300,
    "Intel Core i5-13450HX": 24437,
    "AMD Ryzen 5 5500U": 13000,
}


class TestDeviceMatching(unittest.TestCase):
    def m(self, q):
        got = b.match_device(q, DEVICES)
        return got[0] if got else None

    def test_exact(self):
        self.assertEqual(self.m("Samsung Galaxy A54"), "Samsung Galaxy A54")

    def test_missing_filler_word_still_matches(self):
        """'Samsung S21' is the same phone as 'Samsung Galaxy S21'."""
        self.assertEqual(self.m("Samsung S21"), "Samsung Galaxy S21 5G (Exynos)")

    def test_case_and_accents_ignored(self):
        self.assertEqual(self.m("PREDÁM samsung galaxy a54 super stav"),
                         "Samsung Galaxy A54")

    def test_capacity_pair_ignored(self):
        for q in ("samsung a54 8/256", "Samsung A54 8GB/256GB", "Samsung A54 8 GB / 256 GB"):
            self.assertEqual(self.m(q), "Samsung Galaxy A54", q)

    def test_variant_is_not_the_base_model(self):
        self.assertEqual(self.m("Samsung Galaxy S21 Ultra"), "Samsung Galaxy S21 Ultra 5G")
        self.assertEqual(self.m("Huawei P30 Pro"), "Huawei P30 Pro")
        self.assertEqual(self.m("Huawei P30"), "Huawei P30")

    def test_different_model_number_never_matches(self):
        self.assertIsNone(self.m("Samsung Galaxy S22"))
        self.assertIsNone(self.m("Xiaomi 13"))

    def test_absent_model_returns_none_not_a_guess(self):
        """A wrong benchmark is worse than no benchmark."""
        self.assertIsNone(self.m("Xiaomi Redmi Note 12 Pro"))
        self.assertIsNone(self.m("Samsung Galaxy A05s"))

    def test_note_line_not_confused_with_base(self):
        self.assertIsNone(self.m("Xiaomi 11"))
        self.assertEqual(self.m("Xiaomi Redmi Note 11"), "Xiaomi Redmi Note 11")

    def test_empty_query(self):
        self.assertIsNone(self.m(""))


class TestCpuMatching(unittest.TestCase):
    def m(self, q):
        got = b.match_component(q, CPUS)
        return got[0] if got else None

    def test_cpu_inside_a_long_listing(self):
        self.assertEqual(self.m("Lenovo ThinkPad T480 i5-8350U 16GB"),
                         "Intel Core i5-8350U @ 1.70GHz")

    def test_slovak_listing_with_slashes(self):
        self.assertEqual(self.m("Dell Latitude 5580 / i5 7300U / 8GB DDR4 / 256GB NVME"),
                         "Intel Core i5-7300U @ 2.60GHz")

    def test_vendor_words_optional(self):
        self.assertEqual(self.m("HP 15, 11th Gen Intel Core i5-1135G7, 16GB RAM"),
                         "Intel Core i5-1135G7 @ 2.40GHz")

    def test_hx_part_number(self):
        self.assertEqual(self.m("Herný notebook Lenovo LOQ - i5-13450HX RTX 4050"),
                         "Intel Core i5-13450HX")

    def test_amd(self):
        self.assertEqual(self.m("Acer Aspire 5 Ryzen 5 5500U 16GB"), "AMD Ryzen 5 5500U")

    def test_no_cpu_named(self):
        self.assertIsNone(self.m("Lenovo ThinkPad, dobry stav"))


if __name__ == "__main__":
    unittest.main(verbosity=1)


class TestSlovakTransliteration(unittest.TestCase):
    """Every Slovak diacritic must fold to its plain ASCII letter."""

    PAIRS = [
        ("á", "a"), ("ä", "a"), ("č", "c"), ("ď", "d"), ("é", "e"), ("í", "i"),
        ("ĺ", "l"), ("ľ", "l"), ("ň", "n"), ("ó", "o"), ("ô", "o"), ("ŕ", "r"),
        ("š", "s"), ("ť", "t"), ("ú", "u"), ("ý", "y"), ("ž", "z"),
    ]

    def test_every_letter_lower_and_upper(self):
        for ch, plain in self.PAIRS:
            self.assertEqual(b._norm(ch), plain, ch)
            self.assertEqual(b._norm(ch.upper()), plain, ch.upper())

    def test_no_letter_is_dropped(self):
        for ch, _ in self.PAIRS:
            self.assertTrue(b._norm(ch), f"{ch} vanished")

    def test_real_words(self):
        for word, plain in [
            ("Predám", "predam"), ("Úplne nový", "uplne novy"),
            ("Sušička", "susicka"), ("nefunkčný", "nefunkcny"),
            ("Príslušenstvo", "prislusenstvo"), ("čierny", "cierny"),
        ]:
            self.assertEqual(b._norm(word), plain, word)

    def test_accented_listing_matches_plain_table_entry(self):
        got = b.match_device("Predám Samsung Galaxy A54 – zánovný", DEVICES)
        self.assertEqual(got[0], "Samsung Galaxy A54")


class TestLearnedEntries(unittest.TestCase):
    """Web lookups must enrich the local database, and only when sane."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = (b.CACHE_DIR, b.LOCAL_PATH, b.DB_PATH)
        b.CACHE_DIR = Path(self.tmp.name)
        b.LOCAL_PATH = b.CACHE_DIR / "benchmarks_local.json"
        b.DB_PATH = b.CACHE_DIR / "benchmarks.json"
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        b.CACHE_DIR, b.LOCAL_PATH, b.DB_PATH = self._orig

    def test_plausible_value_is_stored(self):
        self.assertTrue(b.add_entry("device", "Honor 8X", 5200))
        self.assertEqual(b.load_local()["device"]["Honor 8X"], 5200)

    def test_implausible_values_are_refused(self):
        for kind, bad in (("device", 9_000_000), ("cpu", -3), ("antutu", 5),
                          ("gpu", 0)):
            self.assertFalse(b.add_entry(kind, "Bogus", bad), (kind, bad))
        self.assertEqual(b.load_local(), {})

    def test_bool_is_not_a_score(self):
        self.assertFalse(b.add_entry("device", "Bogus", True))

    def test_empty_name_refused(self):
        self.assertFalse(b.add_entry("device", "   ", 5000))

    def test_learned_entry_is_matchable(self):
        b.add_entry("device", "Samsung Galaxy A05s", 4200)
        b.save({"device": {}})           # upstream table has nothing
        got = b.match_device("Predám Samsung Galaxy A05s 4/64",
                             b.load(auto_download=False).get("device", {}))
        self.assertIsNotNone(got)
        self.assertEqual(got[1], 4200)

    def test_local_survives_upstream_refresh(self):
        b.add_entry("device", "Honor 8X", 5200)
        b.save({"device": {"Something Else": 1}})   # simulate a re-download
        self.assertEqual(b.load(auto_download=False)["device"]["Honor 8X"], 5200)

    def test_local_overrides_upstream(self):
        b.save({"device": {"Honor 8X": 1}})
        b.add_entry("device", "Honor 8X", 5200)
        self.assertEqual(b.load(auto_download=False)["device"]["Honor 8X"], 5200)
