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
