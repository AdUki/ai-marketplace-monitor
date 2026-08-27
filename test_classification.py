#!/usr/bin/env python3
"""Classification and benchmark tests for device_facts.

The corpus is real listing titles taken from the running monitor's own
queue -- Slovak, inconsistent, full of accessories that name the device
they attach to. Invented examples were exactly why three classification
bugs shipped: they were all too tidy.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ai_marketplace_monitor import device_facts as df  # noqa: E402

# (title, expected classification) -- all observed in production.
CORPUS = [
    # --- phones -----------------------------------------------------------
    ("Samsung galaxy S25+", "phone"),
    ("Úplne nový Samsung A26 5G 6GB/128GB", "phone"),
    ("Samsung galaxy A05s", "phone"),
    ("Samsung Galaxy S20+ 128 GB", "phone"),
    ("iPhone 13 Pro 256 GB", "phone"),
    ("Iphone 13 Mini 256 GB", "phone"),
    ("iPhone 13 Mini 128Gb 100% Zdravie Batérie", "phone"),
    ("- iPhone 13 5G 128GB", "phone"),
    ("Iphone 14 pro 128 Gold", "phone"),
    ("Predám Huawei P30 128GB", "phone"),
    ("Huawei p20 lite", "phone"),
    ("Honor 8x", "phone"),
    ("Honor 20 lite", "phone"),
    ("Mobil - HONOR 400 5G 8GB/512GB Midnight Black", "phone"),
    ("HONOR Magic V6 16 GB/512 GB", "phone"),
    ("Mobilný telefón Xiaomi Redmi Note 15 Pro 5G 8 GB / 256 GB (71571) čierny", "phone"),
    ("Huawei pura X", "phone"),
    ("Apple iPhone Air 256GB Space Black", "phone"),
    # --- tablets: mobile SoC, AnTuTu applies ------------------------------
    ("Samsung Galaxy Tab S10 FE 8 GB/128 GB Gray", "phone"),
    ("Samsung Galaxy Tab S9 FE 5G – ako nový", "phone"),
    ("Apple iPad Pro 11\" (2. generácie) Space Gray 256 GB", "phone"),
    ("iPad Pro 12.9 3rd 256 gb A0024", "phone"),
    # --- laptops ----------------------------------------------------------
    ("Lenovo Yoga 9 14IAP7", "laptop"),
    ("ASUS X556UQ – notebook", "laptop"),
    ("Notebook Lenovo 80G0", "laptop"),
    ("Herný notebook Lenovo LOQ - i5-13450HX RTX 4050 16/1TB", "laptop"),
    ("MacBook Air 11” 2015 – i5, 4 GB RAM, 128 GB SSD", "laptop"),
    ("MacBook Pro 13 Retina, Early 2015", "laptop"),
    ('MacBook Pro 16" i7-2.6GHz,6j/32GB/512GB, Výdrž batérie 5 h.', "laptop"),
    ("Dell Latitude 5580 / i5 7300U / 8GB DDR4 / 256GB NVME / Full HD", "laptop"),
    ("Surface Laptop 4 15, 16GB RAM, 512GB SSD, 11th gen intel i7", "laptop"),
    ("Alienware M17 Laptop", "laptop"),
    ("Asus ROG Strix Gaming Laptop 17.3 I7-7700HQ GTX 1070 16GB Ram", "laptop"),
    ("Acer Predator Helios 300 Gaming Laptop, Intel Core i7-9750H", "laptop"),
    ("Lenovo ThinkPad X13 13.3 FHD Laptop, Intel Core i5-10210U", "laptop"),
    ("Lenovo ideapad 1 14IGL05", "laptop"),
    ("MSI GF75 Thin Gaming Laptop 17.3 144Hz", "laptop"),
    ("HP OMEN Gaming Laptop Black 512 GB ssd plus 1TB SSD", "laptop"),
    # --- accessories: name a device, ARE NOT one --------------------------
    ("Podstavec pre notebook", "other"),
    ("Asus Notebook SSD disk", "other"),
    ("Sieťový adaptér Asus ROG 240W CP (90XB095N-MPW000) čierny", "other"),
    ("Dokovací stanice OWC Thunderbolt 4", "other"),
    ("apple pencil", "other"),
    ("Grafický tablet Wacom Intuos Pro M (PTH-660) - TOP stav + 10x hrot", "other"),
    ("Bezdrôtový in-ear odposluch JTS SIEM-2T + NOVÉ slúchadlá KZ", "other"),
    ("Apple MacBook Air + Apple Magic Mouse", "laptop"),  # the laptop is the item
    # --- consoles / displays / appliances: no comparable benchmark --------
    ("PlayStation 5 Console", "other"),
    ("Playstation 4 slim (ps4)", "other"),
    ("Xbox Series X console", "other"),
    ("Microsoft Xbox Series S Console 512GB", "other"),
    ("Nintendo 3DS XL (Červeno-čierna verzia)", "other"),
    ("MSI Optix G27C6 – 27”, 165 Hz", "other"),
    ("ViewSonic VX3276-2K-mhd", "other"),
    ('Samsung 32" TV / Monitor (+ original remote)', "other"),
    ("Smart TV samsung", "other"),
    ("Sušička 7Kg", "other"),
    ("Chladnička", "other"),
    ("Mraznička", "other"),
    ("Huawei band 10", "other"),
    # --- not electronics at all -------------------------------------------
    ("Detská stolička", "other"),
    ("Dievčenské šaty", "other"),
    ("Kuchynská linka", "other"),
    ("Školská taška", "other"),
    ("Monitor dychu babysense", "other"),
    ("Philips avent baby monitor", "other"),
    ("Balik 6kniziek Labkova Patrola", "other"),
    ("Hojdacia stolička", "other"),
    ("Spinning Bike – Spinningový bicykel Echelon Connect Sport", "other"),
    ("Elektroakustická gitara Ibanez AE10, ročník 1995", "other"),
]


class TestClassification(unittest.TestCase):
    def test_corpus(self):
        wrong = [
            (t, want, df.classify(t))
            for t, want in CORPUS
            if df.classify(t) != want
        ]
        if wrong:
            msg = "\n".join(f"    {t!r}\n      want {w}, got {g}" for t, w, g in wrong)
            self.fail(f"{len(wrong)}/{len(CORPUS)} misclassified:\n{msg}")

    def test_accessories_never_queued(self):
        for title, want in CORPUS:
            if want == "other":
                self.assertFalse(df.looks_benchmarkable(title), title)

    def test_devices_always_queued(self):
        for title, want in CORPUS:
            if want != "other":
                self.assertTrue(df.looks_benchmarkable(title), title)

    def test_kind_matches_classification(self):
        for title, want in CORPUS:
            if want == "laptop":
                self.assertEqual(df.guess_kind(title), "laptop", title)
            elif want == "phone":
                self.assertEqual(df.guess_kind(title), "phone", title)


class TestBenchmarkSelection(unittest.TestCase):
    """The right benchmark, and the right scale for it."""

    def test_phone_uses_antutu_scale(self):
        r = df.quality_score({"kind": "phone", "benchmark_name": "antutu_v10",
                              "benchmark_score": 1_000_000, "ram_gb": [8]})
        self.assertEqual(r["confidence"], "high")
        self.assertTrue(40 <= r["score"] <= 75, r)

    def test_laptop_uses_passmark_scale(self):
        r = df.quality_score({"kind": "laptop", "benchmark_name": "passmark_cpu",
                              "benchmark_score": 15_000, "ram_gb": [16]})
        self.assertTrue(50 <= r["score"] <= 85, r)

    def test_same_number_different_meaning(self):
        n = 20_000
        phone = df.quality_score({"kind": "phone", "benchmark_name": "antutu_v10",
                                  "benchmark_score": n, "ram_gb": [8]})["score"]
        laptop = df.quality_score({"kind": "laptop", "benchmark_name": "passmark_cpu",
                                   "benchmark_score": n, "ram_gb": [8]})["score"]
        self.assertGreater(laptop, phone + 20)

    def test_antutu_v9_scaled_against_its_own_ceiling(self):
        v9 = df.quality_score({"kind": "phone", "benchmark_name": "antutu_v9",
                               "benchmark_score": 500_000, "ram_gb": [8]})["score"]
        v10 = df.quality_score({"kind": "phone", "benchmark_name": "antutu_v10",
                                "benchmark_score": 500_000, "ram_gb": [8]})["score"]
        self.assertGreater(v9, v10, "a v9 score is worth more than the same v10 score")

    def test_flagship_beats_midrange_beats_budget(self):
        def score(n):
            return df.quality_score({"kind": "phone", "benchmark_name": "antutu_v10",
                                     "benchmark_score": n, "ram_gb": [8]})["score"]
        self.assertGreater(score(1_800_000), score(700_000))
        self.assertGreater(score(700_000), score(250_000))

    def test_missing_benchmark_is_not_confident(self):
        r = df.quality_score({"kind": "laptop", "ram_gb": [32], "storage_gb": [2000]})
        self.assertEqual(r["confidence"], "low")
        self.assertLessEqual(r["score"], 45)


if __name__ == "__main__":
    unittest.main(verbosity=1)
