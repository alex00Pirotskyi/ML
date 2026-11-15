#!/usr/bin/env python3
"""Synthetic unit tests for conservative LP pseudo-label generation."""
import csv
import gzip
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_rest_labeler import compute_labels, label_one, OUTPUT_COLUMNS


def signal(n=800, active=None, disconnected=(), quiet=(), psd=True):
    rng = np.random.default_rng(77)
    df = pd.DataFrame({
        "Timestamp": 1000000 + (np.arange(n) * 6).astype("int64"),
        "PointerState": np.where(np.arange(n) % 17 == 0, 1, 0),
        "TriggerPressure": np.where(np.arange(n) % 17 == 0, 1, 0),
        "User": ["Synthetic"] * n,
    })
    for i in (1, 2, 3):
        if i in disconnected:
            wl = 11000 * (1 + .01 * rng.normal(size=n))
            ps = 90000 * (1 + .01 * rng.normal(size=n))
        elif i in quiet:
            wl = 12 + rng.normal(size=n)
            ps = 4 + rng.normal(size=n)
        else:
            wl = 500 * (1 + .03 * rng.normal(size=n))
            ps = 1000 * (1 + .03 * rng.normal(size=n))
        if active and i in active["channels"]:
            sl = slice(active["start"], active["end"])
            wl[sl] *= active["scale"]
            ps[sl] *= active["scale"] ** 2
        df[f"p_core->wl[CH{i}]"] = wl
        if psd:
            for z, share in enumerate((.20, .28, .30, .11, .11)):
                df[f"p_core->emg_psd_mean[CH{i}][{z}]"] = ps * share
    return df


class TestRestTeacher(unittest.TestCase):
    def check_partition(self, labels):
        self.assertFalse(labels.isna().any().any())
        self.assertTrue(labels["lp_state"].isin((0, 1)).all())
        self.assertTrue(labels["lp_state_weight"].between(0, 1).all())
        for i in (1, 2, 3):
            names = [
                f"lp_{kind}_ch{i}"
                for kind in ("resting", "high_activity", "disconnection", "unknown")
            ]
            self.assertTrue(all(labels[c].between(0, 1).all() for c in names))
            np.testing.assert_allclose(labels[names].sum(axis=1), 1, atol=1e-10)

    def test_rest_is_one_and_pointer_ignored(self):
        raw = signal()
        label, _ = compute_labels(raw)
        self.check_partition(label)
        self.assertGreater(label["lp_state"].iloc[300:].mean(), .95)
        alternate = raw.copy()
        alternate["PointerState"] = 1 - alternate["PointerState"]
        label2, _ = compute_labels(alternate)
        pd.testing.assert_frame_equal(label, label2)

    def test_activation_and_release_one_channel(self):
        label, _ = compute_labels(signal(
            n=1500, active={"channels": (1,), "start": 300, "end": 950, "scale": 4}
        ))
        self.check_partition(label)
        self.assertGreater(label["lp_high_activity_ch1"].iloc[430:700].median(), .9)
        self.assertLess(label["lp_state"].iloc[430:700].mean(), .1)
        self.assertGreater(label["lp_state"].iloc[1150:1450].mean(), .75)

    def test_stationary_noise_is_not_rest(self):
        label, _ = compute_labels(signal(n=1600, disconnected=(2,)))
        self.check_partition(label)
        self.assertGreater(label["lp_disconnection_ch2"].iloc[750:].median(), .75)
        self.assertGreater(label["lp_resting_ch1"].iloc[750:].median(), .75)
        self.assertEqual(label["lp_state"].iloc[750:].sum(), 0)

    def test_quiet_untrusted_channel(self):
        label, _ = compute_labels(signal(n=1000, quiet=(3,)))
        self.check_partition(label)
        self.assertGreater(label["lp_unknown_ch3"].iloc[550:].median(), .6)
        self.assertLess(label["lp_disconnection_ch3"].iloc[550:].median(), .2)
        self.assertEqual(label["lp_state"].iloc[550:].sum(), 0)

    def test_no_psd(self):
        label, _ = compute_labels(signal(n=120, psd=False))
        self.check_partition(label)
        self.assertTrue(all(label[f"lp_unknown_ch{i}"].max() >= .2 for i in (1, 2, 3)))

    def test_original_field_values_and_skip(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "original"
            dest = Path(temporary) / "labels"
            (source / "CONNECTED").mkdir(parents=True)
            df = signal(n=550)
            df["Note"] = ["text, with comma"] * len(df)
            path = source / "CONNECTED" / "a.csv.gz"
            df.to_csv(path, compression="gzip", index=False)
            original_bytes = path.read_bytes()
            args = (str(path), str(source), str(dest), False)
            self.assertEqual(label_one(args)["status"], "labeled")
            target = dest / "CONNECTED" / "a.csv.gz"
            self.assertTrue(target.exists())
            self.assertEqual(path.read_bytes(), original_bytes)
            with gzip.open(path, "rt", newline="") as handle:
                old = list(csv.reader(handle))
            with gzip.open(target, "rt", newline="") as handle:
                new = list(csv.reader(handle))
            self.assertEqual(len(old), len(new))
            self.assertEqual(new[0][-len(OUTPUT_COLUMNS):], list(OUTPUT_COLUMNS))
            for a, b in zip(old, new):
                self.assertEqual(a, b[:len(a)])
            self.assertEqual(label_one(args)["status"], "skipped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
