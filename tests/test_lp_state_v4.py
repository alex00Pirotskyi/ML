#!/usr/bin/env python3
"""Synthetic contract tests for a conservative LP labeler (not a real-world score)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import VERSION, label_recording, main


def synthetic(n=900, press=(240, 575), signal=(220, 590)):
    rng = np.random.default_rng(193)
    d = {"Timestamp": 1700000000 + np.arange(n) * 6,
         "PointerState": np.zeros(n, int)}
    d["PointerState"][slice(*press)] = 1
    for ch in range(1, 4):
        w = 500 + rng.normal(0, 9, n)
        p = np.full(n, 6600.) + rng.normal(0, 12, n)
        if ch == 1:
            w[slice(*signal)] *= 4.0
            p[slice(*signal)] += 4400
        d[f"p_core->wl[CH{ch}]"] = w
        for z in range(5):
            d[f"p_core->emg_psd_mean[CH{ch}][{z}]"] = p + z * 60
    return pd.DataFrame(d)


class TestStrictState(unittest.TestCase):
    def test_confident_hold_and_rest(self):
        df = synthetic()
        y, audit = label_recording(df, "MID_LONG_PRESS")
        self.assertEqual(audit["verified_events"], 1)
        self.assertGreater(np.mean(y.lp_state.iloc[345:465] == 0), .90)
        self.assertGreater(np.mean(y.lp_state.iloc[75:160] == 1), .90)
        self.assertGreater(np.mean(y.lp_state.iloc[710:820] == 1), .90)
        np.testing.assert_array_equal(y.lp_state_weight.to_numpy(),
                                      (y.lp_state != -1).to_numpy(dtype=float))

    def test_boundaries_are_unknown_with_pointer_offset(self):
        df = synthetic(press=(255, 590), signal=(230, 565))
        y, _ = label_recording(df, "SOFT_LONG_PRESS")
        for transition in (255, 590):
            self.assertTrue((y.lp_state.iloc[transition-20:transition+20] == -1).all())

    def test_false_long_pointer_event_is_not_active(self):
        df = synthetic()
        for ch in range(1, 4):
            df[f"p_core->wl[CH{ch}]"] = 500.
            for z in range(5):
                df[f"p_core->emg_psd_mean[CH{ch}][{z}]"] = 6000.
        y, audit = label_recording(df, "STRONG_LONG_PRESS")
        self.assertTrue((y.lp_state == -1).all())
        self.assertEqual(audit["reason"], "no_verified_lp_event")

    def test_short_pointer_glitch_rejected(self):
        df = synthetic()
        df["PointerState"] = 0
        df.loc[400:405, "PointerState"] = 1
        y, audit = label_recording(df, "MID_LONG_PRESS")
        self.assertTrue((y.lp_state == -1).all())
        self.assertEqual(audit["verified_events"], 0)

    def test_disconnected_channels_abstain(self):
        df = synthetic()
        for ch in (2, 3):
            df[f"p_core->wl[CH{ch}]"] = 0.
        y, audit = label_recording(df, "MID_LONG_PRESS")
        self.assertTrue((y.lp_state == -1).all())
        self.assertEqual(audit["reason"], "fewer_than_two_usable_channels")

    def test_missing_pointer_abstains(self):
        y, audit = label_recording(synthetic().drop(columns="PointerState"),
                                   "MID_LONG_PRESS")
        self.assertEqual(audit["reason"], "no_stable_off_reference")
        self.assertTrue((y.lp_state == -1).all())

    def test_missing_psd_abstains(self):
        df = synthetic().drop(columns="p_core->emg_psd_mean[CH2][0]")
        y, audit = label_recording(df, "MID_LONG_PRESS")
        self.assertEqual(audit["reason"], "missing_emg_features")
        self.assertTrue((y.lp_state == -1).all())

    def test_legacy_psd_marker(self):
        df = synthetic()
        normal, _ = label_recording(df, "MID_LONG_PRESS")
        legacy = df.copy(deep=True)
        for ch in range(1, 4):
            for z, count in enumerate((8, 10, 12, 46, 51)):
                key = f"p_core->emg_psd_mean[CH{ch}][{z}]"
                legacy[key] = legacy[key] / count
        legacy["psd_mean_applied"] = 1
        restored, _ = label_recording(legacy, "MID_LONG_PRESS")
        np.testing.assert_array_equal(normal.lp_state, restored.lp_state)

    def test_write_preserves_original_source_and_rows(self):
        df = synthetic(n=650, press=(220, 470), signal=(200, 490))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "DATA_LONG_PRESS"
            source = root / "MID_LONG_PRESS" / "example.csv.gz"
            source.parent.mkdir(parents=True)
            df.to_csv(source, index=False, compression="gzip")
            before = source.read_bytes()
            out = Path(td) / "LABELED"
            self.assertEqual(main(["--input-dir", str(root), "--output-dir", str(out)]), 0)
            dest = out / "MID_LONG_PRESS" / "example.csv.gz"
            self.assertTrue(dest.is_file())
            self.assertEqual(before, source.read_bytes())
            labeled = pd.read_csv(dest)
            self.assertIn(-1, set(labeled.lp_state))
            self.assertTrue(set(labeled.lp_state).issubset({-1, 0, 1}))
            self.assertEqual(labeled.lp_teacher_version.iloc[0], VERSION)
            self.assertEqual(len(df), len(labeled))

    def test_reject_nested_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "SOURCE"
            root.mkdir()
            with self.assertRaises(SystemExit):
                main(["--input-dir", str(root), "--output-dir", str(root/"NESTED")])

    def test_invalid_time_abstains(self):
        y, audit = label_recording(synthetic().drop(columns="Timestamp"),
                                   "MID_LONG_PRESS")
        self.assertEqual(audit["reason"], "invalid_or_missing_timestamps")
        self.assertTrue((y.lp_state == -1).all())

    def test_connected_control_rest_only(self):
        df = synthetic()
        df["PointerState"] = 0
        for ch in range(1, 4):
            df[f"p_core->wl[CH{ch}]"] = 500.
            for z in range(5):
                df[f"p_core->emg_psd_mean[CH{ch}][{z}]"] = 6000. + z * 60
        y, _ = label_recording(df, "CONNECTED")
        self.assertGreater(np.mean(y.lp_state == 1), .8)
        self.assertFalse((y.lp_state == 0).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
