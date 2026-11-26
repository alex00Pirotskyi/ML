"""Synthetic contract tests for gold_labeler; no repository data needed."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gold_labeler import ACC, CH, REQUIRED_EVENTS, Policy, correct_recording, read_spec, run


def fixture(kind=1, name="TAP/sample_DoubleTaps_240bpm.csv.gz"):
    t = np.arange(0, 5000, 10, dtype=np.int64)
    raw = pd.DataFrame({"Timestamp": t+1000000, "PointerState": 0})
    for col in ACC:
        raw[col] = 100
    labels = pd.DataFrame({"Timestamp": raw.Timestamp, "gesture_kind": kind,
                           "tap_label": -1, "tap_active": -1, "tap_event_id": -1,
                           "hold_label": 0, "n_ch_off": 0})
    for col in CH:
        labels[col] = 0
    ev = pd.DataFrame(columns=REQUIRED_EVENTS)
    if kind == 1:
        q = (t >= 1200) & (t < 1260)
        labels.loc[q, "tap_label"] = 1
        extent = (t >= 1200) & (t < 1700)
        labels.loc[extent, "tap_active"] = 1
        labels.loc[extent, "tap_event_id"] = 7
        raw.loc[extent, "PointerState"] = 1
        labels.loc[(t >= 1430) & (t <= 1480), CH[0]] = 1
        ev = pd.DataFrame([
            dict(recording=name, type="tap_press", tier="PRESS", t_on=1200,
                 t_ptr=1200, t_rel=1700, score=.98, labeled=1, k=7, t_off=1700),
            dict(recording=name, type="tap_sensor_only", tier="INPRESS", t_on=1450,
                 t_ptr=np.nan, t_rel=np.nan, score=.99, labeled=-1, k=-1, t_off=1600),
        ])
    return t, raw, labels, ev


class GoldContracts(unittest.TestCase):
    def setUp(self):
        self.name = "TAP/sample_DoubleTaps_240bpm.csv.gz"
        self.policy = Policy(.96)

    def test_merged_pointer_recovers_distinct_event_and_id(self):
        t, raw, base, ev = fixture(name=self.name)
        before = base.copy(deep=True)
        events_before = ev.copy(deep=True)
        out, events, report = correct_recording(raw, base, ev, self.name, self.policy)
        q = (t >= 1450) & (t < 1510)
        self.assertTrue((out.loc[q, "tap_label"] == 1).all())
        self.assertTrue((out.loc[q, "tap_event_id"] == 8).all())
        self.assertEqual(events.iloc[1].labeled, 1)
        self.assertEqual(events.iloc[1].k, 8)
        self.assertTrue(pd.isna(events.iloc[1].t_off))
        self.assertEqual(report["recovered_events"], 1)
        self.assertEqual(report["recovered_with_extra_emg_activity"], 1)
        pd.testing.assert_frame_equal(base, before)
        pd.testing.assert_frame_equal(ev, events_before)
        self.assertTrue((out.loc[base.tap_label == 1, "tap_label"] == 1).all())

    def test_240_bpm_not_only_double_filename(self):
        name = "TAP/sample_SoftTap_240bpm.csv.gz"
        _, raw, base, ev = fixture(name=name)
        self.assertEqual(correct_recording(raw, base, ev, name, self.policy)[2]["recovered_events"], 1)
        name2 = "TAP/sample_SoftTap_60bpm.csv.gz"
        ev["recording"] = name2
        self.assertEqual(correct_recording(raw, base, ev, name2, self.policy)[2]["recovered_events"], 0)

    def test_weak_candidate_and_ringing_not_promoted(self):
        _, raw, base, ev = fixture(name=self.name)
        ev.loc[1, "score"] = .7
        self.assertEqual(correct_recording(raw, base, ev, self.name, self.policy)[2]["recovered_events"], 0)
        ev.loc[1, ["score", "t_on"]] = [.99, 1300]
        self.assertEqual(correct_recording(raw, base, ev, self.name, self.policy)[2]["recovered_events"], 0)

    def test_never_chain_promoted_anchors(self):
        _, raw, base, ev = fixture(name=self.name)
        extra = ev.iloc[1].copy()
        extra["t_on"] = 1700  # 250 after recovered but 500 from original
        extra["score"] = .98
        ev = pd.concat([ev, extra.to_frame().T], ignore_index=True)
        self.assertEqual(correct_recording(raw, base, ev, self.name, self.policy)[2]["recovered_events"], 1)

    def test_known_zero_and_bad_contact_not_overwritten(self):
        t, raw, base, ev = fixture(name=self.name)
        q = (t >= 1450) & (t < 1510)
        base.loc[q, "tap_label"] = 0
        out, _, report = correct_recording(raw, base, ev, self.name, self.policy)
        self.assertEqual(report["recovered_events"], 0)
        self.assertTrue((out.loc[q, "tap_label"] == 0).all())
        base.loc[q, "tap_label"] = -1
        base.loc[q, CH[:2]] = 2
        base.loc[q, "n_ch_off"] = 2
        out, _, report = correct_recording(raw, base, ev, self.name, self.policy)
        self.assertEqual(report["recovered_events"], 0)
        self.assertTrue((out.loc[q, "tap_label"] == 0).all())

    def test_negative_folder_and_hold_preserved(self):
        for name, kind in (("FINGER/fist.csv.gz", 0), ("LONG_PRESS/hold.csv.gz", 3)):
            t, raw, base, ev = fixture(kind=kind, name=name)
            if kind == 3:
                base.loc[(t >= 1200) & (t < 2800), ["hold_label", "tap_label"]] = [1, 0]
            out, _, _ = correct_recording(raw, base, ev, name, Policy(.96, mode="both"))
            if kind == 0:
                self.assertTrue((out.tap_label == 0).all())
            else:
                np.testing.assert_array_equal(out.tap_label, base.tap_label)
                np.testing.assert_array_equal(out.hold_label, base.hold_label)

    def test_background_restored_only_outside_guards(self):
        t, raw, base, ev = fixture(name=self.name)
        out, _, report = correct_recording(raw, base, ev, self.name, Policy(.96, mode="background"))
        self.assertGreater(report["restored_quiet_rows"], 0)
        self.assertEqual(int(out.loc[t == 1450, "tap_label"].iloc[0]), -1)
        self.assertEqual(int(out.loc[t == 3000, "tap_label"].iloc[0]), 0)
        self.assertEqual(int(out.loc[t == 100, "tap_label"].iloc[0]), -1)
        self.assertEqual(report["recovered_events"], 0)

    def test_extra_emg_is_optional_stricter_ablation(self):
        _, raw, base, ev = fixture(name=self.name)
        base.loc[:, CH] = 0
        self.assertEqual(correct_recording(raw, base, ev, self.name, self.policy)[2]["recovered_events"], 1)
        self.assertEqual(correct_recording(raw, base, ev, self.name, Policy(.96, require_emg_active=True))[2]["recovered_events"], 0)

    def test_mismatched_timestamp_and_person_spec_fail(self):
        _, raw, base, ev = fixture(name=self.name)
        base.loc[0, "Timestamp"] += 1
        with self.assertRaisesRegex(ValueError, "Timestamp mismatch"):
            correct_recording(raw, base, ev, self.name, self.policy)
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d)/"spec.json"
            spec.write_text(json.dumps({"split_policy":"person", "feature_families":["ACC","EMG"], "run_id":"x", "strong_score":.96}))
            with self.assertRaisesRegex(ValueError, "gesture_aware_file"):
                read_spec(spec)

    def test_cli_output_and_frozen_reference(self):
        _, raw, base, ev = fixture(name=self.name)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, labels, out = root/"raw", root/"base", root/"out"
            raw_path = source/self.name
            raw_path.parent.mkdir(parents=True)
            raw.to_csv(raw_path, index=False)
            sidecar = labels/self.name.replace(".csv.gz", ".labels.csv.gz")
            sidecar.parent.mkdir(parents=True)
            base.to_csv(sidecar, index=False)
            events_path = labels/"_events.csv.gz"
            ev.to_csv(events_path, index=False)
            (labels/"_summary.json").write_text(json.dumps({"totals":{"failed":0,"files":1}, "recordings":{self.name:{}}}))
            spec = {"run_id":"test", "split_policy":"gesture_aware_file", "feature_families":["ACC","EMG"], "strong_score":.96}
            stats = run(source, labels, out, spec, self.policy)
            self.assertEqual(stats["recovered_events"], 1)
            self.assertEqual(events_path.read_bytes(), (out/"_events_reference.csv.gz").read_bytes())
            self.assertTrue((out/"_gold_report.json").exists())
            with self.assertRaises(ValueError):
                run(source, labels, out, spec, self.policy)


if __name__ == "__main__":
    unittest.main()
