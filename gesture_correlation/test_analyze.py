from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as g


class GestureCorrelationTests(unittest.TestCase):
    def test_participant_alias_from_filename(self):
        self.assertEqual(
            g.participant(Path("Alex_left_IndexFinger_Horizontal_Default_60bpm_2026-02-02_14-36-42.csv.gz")),
            "alex",
        )
        self.assertEqual(
            g.participant(Path("AdiEdelshtein_MidTap_Horizontal_Default_60bpm_2026-07-01_11-49-42.csv.gz")),
            "adi",
        )
        self.assertEqual(
            g.participant(Path("Ohad-right-noise_Nothing_Default_NONE_60bpm_2026-01-02_18-31-15.csv.gz")),
            "ohad",
        )

    def test_subtype_parser_skips_hand_side_and_stops_at_protocol_metadata(self):
        self.assertEqual(
            g.subtype(Path("Alex_left_IndexFinger_Horizontal_Default_60bpm_2026-02-02_14-36-42.csv.gz")),
            "IndexFinger",
        )
        self.assertEqual(
            g.subtype(Path("AdiEdelshtein_MidTap_Horizontal_Default_60bpm_2026-07-01_11-49-42.csv.gz")),
            "MidTap",
        )
        self.assertEqual(
            g.subtype(Path("Ohad-right-noise_Nothing_Default_NONE_60bpm_2026-01-02_18-31-15.csv.gz")),
            "Nothing",
        )

    def test_descriptor_schema_includes_every_input_feature(self):
        features = [
            "p_core->acc_psd[0][X][0]",
            "p_core->emg_psd[0][CH1][0]",
            "p_core->cload_lean_8.lw[CH1]",
        ]
        schema = g.schema_for(features, [100, 200])
        self.assertEqual(len(schema["groups"]), len(features))
        self.assertEqual(len(schema["model_base"]), len(schema["model_indices"]))
        self.assertTrue(all(schema["names"][i].startswith(features[j] + "::")
                            for j, group in enumerate(schema["groups"]) for i in group))
        self.assertIn((2, "w200_median_p90"), schema["index"])

    def test_recording_descriptor_is_bag_level_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "TAP" / "Alex_MidTap_Horizontal_Default_60bpm_2026-01-01_10-00-00.csv.gz"
            raw.parent.mkdir(parents=True)
            t = np.arange(0, 3000, 5, dtype=np.int64)
            phase = np.arange(len(t), dtype=np.float32)
            df = pd.DataFrame({
                "Timestamp": t,
                "p_core->acc_psd[0][X][0]": np.sin(phase / 8.0) * 10.0,
                "p_core->emg_psd[0][CH1][0]": np.cos(phase / 11.0) * 4.0,
                "p_core->wl[CH1]": 100.0 + np.sin(phase / 30.0),
            })
            with gzip.open(raw, "wt") as f:
                df.to_csv(f, index=False)
            features = list(df.columns[1:])
            schema = g.schema_for(features, [100, 200])
            rec = {
                "raw": raw, "feature": raw, "root": root,
                "class": "TAP", "subtype": "MidTap", "participant": "alex",
                "sha": g.sha256(raw),
            }
            values, meta = g.describe(rec, features, [100, 200], schema)
            self.assertEqual(values.shape, (len(schema["names"]),))
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(meta["class"], "TAP")
            self.assertEqual(meta["subtype"], "MidTap")
            self.assertNotEqual(meta["window_counts"], "{}")
            self.assertAlmostEqual(meta["duration_ms"], 2995.0)

    def test_exact_archive_dedup_and_conflict_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "TAP" / "a.csv.gz"
            b = root / "TAP" / "b.csv.gz"
            c = root / "FINGER" / "c.csv.gz"
            for p in (a, b, c):
                p.parent.mkdir(parents=True, exist_ok=True)
            payload = b"same archive bytes"
            a.write_bytes(payload)
            b.write_bytes(payload)
            c.write_bytes(payload)
            rows = [
                {"raw": a, "class": "TAP"},
                {"raw": b, "class": "TAP"},
                {"raw": c, "class": "FINGER"},
            ]
            kept, duplicates, conflicts = g.deduplicate(rows)
            self.assertEqual(len(kept), 0)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(len(conflicts), 1)

    def test_paired_participant_effects_and_subtype_effects(self):
        features = ["p_core->acc_psd[0][X][0]", "p_core->emg_psd[0][CH1][0]"]
        schema = g.schema_for(features, [100])
        rng = np.random.default_rng(7)
        X = rng.normal(size=(15, len(schema["names"]))).astype(np.float32)
        meta = pd.DataFrame({
            "class": ["TAP"] * 10 + ["FINGER"] * 5,
            "subtype": ["MidTap"] * 5 + ["SoftTap"] * 5 + ["IndexFinger"] * 5,
            "participant": [f"person{i}" for i in range(5)] * 3,
        })
        paired = g.paired_effects(X, meta, features, schema, [100], min_pairs=3)
        self.assertFalse(paired.empty)
        self.assertTrue((paired["n_paired_participants"] == 5).all())
        subtype = g.subtype_effects(X, meta, features, schema, [100])
        self.assertFalse(subtype.empty)
        self.assertEqual(set(subtype["class"]), {"TAP"})

    def test_within_class_correlation_shape(self):
        features = ["p_core->acc_psd[0][X][0]", "p_core->emg_psd[0][CH1][0]", "p_core->wl[CH1]"]
        schema = g.schema_for(features, [100])
        rng = np.random.default_rng(11)
        X = rng.normal(size=(12, len(schema["names"]))).astype(np.float32)
        meta = pd.DataFrame({"class": ["TAP"] * 6 + ["FINGER"] * 6})
        corr = g.within_class_corr(X, meta, features, schema)
        self.assertEqual(corr.shape, (3, 3))
        self.assertTrue(np.isfinite(corr).all())
        self.assertTrue(np.allclose(np.diag(corr), 1.0))

    def test_bh_adjustment_is_bounded_and_monotonic_in_rank_order(self):
        p = np.asarray([0.04, 0.001, 0.02, np.nan])
        q = g.bh(p)
        self.assertTrue(np.isnan(q[-1]))
        self.assertTrue(np.all(q[:3] >= 0))
        self.assertTrue(np.all(q[:3] <= 1))
        order = np.argsort(p[:3])
        self.assertTrue(np.all(np.diff(q[:3][order]) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
