import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from claude_labeling.tap_snapshot import Snapshot, pack
from claude_labeling.tap_only import COLUMNS, label_rows
from claude_labeling.score_tap_reviews import score


class TapReviewTest(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(dict(Timestamp=1000 + np.arange(21) * 10,
                                      tap_label=[0, 1, -1] + [0] * 18, n_ch_off=[2] + [0] * 20))

    def review(self, rows):
        return pd.DataFrame([["TAP/clip.csv", *row, "A+B", "contact-1"] for row in rows],
                            columns=COLUMNS)

    def test_swipe_and_disconnect(self):
        self.assertEqual(label_rows(self.base, "SWIPE", None).tap_label[:3].tolist(), [-1, 0, -1])
        self.assertEqual(label_rows(self.base, "FINGER", None).finger.unique().tolist(), ["unknown"])

    def test_silent_soft_contact(self):
        self.base["tap_label"] = 0
        r = self.review([[60, 60, "tap", "ring", "soft"]])
        out = label_rows(self.base, "TAP", r)
        self.assertTrue((out.loc[6:14, "tap_label"] == 1).all())
        self.assertTrue((out.loc[6:14, "finger"] == "ring").all())
        self.assertEqual(out.tap_label.value_counts().to_dict(), {-1: 12, 1: 9})
        self.assertEqual(out.Timestamp.tolist(), self.base.Timestamp.tolist())
        with self.assertRaises(ValueError):
            label_rows(self.base, "TAP", pd.concat([r, r]))

    def test_adjacent_reviewed_contacts(self):
        r = self.review([[60, 60, "tap", "ring", "soft"], [90, 90, "tap", "little", "soft"]])
        out = label_rows(self.base, "TAP", r)
        self.assertEqual(out.finger.iloc[8:10].tolist(), ["ring", "little"])

    def test_snapshot_checks_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "TAP" / "a.csv"
            src.parent.mkdir()
            self.base.to_csv(src, index=False)
            dest = root / "labels" / "TAP"
            dest.mkdir(parents=True)
            expected = label_rows(self.base, "TAP", None)
            expected.to_csv(dest / "a.csv.labels.csv.gz", index=False)
            archive = root / "snapshot.gz"
            pack(root, dest.parent, archive)
            snapshot = Snapshot(archive)
            result = snapshot.labels(root, "TAP/a.csv")
            pd.testing.assert_frame_equal(expected, result, check_dtype=False)
            src.write_text(src.read_text() + "\n")
            with self.assertRaises(ValueError):
                snapshot.labels(root, "TAP/a.csv")

    def test_one_emission_cannot_hit_two_taps(self):
        r = self.review([[10, 10, "tap", "ring", "soft"], [100, 100, "tap", "index", "normal"],
                         [181, 221, "negative", "unknown", "unknown"]])
        fires = pd.DataFrame(dict(recording=["TAP/clip.csv"] * 2, fire_ms=[70, 200],
                                  finger=["index", "ring"]))
        result = score(r, fires, 40, 60)
        self.assertEqual(result["slices"]["all"]["hits"], 1)
        self.assertEqual(result["slices"]["all"]["negative_ms"], 40)
        self.assertEqual(result["slices"]["all"]["false_on_negative"], 1)
        self.assertEqual(result["finger_confusion"], {"index/index": 1})


if __name__ == "__main__":
    unittest.main()
