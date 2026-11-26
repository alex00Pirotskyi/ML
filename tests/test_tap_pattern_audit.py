import unittest

import numpy as np
import pandas as pd

from claude_labeling import audit_tap_patterns as audit
from claude_labeling import label_gestures as teacher


class MotionModel:
    spec = {"thr_at_neg_false_per_min": {"1": 0.5}}

    def predict(self, window, group):
        # Detect a decrease from the center value to the value 80 ms later.
        center = np.where(teacher.O_A == 0)[0][0] * 10
        later = np.where(teacher.O_A == 8)[0][0] * 10
        return (window[:, center] - window[:, later] > 1).astype(float), 0


class PatternAuditTest(unittest.TestCase):
    def test_shape_dependent_peak_without_pointer(self):
        df = pd.DataFrame(0., index=np.arange(120), columns=teacher.TAP_COLS)
        df["Timestamp"] = df.index * 10
        df.loc[40:52, teacher.TAP_COLS[0]] = 1000
        rows, counts = audit.audit_recording(df, MotionModel())
        self.assertEqual(counts["candidates"], 1)
        self.assertEqual(counts["flat_candidates"], 0)
        self.assertTrue(rows[0]["flat_below"])
        self.assertTrue(rows[0]["after_below"])


if __name__ == "__main__":
    unittest.main()
