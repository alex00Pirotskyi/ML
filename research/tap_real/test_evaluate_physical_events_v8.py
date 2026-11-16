import csv
import tempfile
import unittest
from pathlib import Path
from evaluate_physical_events_v8 import match_events, score

class EventAuditTests(unittest.TestCase):
    def test_one_to_one(self):
        self.assertEqual(len(match_events([100, 200], [105, 110, 202], 20)), 2)

    def test_timestamp_tolerance(self):
        self.assertEqual(match_events([100], [181], 80), [])
        self.assertEqual(match_events([100], [180], 80), [(100, 180)])

    def test_optimal_non_greedy(self):
        self.assertEqual(match_events([100, 160], [145, 165], 50), [(100, 145), (160, 165)])

    def test_double_and_hard_negative(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, fields, data in [
                ("truth", ["recording_id","time_ms","event_type","pair_id"], [
                    ["a",100,"tap","p1"], ["a",280,"tap","p1"], ["a",900,"tap",""]]),
                ("pred", ["recording_id","time_ms","score"], [
                    ["a",105,"0.9"], ["a",275,"0.9"], ["a",900,"0.9"],
                    ["b",100,"0.9"], ["b",300,"0.1"]]),
                ("recordings", ["recording_id","duration_s","category","participant","session","contact"], [
                    ["a",10,"TAP","p1","s1","good"], ["b",20,"FINGER","p2","s2","partial"]])
            ]:
                with (d / (name+".csv")).open("w", newline="") as f:
                    w=csv.writer(f);w.writerow(fields);w.writerows(data)
            r=score(d/"truth.csv",d/"pred.csv",d/"recordings.csv",80,0.5)
            self.assertEqual((r["tp"],r["fp"],r["fn"]),(3,1,0))
            self.assertEqual(r["double_pairs_both_recovered"],1)
            self.assertAlmostEqual(r["negative_folder_fp_per_hour"],180.0)
            self.assertEqual(r["by_category"]["FINGER"]["fp"],1)

if __name__ == "__main__":
    unittest.main()
