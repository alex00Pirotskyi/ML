"""Regression tests for one-channel LP state and noisy pointer independence."""
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_state_v4_partial import label_recording


def recording(n=1000, active=(260,700), channels=(0,)):
    rng=np.random.default_rng(14)
    data={"Timestamp":17000000+np.arange(n)*6,"PointerState":np.zeros(n,int)}
    data["PointerState"][slice(*active)]=1
    for ch in range(3):
        if ch in channels:
            wl=520.+rng.normal(0,12,n)
            power=np.full(n,2200.)+rng.normal(0,4,n)
            wl[slice(*active)]*=2.5
            power[slice(*active)]+=950.
        else:
            # Deliberately "disconnected" CH, very noisy/high baseline.
            wl=12000.+rng.normal(0,20,n)
            power=np.full(n,14000.)+rng.normal(0,10,n)
        data[f"p_core->wl[CH{ch+1}]"]=wl
        for zone in range(5):
            data[f"p_core->emg_psd_mean[CH{ch+1}][{zone}]"]=power+zone*50
    return pd.DataFrame(data)


class TestPartialContact(unittest.TestCase):
    def test_one_channel_is_enough(self):
        lab,audit=label_recording(recording())
        self.assertEqual(audit["responsive_channels"],1)
        self.assertGreater(np.mean(lab.lp_state.iloc[360:600]==0),.9)
        self.assertGreater(np.mean(lab.lp_state.iloc[90:190]==1),.9)
        self.assertGreater(np.mean(lab.lp_state.iloc[810:920]==1),.9)
        self.assertTrue(audit["rest_after_first_active"])

    def test_pointer_removed_does_not_change_labels(self):
        df=recording()
        a,_=label_recording(df)
        b,_=label_recording(df.drop(columns="PointerState"))
        np.testing.assert_array_equal(a.lp_state,b.lp_state)

    def test_pointer_false_positives_and_negatives_do_not_change_labels(self):
        original=recording()
        expected,_=label_recording(original)
        corrupt=original.copy()
        corrupt["PointerState"]=0
        corrupt.loc[180:220,"PointerState"]=1 # false positive
        corrupt.loc[480:570,"PointerState"]=1 # true, but most positives missing
        corrupt.loc[850:940,"PointerState"]=1 # false positive
        observed,_=label_recording(corrupt)
        np.testing.assert_array_equal(expected.lp_state,observed.lp_state)

    def test_pointer_shift_does_not_change_labels(self):
        original=recording()
        expected,_=label_recording(original)
        delayed=original.copy()
        delayed["PointerState"]=np.roll(original.PointerState.to_numpy(),80)
        observed,_=label_recording(delayed)
        np.testing.assert_array_equal(expected.lp_state,observed.lp_state)

    def test_no_activity_means_no_invented_hold(self):
        df=recording()
        for ch in range(3):
            arr=df[f"p_core->wl[CH{ch+1}]"].to_numpy().copy()
            if ch==0: arr[:]=520.
            df[f"p_core->wl[CH{ch+1}]"]=arr
            for z in range(5):
                key=f"p_core->emg_psd_mean[CH{ch+1}][{z}]"
                if ch==0:df[key]=2200.+z*50
        lab,audit=label_recording(df)
        self.assertEqual(audit["responsive_channels"],0)
        self.assertTrue((lab.lp_state==-1).all())

    def test_two_corrupt_channels_do_not_block_good_one(self):
        df=recording(channels=(0,1,2))
        full,_=label_recording(df)
        corrupted=df.copy()
        for ch in (1,2):
            corrupted[f"p_core->wl[CH{ch+1}]"]=0
            for z in range(5):
                corrupted[f"p_core->emg_psd_mean[CH{ch+1}][{z}]"]=0
        lab,st=label_recording(corrupted)
        self.assertEqual(st["responsive_channels"],1)
        self.assertGreater(np.mean(lab.lp_state.iloc[360:600]==0),.9)
        self.assertGreater(np.mean(full.lp_state.iloc[360:600]==0),.9)

    def test_legacy_psd_conversion_does_not_change_labels(self):
        raw=recording()
        a,_=label_recording(raw)
        legacy=raw.copy()
        for ch in range(1,4):
            for z,count in enumerate([8,10,12,46,51]):
                key=f"p_core->emg_psd_mean[CH{ch}][{z}]"
                legacy[key]=legacy[key]/count
        legacy["psd_mean_applied"]=1
        b,_=label_recording(legacy)
        np.testing.assert_array_equal(a.lp_state,b.lp_state)

    def test_missing_timestamps_means_unknown(self):
        lab,stat=label_recording(recording().drop(columns="Timestamp"))
        self.assertEqual(stat["reason"],"invalid_timing")
        self.assertTrue((lab.lp_state==-1).all())

    def test_mixed_psd_encoding_means_unknown(self):
        df=recording()
        df["psd_mean_applied"]=0
        df.loc[500:,"psd_mean_applied"]=1
        lab,stat=label_recording(df)
        self.assertEqual(stat["reason"],"mixed_psd_provenance")
        self.assertTrue((lab.lp_state==-1).all())

    def test_noisy_static_channel_is_unknown(self):
        df=recording()
        for ch in range(3):
            df[f"p_core->wl[CH{ch+1}]"]=16000
            for z in range(5):
                df[f"p_core->emg_psd_mean[CH{ch+1}][{z}]"]=15000
        lab,stat=label_recording(df)
        self.assertEqual(stat["responsive_channels"],0)
        self.assertTrue((lab.lp_state==-1).all())


if __name__=="__main__":
    unittest.main(verbosity=2)
