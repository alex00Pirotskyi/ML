#!/usr/bin/env python3
"""Causal and firmware PSD unit tests. Run from repo root."""
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_rest_labeler import compute_labels
from lp_causal_fw_teacher import BIN_COUNTS, VERSION


def make_df(n=1600,step=6,wl=520.):
    rng=np.random.default_rng(3)
    df=pd.DataFrame({"Timestamp":np.arange(n)*step+1000000,
                     "User":["synthetic"]*n,
                     "PointerState":[i%19==0 for i in range(n)]})
    for ch in range(1,4):
        df[f"p_core->wl[CH{ch}]"]=wl*(1+rng.normal(0,.008,n))
        for z in range(5):
            df[f"p_core->emg_psd_mean[CH{ch}][{z}]"] = (
                6000. + 550.*z+rng.normal(0,40,n))
    return df


def scores(df,mode="auto"):
    labels,stats=compute_labels(df,psd_mode=mode)
    for ch in range(1,4):
        names=[f"lp_{x}_ch{ch}" for x in
               ("resting","high_activity","disconnection","unknown")]
        assert np.allclose(labels[names].sum(axis=1),1,atol=2e-5)
        assert all(labels[k].between(0,1).all() for k in names)
    assert labels["lp_state"].isin([0,1]).all()
    return labels,stats


class TestCausalFwTeacher(unittest.TestCase):
    def test_steady_rest_after_causal_calibration(self):
        labels,st=scores(make_df())
        self.assertTrue(st["causal"])
        self.assertEqual(st["version"],VERSION)
        self.assertGreater(labels["lp_state"].iloc[600:].mean(),.90)
        self.assertEqual(labels["lp_state"].iloc[:30].sum(),0)
        self.assertLess(labels["lp_state_weight"].iloc[:30].max(),.03)

    def test_sustained_single_channel_activity_and_return(self):
        df=make_df(n=2100)
        df.loc[600:1350,"p_core->wl[CH1]"]*=3.7
        for z in range(5):
            df.loc[600:1350,f"p_core->emg_psd_mean[CH1][{z}]"]+=6000
        label,_=scores(df)
        self.assertGreater(label["lp_high_activity_ch1"].iloc[760:1150].median(),.6)
        self.assertLess(label["lp_state"].iloc[760:1150].mean(),.1)
        self.assertGreater(label["lp_state"].iloc[1800:].mean(),.8)

    def test_stationary_high_noise_on_one_channel(self):
        df=make_df(n=1800)
        df["p_core->wl[CH2]"]=12500.
        for z in range(5):
            df[f"p_core->emg_psd_mean[CH2][{z}]"]=22000.
        label,_=scores(df)
        self.assertGreater(label["lp_disconnection_ch2"].iloc[1000:].median(),.85)
        self.assertGreater(label["lp_resting_ch1"].iloc[600:].median(),.75)
        self.assertLess(label["lp_state"].iloc[600:].mean(),.02)

    def test_very_quiet_channel_is_unknown(self):
        df=make_df()
        df["p_core->wl[CH3]"]=12.
        label,_=scores(df)
        self.assertGreater(label["lp_unknown_ch3"].iloc[500:].median(),.7)
        self.assertEqual(int(label["lp_state"].iloc[500:].sum()),0)

    def test_recording_starts_active_then_releases(self):
        df=make_df(n=2000)
        df.loc[:650,"p_core->wl[CH1]"]=13000.
        for z in range(5):
            df.loc[:650,f"p_core->emg_psd_mean[CH1][{z}]"]+=11000
        label,_=scores(df)
        self.assertEqual(int(label["lp_state"].iloc[:650].sum()),0)
        self.assertGreater(label["lp_unknown_ch1"].iloc[100:200].median(),.8)
        self.assertGreater(label["lp_state"].iloc[780:930].mean(),.70)
        self.assertGreater(label["lp_state"].iloc[1400:].mean(),.75)

    def test_prefix_invariant_to_all_future_data(self):
        df=make_df(2100)
        df.loc[1100:,"p_core->wl[CH1]"]=24000.
        for z in range(5):
            df.loc[1100:,f"p_core->emg_psd_mean[CH1][{z}]"]=33000.
        early,_=scores(df.iloc[:1100].copy())
        full,_=scores(df)
        first=full.iloc[:1100].reset_index(drop=True)
        pd.testing.assert_frame_equal(early,first,check_exact=True)

    def test_recognize_legacy_post_log_division_by_flag(self):
        df=make_df(1450)
        df.loc[700:950,"p_core->wl[CH1]"]*=3
        for z in range(5):
            df.loc[700:950,f"p_core->emg_psd_mean[CH1][{z}]"]+=5000
        direct,_=scores(df)
        old=df.copy()
        for ch in range(1,4):
            for z in range(5):
                key=f"p_core->emg_psd_mean[CH{ch}][{z}]"
                old[key]=np.round(old[key]/BIN_COUNTS[z])
        old["psd_mean_applied"]=1
        restored,meta=scores(old)
        self.assertTrue(meta["legacy_undo_applied"])
        np.testing.assert_allclose(
            direct["lp_high_activity_ch1"],restored["lp_high_activity_ch1"],
            atol=.012,rtol=0)

    def test_no_psd_is_unknown_not_confident_rest(self):
        df=make_df()
        df=df.drop(columns=[c for c in df if "emg_psd_mean" in c])
        label,_=scores(df)
        self.assertEqual(int(label["lp_state"].sum()),0)
        self.assertTrue((label["lp_unknown_ch1"]>.95).all())

    def test_pointer_irrelevant(self):
        df=make_df()
        first,_=scores(df)
        df["PointerState"]=~df["PointerState"]
        second,_=scores(df)
        pd.testing.assert_frame_equal(first,second)


if __name__=="__main__":
    unittest.main(verbosity=2)
