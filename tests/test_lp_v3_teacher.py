#!/usr/bin/env python3
"""Physical consistency tests for offline LP V3 teacher (no fake real truth)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_v3_teacher import label_recording,VERSION

def synthetic(n=2000,step=6):
    rg=np.random.default_rng(38)
    df=pd.DataFrame({"Timestamp":np.arange(n)*step+100000,
                     "User":["SYNTH"]*n,"PointerState":np.arange(n)%11==0,
                     "TriggerPressure":np.arange(n)%11==0})
    for c in range(1,4):
        wl=600*(1+rg.normal(0,.02,n))
        df[f"p_core->wl[CH{c}]"]=wl
        df[f"p_core->wl_head[CH{c}]"]=wl/2
        df[f"p_core->wl_tail[CH{c}]"]=wl/2
        for z in range(5):
            p=6000+z*800+rg.normal(0,35,n)
            df[f"p_core->emg_psd_mean[CH{c}][{z}]"]=p
            df[f"p_core->emg_psd[0][CH{c}][{z}]"]=p
    for ax in "XYZ":
        for z in range(6):
            df[f"p_core->acc_rate[{ax}][{z}]"]=1000+rg.normal(0,10,n)
    return df

def activate(df,ch,start,end,scale=4):
    sl=df.index[start:end]
    df.loc[sl,f"p_core->wl[CH{ch}]"]*=scale
    df.loc[sl,f"p_core->wl_head[CH{ch}]"]*=scale
    df.loc[sl,f"p_core->wl_tail[CH{ch}]"]*=scale
    for z in range(5):
        for key in ("p_core->emg_psd_mean","p_core->emg_psd[0]"):
            if key=="p_core->emg_psd_mean":
                name=f"p_core->emg_psd_mean[CH{ch}][{z}]"
            else:
                name=f"p_core->emg_psd[0][CH{ch}][{z}]"
            df.loc[sl,name]+=5000

def run(d,contact=(.02,.02,.02)):
    y,stats=label_recording(d,contact_prob=contact,
                            good_reference_wl=(1800,1800,1800))
    assert len(y)==len(d)
    assert y["lp_state"].isin((0,1)).all()
    assert y["lp_state_weight"].between(0,1).all()
    for c in range(1,4):
        names=[f"lp_{kind}_ch{c}" for kind in ("resting","high_activity",
                    "disconnection","unknown")]
        assert all(y[name].between(0,1).all() for name in names)
        np.testing.assert_allclose(y[names].sum(axis=1),1,atol=1e-5)
    return y,stats

class TestV3(unittest.TestCase):
    def test_quiet_healthy_rest_is_one(self):
        y,s=run(synthetic())
        self.assertEqual(s["version"],VERSION)
        self.assertGreater(y["lp_state"].iloc[500:].mean(),.9)
        self.assertFalse(s["causal"])

    def test_start_rest_contract_and_return_rest(self):
        d=synthetic()
        activate(d,1,400,1200)
        y,s=run(d)
        self.assertTrue(s["channel_details"][0]["credible"])
        self.assertGreater(y["lp_state"].iloc[150:300].mean(),.6)
        self.assertLess(y["lp_state"].iloc[550:1050].mean(),.15)
        self.assertGreater(y["lp_high_activity_ch1"].iloc[550:1050].median(),.60)
        self.assertGreater(y["lp_state"].iloc[1500:1900].mean(),.55)

    def test_start_active_then_relax(self):
        d=synthetic(n=2000)
        activate(d,1,0,650,scale=4.5)
        y,s=run(d)
        self.assertLess(y["lp_state"].iloc[140:500].mean(),.1)
        self.assertGreater(y["lp_state"].iloc[1100:1900].mean(),.6)

    def test_stationary_bad_contact(self):
        d=synthetic()
        d["p_core->wl[CH2]"]=12500
        d["p_core->wl_head[CH2]"]=6250
        d["p_core->wl_tail[CH2]"]=6250
        for z in range(5):
            d[f"p_core->emg_psd_mean[CH2][{z}]"]=23000
            d[f"p_core->emg_psd[0][CH2][{z}]"]=23000
        y,s=run(d,contact=(.02,.98,.02))
        self.assertGreater(y["lp_disconnection_ch2"].median(),.9)
        self.assertEqual(int(y["lp_state"].sum()),0)

    def test_quiet_low_contact_signal_abstains(self):
        d=synthetic()
        d["p_core->wl[CH3]"]=12
        for z in range(5):
            d[f"p_core->emg_psd_mean[CH3][{z}]"]=2
        y,s=run(d,contact=(.02,.1,.85))
        self.assertGreater(y["lp_unknown_ch3"].median(),.75)
        self.assertEqual(int(y["lp_state"].sum()),0)

    def test_contact_suspicion_during_responsive_activity(self):
        d=synthetic()
        activate(d,1,400,1350,scale=5)
        y,s=run(d,contact=(.9,.02,.02))
        self.assertTrue(s["channel_details"][0]["credible"])
        self.assertLess(y["lp_disconnection_ch1"].mean(),.25)
        self.assertGreater(y["lp_unknown_ch1"].mean(),.2)

    def test_pointer_irrelevant(self):
        d=synthetic(n=800)
        activate(d,2,270,470)
        a,_=run(d)
        d["PointerState"]=~d["PointerState"]
        d["TriggerPressure"]=~d["TriggerPressure"]
        b,_=run(d)
        pd.testing.assert_frame_equal(a,b)

    def test_legacy_postlog_transform(self):
        d=synthetic(n=1700)
        activate(d,1,550,900)
        normal,_=run(d)
        legacy=d.copy()
        for c in range(1,4):
            for z,count in enumerate((8,10,12,46,51)):
                for nm in (f"p_core->emg_psd_mean[CH{c}][{z}]",
                           f"p_core->emg_psd[0][CH{c}][{z}]"):
                    legacy[nm]=np.round(legacy[nm]/count)
        legacy["psd_mean_applied"]=1
        restored,_=run(legacy)
        self.assertLess(float(np.mean(
             np.abs(normal["lp_high_activity_ch1"]-restored["lp_high_activity_ch1"])
            )),.1)

if __name__=="__main__":
    unittest.main(verbosity=2)
