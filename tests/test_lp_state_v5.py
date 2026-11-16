"""Synthetic contract tests; passing these is NOT accuracy on real people."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_state_v5 import label_recording,VERSION,Settings


def mock(n=1850,events=((200,400),(660,860),(1100,1300)),good_ch=(0,)):
    rng=np.random.default_rng(987)
    data={'Timestamp':1700000000000+np.arange(n)*6,
          'PointerState':np.zeros(n,dtype=int),
          'class_id':np.ones(n,dtype=int),
          'TriggerPressure':np.zeros(n)}
    for a,b in events:
        data['PointerState'][a:b]=1
        data['TriggerPressure'][a:b]=1
    for ch in range(3):
        if ch in good_ch:
            wl=540+rng.normal(0,9,n)
            psd=np.full(n,2400.)+rng.normal(0,8,n)
            for a,b in events:
                wl[a:b]*=2.8
                psd[a:b]+=2100
        else:
            wl=13000+rng.normal(0,30,n)
            psd=np.full(n,14500.)+rng.normal(0,40,n)
        data[f'p_core->wl[CH{ch+1}]']=wl
        for z in range(5):
            data[f'p_core->emg_psd_mean[CH{ch+1}][{z}]']=psd+z*70
    return pd.DataFrame(data)


class V5Tests(unittest.TestCase):
    def test_single_good_channel_survives_two_bad(self):
        labels,st=label_recording(mock(),[.08,.99,.98])
        self.assertEqual(st['strict_accepted_channels'],1)
        self.assertTrue(st['single_channel'])
        self.assertGreater(np.mean(labels.lp_state.iloc[255:355]==0),.9)
        self.assertGreater(np.mean(labels.lp_state.iloc[480:570]==1),.9)
        self.assertGreater(np.mean(labels.lp_state.iloc[1450:1620]==1),.9)
        self.assertEqual(st['reason'],'strict_emg_events')
        self.assertEqual(labels.lp_ch1_event_supported.sum()>0,True)
        self.assertEqual(labels.lp_ch2_event_supported.sum(),0)

    def test_pointer_dropout_fp_fn_shift_have_no_effect(self):
        source=mock()
        a,_=label_recording(source,[.1,.95,.98])
        b,_=label_recording(source.drop(columns=['PointerState','TriggerPressure','class_id']),[.1,.95,.98])
        broken=source.copy()
        broken.PointerState=0
        broken.loc[2:30,'PointerState']=1
        broken.loc[1600:1740,'PointerState']=1
        broken.TriggerPressure=1
        broken.class_id=5
        c,_=label_recording(broken,[.1,.95,.98])
        delayed=source.copy()
        delayed.PointerState=np.roll(source.PointerState.to_numpy(),130)
        d,_=label_recording(delayed,[.1,.95,.98])
        for x in (b,c,d):
            np.testing.assert_array_equal(a.lp_state,x.lp_state)
            for ch in (1,2,3):
                np.testing.assert_array_equal(a[f'lp_high_activity_ch{ch}'],x[f'lp_high_activity_ch{ch}'])

    def test_single_event_fails_closed(self):
        labels,st=label_recording(mock(events=((250,480),)),[.1,.99,.99])
        self.assertEqual(st['reason'],'no_confirmed_event_family')
        self.assertTrue((labels.lp_state==-1).all())
        self.assertTrue((labels.lp_state_weight==0).all())

    def test_untrusted_three_events_fail(self):
        labels,st=label_recording(mock(),[.9,.98,.99])
        self.assertEqual(st['strict_accepted_channels'],0)
        self.assertTrue((labels.lp_state==-1).all())

    def test_three_bad_channels_do_not_invent_training_truth(self):
        labels,st=label_recording(mock(good_ch=()),[.05,.05,.05])
        self.assertEqual(st['strict_accepted_channels'],0)
        self.assertTrue((labels.lp_state==-1).all())

    def test_scores_bounded_channel_independent(self):
        lab,_=label_recording(mock(),[.1,.9,.98])
        for ch in (1,2,3):
            for key in ('resting','high_activity','signal_quality','unknown'):
                c=lab[f'lp_{key}_ch{ch}']
                self.assertGreaterEqual(float(c.min()),0)
                self.assertLessEqual(float(c.max()),1)
            self.assertTrue((lab[f'lp_resting_ch{ch}']+lab[f'lp_high_activity_ch{ch}'] <= lab[f'lp_signal_quality_ch{ch}']+1e-6).all())
            self.assertTrue(((lab[f'lp_target_weight_ch{ch}']>=0)&(lab[f'lp_target_weight_ch{ch}']<=1)).all())
        self.assertTrue(set(lab.lp_state.unique()).issubset({-1,0,1}))
        self.assertTrue(np.all(lab.lp_state_weight[lab.lp_state<0]==0))
        self.assertTrue((lab.lp_teacher_version==VERSION).all())

    def test_quality_proxy_independent_of_activity_label(self):
        lab,st=label_recording(mock(),[.1,.99,.99])
        self.assertGreater(lab.lp_signal_quality_ch1.mean(),0.88)
        self.assertLess(lab.lp_signal_quality_ch2.mean(),.02)
        self.assertTrue((lab.lp_high_activity_ch2==0).all())

    def test_connected_quiet_reference_weak_weight(self):
        df=mock(events=())
        lab,st=label_recording(df,[.1,.9,.9],group='CONNECTED')
        self.assertEqual(st['reason'],'weak_control_rest')
        self.assertGreater(np.mean(lab.lp_state==1),.7)
        self.assertTrue((lab.lp_state_weight[lab.lp_state==1]<=.55+1e-7).all())
        self.assertTrue((lab.lp_high_activity_ch1==0).all())

    def test_missing_sensor_fails_closed(self):
        df=mock().drop(columns=['p_core->emg_psd_mean[CH2][3]'])
        lab,st=label_recording(df,[.1,.9,.9])
        self.assertEqual(st['reason'],'missing_features_or_short')
        self.assertTrue((lab.lp_state==-1).all())

    def test_invalid_quality_inputs_rejected(self):
        with self.assertRaises(ValueError):label_recording(mock(),[.1,.2])
        with self.assertRaises(ValueError):label_recording(mock(),[.1,np.nan,.3])
        with self.assertRaises(ValueError):label_recording(mock(),[-.1,.2,.3])

    def test_legacy_psd_mean_flag_equivalence(self):
        src=mock()
        a,_=label_recording(src,[.1,.99,.99])
        legacy=src.copy(deep=True)
        for ch in (1,2,3):
            for z,cnt in enumerate((8,10,12,46,51)):
                key=f'p_core->emg_psd_mean[CH{ch}][{z}]'
                legacy[key]=legacy[key]/cnt
        legacy['psd_mean_applied']=1
        b,_=label_recording(legacy,[.1,.99,.99])
        np.testing.assert_array_equal(a.lp_state,b.lp_state)

    def test_noncausal_settings_validate(self):
        with self.assertRaises(ValueError):label_recording(mock(),[.1,.9,.9],settings=Settings(min_events=0))


if __name__=='__main__':unittest.main(verbosity=2)
