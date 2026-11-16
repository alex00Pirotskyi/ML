"""Protect label export, source integrity and corrected PSD quality probe."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_v5_labeler import write_labeled,checksum
from lp_state_v5 import label_recording
from scripts.lp_v3_feature_probe import record, WH, WT, ACC, PI
from test_lp_state_v5 import mock


class V5Export(unittest.TestCase):
    def test_writer_does_not_change_source_csv(self):
        data=mock()
        lab,_=label_recording(data,[.1,.99,.99])
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=root/'SOFT_LONG_PRESS'/'person.csv.gz'
            source.parent.mkdir()
            data.to_csv(source,index=False,compression='gzip')
            before=checksum(source)
            out=root/'OUTPUT'/'SOFT_LONG_PRESS'/'person.csv.gz'
            write_labeled(source,out,lab)
            self.assertEqual(before,checksum(source))
            dest=pd.read_csv(out,compression='gzip')
            self.assertEqual(len(dest),len(data))
            for c in data.columns:self.assertIn(c,dest)
            for c in lab.columns:self.assertIn(c,dest)
            self.assertTrue(set(dest.lp_state.unique()).issubset({-1,0,1}))
            self.assertEqual(len([k for k in lab.columns if k.startswith('lp_signal_quality_ch')]),3)
            self.assertEqual(len([k for k in lab.columns if k.startswith('lp_resting_ch')]),3)
            self.assertEqual(len([k for k in lab.columns if k.startswith('lp_high_activity_ch')]),3)

    def test_probe_legacy_psd_transform_matches_raw(self):
        base=mock()
        n=len(base)
        for ch in (1,2,3):
            base[WH[ch-1]]=np.maximum(base[f'p_core->wl[CH{ch}]']*.8,0)
            base[WT[ch-1]]=np.maximum(base[f'p_core->wl[CH{ch}]']*.7,0)
            for z in range(5):
                base[PI[ch-1][z]]=base[f'p_core->emg_psd_mean[CH{ch}][{z}]']+100
        for c in ACC:base[c]=100
        base['User']='Synthetic subject'
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            rawpath=td/'raw.csv.gz'
            legacy=td/'legacy.csv.gz'
            base.to_csv(rawpath,index=False,compression='gzip')
            d=base.copy()
            for ch in (1,2,3):
                for z,cnt in enumerate((8,10,12,46,51)):
                    for field in (f'p_core->emg_psd_mean[CH{ch}][{z}]',
                                  f'p_core->emg_psd[0][CH{ch}][{z}]'):
                        d[field]=d[field]/cnt
            d['psd_mean_applied']=1
            d.to_csv(legacy,index=False,compression='gzip')
            r0=record(rawpath,'MID_LONG_PRESS')
            r1=record(legacy,'MID_LONG_PRESS')
            for ch in range(3):
                np.testing.assert_allclose(r0['ch'][ch]['feat'],r1['ch'][ch]['feat'],atol=1e-7,rtol=1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
