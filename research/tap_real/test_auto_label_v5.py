#!/usr/bin/env python3
"""Synthetic regression for V5's recording-level TAP/SWIPE class semantics."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from auto_label_v5 import _decode_one,export,POSITIVE,NEGATIVE

T0=1_776_000_000_000.
def recording(offset=0,pointer=True):
    ts=T0+np.arange(0,9000,10,dtype=float)
    ptr=np.zeros(len(ts))
    for t in [2000,3000,4000,5000,6000,7000]:
        if pointer:ptr[(ts>=T0+t+80)&(ts<T0+t+130)]=1
    return pd.DataFrame({'Timestamp':ts,'PointerState':ptr,'Metronome':np.zeros(len(ts))+offset})

def candidates(category='TAP',times=(2000,3000,4000,5000,6000,7000),score=.9):
    return pd.DataFrame([{'file':f'{category}/synthetic.csv.gz','category':category,
        'candidate_ordinal':i+1,'peak_ms':T0+t+45,'onset_ms':T0+t-25,
        'end_ms':T0+t+120,'p_target':score,
        'p_tap':.70 if category=='TAP' else .2,
        'p_swipe':.75 if category=='SWIPE' else .2,
        'shape_score':.86,'fall_fraction':.8,'activity':1.5,
        'rise_time_ms':70,'fall_time_ms':75,'duration_ms':145}
        for i,t in enumerate(times)])

def main():
    assert POSITIVE=={'TAP','SWIPE'} and 'SWIPE' not in NEGATIVE
    for cat in sorted(POSITIVE):
        labels,summary,_=_decode_one(recording(),cat,candidates(cat),f'{cat}/synthetic.csv.gz')
        assert summary['positive_events']>=4
        assert (labels.gesture_label==1).sum()>10
        assert (labels.gesture_label==0).sum()==0
        assert set(labels[labels.gesture_label==1].gesture_kind)=={1 if cat=='TAP' else 2}
        assert (labels[labels.gesture_label==-1].gesture_weight==0).all()
    for cat in sorted(NEGATIVE):
        labels,summary,_=_decode_one(recording(),cat,candidates('TAP'),f'{cat}/synthetic.csv.gz')
        assert summary['positive_events']==0
        assert (labels.gesture_label==1).sum()==0
        assert (labels.gesture_label==0).sum()>0
    stuck=recording(pointer=False);stuck.PointerState=1
    weak=candidates(score=.02);weak.p_tap=.01;weak.p_swipe=.01;weak.shape_score=.01
    labelled,summary,_=_decode_one(stuck,'TAP',weak,'TAP/stuck.csv.gz')
    assert summary['positive_events']==0
    times=(1900,2160,3000,3250,4200,4450,5300,5540)
    label,summary,_=_decode_one(recording(),'TAP',candidates(times=times),'TAP/DoubleTaps.csv.gz')
    assert summary['double_pairs']>=1
    for pairid in set(label.gesture_pair_id[label.gesture_pair_id>0]):
        assert label.loc[label.gesture_pair_id==pairid,'gesture_event_id'].nunique()==2
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)/'DATA_TAP';root.mkdir()
        frames=[]
        for idx,cat in enumerate(('TAP','SWIPE','FINGER')):
            (root/cat).mkdir()
            raw=recording(offset=idx)
            raw.to_csv(root/cat/'synthetic.csv.gz',index=False,
                       compression={'method':'gzip','mtime':0})
            frames.append(candidates(cat))
        scores=Path(directory)/'scores.csv.gz'
        pd.concat(frames).to_csv(scores,index=False,compression='gzip')
        dest=Path(directory)/'labels'
        report=export(root,scores,dest)
        assert report['positive_bags']==2 and report['covered_positive_bags']==2
        assert report['negative_bag_positive_events']==0
        assert report['rows_positive']>0
        for cat in ('TAP','SWIPE','FINGER'):
            lab=pd.read_csv(dest/cat/'synthetic.labels.csv.gz')
            assert lab.gesture_label.isin([-1,0,1]).all()
            if cat in POSITIVE:assert (lab.gesture_label==1).any()
            if cat in NEGATIVE:assert not (lab.gesture_label==1).any()
    print('ALL_V5_TESTS_PASSED')

if __name__=='__main__':main()
