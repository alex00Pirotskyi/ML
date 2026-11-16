#!/usr/bin/env python3
"""Synthetic safety regression tests; no physical-event accuracy claims."""
from __future__ import annotations
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from tap_event_shape_v3 import extract_events, extract_signals, label_recording

def recording(centers=(2000,2200),amps=(1.,.35),width=24,dt=10,pointer="none"):
    t=np.arange(0,4400,dt,dtype=float)
    cols={"Timestamp":1782000000000.+t,"PointerState":np.zeros(len(t)),
          "Metronome":np.zeros(len(t)),"TriggerPressure":np.zeros(len(t))}
    if pointer=="stuck":cols["PointerState"][:]=1
    if pointer=="delayed":
        for c in centers:cols["PointerState"][(t>=c+85)&(t<c+150)]=1
    envelope=sum(a*np.exp(-.5*((t-c)/width)**2) for c,a in zip(centers,amps))
    for ch in [1,2,3]:
        for zone in range(5):
            cols[f"p_core->emg_psd[0][CH{ch}][{zone}]"]=(
                450+60*ch+25*zone+(650 if ch==1 else 280)*envelope)
        cols[f"p_core->wl[CH{ch}]"]=250.+500.*envelope
    return pd.DataFrame(cols)

def match(pred,truth,tol=90):
    pairs=sorted((abs(a-b),i,j) for i,a in enumerate(pred)
                 for j,b in enumerate(truth) if abs(a-b)<=tol)
    used_pred=set();used_true=set()
    for _,i,j in pairs:
        if i not in used_pred and j not in used_true:
            used_pred.add(i);used_true.add(j)
    return len(used_pred),len(pred)-len(used_pred),len(truth)-len(used_true)

def test_no_false_double():
    for width,amp in product([18,24,40,60,90,125],[.25,.35,.65,1.,2.]):
        df=recording(centers=(2000,),amps=(amp,),width=width,dt=5)
        ev=extract_events(extract_signals(df))
        near=[e for e in ev if 1700<e.peak_ms-1782000000000<2600
              and e.provisional=="TAP_SHAPE"]
        assert len(near)<=1,(width,amp,[(x.peak_ms,x.provisional) for x in near])

def test_rearm_soft_soft():
    for gap in (90,120,180,250,400):
        df=recording(centers=(2000,2000+gap),amps=(.35,.35))
        ev=extract_events(extract_signals(df))
        ts=[e.peak_ms-1782000000000 for e in ev if e.provisional=="TAP_SHAPE"]
        n,fp,fn=match(ts,[2000,2000+gap])
        assert n==2 and fp==0 and fn==0,(gap,ts)
        paired=[e for e in ev if e.pair_id>0 and e.provisional=="TAP_SHAPE"]
        assert len(paired)==2 and paired[0].pair_id==paired[1].pair_id,(gap,paired)
        labels,_,qc=label_recording(df,"TAP","test_DoubleTaps.csv.gz")
        assert qc["double_pair_candidates"]==1
        assert int(labels.tap_pair_id.max())==1
        assert "tap_candidate_id" in labels.columns

def test_external_cues_not_features():
    df=recording(pointer="stuck")
    before=extract_events(extract_signals(df))
    df["PointerState"]=0
    after=extract_events(extract_signals(df))
    assert [e.peak_ms for e in before]==[e.peak_ms for e in after]

def test_negative_classes():
    df=recording()
    for category in ("SWIPE","FINGER","WRIST","ARM","BAND","NOTHING"):
        labels,events,qc=label_recording(df,category)
        assert not np.any(labels.tap_label==1)

def synthetic_benchmark():
    patterns={"soft_soft":(.35,.35),"strong_soft":(1.,.25),
              "soft_strong":(.25,1.),"strong_strong":(1.,1.),
              "mid_mid":(.65,.65)}
    tp=fp=fn=two=0;total=0
    for pattern,gap,width,dt in product(patterns,[70,90,120,180,250,400],
                                         [18,24,40,60],[5,10,20,30]):
        df=recording(centers=(2000,2000+gap),amps=patterns[pattern],
                     width=width,dt=dt)
        events=extract_events(extract_signals(df))
        pred=[e.peak_ms-1782000000000 for e in events
              if e.provisional=="TAP_SHAPE"]
        t,f,n=match(pred,[2000,2000+gap])
        tp+=t;fp+=f;fn+=n;total+=1;two+=int(t==2 and f==0)
    result={"cases":total,"tp":tp,"fp":fp,"fn":fn,"exact_two":two,
            "recall":tp/(tp+fn),"F1":2*tp/(2*tp+fp+fn),
            "note":"SIMULATED Gaussian pulses only; NOT human tap performance"}
    print(json.dumps(result),flush=True)
    assert total==480
    assert tp>=500
    assert fp<=20
    return result

if __name__=="__main__":
    test_no_false_double()
    test_rearm_soft_soft()
    test_external_cues_not_features()
    test_negative_classes()
    synthetic_benchmark()
    print("ALL_V3_SYNTHETIC_CHECKS_PASSED",flush=True)
