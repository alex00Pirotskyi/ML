#!/usr/bin/env python3
"""Deterministic V4 failure-mode tests; not physical tap validation."""
from __future__ import annotations
import csv,gzip,hashlib,io,shutil,tempfile
from pathlib import Path
import numpy as np,pandas as pd
import production_labeler_v4 as mod
from tap_event_shape_v3 import EMG,extract_signals

SOURCE=Path(__file__).parent/'DATA'/'DATA_TAP'
if not SOURCE.is_dir():SOURCE=Path('DATA/DATA_TAP').resolve()

def synthetic_df(n=360,dt=10):
    t=1700000000000.+np.arange(n)*dt
    x={'Timestamp':t,'PointerState':np.zeros(n,dtype=int)}
    for ch in range(3):
        for z in range(5):
            x[EMG[ch][z]]=np.full(n,600+50*ch+10*z,dtype=float)
    return pd.DataFrame(x)

def test_consensus():
    p=mod.ExportPolicy()
    src='TAP/synthetic.csv.gz'
    def row(t,reviewer,decision='TAP',evidence='synchronized_video',err=2):
        return mod.EvidenceRow(src,'0'*64,t,decision,reviewer,evidence,err,'independent test')
    gold,abstain=mod.consensus([row(1000,'A'),row(1010,'B')],5000,p)
    assert len(gold)==1 and not abstain
    gold,abstain=mod.consensus([row(1000,'A'),row(1010,'A')],5000,p)
    assert not gold and abstain[0]['reason']=='repeated_reviewer'
    gold,abstain=mod.consensus([row(1000,'A'),row(1010,'B','NOT_TAP')],5000,p)
    assert not gold and abstain[0]['reason']=='conflicting_or_nonpositive_review'
    gold,abstain=mod.consensus([row(1000,'A'),row(1010,'B'),row(1050,'C','TAP')],5000,p)
    assert len(gold)<=1
    gold,abstain=mod.consensus([row(1000,'sensor','TAP','physical_contact_sensor',3.)],5000,p)
    assert len(gold)==1
    gold,abstain=mod.consensus([row(1000,'sensor','TAP','physical_contact_sensor',11.)],5000,p)
    assert not gold and abstain[0]['reason']=='hardware_clock_uncertain'
    gold,abstain=mod.consensus([row(1000,'A'),row(1005,'B'),row(1060,'C'),row(1065,'D')],5000,p)
    assert len(gold)==0 and len(abstain)==2  # sub-80ms gold onsets are ambiguous
    gold,abstain=mod.consensus([row(1000,'A'),row(1005,'B'),row(1100,'C'),row(1110,'D')],5000,p)
    assert len(gold)==2
    gold,abstain=mod.consensus([row(1000,'A'),row(1005,'B'),row(1600,'C'),row(1610,'D')],5000,p)
    assert len(gold)==2  # two independent taps but NOT double pair; gap > 500ms

def test_quality():
    df=synthetic_df()
    S=extract_signals(df)
    # Category negatives are excluded if every channel is frozen.
    out,events,stats=mod.build_labels(df,'FINGER',[],{'status':'REJECT','hits':[]},mod.ExportPolicy(),sensor=S,candidates=[])
    assert set(out.tap_label.unique())=={-1}
    assert stats['healthy_emg_channels']==0
    rows=[mod.EvidenceRow('TAP/synth.csv.gz','0'*64,1000,'TAP',r,'synchronized_video',3,'test') for r in ['A','B']]
    tap,ev,stats=mod.build_labels(df,'TAP',rows,{'status':'REJECT','hits':[]},mod.ExportPolicy(),sensor=S,candidates=[])
    assert set(tap.tap_label.unique())=={-1} and stats['gold_rejected']>=1
    assert (tap.tap_event_id==-1).all()
    # PointerState alone cannot promote a positive, even if stuck high or spoofed.
    df['PointerState']=1
    S=extract_signals(df)
    tap,_,stats=mod.build_labels(df,'TAP',[],{'status':'SUPPORTED','hits':[{'candidate_onset_ms':S['ts'][10]}]},mod.ExportPolicy(),S,[])
    assert not (tap.tap_label==1).any()
    assert stats['gold_events']==0

def test_duplicates_and_atomic():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)/'DATA_TAP';(root/'TAP').mkdir(parents=True)
        src=sorted((SOURCE/'TAP').glob('*.csv.gz'))[0]
        raw=gzip.open(src,'rb').read()
        a=root/'TAP'/'test1.csv.gz';b=root/'TAP'/'test2.csv.gz'
        a.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
        b.write_bytes(gzip.compress(raw,compresslevel=6,mtime=37))
        assert a.read_bytes()!=b.read_bytes() and mod.input_hash(a)==mod.input_hash(b)
        stats=mod.export(root,Path(d)/'negative_only',None,'candidate')
        assert stats['files_ok']==1 and stats['duplicates']==1
        assert stats['gold_events']==0 and stats['total_positive_rows']==0
        assert len(list((Path(d)/'negative_only').rglob('*.labels.csv.gz')))==1
        # Bad input must fail and atomically remove entire stage.
        c=root/'TAP'/'corrupt.csv.gz';c.write_bytes(b'not a gzip stream')
        try:mod.export(root,Path(d)/'corrupt_output',None,'candidate')
        except RuntimeError as exc:assert 'failures' in str(exc)
        else:raise AssertionError('Corruption must stop export')
        assert not (Path(d)/'corrupt_output').exists()
        assert not (Path(d)/'corrupt_output.stage').exists()

def main():
    test_consensus();test_quality();test_duplicates_and_atomic()
    print('V4_EDGE_CASE_TESTS_OK')

if __name__=='__main__':main()
