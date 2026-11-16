#!/usr/bin/env python3
"""Safety and schema checks (NOT claims about physical tap accuracy)."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil,tempfile
import numpy as np,pandas as pd
import production_labeler_v4 as p
from tap_event_shape_v3 import extract_events,extract_signals

SOURCE=Path(__file__).parent/'DATA'/'DATA_TAP'
if not SOURCE.is_dir():SOURCE=Path('DATA/DATA_TAP').resolve()

def create_fixture(base):
    root=base/'DATA_TAP'; (root/'TAP').mkdir(parents=True);(root/'FINGER').mkdir()
    target=sorted((SOURCE/'TAP').glob('*DoubleTaps*.csv.gz'))[0]
    finger=sorted((SOURCE/'FINGER').glob('*.csv.gz'))[0]
    a=root/'TAP'/target.name;b=root/'FINGER'/finger.name
    shutil.copyfile(target,a);shutil.copyfile(finger,b)
    return root,a,b

def annot(source,sha,t,reviewer='A',decision='TAP',evidence='synchronized_video',sync=5.):
    return {'source':source,'source_sha256':sha,'onset_rel_ms':t,'decision':decision,
            'reviewer_id':reviewer,'evidence_type':evidence,'sync_error_ms':sync,'notes':'synthetic test annotation, NOT an actual reviewer'}

def write_ann(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=p.ANNOTATION_FIELDS);w.writeheader();w.writerows(rows)

def check_synthetics():
    with tempfile.TemporaryDirectory() as d:
        base=Path(d); root,a,b=create_fixture(base)
        rel=a.relative_to(root).as_posix();sha=p.input_hash(a)
        data=pd.read_csv(a,compression='gzip');S=extract_signals(data);ev=extract_events(S)
        times=sorted(x.peak_ms-S['ts'][0] for x in ev)
        good=[]
        for t in times:
            if 450<t<S['ts'][-1]-S['ts'][0]-300 and len(good)==0 or (len(good)==1 and 100<=t-good[0]<=490):
                good.append(float(t))
                if len(good)==2:break
        if len(good)<2:
            good=[1500.,1800.]
        assert 80<=good[1]-good[0]<=500,good
        good=[round(x,2) for x in good]
        before_sha=p.input_hash(a);before_stat=a.stat()
        # V4 must not create positive training labels when no independent review exists.
        out=base/'candidate'
        result=p.export(root,out,None)
        assert result['gold_events']==0 and result['total_positive_rows']==0
        labels=pd.read_csv(next((out/'TAP').glob('*.labels.csv.gz')))
        assert set(labels.tap_label.unique())=={-1}
        assert (labels.tap_event_id==-1).all()
        neg=pd.read_csv(next((out/'FINGER').glob('*.labels.csv.gz')))
        assert set(neg.tap_label.unique())<= {0,-1}
        assert (neg.tap_label==0).sum()>0
        assert result['failed']==0 and result['files_ok']==2
        assert p.input_hash(a)==before_sha and a.stat().st_mtime_ns==before_stat.st_mtime_ns
        # A single video reviewer cannot create a GOLD positive.
        csvpath=base/'annotation.csv'
        write_ann(csvpath,[annot(rel,sha,good[0])])
        try:p.export(root,base/'single',csvpath,'production')
        except ValueError as e:assert 'ZERO independently' in str(e)
        else:raise AssertionError('single reviewer should fail closed')
        assert not (base/'single').exists()
        assert not (base/'single.stage').exists()
        # Even a perfectly matching pointer edge has no authority to create GOLD.
        try:p.export(root,base/'noann',None,'production')
        except ValueError as e:assert 'annotation' in str(e)
        else:raise AssertionError('production without source of truth must be blocked')
        # Two independent reviewers, synchronized evidence; 2 gold taps -> 1 pair.
        rows=[annot(rel,sha,t,reviewer=r) for t in good for r in ('Reviewer-A','Reviewer-B')]
        write_ann(csvpath,rows)
        prod=p.export(root,base/'production',csvpath,'production')
        assert prod['gold_events']==2,prod
        assert prod['gold_pairs']==1,prod
        assert prod['total_positive_rows']>=4,prod
        lab=pd.read_csv(next((base/'production'/'TAP').glob('*.labels.csv.gz')))
        assert set(lab.tap_label.unique())<= {1,-1}
        assert sorted(lab.loc[lab.tap_label==1,'tap_event_id'].unique().tolist())==[1,2]
        assert set(lab.loc[lab.tap_label==1,'tap_pair_id'])=={1}
        assert set(lab.loc[lab.tap_label==1,'tap_provenance'])=={'VERIFIED_GOLD'}
        assert not (lab.loc[lab.tap_label!=1,'tap_pair_id']!=0).any()
        assert not (lab.loc[lab.tap_label!=1,'tap_event_id']!=-1).any()
        assert (lab.loc[lab.tap_label==-1,'tap_label_weight']==0).all()
        # Physical onsets must not depend on the candidate detector recalling
        # them: independently verified soft taps can still be labeled.
        from production_labeler_v4 import read_annotations,build_labels,ExportPolicy
        no_candidates,_,nostats=build_labels(data,'TAP',read_annotations(csvpath)[rel],
                                            {'status':'REJECT','hits':[],'reason':'none'},
                                            ExportPolicy(),sensor=S,candidates=[])
        assert nostats['gold_events']==2 and int((no_candidates.tap_label==1).sum())>0
        assert set(no_candidates.tap_candidate_id.unique())=={-1}
        # The candidate miner may miss gold; do not discard independent expert truth.
        early=1000.0
        # An out-of-range annotation fails before producing a public dataset.
        bad=[annot(rel,sha,S['ts'][-1]-S['ts'][0]+500,'A'),annot(rel,sha,S['ts'][-1]-S['ts'][0]+500,'B')]
        write_ann(csvpath,bad)
        try:p.export(root,base/'outofrange',csvpath,'production')
        except RuntimeError as e:assert 'Out-of-bounds' in str(e)
        else:raise AssertionError('expected strict physical event bound')
        assert not (base/'outofrange').exists()
        # Tampering with source checksum must fail even if timestamps look valid.
        bad=[annot(rel,'0'*64,good[0],'A'),annot(rel,'0'*64,good[0],'B')]
        write_ann(csvpath,bad)
        try:p.export(root,base/'wronghash',csvpath,'production')
        except RuntimeError as e:assert 'Stale/tampered' in str(e)
        else:raise AssertionError('stale annotations should fail')
        assert not (base/'wronghash').exists()
        # Conflicting independent reviews can never promote positives.
        write_ann(csvpath,[annot(rel,sha,good[0],'A'),annot(rel,sha,good[0],'B',decision='NOT_TAP')])
        try:p.export(root,base/'conflict',csvpath,'production')
        except ValueError as e:assert 'ZERO independently' in str(e)
        else:raise AssertionError('conflicting reviews should fail')
        assert not (base/'conflict').exists()
        # Reject leaking path out of DATA_TAP and untrusted pointer as evidence.
        for badsource in ['../TAP/a.csv.gz','TAP/../../a.csv.gz','/TAP/a.csv.gz','SWIPE/../TAP/a.csv.gz']:
            try:p.validate_relative_path(badsource)
            except ValueError:pass
            else:raise AssertionError('path traversal accepted')
        write_ann(csvpath,[annot(rel,sha,good[0],'A',evidence='pointer'),annot(rel,sha,good[0],'B',evidence='pointer')])
        try:p.read_annotations(csvpath)
        except ValueError:pass
        else:raise AssertionError('Oculus accepted as verified ground truth')
        print('SAFE_LABELING_TESTS_OK',json.dumps({'gold_pair_test':prod['gold_pairs'],
                 'positive_rows_test_only':prod['total_positive_rows'],
                 'negative_rows_test_only':result['total_negative_rows']}),flush=True)

if __name__=='__main__':check_synthetics()
