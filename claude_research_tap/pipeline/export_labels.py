"""Turn the event table into per-row training labels.  No model needed.
usage: export_labels.py <DATA_TAP dir> <events.csv[.gz]> <summary.json> <out dir> [--win A B] [--use-weak] [--promote-double-partners]

Output per recording: <name>.labels.csv.gz with one row per source row, in source order:
  Timestamp        copied from the source
  onset_label      1 inside [onset+A, onset+B] ms of a confirmed gesture, 0 trusted negative, -1 unknown (exclude from loss)
  active_label     1 from sensor onset to (shifted) pointer release, 0 trusted negative, -1 unknown
  gesture_kind     0 none, 1 TAP, 2 SWIPE (folder truth)
  event_id         index of the pointer press in the recording, -1 otherwise
  weight           1.0 strong press; p_real for weaker presses when --use-weak; 0 where both labels are -1
Negative folders (FINGER/WRIST/ARM/BAND/NOTHING) are 0 everywhere."""
import sys, os, json, glob, gzip, hashlib, argparse
import numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('events'); ap.add_argument('summary'); ap.add_argument('out')
ap.add_argument('--win',nargs=2,type=int,default=[0,80]); ap.add_argument('--use-weak',action='store_true'); ap.add_argument('--promote-double-partners',action='store_true')
a=ap.parse_args(); A,B=a.win
EV=pd.read_csv(a.events,low_memory=False); S=json.load(open(a.summary)); T_HI=S['thr_at_neg_false_per_min']['1']; T_ORPH=S['thr_at_neg_false_per_min']['5']
evg=dict(tuple(EV[EV.tier!='NEG'].groupby('name'))); known=set(EV.name.unique())
os.makedirs(a.out,exist_ok=True); tot=dict(files=0,onset_pos=0,onset_neg=0,onset_unk=0,events_pos=0,promoted=0); seen=set()
for path in sorted(glob.glob(os.path.join(a.data,'*','*.csv.gz'))):
    name=os.path.basename(path); cat=os.path.basename(os.path.dirname(path))
    h=hashlib.sha256(gzip.open(path,'rb').read()).hexdigest()
    if h in seen: continue                      # exact duplicate recording
    seen.add(h)
    ts=pd.read_csv(path,usecols=['Timestamp'])['Timestamp'].to_numpy(np.int64); t=(ts-ts[0]).astype(float); n=len(t)
    on=np.zeros(n,np.int8); act=np.zeros(n,np.int8); kind=np.zeros(n,np.int8); eid=np.full(n,-1,np.int32); w=np.ones(n,np.float32)
    if cat in ('TAP','SWIPE'):
        if name not in evg:                       # recording too short to score
            on[:]=-1; act[:]=-1
        else:
            e=evg[name]; P=e[e.tier.isin(['PRESS','EDGE'])]; kd=1 if cat=='TAP' else 2
            good=lambda x:(x.tier=='PRESS') and ((x.score>=T_HI) or (a.use_weak and x.p_real>=0.5))
            m_=lambda lo,hi:(t>=lo)&(t<=hi)
            for x in P.itertuples():
                q=m_(min(x.t_ptr,x.t_on)-150,max(x.t_rel,x.t_off)+150); on[q]=-1; act[q]=-1
            extra=e[e.tier.isin(['ORPHAN','INPRESS'])&(e.score>=T_ORPH)]
            for x in extra.itertuples():
                q=m_(x.t_on-200,x.t_on+250); on[q]=-1; act[q]=-1
            for x in P.itertuples():
                if not good(x): continue
                wt=1.0 if x.score>=T_HI else float(x.p_real)
                q=m_(x.t_on+A,x.t_on+B); on[q]=1; q2=m_(x.t_on,x.t_off); act[q2]=1; kind[q2]=kd; eid[q2]=x.k; w[q|q2]=wt; tot['events_pos']+=1
            if a.promote_double_partners and 'DoubleTaps' in name:
                ok=P[(P.tier=='PRESS')&(P.score>=T_HI)].t_on.values
                for x in extra[extra.score>=T_HI].itertuples():
                    if len(ok) and 150<=np.abs(ok-x.t_on).min()<=450:
                        q=m_(x.t_on+A,x.t_on+B); on[q]=1; q2=m_(x.t_on,x.t_on+120); act[q2]=1; kind[q2]=kd; w[q|q2]=0.7; tot['promoted']+=1
    elif name not in known and cat not in ('FINGER','WRIST','ARM','BAND','NOTHING'): continue
    w[(on==-1)&(act==-1)]=0
    pd.DataFrame(dict(Timestamp=ts,onset_label=on,active_label=act,gesture_kind=kind,event_id=eid,weight=w)).to_csv(os.path.join(a.out,name.replace('.csv.gz','.labels.csv.gz')),index=False)
    tot['files']+=1; tot['onset_pos']+=int((on==1).sum()); tot['onset_neg']+=int((on==0).sum()); tot['onset_unk']+=int((on==-1).sum())
json.dump(dict(args=vars(a),totals=tot),open(os.path.join(a.out,'_export_summary.json'),'w'),indent=1); print(tot)
