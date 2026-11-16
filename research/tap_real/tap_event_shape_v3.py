#!/usr/bin/env python3
"""V3 physics-driven Mudra tap event miner: PSD-zone shape + physiological rearming.

Research only: not a validated clinical/embedded tap detector. Input exclusively
DATA/DATA_TAP. No Oculus/pointer/metronome or class ID in physical features.
Only trusted recording categories gate offline labels, not candidate detection.

Unlike peak-based v1/v2, each trigger initiates a *stateful burst episode*; new
onsets are forbidden until a real decay and renewed rise. Double taps do not
require the first event's slow baseline to recover completely.
"""
from __future__ import annotations
import argparse,hashlib,json,math,gzip
from collections import Counter
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np,pandas as pd

CATEGORIES={'TAP','SWIPE','FINGER','WRIST','ARM','BAND','NOTHING'}
EMG=[[f'p_core->emg_psd[0][CH{ch}][{z}]' for z in range(5)] for ch in range(1,4)]
WL=[f'p_core->wl[CH{ch}]' for ch in range(1,4)]
ACC=[[f'p_core->acc_psd[0][{ax}][{z}]' for z in range(6)] for ax in 'XYZ']

@dataclass
class ShapeEvent:
    onset_ms:float
    peak_ms:float
    end_ms:float
    onset_idx:int
    peak_idx:int
    end_idx:int
    activity:float
    rise:float
    rise_time_ms:float
    fall_time_ms:float
    duration_ms:float
    fall_fraction:float
    compactness:float
    channel_count:int
    zone_count:int
    max_channel_fraction:float
    wl_reactivity:float
    acc_reactivity:float
    shape_score:float
    provisional:str
    decision:str='UNKNOWN'
    event_id:int=-1
    pair_id:int=0
    sample_count:int=0

# All config defaults are exploratory. No physical reference data was used to
# calibrate these values.
@dataclass(frozen=True)
class Config:
    fast_tau_ms:float=17.0
    slow_up_tau_ms:float=850.0
    slow_down_tau_ms:float=230.0
    rise_lag_ms:float=26.0
    activity_on:float=.14
    activity_off:float=.065
    rise_on:float=.10
    rise_retrigger:float=.095
    valley_fraction:float=.40
    valley_absolute:float=.09
    min_tap_gap_ms:float=80.0
    max_burst_ms:float=270.0
    min_rise_fall_ms:float=10.0
    quiet_confirm_ms:float=24.0
    max_gap_ms:float=140.0
    strong_shape_score:float=.70
    strong_activity:float=.18
    warmup_ms:float=170.0


def _read_matrix(df,cols,strict=True):
    if strict and any(c not in df for group in cols for c in group):
        missing=[c for group in cols for c in group if c not in df]
        raise ValueError(f'missing sensor columns: {missing[:4]}')
    if any(c not in df for group in cols for c in group):return None
    arr=np.stack([df[names].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32) for names in cols],axis=1)
    return np.where(np.isfinite(arr),np.clip(arr,0,65534),0.)


def _validate_time(df):
    if 'Timestamp' not in df:raise ValueError('Timestamp column missing')
    ts=pd.to_numeric(df.Timestamp,errors='coerce').to_numpy(np.float64)
    if len(ts)<16 or not np.all(np.isfinite(ts)) or np.any(np.diff(ts)<=0):
        raise ValueError('insufficient or nonmonotone timestamps')
    return ts


def _channels(z):
    """Shape N x (3,5) -> channel activity N x 3 without summing logs."""
    sorted_values=np.sort(z,axis=-1)
    # 3 prominent positive frequency-zone changes must be supported, but one
    # electrode is sufficient. This suppresses single-bin PSD spikes.
    return .35*sorted_values[...,-1]+.65*sorted_values[...,-3:].mean(axis=-1)


def _robust_agg(x):
    x_sorted=np.sort(x,axis=1)
    return .8*x_sorted[:,-1]+.2*x_sorted[:,-2]


def extract_signals(df:pd.DataFrame,cfg:Config=Config()):
    """Timestamp-aware PSD-zone event envelope; no future-signal normalization.

    'activity' is slow-baseline contrast of fast-smoothed log-PSD zones.
    'rise' is 26-ms change of fast zone envelope, permitting a second tap
    while a previous tap's envelope remains above the original baseline.
    """
    ts=_validate_time(df)
    emg=_read_matrix(df,EMG).astype(np.float64)
    n=len(ts)
    fast=emg[0].copy();base=emg[0].copy()
    env=np.zeros(n,np.float64);rise=np.zeros(n,np.float64)
    per_ch=np.zeros((n,3),np.float32);zact=np.zeros((n,3,5),np.float32)
    fast_history=np.empty_like(emg)
    active_history=np.empty_like(emg)
    last_index=0
    for i in range(1,n):
        dt=float(np.clip(ts[i]-ts[i-1],.1,100.))
        fast+=(-math.expm1(-dt/cfg.fast_tau_ms))*(emg[i]-fast)
        tau=np.where(emg[i]>base,cfg.slow_up_tau_ms,cfg.slow_down_tau_ms)
        base+=(-np.expm1(-dt/tau))*(emg[i]-base)
        # Log-coded per-zone PSD; relative differences, no physically invalid
        # sum of independently logarithmically encoded zone powers.
        scale=np.maximum(250.,.24*np.maximum(base,100.))
        act=np.clip((fast-base)/scale,0.,15.)
        fast_history[i]=fast
        active_history[i]=act
        zact[i]=act
        per_ch[i]=_channels(act)
        env[i]=_robust_agg(per_ch[i:i+1])[0]
    fast_history[0]=emg[0]
    for i in range(1,n):
        while last_index<i-1 and ts[last_index]<ts[i]-cfg.rise_lag_ms:
            last_index+=1
        pre=np.maximum(fast_history[last_index],250.)
        # Change in the already-smoothed, per-zone log-power at each channel.
        delta=np.clip((fast_history[i]-fast_history[last_index])/(.26*pre),0.,12.)
        rise[i]=_robust_agg(_channels(delta)[None,:])[0]
    # High frequency ± sample jitter needs an additional 8ms envelope smoothing
    # in timestamps, still causal; raw rise is kept for strong second onset.
    smooth=env.copy()
    for i in range(1,n):
        a=-math.expm1(-float(np.clip(ts[i]-ts[i-1],.1,100.))/9.)
        smooth[i]=smooth[i-1]+a*(env[i]-smooth[i-1])
    wl=_read_matrix(df,[[c] for c in WL],strict=False)
    acc=_read_matrix(df,ACC,strict=False)
    return {'ts':ts,'activity':smooth,'rise':rise,'per_channel':per_ch,'zact':zact,
            'emg_raw':emg,'wl':wl,'acc':acc,'dt_med':float(np.median(np.diff(ts)))}


def _delta_ratio(array,start,peak):
    if array is None:return 0.
    v0=array[start];v1=array[peak]
    if v0.ndim==1:
        increase=np.maximum(0,v1-v0)/np.maximum(25.,.25*np.maximum(v0,1.))
        return float(np.quantile(increase,.75))
    increase=np.maximum(0,v1-v0)/np.maximum(25.,.25*np.maximum(v0,1.))
    return float(np.quantile(increase,.80))


def _event_score(evt):
    peak=evt['activity'];dur=evt['duration_ms'];rise=evt['rise_time_ms']
    fall=evt['fall_fraction'];compact=evt['compactness'];div=evt['zone_count']
    # Shape score deliberately independent of label, time of day, cadence,
    # metronome, pointer or class. No supervised tuning asserted.
    attack=np.clip((rise-6.)/13.,0,1)*np.clip((140.-rise)/60.,0,1)
    compact_shape=np.clip((.82-compact)/.32,0,1)
    decay=np.clip((fall-.22)/.45,0,1)
    compact_duration=np.clip((320.-dur)/160.,0,1)
    zone=np.clip((div-1.)/5.,0,1)
    return float(.24*attack+.26*decay+.20*compact_shape+.20*compact_duration+.10*zone)


def _event_from_state(S,on,peak,end,min_peak,config):
    ts=S['ts'];activity=S['activity'];rise=S['rise'];z=S['zact'];ch=S['per_channel']
    if peak<=on or end<=peak:return None
    seg=activity[on:end+1]
    duration=float(ts[end]-ts[on]);rt=float(ts[peak]-ts[on]);ft=float(ts[end]-ts[peak]);
    if duration<config.min_rise_fall_ms:return None
    a=float(activity[peak]);fall=max(0.,(a-float(activity[end]))/max(a,.0001))
    compact=float(np.trapezoid(seg,ts[on:end+1])/max(1.,duration*a))
    pech=ch[peak];ratio=np.max(pech)/max(1.e-3,np.sum(pech))
    zonecount=int((z[peak]>max(.07,.2*a)).sum())
    evt={'onset_ms':float(ts[on]),'peak_ms':float(ts[peak]),'end_ms':float(ts[end]),
         'onset_idx':int(on),'peak_idx':int(peak),'end_idx':int(end),
         'activity':a,'rise':float(np.max(rise[on:peak+1])),
         'rise_time_ms':rt,'fall_time_ms':ft,'duration_ms':duration,
         'fall_fraction':float(fall),'compactness':compact,
         'channel_count':int(np.sum(pech>=.25*np.max(pech))) if np.max(pech)>0 else 0,
         'zone_count':zonecount,'max_channel_fraction':float(ratio),
         'wl_reactivity':_delta_ratio(S['wl'],on,peak),
         'acc_reactivity':_delta_ratio(S['acc'],on,peak)}
    score=_event_score(evt)
    evt['shape_score']=score
    # Tiny broad transients can be valid but cannot safely be declared real taps.
    if duration>config.max_burst_ms:
        outcome='SUSTAINED'
    elif fall<.20:
        outcome='UNRESOLVED'
    elif rt<config.min_rise_fall_ms or rt>150:
        outcome='NON_TAP_SHAPE'
    elif a<config.strong_activity:
        outcome='LOW_SNR'
    elif score>=config.strong_shape_score:
        outcome='TAP_SHAPE'
    else:
        outcome='AMBIGUOUS_SHAPE'
    evt['provisional']=outcome
    return ShapeEvent(**evt)


def extract_events(S,cfg:Config=Config()):
    """Episode state machine with decay/reversal rearming, not peak NMS.

    States IDLE -> RISE -> FALL -> (REARMED -> RISE for double) or IDLE.
    - In RISE/FALL, a broad contracting activation yields at most one event.
    - A second event requires an observed valley and renewed rise; no tap is
      invented solely because the file name says DoubleTaps or from beat timing.
    - Timestamps gaps reset the state to avoid bridged false events.
    """
    ts=S['ts'];a=S['activity'];dr=S['rise'];n=len(ts)
    state='IDLE';events=[];start=peak=valley=0;low_since=None;armed_since=None;last_event_on=-1e18
    warm=ts[0]+cfg.warmup_ms
    def finalize(end):
        e=_event_from_state(S,start,peak,end,float(a[peak]),cfg)
        if e is not None:events.append(e)
    for i in range(1,n):
        t=ts[i]
        if t<warm:continue
        if ts[i]-ts[i-1]>cfg.max_gap_ms:
            # Don't bridge missing sensor data. Current open event is uncertain.
            if state!='IDLE' and i-1>peak:finalize(i-1)
            state='IDLE';low_since=None;armed_since=None;last_event_on=t
            continue
        if state=='IDLE':
            if dr[i]>=cfg.rise_on and a[i]>=cfg.activity_on and (t-last_event_on)>=cfg.min_tap_gap_ms:
                # Rise onset can precede high-contrast threshold; bound search to
                # 95 ms and last confirmed episode to avoid an earlier tap.
                j=i
                while j>1 and ts[i]-ts[j-1]<=90. and dr[j-1]>.02 and a[j-1]>.035:
                    j-=1
                start=peak=valley=j
                state='RISE';low_since=None;armed_since=None
            continue
        if a[i]>a[peak]:peak=i
        if state=='RISE':
            if a[i]<a[peak]-max(.045,.18*a[peak]):
                valley=i;state='FALL';armed_since=t
        else:
            if a[i]<a[valley]:valley=i
            drop=float(a[peak]-a[valley]);rel=drop/max(.01,a[peak]);
            eligible=(t-ts[start]>=cfg.min_tap_gap_ms and
                      rel>=cfg.valley_fraction and drop>=cfg.valley_absolute and
                      dr[i]>=cfg.rise_retrigger and
                      a[i]>a[valley]+max(.03,.2*a[peak]) and
                      ts[i]-ts[peak]>=cfg.min_tap_gap_ms*.6)
            if eligible:
                # The first episode is closed at the valley, not at second peak.
                e=_event_from_state(S,start,peak,valley,float(a[peak]),cfg)
                if e is not None:events.append(e)
                start=valley
                peak=i
                valley=i
                state='RISE'
                last_event_on=ts[start]
                low_since=None;armed_since=None
                continue
            if a[i]<=cfg.activity_off or (t-ts[start]>cfg.max_burst_ms and dr[i]<cfg.rise_on*.75):
                if low_since is None:low_since=t
                if t-low_since>=cfg.quiet_confirm_ms:
                    finalize(i);last_event_on=ts[start];state='IDLE';low_since=None;armed_since=None
            else:low_since=None
    if state!='IDLE' and n-1>peak:finalize(n-1)
    return events


def label_recording(df,category,recording_name='',cfg:Config=Config()):
    if category not in CATEGORIES:raise ValueError(f'unsupported category {category}')
    S=extract_signals(df,cfg)
    events=extract_events(S,cfg)
    ts=S['ts'];n=len(ts)
    labels=np.full(n,-1,dtype=np.int8);weight=np.zeros(n,dtype=np.float32);ids=np.full(n,-1,dtype=np.int32)
    if category!='TAP':
        labels[:]=0;weight[:]=.80 if category!='NOTHING' else 1.
        for e in events:
            e.decision='HARD_NEGATIVE'
            # Close tap-like morphology negatives get lower weight because
            # selected category may contain involuntary tap-like subevents.
            if e.shape_score>.65:weight[max(0,e.onset_idx-5):min(n,e.end_idx+6)]=.55
    else:
        quiet=S['activity']<.035
        labels[quiet]=0;weight[quiet]=.8
        for e in events:
            lo=max(0,int(e.onset_idx)-5);hi=min(n,int(e.end_idx)+6)
            labels[lo:hi]=-1;weight[lo:hi]=0
        # A TAP file with no confidently observed tap has UNKNOWN labels,
        # preventing undiscovered weak taps from becoming negative truth.
        accepted=0
        for e in events:
            if e.provisional=='TAP_SHAPE':
                e.decision='POSITIVE';e.event_id=accepted+1;accepted+=1
                core=(ts>=e.onset_ms)&(ts<=min(e.onset_ms+75,e.peak_ms+45))
                labels[core]=1;weight[core]=float(np.clip(e.shape_score,.5,.95));ids[core]=e.event_id
            else:e.decision='UNKNOWN'
        if accepted==0:
            labels[:]=-1;weight[:]=0
    # Remove rows around timestamp gaps and warmup including ending events.
    labels[ts<=ts[0]+cfg.warmup_ms]=-1;weight[ts<=ts[0]+cfg.warmup_ms]=0;ids[ts<=ts[0]+cfg.warmup_ms]=-1
    gaps=np.flatnonzero(np.diff(ts)>cfg.max_gap_ms)
    for k in gaps:
        mask=(ts>=ts[k]-20)&(ts<=ts[k+1]+100)
        labels[mask]=-1;weight[mask]=0;ids[mask]=-1
    out=pd.DataFrame({'Timestamp':df.Timestamp,'tap_label':labels,'tap_label_weight':weight,'tap_event_id':ids})
    qc={'rows':n,'events':len(events),'shape_pass_events':int(sum(e.provisional=='TAP_SHAPE' for e in events)),
        'positive_events':int(sum(e.decision=='POSITIVE' for e in events)),'event_rate_hr':round(len(events)*3600000/max(1,ts[-1]-ts[0]),2),
        'positive_rows':int(np.sum(labels==1)),'unknown_rows':int(np.sum(labels==-1)),
        'median_dt_ms':S['dt_med'],'timestamp_gaps_gt_140ms':int(len(gaps)),
        'note':'research pseudo-labels: physical-event precision/recall NOT established'}
    return out,events,qc


def hash_raw(path):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if path.suffix=='.gz' else open(path,'rb')) as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',default='DATA/DATA_TAP');p.add_argument('--out-dir',default='DATA/DATA_TAP_LABELS_SHAPE_V3')
    p.add_argument('--max-files',type=int,default=None);p.add_argument('--only-double-taps',action='store_true')
    p.add_argument('--full-rows',action='store_true')
    p.add_argument('--experimental-positive-labels', action='store_true',
                   help='UNVERIFIED: enable heuristic TAP positives (default outputs only UNKNOWN candidates)')
    p.add_argument('--candidate-only',action='store_true',
                   help='Compatibility alias for conservative default')
    args=p.parse_args()
    root=Path(args.input_dir).expanduser().resolve();out=Path(args.out_dir).expanduser().resolve()
    if not root.is_dir() or root.name!='DATA_TAP':p.error('input must be a DATA_TAP folder')
    if out==root or out in root.parents or root in out.parents:p.error('output must be disjoint from DATA_TAP')
    if out.exists() and any(out.iterdir()):p.error('output must be empty: use a versioned directory')
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.rglob('*.csv.gz'))+sorted(root.rglob('*.csv'))
    files=[f for f in files if f.parent.name in CATEGORIES]
    if args.only_double_taps:files=[f for f in files if f.parent.name=='TAP' and 'doubletap' in f.name.casefold()]
    if args.max_files:files=files[:args.max_files]
    seen={};manifest=[]
    for f in files:
        rel=f.relative_to(root);entry={'source':str(rel),'category':f.parent.name}
        try:
            digest=hash_raw(f);entry['sha256']=digest
            if digest in seen:entry.update(status='DUPLICATE',same_as=seen[digest]);manifest.append(entry);continue
            seen[digest]=str(rel)
            df=pd.read_csv(f,low_memory=False)
            label,events,qc=label_recording(df,f.parent.name,f.name)
            if not args.experimental_positive_labels or args.candidate_only:
                label.loc[label.tap_label==1,'tap_label']=-1
                label.loc[label.tap_label==-1,'tap_label_weight']=0
                label.loc[label.tap_label==-1,'tap_event_id']=-1
                for e in events:
                    if e.decision=='POSITIVE':e.decision='CANDIDATE'
            folder=out/f.parent.name;folder.mkdir(exist_ok=True)
            stem=f.name.replace('.csv.gz','').replace('.csv','')
            target=folder/(stem+'.labels.csv.gz')
            if args.full_rows:label=pd.concat([df.reset_index(drop=True),label.drop(columns=['Timestamp'])],axis=1)
            label.to_csv(target,index=False,compression={'method':'gzip','mtime':0})
            pd.DataFrame([asdict(e) for e in events],columns=list(ShapeEvent.__dataclass_fields__)).to_csv(folder/(stem+'.events.csv'),index=False)
            entry.update(status='OK',**qc)
        except Exception as exc:entry.update(status='FAILED',error=str(exc)[:200]);print('FAILED',rel,exc,flush=True)
        manifest.append(entry)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in manifest))
    counts=dict(Counter(x['status'] for x in manifest))
    print(json.dumps({'files_seen':len(files),'status_counts':counts,'positive_events':sum(x.get('positive_events',0) for x in manifest)}))

if __name__=='__main__':main()