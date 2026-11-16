"""Exploratory pointer-consistency support for V4 tap candidates, never gold."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class AnchorConfig:
    min_edges: int=7
    max_edges: int=30
    min_lag_ms: float=20
    max_lag_ms: float=180
    lag_step_ms: float=10
    tolerance_ms: float=55
    max_lag_disagreement_ms: float=50
    min_coverage: float=.70
    min_half_coverage: float=.55
    max_median_error_ms: float=32.
    max_ambig_candidate_count: int=1
    min_shape_score: float=.46
    min_fall_fraction: float=.28
    min_activity: float=.06
    max_duration_ms:float=310

def _edges(ts:np.ndarray, val:np.ndarray)->np.ndarray:
    val=np.asarray(val,np.float64)
    if len(val)!=len(ts) or not np.isfinite(val).all():return np.array([])
    high=val>0.5
    return ts[high & ~np.r_[False,high[:-1]]]

def _event_filter(ev, cfg):
    return bool(ev.shape_score>=cfg.min_shape_score and ev.fall_fraction>=cfg.min_fall_fraction and
            ev.activity>=cfg.min_activity and ev.duration_ms<=cfg.max_duration_ms and
            10<=ev.rise_time_ms<=170 and ev.zone_count>=2 and
            ev.provisional in ('TAP_SHAPE','AMBIGUOUS_SHAPE','LOW_SNR'))

def _aligned(candidates, peaks, lag, tol):
    """Disjoint chronological matching: pointer edges must each match at most one EMG episode."""
    used=set();result=[]
    for p in peaks:
        options=sorted((abs(ev.peak_ms-(p-lag)),i) for i,ev in enumerate(candidates)
                       if i not in used and abs(ev.peak_ms-(p-lag))<=tol)
        if options:
            err,i=options[0]
            used.add(i)
            result.append((float(p),int(i),float(err)))
    return result

def _best_fit(candidates, pointers, cfg):
    if len(pointers)<3:return (np.nan,[],0.)
    best=(-1e9,None,[],0.)
    for lag in np.arange(cfg.min_lag_ms,cfg.max_lag_ms+1e-3,cfg.lag_step_ms):
        hits=_aligned(candidates,pointers,lag,cfg.tolerance_ms)
        # Penalize high residual and lag drift away from typical center.
        score=len(hits)*5. - sum(x[2] for x in hits)/max(1,len(hits))*.034
        if score>best[0]:best=(score,lag,hits,len(hits)/len(pointers))
    return best[1],best[2],best[3]

def evaluate(events, ts, pointer, category='TAP', cfg:AnchorConfig=AnchorConfig())->dict[str,Any]:
    ptr=_edges(ts,pointer)
    result={'status':'REJECT','reason':'unknown','pointers':len(ptr),'candidates':len(events),
            'pointer_coverage':0.,'lag_ms':None,'offset_split_error_ms':None,
            'hits':[],'quality_count':0}
    if len(ptr)<cfg.min_edges or len(ptr)>cfg.max_edges:
        result['reason']='pointer_count_invalid';return result
    if np.mean(np.asarray(pointer)>0.5)>.72:
        result['reason']='pointer_stuck';return result
    good=[ev for ev in events if _event_filter(ev,cfg)]
    result['quality_count']=len(good)
    if len(good)<max(4,len(ptr)*.55):
        result['reason']='insufficient_EMG_events';return result
    if len(good)>3.0*len(ptr):
        result['reason']='excessive_EMG_candidates';return result
    # Lock offset separately on the first and second 50% of pointer edges,
    # independently of file-level category. Reject highly variable offsets.
    split=len(ptr)//2
    left,right=ptr[:split],ptr[split:]
    lag_l,_,_= _best_fit(good,left,cfg)
    lag_r,_,_= _best_fit(good,right,cfg)
    if not np.isfinite(lag_l) or not np.isfinite(lag_r):
        result['reason']='insufficient_pointer_edges';return result
    result['offset_split_error_ms']=abs(float(lag_l-lag_r))
    if abs(lag_l-lag_r)>cfg.max_lag_disagreement_ms:
        result['reason']='unstable_pointer_EMG_delay';return result
    lag=float(np.median([lag_l,lag_r]))
    hit=_aligned(good,ptr,lag,cfg.tolerance_ms)
    halves=(_aligned(good,left,lag,cfg.tolerance_ms),_aligned(good,right,lag,cfg.tolerance_ms))
    result['pointer_coverage']=len(hit)/len(ptr)
    result['lag_ms']=lag
    if min(len(halves[0])/len(left),len(halves[1])/len(right))<cfg.min_half_coverage:
        result['reason']='low_coverage_by_half';return result
    if result['pointer_coverage']<cfg.min_coverage:
        result['reason']='low_pointer_coverage';return result
    if np.median([x[2] for x in hit])>cfg.max_median_error_ms:
        result['reason']='high_alignment_error';return result
    # Confirm each pointer has a uniquely interpretable EMG candidate.
    unique=[];ambig=0
    for p,i,err in hit:
        rivals=[ev for ev in good if abs(ev.peak_ms-(p-lag))<cfg.tolerance_ms]
        if len(rivals)>cfg.max_ambig_candidate_count:
            ambig+=1;continue
        e=good[i]
        unique.append({'candidate_peak_ms':float(e.peak_ms),'candidate_onset_ms':float(e.onset_ms),
                       'pointer_ms':float(p),'error_ms':err,'shape_score':float(e.shape_score)})
    if len(unique)<max(4,int(cfg.min_coverage*len(ptr))):
        result['reason']='too_few_unique_matches';return result
    result.update(status='SUPPORTED',reason='stable_unique_alignment',hits=unique,
                  ambiguous_pointer_edges=ambig,uniquely_matched=len(unique))
    return result
