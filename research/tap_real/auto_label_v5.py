#!/usr/bin/env python3
"""Automatic, recording-supervised TAP+SWIPE weak-label builder (Mudra).

Recording-level facts supplied by the experimenter:
- TAP and SWIPE recordings CONTAIN the intended gestures.
- FINGER, WRIST, ARM, BAND, NOTHING contain NO TAP or SWIPE gestures.

Temporal positive labels are model-derived SILVER pseudo-labels, not independently
verified physical event ground truth. The source class never enters the student
model features, but it can gate the offline labeler's supervised decoder.

Two independent stages:
  fit: train/evaluate event-shape teacher with participant-block OOF scoring
  export: use folder truth + OOF teacher + pointer/EMG onset matching to emit
          0/1/-1 labels, event IDs, pair IDs, weights and provenance.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold

VERSION='5.0.0-research'
POSITIVE={'TAP','SWIPE'}
NEGATIVE={'FINGER','WRIST','ARM','BAND','NOTHING'}
ALLOWED=POSITIVE|NEGATIVE
META={'file','category','tap','person','double','peak_ms','shape_pass','session_seconds','clip_frac'}
PERIODS=['Timestamp','PointerState','Metronome']
GROUP_SPLITS=5
RANDOM_SEED=20261116


def sha256(path:Path,uncompressed:bool=False)->str:
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if uncompressed and path.suffix=='.gz' else path.open('rb')) as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_events(path:Path)->tuple[pd.DataFrame,list[str]]:
    data=pd.read_csv(path,compression='infer')
    needed=META-{'clip_frac','double','tap','shape_pass'}
    if not needed.issubset(data.columns):raise ValueError(f'Event feature schema missing {needed-set(data.columns)}')
    if not set(data.category.unique())<=ALLOWED:raise ValueError('Unsupported category in features')
    if not (data.file.str.split('/').str[0]==data.category).all():raise ValueError('Inconsistent category/file source')
    # Allowed models consume only logged EMG/ACC/WL event shapes; exclude subject,
    # class, recording metadata, pointer, external triggers, tempo and file time.
    columns=[c for c in data.columns if c not in META]
    if set(columns)&{'PointerState','Metronome','TriggerPressure','User','class_id'}:
        raise ValueError('Forbidden feature leakage')
    if not all(pd.api.types.is_numeric_dtype(data[c]) for c in columns):
        raise ValueError('All model features must be numeric')
    return data,columns


def _teacher():
    return LGBMClassifier(objective='multiclass',num_class=3,n_estimators=140,
       learning_rate=.075,num_leaves=15,max_depth=6,min_child_samples=50,
       reg_lambda=18.0,verbosity=-1,n_jobs=3,random_state=RANDOM_SEED)


def _weights(categories:np.ndarray,files:np.ndarray)->np.ndarray:
    # Balance *recordings*, not EMG frames or rare heavy subjects.
    counts=Counter(files)
    result=np.asarray([1./counts[f] for f in files],dtype=float)
    for c in np.unique(categories):result[categories==c]/=result[categories==c].sum()
    return result/result.mean()


def _classes(category):return {'TAP':1,'SWIPE':2}.get(str(category),0)


def _bag_metrics(predictions:pd.DataFrame)->dict:
    group=predictions.groupby('file',sort=False)
    ag=group.agg(category=('category','first'),person=('person','first'),
        seconds=('session_seconds','first'),mean_p=('p_target','mean'),
        max_p=('p_target','max'),n=('p_target','size'))
    ag['positive']=ag.category.isin(POSITIVE).astype(int)
    out={'files':int(len(ag)),
         'AUC_mean':float(roc_auc_score(ag.positive,ag.mean_p)),
         'AP_mean':float(average_precision_score(ag.positive,ag.mean_p))}
    for grp,sub in ag.groupby('category'):
        out[grp]={'files':len(sub),'p_mean':float(sub.mean_p.mean()),'p_median':float(sub.mean_p.median())}
    return out


def fit(features:Path,out:Path,folds:int=5,source_root:Path|None=None)->dict:
    if out.exists():raise ValueError('Output directory must not already exist')
    data,cols=read_events(features)
    y=data.category.map(_classes).to_numpy('int8')
    person=data.person.astype(str).to_numpy()
    file=data.file.astype(str).to_numpy()
    x=data[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy('float32')
    if len(set(person))<folds:raise ValueError('Insufficient independent participant groups')
    print('FIT_START',len(data),len(cols),flush=True)
    prediction=np.full((len(data),3),np.nan,dtype=np.float32)
    splits=[]
    for j,(tr,te) in enumerate(GroupKFold(folds).split(x,y,person),1):
        if set(person[tr])&set(person[te]):raise AssertionError('Participant leakage!')
        model=_teacher();model.fit(x[tr],y[tr],sample_weight=_weights(y[tr],file[tr]))
        prediction[te]=model.predict_proba(x[te]).astype('float32')
        print('FIT_FOLD',j,'done',len(te),flush=True)
        splits.append({'fold':j,'train_events':len(tr),'test_events':len(te),
             'train_people':len(set(person[tr])),'test_people':len(set(person[te])),
             'train_files':len(set(file[tr])),'test_files':len(set(file[te]))})
    if not np.isfinite(prediction).all():raise AssertionError('Missing OOF prediction')
    data_scored=data[['file','person','category','peak_ms','rise_time_ms','fall_time_ms',
                      'fall_fraction','duration_ms','shape_score','activity','session_seconds']].copy()
    data_scored['candidate_ordinal']=data.groupby('file',sort=False).cumcount()+1
    data_scored['p_none']=prediction[:,0]
    data_scored['p_tap']=prediction[:,1]
    data_scored['p_swipe']=prediction[:,2]
    data_scored['p_target']=prediction[:,1]+prediction[:,2]
    data_scored['onset_ms']=data_scored.peak_ms-data_scored.rise_time_ms
    data_scored['end_ms']=data_scored.peak_ms+data_scored.fall_time_ms
    # Candidate-time integrity guard; feature extractor must never invent dates.
    if (data_scored.end_ms<data_scored.onset_ms).any():raise AssertionError('Inverted episode times')
    print('FIT_OOF_SUMMARY',flush=True)
    source_digests=[]
    if source_root is not None:
        source_root=source_root.resolve()
        if source_root.name!='DATA_TAP' or not source_root.is_dir():
            raise ValueError('Source hash binding requires DATA_TAP directory')
        wanted=set(data_scored.file)
        sha_cache={}
        for file in _source_files(source_root):
            rel=file.relative_to(source_root).as_posix()
            h=sha256(file,uncompressed=True)
            if h in sha_cache and rel in wanted:
                raise ValueError('Model training contains duplicate source under a distinct file path: '+rel)
            sha_cache[h]=rel
            if rel in wanted:source_digests.append({'file':rel,'sha256':h})
        found={x['file'] for x in source_digests}
        if found!=wanted:raise ValueError('Training sources absent from DATA_TAP: '+str(sorted(wanted-found)[:6]))
    manifest={'labeler':VERSION,'source_assumption':
       'TAP,SWIPE positive recording bags; FINGER,WRIST,ARM,BAND,NOTHING negative bags',
       'feature_sha256':sha256(features),'source_hash_binding':bool(source_digests),
       'source_sha_count':len(source_digests),'features':cols,'rows':len(data_scored),
       'files':int(data_scored.file.nunique()),'people_aliases':int(data_scored.person.nunique()),
       'folds':splits,'metrics':_bag_metrics(data_scored),
       'scores_interpretation':'Out-of-fold model scores, NOT calibrated true physical onset probabilities'}
    tmp=out.with_name(out.name+'.stage')
    if tmp.exists():raise ValueError('Staging directory already exists')
    tmp.mkdir(parents=True)
    try:
        print('FIT_WRITE_SCORES',flush=True)
        data_scored.to_csv(tmp/'oof_event_scores.csv.gz',index=False,
            compression={'method':'gzip','mtime':0})
        print('FIT_TRAIN_FULL',flush=True)
        model=_teacher();model.fit(x,y,sample_weight=_weights(y,file))
        model.booster_.save_model(str(tmp/'lgbm_teacher_v5.txt'))
        if source_digests:
            pd.DataFrame(source_digests).to_csv(tmp/'source_sha256.csv',index=False)
        (tmp/'training_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
        print('FIT_PUBLISH',flush=True)
        os.replace(tmp,out)
    except BaseException:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    return manifest


def _rising_edges(ts:np.ndarray,x:np.ndarray)->np.ndarray:
    if len(x)!=len(ts):return np.empty(0,float)
    signal=np.isfinite(x)&(x>.5)
    return ts[signal&~np.r_[False,signal[:-1]]]


def _fit_pointer_candidates(g:pd.DataFrame,ptr:np.ndarray,pointer_duty:float)->dict:
    """Fit a single lag on first-half pointer edges, evaluate on held-out second.

    This is *corroboration*, not an independent ground truth proof. Do not
    optimize over every pointer using high-dimensional EMG threshold overfit.
    """
    out={'status':'NO_VALID_POINTER','lag_ms':None,'n_edges':len(ptr),'heldout_coverage':0.,
         'train_coverage':0.,'matched_ids':[], 'median_error_ms':None}
    if len(ptr)<4 or len(ptr)>100 or pointer_duty>=.82:return out
    v=g[(g.p_target>=.36)&(g.p_class>=.13)&(g.shape_score>=.27)&
         (g.rise_time_ms<=450)&(g.fall_fraction>=.20)]
    if len(v)<3:return out
    pp=v.peak_ms.to_numpy(float)
    pp_cls=v.p_class.to_numpy(float)
    pp_tot=v.p_target.to_numpy(float)
    ords=v.candidate_ordinal.to_numpy(int)
    train=ptr[:max(2,len(ptr)//2)]
    test=ptr[max(2,len(ptr)//2):]
    if len(test)<2:return out
    def assign(peaks,lag):
        # Each physiological episode can correspond to only one pointer edge.
        used=set();hits=[]
        for p in peaks:
            dist=np.abs(pp-(p-lag))
            preference=dist/100.+.20*(1.0-pp_tot)+.08*(1.0-pp_cls)
            order=np.argsort(preference)[:8]
            for j in order:
                if int(j) in used or dist[j]>105:continue
                used.add(int(j));hits.append((int(ords[j]),float(dist[j]),float(pp_tot[j])))
                break
        return hits
    best=(-1e9,None,[])
    for lag in np.arange(-300,361,10,dtype=int):
        hits=assign(train,float(lag))
        quality=sum(.60+.40*h[2] - .5*(h[1]/105.) for h in hits)
        # Prefers repeatable matches not arbitrary phase; DO NOT set lag using test half.
        val=quality/len(train)-.07*abs(lag-20.)/300.
        if val>best[0]:best=(val,int(lag),hits)
    lag=best[1]
    trainhits=best[2]
    testhits=assign(test,float(lag))
    # Resolve duplicate assignment across halves (usually disjoint times).
    used=set();matched=[]
    for event_id,error,p in trainhits+testhits:
        if event_id in used:continue
        used.add(event_id);matched.append((event_id,error,p))
    cov_train=len(trainhits)/len(train)
    cov_test=len(testhits)/len(test)
    residual=float(np.median([x[1] for x in matched])) if matched else None
    out.update(lag_ms=lag,train_coverage=cov_train,heldout_coverage=cov_test,
               median_error_ms=residual, matched_ids=[x[0] for x in matched],
               status=('STRONG_ALIGNMENT' if cov_train>=.70 and cov_test>=.70 and residual is not None
                         and residual<=40 and len(matched)>=5 else
                      'CONSISTENT_WEAK' if cov_train>=.42 and cov_test>=.42 and residual is not None
                         and residual<=72 and len(matched)>=4 else 'WEAK_ALIGNMENT'))
    return out


def _quality(e)->bool:
    return bool((e.shape_score>=.24) and (e.fall_fraction>=.20) and
                (e.activity>=.08) and (e.duration_ms<=1500) and
                (8<=e.rise_time_ms<=600))


def _choose(g:pd.DataFrame,pointer:np.ndarray,duty:float,category:str,
            max_events_per_second:float=2.8):
    """Evidence-led, per-recording, one-to-one candidate decoder.

    In positive bags, a trusted folder label guarantees existence of target
    events but NOT that each EMG episode is a target. Preserve that uncertainty.
    """
    data=g.copy().sort_values('peak_ms').reset_index(drop=True)
    data['p_class']=data.p_tap if category=='TAP' else data.p_swipe
    aligned=_fit_pointer_candidates(data,pointer,duty)
    selected=[]
    matched_ids=set(aligned['matched_ids'])
    for row in data.itertuples(index=False):
        if not _quality(row):continue
        why=None
        if row.candidate_ordinal in matched_ids and row.p_target>=.42 and row.p_class>=.13:
            if aligned['status']=='STRONG_ALIGNMENT':why='POINTER_EMG_STRONG'
            elif aligned['status']=='CONSISTENT_WEAK' and row.p_target>=.55 and row.shape_score>=.35:
                why='POINTER_EMG_WEAK'
            elif row.p_target>=.78 and row.p_class>=.35 and row.shape_score>=.55:
                why='MODEL_UNCERTAIN_POINTER'
        elif row.p_target>=.85 and row.p_class>=.44 and row.shape_score>=.56:
            why='MIL_HIGH'
        if why:
            selected.append([row,why])
    # Model scores are weak supervision and may fire repeatedly on the same
    # contraction; suppress overlaps but allow distinct double-tap re-onsets.
    selected=sorted(selected,key=lambda z:(z[1]=='POINTER_EMG_STRONG', z[0].p_class,z[0].p_target),reverse=True)
    kept=[]
    duration=max(.1,(float(data.peak_ms.max())-float(data.peak_ms.min()))/1000)
    limit=max(1,int(math.ceil(duration*max_events_per_second)))
    for row,why in selected:
        if any(abs(row.onset_ms-z[0].onset_ms)<80 for z in kept):continue
        if len(kept)>=limit:continue
        kept.append((row,why))
    if not kept:
        # Positive-bag MIL witness prior: guaranteed activity somewhere in
        # recording. Don't fake an event if even the most plausible episode
        # lacks physiological support. Only low-weight exploratory witnesses.
        feasible=data[(data.p_target>=.40)&(data.shape_score>=.20)&
             (data.duration_ms<=2000)&(data.rise_time_ms>=5)]
        if len(feasible):
            best=feasible.sort_values(['p_class','p_target','shape_score'],ascending=False).iloc[0]
            kept=[(best,'BAG_WITNESS_LOW')]
    return sorted(kept,key=lambda z:z[0].onset_ms),aligned


def _guard_bad_time(ts:np.ndarray)->np.ndarray:
    mask=(ts<ts[0]+170)
    dif=np.diff(ts)
    for i in np.flatnonzero(dif>140):
        mask|=(ts>=ts[i]-30)&(ts<=ts[i+1]+120)
    return mask


def _source_files(root:Path):
    files=sorted(root.rglob('*.csv.gz'))+sorted(root.rglob('*.csv'))
    return [f for f in files if f.parent.name in ALLOWED]


def _decode_one(df:pd.DataFrame,category:str,features:pd.DataFrame,
                review_file:str)->tuple[pd.DataFrame,dict,list]:
    ts=pd.to_numeric(df.Timestamp,errors='coerce').to_numpy('float64')
    if len(ts)<16 or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
        raise ValueError('Nonmonotonic or invalid timestamps')
    n=len(ts)
    y=np.full(n,-1,dtype='int8')
    weight=np.zeros(n,dtype='float32')
    event_id=np.full(n,-1,dtype='int32')
    pair_id=np.zeros(n,dtype='int32')
    candidate_id=np.full(n,-1,dtype='int32')
    kind=np.full(n,0,dtype='int8') # 0 = none; 1 = TAP; 2 = SWIPE
    score=np.zeros(n,dtype='float32')
    corroborated=np.zeros(n,dtype='int8')
    provenance=np.full(n,'UNKNOWN',dtype=object)
    bad=_guard_bad_time(ts)
    if category in NEGATIVE:
        y[:]=0;weight[:]=1;provenance[:]='RECORDING_NEGATIVE'
        selected=[];aligned={'status':'NEGATIVE_BAG','n_edges':0,'matched_ids':[]}
    else:
        if features.empty:raise ValueError('No EMG event candidates for known positive recording')
        if ((features.peak_ms<ts[0]-100)|(features.peak_ms>ts[-1]+100)).any():
            raise ValueError('Features do not match source timestamps (stale artifact)')
        p=pd.to_numeric(df['PointerState'],errors='coerce').fillna(0).to_numpy(float) if 'PointerState' in df else np.zeros(n)
        pointer=_rising_edges(ts,p)
        selected,aligned=_choose(features,pointer,float(np.mean(p>.5)),category)
        k=1 if category=='TAP' else 2
        selected_t=sorted((float(r.onset_ms),float(r.peak_ms),float(r.end_ms),int(r.candidate_ordinal),str(reason),r)
                          for r,reason in selected)
        for eid,(on,peak,end,cid,reason,row) in enumerate(selected_t,1):
            # Distinct event intervals, even for soft/rapid double taps.
            nexton=selected_t[eid][0] if eid<len(selected_t) else float('inf')
            lo=max(on+8.,ts[0]+171.)
            maxafter=100 if category=='TAP' else 220
            hi=min(on+maxafter,peak+(40 if category=='TAP' else 120),end-6.,nexton-18.)
            if hi<=lo:continue
            core=(ts>=lo)&(ts<=hi)&~bad&(event_id<0)
            if int(core.sum())<2:continue
            y[core]=1;event_id[core]=eid;kind[core]=k
            if reason=='POINTER_EMG_STRONG':w=.72+.15*min(1.,float(row.p_target))
            elif reason=='POINTER_EMG_WEAK':w=.42+.14*min(1.,float(row.p_target))
            elif reason=='MODEL_UNCERTAIN_POINTER':w=.30+.14*min(1.,float(row.p_target))
            elif reason=='MIL_HIGH':w=.48+.17*min(1.,float(row.p_target))
            else:w=.15
            weight[core]=w
            score[core]=float(row.p_target)
            provenance[core]='AUTO_'+reason
            corroborated[core]=int(reason.startswith('POINTER_EMG'))
            candidate_id[core]=cid
        # No positive bag frame is called negative: unobserved very soft taps
        # may lie between candidates, and are unknown rather than false zeros.
    # Event labels are invalid near warmup or timestamp discontinuities.
    y[bad]=-1;weight[bad]=0.;event_id[bad]=-1;kind[bad]=0
    score[bad]=0;corroborated[bad]=0;candidate_id[bad]=-1
    provenance[bad]='SIGNAL_UNAVAILABLE'
    if category=='TAP':
        prev=None;next_pair=1
        events=[]
        for eid in sorted(set(event_id[event_id>0])):
            first=np.flatnonzero(event_id==eid)[0]
            events.append((int(eid),float(ts[first])))
        if 'doubletap' in review_file.lower():
            used=set()
            for j in range(len(events)-1):
                id0,t0=events[j];id1,t1=events[j+1]
                if j in used or j+1 in used:continue
                if 65<=t1-t0<=500:
                    pair_id[(event_id==id0)|(event_id==id1)]=next_pair
                    used|={j,j+1};next_pair+=1
    if category in NEGATIVE and (y==1).any():raise AssertionError('Negative bag contains positive label')
    if ((y!=1)&(event_id!=-1)).any():raise AssertionError('Event IDs outside positives')
    if ((y==-1)&(weight!=0)).any():raise AssertionError('Unknown has nonzero weight')
    if ((pair_id!=0)&(y!=1)).any():raise AssertionError('Pair ID outside positives')
    if ((y==1)&(kind==0)).any():raise AssertionError('Missing positive gesture subtype')
    out=pd.DataFrame({'Timestamp':df.Timestamp,'gesture_label':y,'gesture_weight':weight,
       'gesture_kind':kind,'gesture_event_id':event_id,'gesture_pair_id':pair_id,
       'gesture_provenance':provenance,'gesture_candidate_id':candidate_id,
       'gesture_model_score':score,'gesture_pointer_supported':corroborated})
    # Existing TAP/SWIPE student training may expect tap_label=1 for both.
    # Provide alias but state explicitly that this is the COMBINED class.
    out['tap_label']=y;out['tap_label_weight']=weight
    counts={str(k):int(v) for k,v in Counter(provenance[y==1]).items()}
    summary={'source':review_file,'category':category,'rows':n,
       'negative_rows':int((y==0).sum()),'positive_rows':int((y==1).sum()),
       'unknown_rows':int((y==-1).sum()),'positive_events':int(len(set(event_id[event_id>0]))),
       'double_pairs':int(len(set(pair_id[pair_id>0]))),
       'event_provenances':counts,'cue':{k:v for k,v in aligned.items() if k!='matched_ids'},
       'model_candidate_events':len(features)}
    detail=[{'event_id':int(eid),'timestamp_rel_ms':round(float(ts[np.flatnonzero(event_id==eid)[0]]-ts[0]),1),
       'kind':category,'provenance':str(provenance[np.flatnonzero(event_id==eid)[0]]),
       'score':round(float(score[np.flatnonzero(event_id==eid)[0]]),4)} for eid in sorted(set(event_id[event_id>0]))]
    return out,summary,detail


def export(root:Path,score_csv:Path,out:Path,max_files:int|None=None,
           full_rows:bool=False,source_hash_csv:Path|None=None,strict:bool=False)->dict:
    root=root.expanduser().resolve();out=out.expanduser().resolve()
    if root.name!='DATA_TAP' or not root.is_dir():raise ValueError('Input must be DATA_TAP')
    if out==root or root in out.parents or out in root.parents:raise ValueError('Output must be disjoint from source')
    if out.exists():raise ValueError('Output already exists; use a new immutable version')
    if strict and source_hash_csv is None:
        raise ValueError('Strict export requires immutable source SHA-256 manifest')
    expected_hashes={}
    if source_hash_csv is not None:
        h=pd.read_csv(source_hash_csv)
        if not {'file','sha256'}<=set(h.columns) or h.file.duplicated().any():
            raise ValueError('Malformed source hash binding manifest')
        expected_hashes=dict(zip(h.file,h.sha256))
    scored=pd.read_csv(score_csv,compression='infer')
    required={'file','category','candidate_ordinal','peak_ms','onset_ms','end_ms',
              'p_target','p_tap','p_swipe','rise_time_ms','duration_ms','shape_score','fall_fraction','activity'}
    if not required<=set(scored.columns):raise ValueError(f'Score schema missing {required-set(scored.columns)}')
    if not np.isfinite(scored[['peak_ms','onset_ms','end_ms','p_target','p_tap','p_swipe']]).all().all():
        raise ValueError('Nonfinite scores/event timestamps')
    if not (scored.category.str.upper()==scored.file.str.split('/').str[0]).all():
        raise ValueError('Corrupt class/file mapping in score artifact')
    features={key:sub for key,sub in scored.groupby('file',sort=False)}
    files=_source_files(root)
    if max_files:files=files[:max_files]
    staged=out.with_name(out.name+'.stage')
    if staged.exists():raise ValueError('Staging already exists; inspect prior run')
    staged.mkdir(parents=True)
    manifest=[];accepted=[];seen={};errors=[]
    try:
        for idx,f in enumerate(files):
            rel=f.relative_to(root).as_posix();cat=f.parent.name
            entry={'source':rel,'category':cat}
            try:
                before=f.stat();digest=sha256(f,uncompressed=True)
                if digest in seen:
                    othercat,otherfile=seen[digest]
                    if cat!=othercat:raise ValueError(f'Identical CSV under contradictory classes: {otherfile}, {rel}')
                    entry.update(status='DUPLICATE',same_as=otherfile,source_sha256=digest)
                    manifest.append(entry);continue
                seen[digest]=(cat,rel)
                if rel not in features:raise ValueError('No event-feature model scores for source')
                if expected_hashes and expected_hashes.get(rel)!=digest:
                    raise ValueError('Source SHA-256 changed since feature/model extraction')
                keep={'Timestamp','PointerState','Metronome'}
                src=pd.read_csv(f,compression='infer',low_memory=False,
                                usecols=lambda col:col in keep)
                lab,stats,det=_decode_one(src,cat,features[rel],rel)
                if before.st_mtime_ns!=f.stat().st_mtime_ns or before.st_size!=f.stat().st_size:
                    raise ValueError('Source changed during export')
                folder=staged/cat;folder.mkdir(parents=True,exist_ok=True)
                name=f.name.removesuffix('.gz').removesuffix('.csv')
                lab.to_csv(folder/(name+'.labels.csv.gz'),index=False,
                           compression={'method':'gzip','mtime':0})
                if full_rows:
                    raw=pd.read_csv(f,low_memory=False)
                    if len(raw)!=len(lab) or not np.array_equal(raw.Timestamp,lab.Timestamp):
                        raise ValueError('Full-rows original CSV mismatch')
                    raw=pd.concat([raw.reset_index(drop=True),lab.drop(columns=['Timestamp'])],axis=1)
                    raw.to_csv(folder/(name+'.full_labeled.csv.gz'),index=False,
                           compression={'method':'gzip','mtime':0})
                # Detailed event list for traceability.
                pd.DataFrame(det,columns=['event_id','timestamp_rel_ms','kind','provenance','score']).to_csv(
                    folder/(name+'.selected_events.csv'),index=False)
                entry.update(status='OK',source_sha256=digest,**stats)
                accepted.append(entry)
            except Exception as ex:
                entry.update(status='FAILED',error=f'{type(ex).__name__}: {ex}')
                errors.append(entry)
            manifest.append(entry)
            if (idx+1)%250==0:
                print('EXPORT_PROGRESS',idx+1,'/',len(files),'fail',len(errors),flush=True)
        if errors:raise RuntimeError(f'{len(errors)} export failures, first: {errors[0]}')
        output={'version':VERSION,'target_class':{'TAP':1,'SWIPE':1,
                 'FINGER':0,'WRIST':0,'ARM':0,'BAND':0,'NOTHING':0},
           'meaning':'AUTOMATIC recording-supervised pseudo labels, NOT independently verified physical onset truth',
           'files_seen':len(files),'files_ok':len(accepted),
           'duplicate_files':sum(e['status']=='DUPLICATE' for e in manifest),
           'failed_files':len(errors),'sources':str(root),
           'scores_sha256':sha256(score_csv),
           'source_sha256_manifest':sha256(source_hash_csv) if source_hash_csv else None,
           'strict_provenance':strict,'rows_positive':sum(x['positive_rows'] for x in accepted),
           'rows_negative':sum(x['negative_rows'] for x in accepted),
           'rows_unknown':sum(x['unknown_rows'] for x in accepted),
           'events_positive':sum(x['positive_events'] for x in accepted),
           'pairs_double':sum(x['double_pairs'] for x in accepted),
           'positive_bags':sum(x['category'] in POSITIVE for x in accepted),
           'covered_positive_bags':sum(x['category'] in POSITIVE and x['positive_events']>0 for x in accepted),
           'negative_bag_positive_events':sum(x['positive_events'] for x in accepted if x['category'] in NEGATIVE),
           'per_category':{},'provenance_counts':dict(Counter(k for x in accepted for k in
                               [p for p,v in x['event_provenances'].items() for _ in range(v)]))}
        for category in sorted(ALLOWED):
            group=[x for x in accepted if x['category']==category]
            output['per_category'][category]={'files':len(group),
                'covered':sum(x['positive_events']>0 for x in group),
                'event_count':sum(x['positive_events'] for x in group),
                'row_positives':sum(x['positive_rows'] for x in group),
                'pairs':sum(x['double_pairs'] for x in group)}
        if output['negative_bag_positive_events']!=0:raise AssertionError('Negative leakage')
        if output['rows_positive']<=0:raise ValueError('No positive training samples created')
        (staged/'manifest.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in manifest))
        (staged/'summary.json').write_text(json.dumps(output,indent=2,sort_keys=True))
        os.replace(staged,out)
        return output
    except BaseException:
        if staged.exists():shutil.rmtree(staged)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    cmd=ap.add_subparsers(dest='cmd',required=True)
    p=cmd.add_parser('fit');p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--folds',type=int,default=5)
    p.add_argument('--input-dir',type=Path,help='Optional DATA_TAP source for per-file hash binding')
    p=cmd.add_parser('export');p.add_argument('--input-dir',type=Path,default=Path('DATA/DATA_TAP'))
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-files',type=int);p.add_argument('--full-rows',action='store_true')
    p.add_argument('--source-hashes',type=Path)
    p.add_argument('--strict-provenance',action='store_true')
    args=ap.parse_args()
    start=time.monotonic()
    if args.cmd=='fit':result=fit(args.features,args.out_dir,args.folds,args.input_dir)
    else:result=export(args.input_dir,args.scores,args.out_dir,args.max_files,args.full_rows,
                       args.source_hashes,args.strict_provenance)
    print('AUTO_LABEL_V5_RESULT',json.dumps({'seconds':round(time.monotonic()-start,1),**result}),flush=True)

if __name__=='__main__':main()