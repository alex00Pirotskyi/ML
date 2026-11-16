#!/usr/bin/env python3
"""Participant-held-out *teacher imitation* of V5, never physical LP accuracy.

Everything fed to the student is causal and consists of firmware-feasible
current/past WL, logged-PSD, and WL head/tail features. Neither user, folder,
PointerState, future window quantiles, teacher weight nor contact-proxy output
are student inputs.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json
import sys,time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,mean_absolute_error,r2_score,confusion_matrix
import lightgbm as lgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_v5_labeler import INPUT_ROOT,LP_GROUPS,GROUPS,infer_quality
from lp_v3_labeler import train_contact_quality,normalize_person,EXTRA_READ
from lp_state_v5 import label_recording,WL,PSD,BIN_COUNTS,sample_timing


def mat(df,names):
    return np.column_stack([pd.to_numeric(df[n],errors='coerce').fillna(0).to_numpy(float)
                            if n in df else np.zeros(len(df)) for n in names])


def features(df):
    dt,_=sample_timing(df)
    if dt is None:dt=6.
    wl=np.log1p(np.maximum(mat(df,WL),0))
    zone=np.maximum(np.stack([mat(df,c) for c in PSD],axis=1),0)
    instant=np.maximum(np.stack([mat(df,[f'p_core->emg_psd[0][CH{c}][{z}]' for z in range(5)]) for c in (1,2,3)],axis=1),0)
    # Match the offline teacher and contact-quality probe when an old
    # psd_mean.py export divided already-LOG-coded zone values by counts.
    if 'psd_mean_applied' in df:
        marker=pd.to_numeric(df['psd_mean_applied'],errors='coerce')
        if marker.eq(1).all():
            zone=zone*BIN_COUNTS.reshape(1,1,5)
            instant=instant*BIN_COUNTS.reshape(1,1,5)
        elif not marker.eq(0).all():
            raise ValueError('Mixed PSD normalization in student input')
    zone=zone/10000.
    p=np.median(zone,axis=2)
    raw=np.median(instant,axis=2)/10000.
    a=[wl,p,raw]
    for tau in (.25,1.0,3.0):
        alpha=1-np.exp(-dt/(tau*1000))
        smw=pd.DataFrame(wl).ewm(alpha=alpha,adjust=False).mean().to_numpy()
        smp=pd.DataFrame(p).ewm(alpha=alpha,adjust=False).mean().to_numpy()
        a.extend([smw,smp,wl-smw,p-smp])
    wh=mat(df,[f'p_core->wl_head[CH{c}]' for c in (1,2,3)])
    wt=mat(df,[f'p_core->wl_tail[CH{c}]' for c in (1,2,3)])
    a.append(np.log((np.maximum(wh,0)+2)/(np.maximum(wt,0)+2)))
    xx=np.column_stack(a)
    # raw: current WL3, historical logged PSD3, current logged PSD3 = 9
    # compact: all past-only EMAs and WL asymmetry = 48
    return {'raw9':xx[:,:9].astype(np.float32),'compact48':xx.astype(np.float32)}


def main():
    start=time.monotonic()
    root=INPUT_ROOT
    full,models,known,ref,quality=train_contact_quality(root,.7)
    xx=defaultdict(list)
    ys=[];wt=[];people=[];groups=[]
    y9=[];qweight=[]
    for g in GROUPS:
        for p in sorted((root/g).glob('*.csv.gz')):
            df=pd.read_csv(p,compression='gzip',usecols=lambda c:c in EXTRA_READ,low_memory=False)
            bad,src=infer_quality(p,g,df,models,full,known)
            labels,audit=label_recording(df,bad,group=g)
            f=features(df)
            ix=np.unique(np.linspace(0,len(df)-1,min(160,len(df)),dtype=int))
            for k,v in f.items():xx[k].append(v[ix])
            ys.append(labels.lp_state.to_numpy(np.int8)[ix]);wt.append(labels.lp_state_weight.to_numpy(np.float32)[ix])
            y9.append(labels[[f'lp_{name}_ch{ch}' for ch in (1,2,3) for name in ('resting','high_activity','signal_quality')]].to_numpy(np.float32)[ix])
            qweight.append(labels[[f'lp_target_weight_ch{ch}' for ch in (1,2,3)]].to_numpy(np.float32)[ix])
            person=normalize_person(df.User.iloc[0] if 'User' in df else None,p.name)
            people.extend([person]*len(ix));groups.extend([g]*len(ix))
        print('BENCH_EXTRACT',g,flush=True)
    X={k:np.concatenate(v) for k,v in xx.items()}
    y=np.concatenate(ys);w=np.concatenate(wt);yt=np.concatenate(y9);qw=np.concatenate(qweight)
    user=np.array(people);group=np.array(groups)
    splits=list(GroupKFold(n_splits=3).split(X['raw9'],groups=user))
    result={'teacher':'V5 repeated-channel contact proxy','warning':'ONLY pseudo-label imitation. No physical hold/release or quality accuracy.',
            'rows':len(y),'accepted_rows':int((y>=0).sum()),'accepted_fraction':float(np.mean(y>=0)),
            'participants':len(set(user)),'features':{k:v.shape[1] for k,v in X.items()},
            'models':{},'regression':{}}
    for mode in ('raw9','compact48'):
        for config in (('tiny60',60,8),('compact120',120,10)):
            name,trees,leaves=config
            pr=np.full(len(y),np.nan,float);pred=np.full(len(y),-1,np.int8)
            bytes_size=[];notes=[]
            for fold,(tr,te) in enumerate(splits):
                tr=tr[y[tr]>=0];te=te[y[te]>=0]
                if len(np.unique(y[tr]))!=2 or len(np.unique(y[te]))!=2:
                    notes.append({'fold':fold,'warning':'single-class fold'});continue
                model=lgb.LGBMClassifier(n_estimators=trees,learning_rate=.06,
                     num_leaves=leaves,max_depth=5,min_child_samples=70,
                     reg_lambda=4,verbosity=-1,n_jobs=2,random_state=47)
                model.fit(X[mode][tr],y[tr],sample_weight=w[tr])
                score=model.predict_proba(X[mode][te])[:,1]
                pr[te]=score;pred[te]=(score>=.5).astype(np.int8)
                bytes_size.append(len(model.booster_.model_to_string().encode()))
                notes.append({'fold':fold,'test_rows':len(te),'rest_rate':float(np.mean(y[te])),
                              'balanced_accuracy':float(balanced_accuracy_score(y[te],pred[te])),
                              'rest_f1':float(f1_score(y[te],pred[te]))})
            valid=np.isfinite(pr)
            entry={'test_rows':int(valid.sum()),'rest_fraction':float(np.mean(y[valid])),
                   'balanced_accuracy':float(balanced_accuracy_score(y[valid],pred[valid])),
                   'f1_rest':float(f1_score(y[valid],pred[valid])),
                   'precision_rest':float(precision_score(y[valid],pred[valid])),
                   'recall_rest':float(recall_score(y[valid],pred[valid])),
                   'roc_auc':float(roc_auc_score(y[valid],pr[valid])),
                   'confusion_0_1':confusion_matrix(y[valid],pred[valid],labels=[0,1]).tolist(),
                   'mean_text_model_bytes':int(np.mean(bytes_size)), 'folds':notes,
                   'by_group':{}}
            for g in LP_GROUPS+('CONNECTED','NOTHING'):
                mask=valid & (group==g)
                if np.any(mask):
                    entry['by_group'][g]={'rows':int(mask.sum()),'rest_fraction':float(np.mean(y[mask])),
                         'accuracy_teacher':float(np.mean(pred[mask]==y[mask])),
                         'f1_rest':float(f1_score(y[mask],pred[mask],zero_division=0))}
            result['models'][mode+'_'+name]=entry
            print('CAUSAL_STUDENT',mode,name,json.dumps({k:v for k,v in entry.items() if k not in ('folds','by_group')}),flush=True)
    # All held-out participants: nine continuous regression targets.
    # Rest/activity unknown rows are NEVER interpreted as numeric zero truth.
    for c in range(3):
        for o,name in enumerate(('resting','high_activity','signal_quality')):
            index=c*3+o
            predicted=np.full(len(y),np.nan)
            fold_result=[]
            for fold,(train,test) in enumerate(splits):
                if name=='signal_quality':
                    tr=train;te=test;sample_weight=np.ones(len(tr))
                else:
                    tr=train[qw[train,c]>=.8]
                    te=test[qw[test,c]>=.8]
                    sample_weight=qw[tr,c]
                if len(tr)<100 or len(te)<100:
                    fold_result.append({'fold':fold,'train_rows':len(tr),'test_rows':len(te),'warning':'insufficient confident target windows'})
                    continue
                m=lgb.LGBMRegressor(n_estimators=140,learning_rate=.06,num_leaves=12,
                             max_depth=6,min_child_samples=70,reg_lambda=4,
                             verbosity=-1,n_jobs=2,random_state=47)
                m.fit(X['compact48'][tr],yt[tr,index],sample_weight=sample_weight)
                pred=np.clip(m.predict(X['compact48'][te]),0,1)
                predicted[te]=pred
                truth=yt[te,index]
                fold_result.append({'fold':fold,'test_rows':len(te),
                                    'r2_to_pseudo_teacher':float(r2_score(truth,pred)) if np.var(truth)>1e-9 else None,
                                    'mae':float(mean_absolute_error(truth,pred))})
            valid=np.isfinite(predicted)
            truth=yt[valid,index]
            guess=predicted[valid]
            result['regression'][f'{name}_ch{c+1}']={
                 'r2_to_pseudo_teacher':float(r2_score(truth,guess)) if len(truth)>0 and np.var(truth)>1e-9 else None,
                 'mae_to_pseudo_teacher':float(mean_absolute_error(truth,guess)) if len(truth)>0 else None,
                 'test_rows':int(valid.sum()),
                 'evaluated_only_on_confident_channel_targets':name!='signal_quality',
                 'folds':fold_result}
            print('CAUSAL_Q_REG',name,c+1,json.dumps({k:v for k,v in result['regression'][f'{name}_ch{c+1}'].items() if k!='folds'}),flush=True)
    result['elapsed_s']=round(time.monotonic()-start,1)
    Path('/tmp/lp_v5_student_results.json').write_text(json.dumps(result,indent=2))
    print('V5_CAUSAL_STUDENT_DONE',result['elapsed_s'],flush=True)
if __name__=='__main__':main()
