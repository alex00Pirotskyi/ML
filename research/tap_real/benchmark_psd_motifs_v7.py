#!/usr/bin/env python3
"""Reproducible V7 full-source PSD motif ablations; outputs NOT physical onset truth.

Requires source-hash-audited extract_psd_motifs_v7.py artifact containing
72,016 waveform episodes from all 2,790 DATA_TAP sources.
"""
import argparse,json,time,warnings,collections
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore",message="X does not have valid feature names")

def work(motifs,out,methods):
    meta=pd.read_csv(motifs/"events_meta.csv.gz")
    z=np.load(motifs/"psd_motifs.npz")
    wave=z["waveform"].astype(np.float32)
    compact=z["compact"].astype(np.float32)
    if not (len(meta)==len(wave)==len(compact)):raise ValueError("Waveform and metadata rows differ")
    strong=((meta.category=="TAP") & meta.v5_provenance.eq("AUTO_POINTER_EMG_STRONG")).to_numpy()
    hard=meta.category.isin(["FINGER","WRIST","ARM","BAND","NOTHING"]).to_numpy()
    trainable=strong|hard
    extra=meta[["activity","rise_time_ms","fall_time_ms","duration_ms",
       "fall_fraction","compactness","shape_score","wl_reactivity",
       "acc_reactivity","normalized_psd_scale"]].to_numpy(np.float32)
    choices={"amplitude_morphology":extra,
      "compact31":compact,"waveform345":wave,
      "waveform345_morphology":np.concatenate([wave,extra],axis=1)}
    group=meta.person.to_numpy();file=meta.file.to_numpy();category=meta.category.to_numpy()
    folds=list(GroupKFold(5).split(np.arange(len(meta)),groups=group))
    out.mkdir(parents=True,exist_ok=True)
    result={"source_files":int(meta.file.nunique()),"waveform_events":len(meta),
       "high_confidence_tap_pseudo_positive":int(strong.sum()),
       "negative_category_events":int(hard.sum()),"participant_string_groups":int(len(set(group))),
       "scores_are_physical_truth":False,"methods":{}}
    all_scores={}
    start=time.time()
    for name in methods:
        X=choices[name]
        predicted=np.full(len(meta),np.nan,np.float32)
        info=[]
        for fold,(tr,te) in enumerate(folds):
            tid=tr[trainable[tr]];y=strong[tid].astype(int)
            if len(set(y))!=2:raise ValueError("Missing class in grouped training split")
            counts=collections.Counter(file[tid])
            w=np.array([1/np.sqrt(counts[f]) for f in file[tid]],dtype=np.float32)
            for cls in (0,1):
                flag=y==cls
                w[flag]*=len(w)/(2*max(1e-9,float(w[flag].sum())))
            model=LGBMClassifier(n_estimators=130,num_leaves=7,max_depth=5,
                learning_rate=.052,min_child_samples=45,reg_lambda=22,
                verbosity=-1,n_jobs=3,random_state=20261116+fold)
            model.fit(X[tid],y,sample_weight=w)
            predicted[te]=model.predict_proba(X[te])[:,1]
            info.append({"fold":fold,"train_candidate_events":len(tid),
               "heldout_candidate_events":len(te),
               "heldout_participant_strings":len(set(group[te]))})
        if not np.isfinite(predicted).all():raise ValueError("Missing heldout probabilities")
        lab=strong|hard
        event_auc=float(roc_auc_score(strong[lab],predicted[lab]))
        event_ap=float(average_precision_score(strong[lab],predicted[lab]))
        bag=meta[["file","category","person"]].copy()
        bag["score"]=predicted
        bags=bag.groupby("file").agg(category=("category","first"),person=("person","first"),score=("score","max"))
        pos=bags.category=="TAP";finger=bags.category=="FINGER";all_neg=bags.category.isin(["FINGER","WRIST","ARM","BAND","NOTHING"])
        bag_auc=float(roc_auc_score(pos[pos|finger].astype(int),bags.loc[pos|finger,"score"]))
        bag_all_auc=float(roc_auc_score(pos[pos|all_neg].astype(int),bags.loc[pos|all_neg,"score"]))
        # Descriptive, post-hoc thresholds; do not interpret as prospective specificity!
        t5=float(np.quantile(bags.loc[finger,"score"],.95))
        n99=float(np.quantile(predicted[hard],.99))
        result["methods"][name]={"heldout_pseudo_event_auc":event_auc,"heldout_pseudo_event_ap":event_ap,
          "heldout_tap_finger_bag_auc":bag_auc,
          "heldout_tap_all_hard_negative_bag_auc":bag_all_auc,
          "posthoc_tap_bag_hit_at_finger_5pct_bag_tail":float(np.mean(bags.loc[pos,"score"]>t5)),
          "posthoc_pseudo_tap_event_recall_at_1pct_neg_event_tail":float(np.mean(predicted[strong]>n99)),
          "folds":info}
        print("V7_PATTERN_METHOD",name,json.dumps(result["methods"][name]),flush=True)
        all_scores[name]=predicted
        (out/"summary.json").write_text(json.dumps(result,indent=2))
    # Paired participant-block bootstrap of the best waveform vs generic morphology.
    if {"amplitude_morphology","waveform345_morphology"}<=set(all_scores):
        df=meta[["file","category","person"]].copy()
        for key in ("amplitude_morphology","waveform345_morphology"):
            df[key]=all_scores[key]
        g=df.groupby("file").agg(category=("category","first"),person=("person","first"),
                baseline=("amplitude_morphology","max"),pattern=("waveform345_morphology","max"))
        pos=g[g.category=="TAP"];neg=g[g.category=="FINGER"]
        pp={k:v[["baseline","pattern"]].to_numpy() for k,v in pos.groupby("person")}
        nn={k:v[["baseline","pattern"]].to_numpy() for k,v in neg.groupby("person")}
        rng=np.random.default_rng(20261116);rep=[];ppl=list(pp);npl=list(nn)
        for i in range(1200):
            p=np.vstack([pp[x] for x in rng.choice(ppl,len(ppl),replace=True)])
            n=np.vstack([nn[x] for x in rng.choice(npl,len(npl),replace=True)])
            y=np.r_[np.ones(len(p)),np.zeros(len(n))];A=np.vstack([p,n])
            rep.append([roc_auc_score(y,A[:,0]),roc_auc_score(y,A[:,1])])
        r=np.array(rep)
        result["subject_block_bootstrap"]={"n":len(r),
           "baseline_95ci":np.percentile(r[:,0],[2.5,97.5]).tolist(),
           "waveform_95ci":np.percentile(r[:,1],[2.5,97.5]).tolist(),
           "improvement_95ci":np.percentile(r[:,1]-r[:,0],[2.5,97.5]).tolist()}
    result["elapsed_seconds"]=round(time.time()-start,2)
    result["limitations"]=[
      "V5-selected pointer+EMG candidates are weak pseudo positives, not physical tap onset ground truth",
      "Training a waveform discriminator cannot establish actual per-event recall or latency",
      "Source-folder TAP/FINGER comparison is true recording-level ground truth only",
      "Each waveform includes up to +240ms after candidate peak: offline teacher only",
      "Subject aliases are imperfect and same-session/different-day shifts are significant",
      "Post-hoc thresholds are diagnostic and not calibrated prospective production thresholds"]
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame({"file":file,"category":category,"person":group,
       **{name:score for name,score in all_scores.items()}}).to_csv(out/"out_of_fold_event_scores.csv.gz",index=False)
    print("V7_PATTERN_BENCHMARK_SUCCESS",json.dumps({"waveforms":len(meta),"methods":list(all_scores),"total_elapsed":result["elapsed_seconds"]}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--motif-dir",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--methods",default="amplitude_morphology,compact31,waveform345,waveform345_morphology")
    x=p.parse_args()
    choices=x.methods.split(",")
    if not choices:raise ValueError("Need methods")
    work(x.motif_dir,x.out_dir,choices)
if __name__=="__main__":main()
