#!/usr/bin/env python3
"""Exhaustive event-aligned raw PSD morphology extraction, audit bound to V5 source hashes.

Input DATA/DATA_TAP/ original gzip CSVs and V5 labels (only for comparison).
No V5 selected labels or pointer/metronome may influence waveform extraction.
Events are V3 physiological candidates; true onset status unknown.
"""
import sys,json,gzip,hashlib,time,argparse,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from tap_event_shape_v3 import extract_signals,extract_events
EMG=[f"p_core->emg_psd[0][CH{ch}][{z}]" for ch in [1,2,3] for z in range(5)]
WL=[f"p_core->wl[CH{ch}]" for ch in [1,2,3]]
ACC=[f"p_core->acc_psd[0][{axis}][{z}]" for axis in ["X","Y","Z"] for z in range(6)]
TIME=np.arange(-200,241,20,dtype=float)
def sha256_uncompressed(f):
    h=hashlib.sha256()
    with gzip.open(f,"rb") as io:
        for chunk in iter(lambda:io.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def canon(user,filename):
    s=re.sub("[^a-z]","",str(user).lower())
    s=re.sub(r"(lefthand|righthand|left|right|noise|psd\d+|psd|hand)$","",s)
    for prefix in ("ohad","alex","neomi","yossef","noam","michael","perry",
                   "nikol","shay","eitan","kirill","sarina","adi","samuel"):
        if s.startswith(prefix):return prefix
    return s or filename.split("_")[0].lower()
def calculate(df,items,meta,rel):
    ts=pd.to_numeric(df.Timestamp,errors="raise").to_numpy(float)
    if len(ts)<80 or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
        raise ValueError("Insufficient/non-monotonic source timestamps")
    dt=float(np.median(np.diff(ts)))
    raw=gaussian_filter1d(df[EMG].to_numpy(dtype=float),sigma=max(1.,12./max(dt,1)),axis=0)
    user=canon(df.User.iloc[0],Path(rel).name) if "User" in df else Path(rel).name.split("_")[0]
    V=extract_signals(df)
    E=extract_events(V)
    rows=[];vector=[];compact=[]
    for ordinal,e in enumerate(E,1):
        peak=float(e.peak_ms)
        if peak-ts[0]<380 or ts[-1]-peak<250:continue
        baseline_range=(ts>=peak-320)&(ts<=peak-170)
        if np.sum(baseline_range)<6:continue
        baseline=np.median(raw[baseline_range],axis=0)
        relative=(raw-baseline[None,:]).reshape(len(ts),3,5)
        wave=np.vstack([np.interp(peak+TIME,ts,relative[:,ch,z])
                 for ch in range(3) for z in range(5)]).T
        win=wave[(TIME>=-20)&(TIME<=80)].reshape(-1,3,5)
        recruitment=np.maximum(0,win).mean(axis=(0,2))
        order=np.argsort(recruitment)[::-1]
        waveform=wave.reshape(len(TIME),3,5)[:,order,:]
        divisor=float(np.percentile(np.maximum(0,waveform),96))
        if divisor<3:continue
        normalized=np.clip(waveform/max(divisor,10.),-1,4)
        positive=np.maximum(0,normalized)
        timecurve=positive.sum(axis=(1,2))/15.
        spectrum=positive[(TIME>=-40)&(TIME<=100)].mean(axis=(0,1))
        spectrum=spectrum/max(1e-6,float(spectrum.sum()))
        channels=positive[(TIME>=-40)&(TIME<=100)].mean(axis=(0,2))
        channels=channels/max(1e-6,float(channels.sum()))
        shape=np.concatenate([timecurve,spectrum,channels])
        picked=items.get(ordinal)
        rows.append({"file":rel,"category":meta["category"],"person":user,
             "candidate_ordinal":ordinal,"peak_ms":peak,
             "selected_v5":int(picked is not None),
             "v5_provenance":picked[0] if picked else "NONE",
             "v5_weight":float(picked[1]) if picked else 0.,
             "shape_score":float(e.shape_score),
             "activity":float(e.activity),"rise_time_ms":float(e.rise_time_ms),
             "fall_time_ms":float(e.fall_time_ms),
             "duration_ms":float(e.duration_ms),"fall_fraction":float(e.fall_fraction),
             "compactness":float(e.compactness),
             "wl_reactivity":float(e.wl_reactivity),"acc_reactivity":float(e.acc_reactivity),
             "normalized_psd_scale":float(divisor),
             "sample_dt_ms":dt})
        vector.append(normalized.astype(np.float16).reshape(-1))
        compact.append(shape.astype(np.float16))
    return rows,vector,compact,len(E)
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input-dir",type=Path,required=True)
    a.add_argument("--labels-dir",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    a.add_argument("--max-files",type=int,default=0)
    args=a.parse_args()
    root=args.input_dir.resolve();labels=args.labels_dir.resolve()
    out=args.out_dir.resolve()
    if root.name!="DATA_TAP" or root in out.parents or out==root:raise ValueError("Unsafe source path")
    if out.exists() and any(out.iterdir()):raise ValueError("Output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    entries=[json.loads(s) for s in (labels/"manifest.jsonl").read_text().splitlines()]
    entries=[e for e in entries if e.get("status")=="OK"]
    if args.max_files:
        by=defaultdict(list)
        for e in entries:by[e["category"]].append(e)
        entries=sorted((e for rows in by.values() for e in rows[:args.max_files]),key=lambda x:x["source"])
    recs=[];vectors=[];compact=[];audit=Counter();start=time.time()
    for i,meta in enumerate(entries,1):
        rel=meta["source"];source=root/rel
        if sha256_uncompressed(source)!=meta["source_sha256"]:
            raise ValueError("Source SHA mismatch: "+rel)
        cat=meta["category"]
        stem=Path(rel).name.removesuffix(".csv.gz")
        sidecar=labels/cat/(stem+".labels.csv.gz")
        lab=pd.read_csv(sidecar,compression="gzip",usecols=["gesture_label","gesture_candidate_id","gesture_provenance","gesture_weight"])
        selected=lab[(lab.gesture_label==1)&(lab.gesture_candidate_id>0)]
        items={int(k):(str(group.gesture_provenance.iloc[0]),float(group.gesture_weight.mean()))
               for k,group in selected.groupby("gesture_candidate_id")}
        df=pd.read_csv(source,compression="gzip",usecols=lambda x:x in
                       set(["User","Timestamp"]+EMG+WL+ACC),low_memory=False)
        if len(df)!=len(lab):raise ValueError("source/label length mismatch: "+rel)
        if not set(EMG).issubset(df.columns):raise ValueError("Missing EMG PSD columns: "+rel)
        rows,full,small,original=calculate(df,items,meta,rel)
        recs.extend(rows);vectors.extend(full);compact.extend(small)
        audit[cat+"_files"]+=1;audit[cat+"_events"]+=original
        audit["matched_v5"]+=sum(x["selected_v5"] for x in rows)
        if i%200==0:
            print("PATTERN_EXTRACTION",i,len(entries),"extracted",len(recs),"sec",round(time.time()-start,1),flush=True)
    d=pd.DataFrame(recs)
    F=np.asarray(vectors,dtype=np.float16)
    C=np.asarray(compact,dtype=np.float16)
    assert len(d)==len(F)==len(C)
    d.to_csv(out/"events_meta.csv.gz",index=False)
    np.savez_compressed(out/"psd_motifs.npz",waveform=F,compact=C)
    summary={"n_files":len(entries),"candidate_events_original":int(sum(v for k,v in audit.items() if k.endswith("_events"))),
       "waveform_events":len(d),"selected_v5_matched":int(audit["matched_v5"]),
       "per_class_counts":dict(audit),"waveform_timesteps_ms":TIME.tolist(),
       "waveform_order":"time, ranked channels, spectral zones",
       "waveform_features":int(F.shape[1]),"compact_features":int(C.shape[1]),
       "source":"strict DATA/DATA_TAP V5 source_sha256 bound, no original files changed",
       "extraction":"V3 EMG onset proposals, offline local prebaseline subtraction and channel ranking; no known physical event times",
       "duration_s":round(time.time()-start,1)}
    (out/"extraction_summary.json").write_text(json.dumps(summary,indent=2))
    print("V7_MOTIF_EXTRACTION_END",json.dumps(summary),flush=True)
if __name__=="__main__":main()
