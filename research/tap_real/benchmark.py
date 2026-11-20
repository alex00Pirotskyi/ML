import csv,gzip,hashlib,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,pandas as pd
from scipy.signal import find_peaks
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tap_event_labeler_v1 as v1,tap_event_labeler_v2 as v2

root=Path("DATA/DATA_TAP")
out=Path("research/tap_real/results");out.mkdir(parents=True,exist_ok=True)
categories={"TAP","SWIPE","FINGER","WRIST","ARM","BAND","NOTHING"}
cols={"Timestamp","PointerState","Metronome","TriggerPressure","User"}
cols.update(k for q in v2.EMG_HEAD+v2.ACC_HEAD for k in q)
cols.update(v2.WL)
files=sorted(p for p in root.rglob("*.csv.gz") if p.parent.name in categories)
limit=int(sys.argv[1]) if len(sys.argv)>1 else 0
if limit: # across classes + double taps
    pool=[p for p in files if "DoubleTaps" in p.name or p.parent.name=="NOTHING"]
    for cat in sorted(categories):
        pool+= [p for i,p in enumerate(x for x in files if x.parent.name==cat) if i%max(1,sum(x.parent.name==cat for x in files)//limit)==0][:limit]
    files=sorted(set(pool))
print("SOURCE_FILES",len(files),flush=True)
def matches(a,b,tol=125):
    pairs=sorted((abs(x-y),i,j) for i,x in enumerate(a) for j,y in enumerate(b) if abs(x-y)<=tol)
    ai=set();bi=set();n=0
    for _,i,j in pairs:
        if i not in ai and j not in bi:ai.add(i);bi.add(j);n+=1
    return n
def pairs(xs):
    xs=sorted(xs);i=0;n=0;gaps=[]
    while i<len(xs)-1:
        gap=xs[i+1]-xs[i]
        if 65<=gap<=500:n+=1;gaps.append(float(gap));i+=2
        else:i+=1
    return n,gaps
def risefilter(t,s,threshold):
    ix,_=find_peaks(s,height=threshold,prominence=max(.025,threshold*.6),distance=2)
    ranked=sorted(ix,key=lambda i:s[i],reverse=True)
    result=[]
    for i in ranked:
        if all(abs(t[i]-t[j])>=65 for j in result):result.append(i)
    return sorted(float(t[i]) for i in result)
def summarize(g):
    duration=sum(x['seconds'] for x in g);r={"files":len(g),"hours":round(duration/3600,4)}
    for z in ["v1","v2","rise08","rise12","rise20","rise35"]:
        count=sum(x[z] for x in g)
        r[z+"_events"]=count;r[z+"_events_per_hr"]=round(count*3600/max(.01,duration),2)
        if z in ("v1","v2"):r[z+"_pairs"]=sum(x[z+"_pairs"] for x in g)
    r["pointer_edges"]=sum(x["pointer"] for x in g)
    return r
rows=[];bad=[];seen={};dups=0;start=time.monotonic()
for i,f in enumerate(files):
    try:
        dig=hashlib.sha256(f.read_bytes()).hexdigest()
        if dig in seen:dups+=1;continue
        seen[dig]=str(f)
        df=pd.read_csv(f,usecols=lambda x:x in cols,compression="gzip",low_memory=False)
        t=pd.to_numeric(df.Timestamp,errors="coerce").to_numpy(float)
        if len(t)<15 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError("bad timestamps")
        category=f.parent.name;dt=(t[-1]-t[0])/1000
        ptr=v2.rising_times(pd.to_numeric(df.PointerState,errors="coerce").fillna(0).to_numpy(float),t)
        beat=v2.rising_times(pd.to_numeric(df.Metronome,errors="coerce").fillna(0).to_numpy(float),t)
        entry={"file":str(f),"category":category,"person":str(df.User.iloc[0]) if "User" in df else f.name.split("_")[0],"seconds":float(dt),"pointer":len(ptr),"beats":len(beat),"pointer_duty":float(np.mean(df.PointerState>0)),"double":bool(category=="TAP" and "doubletap" in f.name.lower()),"soft":bool(category=="TAP" and "softtap" in f.name.lower()),"median_dt":float(np.median(np.diff(t)))}
        for tag,fn in (("v1",v1),("v2",v2)):
            # Counterfactual TAP mode for negatives, avoiding class-gate artificial zeroes
            lab,events,q=fn.label_recording(df,"TAP",f.name if category=="TAP" else "") if tag=="v2" else fn.label_recording(df,"TAP")
            accepted=sorted(c.peak_ms for c in events if c.decision=="POSITIVE")
            pn,gaps=pairs(accepted) if entry["double"] else (0,[])
            entry[tag]=len(accepted);entry[tag+"_pairs"]=pn
            entry[tag+"_candidates"]=len(events)
            entry[tag+"_pointer_matches"]=matches(accepted,list(ptr))
            entry[tag+"_unknown_frac"]=float(np.mean(lab.tap_label==-1))
            entry[tag+"_event_ids"]=len(set(lab.loc[lab.tap_label==1,"tap_event_id"]))
            if tag=="v2":entry["v2_tag_pairs"]=q["distinct_double_pairs"]
        emg=v2.emg_matrix(df)
        rises=np.max(v2.short_rise_signal(emg,t),axis=1)
        for z,threshold in (("rise08",.08),("rise12",.12),("rise20",.20),("rise35",.35)):
            ev=risefilter(t,rises,threshold)
            entry[z]=len(ev)
            entry[z+"_pointer_matches"]=matches(ev,list(ptr))
            if entry["double"]:entry[z+"_pairs"]=pairs(ev)[0]
        rows.append(entry)
    except Exception as e:bad.append({"file":str(f),"error":str(e)[:240]})
    if (i+1)%100==0: print("PROGRESS",i+1,len(files),len(rows),len(bad),"seconds",round(time.monotonic()-start),flush=True)
df=pd.DataFrame(rows)
df.to_csv(out/"per_recording.csv",index=False)
groups={cat:summarize([r for r in rows if r["category"]==cat]) for cat in sorted(categories)}
for cat in ("DOUBLE","SOFT","OTHER_TAP"):
    sub=[r for r in rows if (r['double'] if cat=="DOUBLE" else r['soft'] if cat=="SOFT" else (r['category']=="TAP" and not r['double'] and not r['soft']))]
    groups[cat]=summarize(sub)
double=[r for r in rows if r["double"]]
checks={tag:{"at_least_one_pair":sum(r.get(tag+"_pairs",0)>0 for r in double),"at_least_five_pairs":sum(r.get(tag+"_pairs",0)>=5 for r in double),"total_pair_proxies":sum(r.get(tag+"_pairs",0) for r in double)} for tag in ("v1","v2","rise08","rise12","rise20","rise35")}
result={"scope":"real files, not independent physical ground truth","source_files":len(files),"analyzed":len(rows),"duplicate_compressed":dups,"failures":bad[:30],"n_fail":len(bad),"seconds":round(time.monotonic()-start,1),"groups":groups,"double_check":checks,"v2_tagged_pairs":sum(r.get('v2_tag_pairs',0) for r in double),"note":"negative counts are counterfactual labeler taps if these files were classified as TAP, NOT firmware false positives. Pair proxies are not verified true double taps."}
(out/"summary.json").write_text(json.dumps(result,indent=2))
print("RESULT_BEGIN",flush=True);print(json.dumps(result),flush=True);print("RESULT_END",flush=True)
