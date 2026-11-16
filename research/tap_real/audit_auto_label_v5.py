#!/usr/bin/env python3
"""Independent V5 TAP+SWIPE output audit; checks serialization and class isolation.

This is not physical-event ground truth validation.
"""
import argparse,csv,gzip,json
from collections import Counter,defaultdict
from pathlib import Path

POS={"TAP":1,"SWIPE":2}
NEG={"FINGER","WRIST","ARM","BAND","NOTHING"}

def audit(root,out,min_coverage=.7):
    root=Path(root).resolve();out=Path(out).resolve()
    if root.name!="DATA_TAP" or out==root or root in out.parents:
        raise ValueError("Only disjoint DATA_TAP source allowed")
    summary=json.loads((out/"summary.json").read_text())
    expected={"TAP":1,"SWIPE":1,"FINGER":0,"WRIST":0,"ARM":0,"BAND":0,"NOTHING":0}
    if summary.get("target_class")!=expected:raise AssertionError("Incorrect source-class mapping")
    records=[json.loads(x) for x in (out/"manifest.jsonl").read_text().splitlines() if x.strip()]
    ok=[r for r in records if r["status"]=="OK"]
    dups=[r for r in records if r["status"]=="DUPLICATE"]
    failed=[r for r in records if r["status"]=="FAILED"]
    if failed or len(ok)!=summary["files_ok"] or len(dups)!=summary["duplicate_files"]:
        raise AssertionError("Failed or missing records")
    totals=Counter();groups=defaultdict(Counter);uncovered=[]
    for idx,e in enumerate(ok):
        cat=e["category"];rel=e["source"]
        if cat not in POS and cat not in NEG:raise AssertionError("Invalid class")
        original=root/rel
        stem=Path(rel).name.removesuffix(".gz").removesuffix(".csv")
        labelled=out/cat/(stem+".labels.csv.gz")
        if not original.is_file() or not labelled.is_file():
            raise FileNotFoundError(str(labelled))
        src=(gzip.open(original,"rt",newline="") if original.suffix==".gz" else
             original.open("r",newline=""))
        lab=gzip.open(labelled,"rt",newline="")
        seen_events=set();pair_events=defaultdict(set);positive_rows=negative_rows=unknown_rows=0
        try:
            a=csv.DictReader(src);b=csv.DictReader(lab)
            required={"Timestamp","gesture_label","gesture_weight","gesture_kind",
              "gesture_event_id","gesture_pair_id","gesture_provenance",
              "gesture_candidate_id","gesture_model_score","tap_label","tap_label_weight"}
            if not required.issubset(b.fieldnames or []):raise AssertionError("Missing labels")
            first_time=None
            for row,(x,y) in enumerate(zip(a,b),1):
                if float(x["Timestamp"])!=float(y["Timestamp"]):
                    raise AssertionError(f"Timestamp mismatch {rel} at row {row}")
                if first_time is None:first_time=float(x["Timestamp"])
                label=int(y["gesture_label"]);kind=int(y["gesture_kind"])
                eid=int(y["gesture_event_id"]);pair=int(y["gesture_pair_id"])
                weight=float(y["gesture_weight"])
                if label not in (-1,0,1) or not 0<=weight<=1:
                    raise AssertionError("Bad label or weight")
                if int(y["tap_label"])!=label or abs(float(y["tap_label_weight"])-weight)>1e-6:
                    raise AssertionError("Legacy field mismatch")
                if label==1:
                    positive_rows+=1
                    if (cat not in POS or kind!=POS[cat] or eid<=0 or
                            int(y["gesture_candidate_id"])<=0):
                        raise AssertionError("Positive in wrong category/without event")
                    if not y["gesture_provenance"].startswith("AUTO_"):
                        raise AssertionError("Unknown positive provenance")
                    if float(x["Timestamp"])<first_time+170:
                        raise AssertionError("Warmup positive")
                    score=float(y["gesture_model_score"])
                    if not 0<=score<=1:raise AssertionError("Invalid model score")
                    seen_events.add(eid)
                    if pair:pair_events[pair].add(eid)
                elif label==0:
                    negative_rows+=1
                    if eid!=-1 or pair or kind:raise AssertionError("Negative event leakage")
                else:
                    unknown_rows+=1
                    if weight!=0 or eid!=-1 or pair or kind:
                        raise AssertionError("Unknown event/weight leakage")
            if next(a,None) is not None or next(b,None) is not None:
                raise AssertionError("Source/output row count mismatch")
        finally:
            src.close();lab.close()
        if cat in NEG and positive_rows:raise AssertionError("Negative category positive leakage")
        if cat not in POS and seen_events:raise AssertionError("Negative event leakage")
        if len(seen_events)!=int(e["positive_events"]) or positive_rows!=int(e["positive_rows"]):
            raise AssertionError("Manifest / serialized data disagreement")
        for pair,evs in pair_events.items():
            if len(evs)!=2 or cat!="TAP" or "doubletap" not in rel.lower():
                raise AssertionError("Unverified/spurious pair structure")
        totals.update({"rows_positive":positive_rows,"rows_negative":negative_rows,
                       "rows_unknown":unknown_rows,"events_positive":len(seen_events),
                       "pairs_double":len(pair_events)})
        groups[cat]["files"]+=1;groups[cat]["covered"]+=bool(seen_events)
        groups[cat]["events"]+=len(seen_events)
        if cat in POS and not seen_events:uncovered.append(rel)
        if (idx+1)%250==0:print("V5_AUDIT_PROGRESS",idx+1,len(ok),flush=True)
    for key in ("rows_positive","rows_negative","rows_unknown","events_positive","pairs_double"):
        if int(summary[key])!=totals[key]:
            raise AssertionError(f"Summary mismatch: {key} {summary[key]} != {totals[key]}")
    target_files=sum(groups[x]["files"] for x in POS)
    covered=sum(groups[x]["covered"] for x in POS)
    coverage=covered/max(1,target_files)
    if coverage<min_coverage:
        raise AssertionError(f"Positive-bag coverage {coverage:.3f} below {min_coverage:.3f}")
    result={"unique_files_verified":len(ok),"duplicates":len(dups),
       "positive_bags":target_files,"positive_bags_covered":covered,"coverage":coverage,
       "negative_positive_events":sum(groups[x]["events"] for x in NEG),
       "counts":dict(totals),"by_class":{x:dict(v) for x,v in groups.items()},
       "uncovered_bags_preview":uncovered[:20],
       "ground_truth_status":"Trusted recording classes, automatic temporal SILVER pseudo-labels; NOT physical event ground truth"}
    (out/"integrity_summary.json").write_text(json.dumps(result,indent=2))
    print("V5_INTEGRITY_RESULT",json.dumps(result),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--min-target-coverage",type=float,default=.70)
    x=p.parse_args()
    audit(x.input_dir,x.output_dir,x.min_target_coverage)
