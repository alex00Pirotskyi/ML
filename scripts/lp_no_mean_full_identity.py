#!/usr/bin/env python3
"""Verify full decompressed-content identity of 2,431 LP recordings, old vs new."""
from pathlib import Path
import gzip,json,subprocess,time
ROOT=Path("DATA/DATA_LONG_PRESS")
OLD="2e14d95252b2ef94fdd1c8c92c643220ff432855"
def main():
    files=sorted(ROOT.rglob("*.csv.gz"))
    start=time.monotonic()
    identical=0;different=0;failed=0
    diff_names=[];errors=[]
    for n,p in enumerate(files,1):
        try:
            old=subprocess.run(["git","show",f"{OLD}:{p.as_posix()}"],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
            newer=p.read_bytes()
            a=gzip.decompress(old)
            b=gzip.decompress(newer)
            if a==b:identical+=1
            else:
                different+=1
                if len(diff_names)<30:diff_names.append(p.as_posix())
        except Exception as e:
            failed+=1
            if len(errors)<30:errors.append({"path":p.as_posix(),"error":str(e)[:300]})
        if n%200==0 or n==len(files):
            print("CHECK",n,len(files),"identical",identical,"different",different,
                  "errors",failed,"elapsed_s",round(time.monotonic()-start,1),flush=True)
    out={"before":OLD,"after":"36f02ac757166ba6dee848385ce70aa2f276a7d4",
         "files":len(files),"identical_decompressed":identical,
         "different_decompressed":different,"errors":failed,
         "sample_changed_names":diff_names,"error_examples":errors,
         "elapsed_s":round(time.monotonic()-start,1)}
    Path("/tmp/lp_no_mean_entire_dataset.csv_content.json").write_text(json.dumps(out,indent=2))
    print("COMPLETE_DATASET_HASH",json.dumps(out),flush=True)
    if failed:raise SystemExit(2)
if __name__=="__main__":main()
