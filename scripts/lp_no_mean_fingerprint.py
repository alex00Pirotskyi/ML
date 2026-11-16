#!/usr/bin/env python3
"""Compare decompressed CSV byte content from before vs DATA NO PSD MEAN commit."""
import gzip,hashlib,json,subprocess
from pathlib import Path
import numpy as np

OLD="2e14d95252b2ef94fdd1c8c92c643220ff432855"
ROOT=Path("DATA/DATA_LONG_PRESS")
groups=("CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE",
        "MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS",
        "LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE")

def read_old(path):
    r=subprocess.run(["git","show",f"{OLD}:{path.as_posix()}"],
                     capture_output=True,check=True)
    return r.stdout

def main():
    results=[]
    for group in groups:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        indexes=sorted({0,len(paths)//2,len(paths)-1})
        for ix in indexes:
            path=paths[ix]
            current=path.read_bytes()
            old=read_old(path)
            old_unpacked=gzip.decompress(old)
            new_unpacked=gzip.decompress(current)
            out={"file":path.as_posix(),"group":group,
                 "gzip_binary_identical":old==current,
                 "decompressed_csv_identical":old_unpacked==new_unpacked,
                 "before_gz":len(old),"after_gz":len(current),
                 "sha256_old_uncompressed":hashlib.sha256(old_unpacked).hexdigest(),
                 "sha256_new_uncompressed":hashlib.sha256(new_unpacked).hexdigest()}
            results.append(out)
            print("CSV_CONTENT",json.dumps(out),flush=True)
    report={"sampled":len(results),
            "uncompressed_same":sum(x["decompressed_csv_identical"] for x in results),
            "compressed_same":sum(x["gzip_binary_identical"] for x in results),
            "examples":results}
    print("CSV_FINGERPRINT_SUMMARY",json.dumps({k:v for k,v in report.items() if k!="examples"}),flush=True)
    Path("/tmp/lp_gzip_fingerprint.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
