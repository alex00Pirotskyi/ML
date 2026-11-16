#!/usr/bin/env python3
"""Read-only PSD provenance diagnostics: log encoding, optional legacy post-conversion."""
from pathlib import Path
import gzip,json
import pandas as pd
import numpy as np

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE","MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE"]
def q(a,p):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return round(float(np.percentile(a,p)),2) if a.size else None
def main():
    reports=[]
    counts={}
    for group in GROUPS:
        files=sorted((ROOT/group).glob("*.csv.gz"))
        if len(files)>6:
            files=[files[i] for i in np.linspace(0,len(files)-1,6,dtype=int)]
        records=[]
        for path in files:
            try:
                names=[f"p_core->emg_psd_mean[CH1][{z}]" for z in range(5)]
                names0=[f"p_core->emg_psd[0][CH1][{z}]" for z in range(5)]
                namesend=[f"p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH1][{z}]" for z in range(5)]
                cols=names+names0+namesend+["psd_mean_applied"]
                df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in cols)
                if not all(x in df for x in names):raise ValueError("missing PSD")
                values=df[names].to_numpy(float)
                item={"file":path.name,"rows":len(df),"marker":df["psd_mean_applied"].unique().tolist() if "psd_mean_applied" in df else "ABSENT",
                      "median5":[q(values[:,j],50) for j in range(5)],
                      "max5":[q(values[:,j],99) for j in range(5)],
                      "max_any":q(values.ravel(),99),"sat_fraction":round(float(np.mean(values>=65535)),6)}
                if all(x in df for x in names0):
                    new=df[names0].to_numpy(float)
                    item["latest0_median"]=[q(new[:,i],50) for i in range(5)]
                    item["latest0_vs_histmean_diff"]=[q(np.abs(new[:,i]-values[:,i]),50) for i in range(5)]
                records.append(item)
            except Exception as e:
                records.append({"file":path.name,"error":str(e)[:100]})
        counts[group]=len(records)
        reports+=([{"group":group,**rec} for rec in records])
        print("GROUP_PROFILE",group,json.dumps(records[:3]),flush=True)
    print("PROFILE_COUNTS",json.dumps(counts),flush=True)
    print("PROFILE_ALL",json.dumps(reports),flush=True)
if __name__=="__main__":main()
