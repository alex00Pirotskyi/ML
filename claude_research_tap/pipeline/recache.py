"""All p_core firmware columns resampled (zero-order hold) to a 10 ms grid, one npz per recording."""
import sys, gzip, io, json, os
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
G=10
def proc(a):
    i,path,out=a
    df=pd.read_csv(path,low_memory=False)
    cols=[c for c in df.columns if c.startswith('p_core->')]
    t=df['Timestamp'].to_numpy(np.float64); t=t-t[0]
    keep=np.r_[True,np.diff(t)>0]; t=t[keep]
    g=np.arange(0,t[-1],G); idx=np.searchsorted(t,g,side='right')-1
    X=df[cols].to_numpy(np.float32)[keep][idx]
    A=df[['Metronome','PointerState']].to_numpy(np.int8)[keep][idx]
    np.savez(f'{out}/{i:05d}.npz',X=X,A=A)
    return cols
if __name__=='__main__':
    m=pd.read_pickle('meta.pkl'); out='g10'
    with ProcessPoolExecutor(2) as ex: res=list(ex.map(proc,[(r.i,r.path,out) for r in m.itertuples()],chunksize=20))
    assert all(r==res[0] for r in res), 'column mismatch'
    json.dump(res[0],open(f'{out}/cols.json','w')); print(len(res),len(res[0]))
