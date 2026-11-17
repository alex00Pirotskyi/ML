"""Cache every DATA_LONG_PRESS recording: all p_core firmware columns on a 10 ms grid (zero-order hold) + metadata."""
import sys, os, glob, gzip, io, json, hashlib, re
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
G=10
def proc(a):
    i,path,out=a
    raw=gzip.open(path,'rb').read(); sha=hashlib.sha256(raw).hexdigest()
    df=pd.read_csv(io.BytesIO(raw),low_memory=False)
    cols=[c for c in df.columns if c.startswith('p_core->')]
    t=df['Timestamp'].to_numpy(np.float64); t=t-t[0]; keep=np.r_[True,np.diff(t)>0]; t=t[keep]
    g=np.arange(0,t[-1],G); idx=np.searchsorted(t,g,side='right')-1
    X=df[cols].to_numpy(np.float32)[keep][idx]
    A=df[['Metronome','PointerState']].to_numpy(np.float32)[keep][idx]
    np.savez(f'{out}/{i:05d}.npz',X=X,A=A)
    name=os.path.basename(path)
    return dict(i=i,path=path,cat=path.split('/')[-2],name=name,sha=sha,n=len(df),dur=t[-1]/1000,dtmed=float(np.median(np.diff(t))),
                user=str(df['User'].iloc[0]) if 'User' in df else '',hand=str(df['Hand'].iloc[0]) if 'Hand' in df else '',
                class_id=int(df['class_id'].iloc[0]) if 'class_id' in df else -1,class_n=int(df['class_id'].nunique()) if 'class_id' in df else 0,
                ptr_mean=float(df['PointerState'].mean()),cols='|'.join(cols))
if __name__=='__main__':
    root,out=sys.argv[1],sys.argv[2]; os.makedirs(out,exist_ok=True)
    fs=sorted(glob.glob(f'{root}/*/*.csv.gz'))
    with ProcessPoolExecutor(2) as ex: res=list(ex.map(proc,[(i,f,out) for i,f in enumerate(fs)],chunksize=20))
    m=pd.DataFrame(res); assert m.cols.nunique()==1; json.dump(m.cols.iloc[0].split('|'),open(f'{out}/cols.json','w')); m=m.drop(columns='cols')
    m['date']=m.name.str.extract(r'_(\d{4}-\d\d-\d\d)_')[0]; m['orient']=m.name.str.extract(r'_(Horizontal|Vertical|Reverse)_')[0]
    m['u']=m.user.str.lower().str.replace(r'[^a-z]','',regex=True); m['grp']=m.u.str[:4]
    m.to_pickle(f'{out}/meta.pkl'); print(len(m),m.sha.nunique())
