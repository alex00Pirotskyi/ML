import sys, os, glob, gzip, hashlib, re, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT=sys.argv[1]; OUT=sys.argv[2]
def cols():
    c=[]
    for a in 'XYZ':
        for z in range(6): c.append(f'p_core->acc_psd[0][{a}][{z}]')
    for ch in (1,2,3):
        for z in range(5): c.append(f'p_core->emg_psd[0][CH{ch}][{z}]')
    for ch in (1,2,3):
        for k in ('wl','wl_head','wl_tail'): c.append(f'p_core->{k}[CH{ch}]')
    for a in 'XYZ':
        for z in range(6): c.append(f'p_core->acc_rate[{a}][{z}]')
    return c
FEAT=cols()
AUX=['Timestamp','Metronome','PointerState','TriggerPressure','InputX','InputY']
def proc(i_f):
    i,f=i_f
    try:
        raw=gzip.open(f,'rb').read()
        sha=hashlib.sha256(raw).hexdigest()
        import io
        df=pd.read_csv(io.BytesIO(raw),low_memory=False)
        miss=[c for c in FEAT+AUX if c not in df.columns]
        for c in miss: df[c]=np.nan
        X=df[FEAT].to_numpy(np.float32)
        A=df[AUX].to_numpy(np.float64)
        np.savez_compressed(f'{OUT}/{i:05d}.npz',X=X,A=A)
        user=str(df['User'].iloc[0]) if 'User' in df else ''
        cid=df['class_id'].iloc[0] if 'class_id' in df else -1
        return dict(i=i,path=f,cat=f.split('/')[-2],name=os.path.basename(f),sha=sha,n=len(df),ncols=df.shape[1],missing=','.join(miss),user=user,class_id=int(cid),extra=','.join([c for c in df.columns if c not in FEAT+AUX and 'p_core' not in c]))
    except Exception as e:
        return dict(i=i,path=f,err=repr(e))
if __name__=='__main__':
    fs=sorted(glob.glob(f'{ROOT}/*/*.csv.gz'))
    with ProcessPoolExecutor(2) as ex: res=list(ex.map(proc,enumerate(fs),chunksize=20))
    pd.DataFrame(res).to_csv(f'{OUT}/meta.csv',index=False)
    json.dump(FEAT,open(f'{OUT}/feat.json','w'))
    print(len(res))
