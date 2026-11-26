import numpy as np, pandas as pd, re, json
C=__import__('os').environ.get('TAP_CACHE','cache')
FEAT=json.load(open(f'{C}/feat.json'))
ACC=slice(0,18); EMG=slice(18,33); WL=slice(33,42); RATE=slice(42,60)
def norm_user(u):
    u=re.sub(r'[^a-z]','',str(u).lower())
    return u
def meta():
    m=pd.read_csv(f'{C}/meta.csv')
    m=m[m.get('err').isna()] if 'err' in m else m
    m=m.drop_duplicates('sha').reset_index(drop=True)
    p=m.name.str.replace('.csv.gz','',regex=False).str.rsplit('_',n=6,expand=True)
    m['fuser']=p[0]; m['gest']=p[1]; m['orient']=p[2]; m['mode']=p[3]; m['bpm']=p[4].str.replace('bpm','').astype(float); m['date']=p[5]; m['time']=p[6]
    m['pos']=m.cat.isin(['TAP','SWIPE']).astype(int)
    return m
def load(i):
    d=np.load(f'{C}/{i:05d}.npz'); return d['X'],d['A']
