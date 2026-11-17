import numpy as np, pandas as pd, json, os
W=__import__('os').environ.get('LP_WORK','.'); COLS=json.load(open(f'{W}/g10/cols.json'))
def ci(pat): return [k for k,c in enumerate(COLS) if pat in c]
WL=[COLS.index(f'p_core->wl[CH{c}]') for c in (1,2,3)]
WLH=[COLS.index(f'p_core->wl_head[CH{c}]') for c in (1,2,3)]; WLT=[COLS.index(f'p_core->wl_tail[CH{c}]') for c in (1,2,3)]
PSD0=np.array([[COLS.index(f'p_core->emg_psd[0][CH{c}][{z}]') for z in range(5)] for c in (1,2,3)])
PSDM=np.array([[COLS.index(f'p_core->emg_psd_mean[CH{c}][{z}]') for z in range(5)] for c in (1,2,3)]) if any('emg_psd_mean' in c for c in COLS) else None
ACC0=ci('acc_psd[0]')
def meta():
    m=pd.read_pickle(f'{W}/g10/meta.pkl'); m=m.drop_duplicates('sha').reset_index(drop=True)
    m['person']=m.name.str.split('_').str[0].str.lower(); return m
def load(i):
    d=np.load(f'{W}/g10/{i:05d}.npz'); return d['X'],d['A']
