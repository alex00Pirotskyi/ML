#!/usr/bin/env python3
"""Per-channel EMG state from a Mudra recording: REST, ACTIVE (muscle activity) or OFF (electrode not in proper contact).

usage:  python lp_state.py <recording.csv[.gz]> [--model model_full27] [--out out.csv.gz] [--debounce 1.0]

Uses only logged firmware columns (p_core->wl*, p_core->emg_psd*). Everything is causal: a row's state depends only on that row and the past
(up to 10 s). Output has one row per input row:
  Timestamp, and for each channel c in 1..3:  ch<c>_state (REST/ACTIVE/OFF), ch<c>_p_rest, ch<c>_p_active, ch<c>_p_off
OFF is declared only after P(OFF) >= 0.5 has held for --debounce seconds; while that is pending the channel keeps its last confirmed state.
The first second of a recording has incomplete history and is the least reliable."""
import sys, os, json, argparse
import numpy as np, pandas as pd
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
G=10  # ms grid
def lin(v): return np.power(10.0,np.asarray(v,np.float64)/10000.0)-1.0          # firmware code = 10000*log10(1+power)
def db(p): return 10*np.log10(np.maximum(p,1e-4))
def past(x,n,fn):
    o=(n-1)//2
    if fn=='min': return minimum_filter1d(x,n,origin=o,mode='nearest')
    if fn=='max': return maximum_filter1d(x,n,origin=o,mode='nearest')
    mu=uniform_filter1d(x,n,origin=o,mode='nearest')
    if fn=='mean': return mu
    return np.sqrt(np.maximum(uniform_filter1d(x*x,n,origin=o,mode='nearest')-mu*mu,0))
def channel_features(df):
    """dict feature name -> array (n_frames, 3 channels) on the 10 ms grid; also returns row index of each frame"""
    t=df['Timestamp'].to_numpy(np.float64); t=t-t[0]; g=np.arange(0,t[-1]+G,G); idx=np.clip(np.searchsorted(t,g,side='right')-1,0,len(t)-1)
    F={}; ch=(1,2,3)
    wl=np.stack([df[f'p_core->wl[CH{c}]'].to_numpy(np.float64)[idx] for c in ch],1); lw=np.log10(wl+1.0)
    wh=np.stack([df[f'p_core->wl_head[CH{c}]'].to_numpy(np.float64)[idx] for c in ch],1); wt=np.stack([df[f'p_core->wl_tail[CH{c}]'].to_numpy(np.float64)[idx] for c in ch],1)
    pi=np.stack([lin(df[[f'p_core->emg_psd[0][CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in ch],1)      # (n,3,5)
    pm=np.stack([lin(df[[f'p_core->emg_psd_mean[CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in ch],1)
    Pi=db(pi.sum(2)); Pm=db(pm.sum(2)); f=pm/np.maximum(pm.sum(2,keepdims=True),1e-9); lz=db(pm)
    F['lw']=lw; F['ht']=np.log10((wh+1.0)/(wt+1.0)); F['wlq']=lw-Pm/20.0; F['Pi']=Pi; F['Pm']=Pm; F['dPim']=Pi-Pm
    for z in range(5): F[f'f{z}']=f[:,:,z]
    F['r01']=lz[:,:,0]-lz[:,:,1]; F['r21']=lz[:,:,2]-lz[:,:,1]; F['r32']=lz[:,:,3]-lz[:,:,2]; F['r43']=lz[:,:,4]-lz[:,:,3]
    col=lambda fn,x,n: np.stack([past(x[:,c],n,fn) for c in range(3)],1)
    F['sd1']=col('std',lw,100); F['rng1']=col('max',lw,100)-col('min',lw,100); F['min3']=col('min',lw,300); F['max3']=col('max',lw,300); F['sd3']=col('std',lw,300)
    F['min6']=col('min',lw,600); F['min10']=col('min',lw,1000); F['max10']=col('max',lw,1000); F['sd10']=col('std',lw,1000); F['rise10']=lw-F['min10']
    F['Pmin10']=col('min',Pm,1000); F['Prise10']=Pm-F['Pmin10']
    return F,g,t
def decide(P,nd):
    """P: (n,3) probabilities REST/ACTIVE/OFF for one channel -> state per frame.
    OFF needs P(OFF)>=0.5 on nd consecutive frames; while that is pending the last confirmed state is held."""
    n=len(P); raw=P[:,2]>=0.5; held=np.convolve(raw.astype(int),np.ones(nd,int),'full')[:n]>=nd
    base=np.where(P[:,1]>=P[:,0],'ACTIVE','REST').astype(object); out=np.empty(n,object); last=None
    for k in range(n):
        if held[k]: out[k]='OFF'
        elif raw[k]: out[k]=last if last is not None else base[k]      # pending: keep the last confirmed state
        else: out[k]=base[k]
        if not raw[k] or held[k]: last=out[k]
    return out
def predict(df,model_prefix,debounce_s=1.0):
    import lightgbm as lgb
    spec=json.load(open(model_prefix+'.json')); bst=lgb.Booster(model_file=model_prefix+'.txt')
    F,g,t=channel_features(df); n=len(g); P=np.zeros((n,3,3))
    for c in range(3): P[:,c,:]=bst.predict(np.stack([F[k][:,c] for k in spec['features']],1))
    nd=max(1,int(round(debounce_s*1000/G))); state=np.empty((n,3),object)
    for c in range(3):
        state[:,c]=decide(P[:,c,:],nd)
    row=np.clip(np.searchsorted(g,t,side='right')-1,0,n-1)           # grid frame for every input row
    out=pd.DataFrame({'Timestamp':df['Timestamp'].values})
    for c in range(3):
        out[f'ch{c+1}_state']=state[row,c]; out[f'ch{c+1}_p_rest']=P[row,c,0].round(4); out[f'ch{c+1}_p_active']=P[row,c,1].round(4); out[f'ch{c+1}_p_off']=P[row,c,2].round(4)
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--model',default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_full27')); ap.add_argument('--out'); ap.add_argument('--debounce',type=float,default=1.0)
    a=ap.parse_args(); df=pd.read_csv(a.csv,low_memory=False); out=predict(df,a.model,a.debounce)
    dst=a.out or os.path.basename(a.csv).replace('.csv.gz','').replace('.csv','')+'.states.csv.gz'; out.to_csv(dst,index=False)
    for c in (1,2,3): print(f'CH{c}:',out[f'ch{c}_state'].value_counts(normalize=True).round(3).to_dict())
    print('written',dst)
