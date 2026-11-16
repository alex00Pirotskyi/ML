"""Stack all 10 ms frames (126 firmware columns) into one memmap + index."""
import sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
m=pd.read_pickle('meta.pkl'); m=m[m.fold.notna()]
n=[np.load(f'g10/{i:05d}.npz')['A'].shape[0] for i in m.i]; N=sum(n)
X=np.lib.format.open_memmap('st_X.npy',mode='w+',dtype=np.float32,shape=(N,126)); fid=np.zeros(N,np.int32); tt=np.zeros(N,np.int32); ptr=np.zeros(N,np.int8); p=0
for i,k in zip(m.i,n):
    d=np.load(f'g10/{i:05d}.npz'); X[p:p+k]=d['X']; fid[p:p+k]=i; tt[p:p+k]=np.arange(k)*10; ptr[p:p+k]=d['A'][:,1]; p+=k
X.flush(); np.savez('st_idx.npz',fid=fid,t=tt,ptr=ptr); print(N)
