"""Per-channel causal frame features from logged firmware columns (10 ms grid in, strided rows out)."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lpc import *
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
PSDH=np.array([[COLS.index(f'p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH{c}][{z}]') for z in range(5)] for c in (1,2,3)])
ACCM=[k for k,c in enumerate(COLS) if 'acc_psd_mean' in c]
def lin(v): return np.power(10.0,np.asarray(v,np.float64)/10000.0)-1.0
def db(p): return 10*np.log10(np.maximum(p,1e-4))
def past(x,n,fn):
    """causal rolling statistic over the last n frames (including current)"""
    if fn=='min': return minimum_filter1d(x,n,origin=(n-1)//2,mode='nearest')
    if fn=='max': return maximum_filter1d(x,n,origin=(n-1)//2,mode='nearest')
    if fn=='mean': return uniform_filter1d(x,n,origin=(n-1)//2,mode='nearest')
    if fn=='std':
        mu=uniform_filter1d(x,n,origin=(n-1)//2,mode='nearest'); m2=uniform_filter1d(x*x,n,origin=(n-1)//2,mode='nearest'); return np.sqrt(np.maximum(m2-mu*mu,0))
FEATS=['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43','sd1','rng1','min3','max3','sd3','o_lw_lo','o_lw_hi','o_Pm_lo','o_Pm_hi','d_lw_lo','d_lw_hi','acc','accsd','min6','min10','max10','sd10','rise10','Pmin10','Prise10']
def channel_features(X):
    """returns array (n, 3 channels, len(FEATS))"""
    n=len(X); out=np.zeros((n,3,len(FEATS)),np.float32)
    lw=np.log10(X[:,WL]+1.0); Pm=np.stack([db(lin(X[:,PSDM[c]]).sum(1)) for c in range(3)],1)
    acc=np.log10(lin(X[:,ACCM]).sum(1)+1e-3) if False else np.log10(np.clip(X[:,ACCM],0,None).sum(1)+1.0)
    for c in range(3):
        pi=lin(X[:,PSD0[c]]); pm=lin(X[:,PSDM[c]]); Pi=db(pi.sum(1)); f=pm/np.maximum(pm.sum(1,keepdims=True),1e-9); lz=db(pm)
        o=[k for k in range(3) if k!=c]; olw=np.sort(lw[:,o],1); oP=np.sort(Pm[:,o],1)
        cols=[lw[:,c], np.log10((X[:,WLH[c]]+1.0)/(X[:,WLT[c]]+1.0)), lw[:,c]-Pm[:,c]/20.0, Pi, Pm[:,c], Pi-Pm[:,c],
              f[:,0],f[:,1],f[:,2],f[:,3],f[:,4], lz[:,0]-lz[:,1], lz[:,2]-lz[:,1], lz[:,3]-lz[:,2], lz[:,4]-lz[:,3],
              past(lw[:,c],100,'std'), past(lw[:,c],100,'max')-past(lw[:,c],100,'min'), past(lw[:,c],300,'min'), past(lw[:,c],300,'max'), past(lw[:,c],300,'std'),
              olw[:,0],olw[:,1],oP[:,0],oP[:,1], lw[:,c]-olw[:,0], lw[:,c]-olw[:,1], acc, past(acc,100,'std'),
              past(lw[:,c],600,'min'), past(lw[:,c],1000,'min'), past(lw[:,c],1000,'max'), past(lw[:,c],1000,'std'), lw[:,c]-past(lw[:,c],1000,'min'), past(Pm[:,c],1000,'min'), Pm[:,c]-past(Pm[:,c],1000,'min')]
        out[:,c,:]=np.stack(cols,1)
    return out
if __name__=='__main__':
    m=meta(); rows=[];XS=[]
    for r in m.itertuples():
        X,A=load(r.i)
        if len(X)<150: continue
        F=channel_features(X); st=5 if r.cat in ('TAP','SWIPE') else 2
        idx=np.arange(100,len(X),st)       # skip first second (rolling stats warm-up, start transient)
        for c in range(3):
            XS.append(F[idx,c,:]); rows.append(pd.DataFrame(dict(i=r.i,ch=c,t=idx*10,ptr=A[idx,1].astype(np.int8))))
    X=np.concatenate(XS); M=pd.concat(rows,ignore_index=True); np.save('fr_X.npy',X); M.to_pickle('fr_M.pkl'); print(X.shape)
