import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
from scipy.ndimage import uniform_filter1d, minimum_filter1d, gaussian_filter1d
DT=5  # ms grid
def grid(i):
    X,A=load(i); t=A[:,0]-A[0,0]
    keep=np.r_[True,np.diff(t)>0]; X=X[keep];A=A[keep];t=t[keep]
    g=np.arange(0,t[-1],DT); idx=np.searchsorted(t,g,side='right')-1
    return g,X[idx],A[idx]
def acts(X):
    """activation traces on grid: emg (mean log-coded psd over all 15), wl, acc (log1p sum)"""
    e=X[:,EMG].mean(1); w=X[:,33:42:3].mean(1); a=np.log1p(X[:,ACC].sum(1))
    return e,w,a
def onset(x,sm=15,lag=40):
    """positive rise over `lag` ms of smoothed trace"""
    s=gaussian_filter1d(x.astype(float),sm/DT); k=lag//DT
    d=np.zeros_like(s); d[k:]=s[k:]-s[:-k]; return np.clip(d,0,None)
def edges(p):
    d=np.diff(p.astype(int)); return np.where(d>0)[0]+1, np.where(d<0)[0]+1
