import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb, pickle
G=10
O_A=np.arange(-150,251,10)//G; O_R=np.arange(-100,201,20)//G; O_E=np.arange(-240,241,20)//G
def traces2(i):
    X,A=load(i); t=A[:,0]-A[0,0]
    keep=np.r_[True,np.diff(t)>0]; X=X[keep];A=A[keep];t=t[keep]
    g=np.arange(0,t[-1],G); idx=np.searchsorted(t,g,side='right')-1; X=X[idx]
    e=np.log1p(np.clip(X[:,EMG],0,None)).reshape(-1,3,5); e=e-np.percentile(e,10,axis=0,keepdims=True)
    w=np.log1p(np.clip(X[:,33:42:3],0,None)); w=w-np.percentile(w,10,axis=0,keepdims=True)
    a=np.clip(X[:,ACC],0,None).reshape(-1,3,6)
    TA=np.c_[np.log1p(a.sum(1)),np.log1p(a.sum(2)),np.log1p(a.sum((1,2)))].astype(np.float32)   # 10
    TR=np.log1p(a).reshape(-1,18).astype(np.float32)
    es=np.sort(e,axis=1)  # channel-sorted per zone (min, mid, max)
    TE=np.c_[e.mean(1),es[:,2],es[:,0],w.mean(1),w.max(1)].astype(np.float32)  # 17
    s=gaussian_filter1d(TA[:,9],1.0); on=np.r_[np.zeros(2),s[2:]-s[:-2]]
    return g,TA,TR,TE,on
def win(T,c,O):
    n=len(T); return T[np.clip(c[:,None]+O[None,:],0,n-1)].reshape(len(c),-1)
def feats(TA,TR,TE,c):
    A_=win(TA,c,O_A); A2=win(TR,c,O_R); E_=win(TE,c,O_E)
    # baseline-relative copies for ACC total
    return np.c_[A_,A2,E_]
NA=10*len(O_A)+18*len(O_R); NE=17*len(O_E)
if __name__=='__main__':
    m=pd.read_pickle('meta.pkl'); pr=pd.read_pickle('presses.pkl'); prg=dict(tuple(pr.groupby('i')))
    XS=[];MS=[];C={}
    Xmm=np.lib.format.open_memmap('ds2_X.npy',mode='w+',dtype=np.float32,shape=(700000,NA+NE)); pos=[0]
    def put(a):
        Xmm[pos[0]:pos[0]+len(a)]=a; pos[0]+=len(a)
    for r in m.itertuples():
        g,TA,TR,TE,on=traces2(r.i); n=len(g)
        if n<80 or pd.isna(r.fold): continue
        C[r.i]=1
        if r.pos:
            p=prg[r.i]; up=(p.t_up.values//G).astype(int); dn=(p.t_dn.values//G).astype(int)
            snap=[]
            for u in up:
                lo=max(0,u-12); w=on[lo:u+13]
                j=lo+w.argmax() if len(w) and w.max()>0.5 else u
                snap.append(j)
            snap=np.array(snap,int)
            near=np.zeros(n,bool)
            for u,d in zip(np.minimum(up,snap),dn): near[max(0,u-35):min(n,d+36)]=True
            c=np.concatenate([snap-1,snap,snap+1]); k=np.tile(np.arange(len(up)),3); j=np.repeat([-10,0,10],len(up))
            ok=(c>26)&(c<n-27); c=c[ok];k=k[ok];j=j[ok]
            put(feats(TA,TR,TE,c)); MS.append(pd.DataFrame(dict(i=r.i,c=c,kind=1,k=k,jit=j)))
            bg=np.arange(26,n-27,4); bg=bg[~near[bg]]
            if len(bg): put(feats(TA,TR,TE,bg)); MS.append(pd.DataFrame(dict(i=r.i,c=bg,kind=2,k=-1,jit=0)))
        else:
            c=np.arange(26,n-27,2); c=c[np.random.default_rng(r.i).random(len(c))<0.5]
            put(feats(TA,TR,TE,c)); MS.append(pd.DataFrame(dict(i=r.i,c=c,kind=0,k=-1,jit=0)))
    Xmm.flush(); N=pos[0]; del Xmm; X=np.load('ds2_X.npy',mmap_mode='r'); M=pd.concat(MS,ignore_index=True).merge(m[['i','fold']],on='i',how='left'); assert len(M)==N
    print(N,M.kind.value_counts().to_dict(),flush=True)
    fam={'ALL2':np.arange(NA+NE)}   # file-fold retrain: only the family round 2 uses
    models={}
    for f in fam:
        for k in range(5):
            ii=np.where((M.fold!=k).values)[0]; w=np.where(M.kind.values[ii]==2,0.5,1.0)
            mdl=lgb.LGBMClassifier(n_estimators=350,learning_rate=0.07,num_leaves=47,colsample_bytree=0.25,subsample=0.7,subsample_freq=1,min_child_samples=40,verbose=-1,n_jobs=2)
            mdl.fit(np.ascontiguousarray(X[ii][:,:len(fam[f])]),(M.kind.values[ii]==1),sample_weight=w); models[(f,k)]=mdl; print(f,k,flush=True)
    pickle.dump(models,open('models2.pkl','wb'))
    S={}
    for r in m.itertuples():
        if r.i not in C: continue
        g,TA,TR,TE,on=traces2(r.i); n=len(g); c=np.arange(26,n-27,1); W=feats(TA,TR,TE,c)
        S[r.i]=dict(t=c*G,**{f:models[(f,int(r.fold))].predict_proba(W[:,fam[f]])[:,1].astype(np.float32) for f in fam})
    pickle.dump(S,open('scores2.pkl','wb')); print('done')
