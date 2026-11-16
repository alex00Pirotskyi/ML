"""Teacher round 2: same features as v2, training set cleaned with round-1 out-of-fold scores.
 - positives: pointer presses whose round-1 score >= 0.3, anchored at the round-1 peak time (+-10 ms jitter)
 - dropped from training entirely: unsupported presses, and background within 300 ms of an unmatched round-1 peak >= 0.5"""
import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from v2 import *
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
if __name__=='__main__':
    m=pd.read_pickle('meta.pkl'); pr=pd.read_pickle('presses.pkl'); prg=dict(tuple(pr.groupby('i'))); S1=pickle.load(open('scores2.pkl','rb'))
    NF=NA+NE; Xmm=np.lib.format.open_memmap('ds3_X.npy',mode='w+',dtype=np.float32,shape=(440000,NF)); pos=0; MS=[]
    for r in m.itertuples():
        if r.i not in S1: continue
        g,TA,TR,TE,on=traces2(r.i); n=len(g)
        if r.pos:
            s=uniform_filter1d(S1[r.i]['ALL2'],5); pk,_=find_peaks(s,distance=15,height=0.05); pt=S1[r.i]['t'][pk]; ph=s[pk]; used=np.zeros(len(pk),bool)
            p=prg[r.i]; near=np.zeros(n,bool); anc=[]
            for u,d in zip(p.t_up.values,p.t_dn.values):
                c=np.where(np.abs(pt-u)<=150)[0]; a=int(u//G)
                if len(c):
                    j=c[ph[c].argmax()]; used[c]=True
                    if ph[j]>=0.3: anc.append(int(pt[j]//G)); a=min(a,anc[-1])
                near[max(0,a-35):min(n,int(d//G)+36)]=True
            for j in np.where(~used&(ph>=0.5))[0]: near[max(0,int(pt[j]//G)-30):int(pt[j]//G)+31]=True
            anc=np.array(anc,int); c=np.concatenate([anc-1,anc,anc+1]) if len(anc) else np.array([],int); c=c[(c>26)&(c<n-27)]
            if len(c): F=feats(TA,TR,TE,c); Xmm[pos:pos+len(c)]=F; pos+=len(c); MS.append(pd.DataFrame(dict(i=r.i,c=c,kind=1)))
            bg=np.arange(26,n-27,4); bg=bg[~near[bg]]
            if len(bg): Xmm[pos:pos+len(bg)]=feats(TA,TR,TE,bg); pos+=len(bg); MS.append(pd.DataFrame(dict(i=r.i,c=bg,kind=2)))
        else:
            c=np.arange(26,n-27,2); c=c[np.random.default_rng(r.i).random(len(c))<0.5]
            Xmm[pos:pos+len(c)]=feats(TA,TR,TE,c); pos+=len(c); MS.append(pd.DataFrame(dict(i=r.i,c=c,kind=0)))
    Xmm.flush(); del Xmm; X=np.load('ds3_X.npy',mmap_mode='r'); M=pd.concat(MS,ignore_index=True).merge(m[['i','fold']],on='i',how='left'); assert len(M)==pos
    print(pos,M.kind.value_counts().to_dict(),flush=True)
    models={}
    for k in range(5):
        ii=np.where((M.fold!=k).values)[0]; w=np.where(M.kind.values[ii]==2,0.5,1.0)
        mdl=lgb.LGBMClassifier(n_estimators=350,learning_rate=0.07,num_leaves=47,colsample_bytree=0.25,subsample=0.7,subsample_freq=1,min_child_samples=40,verbose=-1,n_jobs=2)
        Xi=np.ascontiguousarray(X[ii]); mdl.fit(Xi,(M.kind.values[ii]==1),sample_weight=w); del Xi; models[k]=mdl; print(k,flush=True)
    pickle.dump(models,open('models3.pkl','wb'))
    S={}
    for r in m.itertuples():
        if r.i not in S1: continue
        g,TA,TR,TE,on=traces2(r.i); n=len(g); c=np.arange(26,n-27,1)
        S[r.i]=dict(t=c*G,ALL3=models[int(r.fold)].predict_proba(feats(TA,TR,TE,c))[:,1].astype(np.float32))
    pickle.dump(S,open('scores3.pkl','wb')); print('done')
