"""Label-design ablations for the per-frame student (126 firmware columns).
variant = S_<a>_<b>[_nobg][_nomask][_hn][_ab]  : positives are frames in [onset+a, onset+b] ms of teacher-confirmed presses
  default positives: strong presses only (score >= threshold at 1 false/min); _ab adds weaker presses with weight p_real
  _nobg   : frames of TAP/SWIPE files outside positives are unknown (negatives come only from negative folders)
  _nomask : orphans / unsupported presses are NOT masked (treated as negatives)
  _hn     : negative-folder frames weighted 1 + 9 * teacher score (teacher false alarms become hard negatives)"""
import sys, os, json, pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.ndimage import maximum_filter1d
def labels(v,EV,m,fid,thr_hi,thr_orph,teacher=None):
    o=v.split('_'); a=int(o[1].replace('m','-')); b=int(o[2]); opt=set(o[3:])
    y=np.zeros(len(fid),np.int8); w=np.ones(len(fid),np.float32)
    st=pd.Series(np.arange(len(fid))).groupby(fid).agg(['first','last']); evg=dict(tuple(EV.groupby('i')))
    def span(i,lo_ms,hi_ms,val,wt=None):
        s,e=st.loc[i,'first'],st.loc[i,'last']; lo=s+max(0,int(np.ceil(lo_ms/10))); hi=min(e,s+int(np.floor(hi_ms/10)))
        if hi>=lo:
            y[lo:hi+1]=val
            if wt is not None: w[lo:hi+1]=wt
    for i in m[m.pos==1].i:
        if i not in evg or i not in st.index: continue
        e=evg[i]; P=e[e.tier.isin(['PRESS','EDGE'])]
        if 'nobg' in opt: span(i,0,1e9,-1)
        good=lambda x:(x.tier=='PRESS') and ((x.score>=thr_hi) or ('ab' in opt and x.p_real>=0.5))
        for x in P.itertuples():
            if good(x) or 'nomask' not in opt: span(i,min(x.t_ptr,x.t_on)-150,max(x.t_rel,x.t_off)+150,-1)
        if 'nomask' not in opt:
            for x in e[(e.tier=='ORPHAN')&(e.score>=thr_orph)].itertuples(): span(i,x.t_on-200,x.t_on+250,-1)
            for x in e[(e.tier=='INPRESS')&(e.score>=thr_orph)].itertuples(): span(i,x.t_on-150,x.t_on+200,-1)
        for x in P.itertuples():
            if good(x): span(i,x.t_on+a,x.t_on+b,1,wt=1.0 if x.score>=thr_hi else x.p_real)
    if 'hn' in opt:
        for i in m[m.pos==0].i:
            s,e=st.loc[i,'first'],st.loc[i,'last']; ts=maximum_filter1d(teacher[i]['ALL3'],11)[:e-s+1]; w[s:s+len(ts)]=1+9*ts
    return y,w
if __name__=='__main__':
    evf,variants=sys.argv[1],sys.argv[2].split(',')
    m=pd.read_pickle('meta.pkl'); m=m[m.fold.notna()]; EV=pd.read_csv(evf,low_memory=False); summ=json.load(open(os.path.join(os.path.dirname(evf),'summary.json')))
    thr_hi=summ['thr_at_neg_false_per_min']['1']; thr_orph=summ['thr_at_neg_false_per_min']['5']
    X=np.load('st_X.npy',mmap_mode='r'); fid=np.load('st_idx.npz')['fid']
    fold=pd.Series(m.fold.values,index=m.i).reindex(fid).values.astype(int); r=np.random.default_rng(0).random(len(fid)); tag=os.path.basename(os.path.dirname(evf))
    teacher=pickle.load(open('scores3f.pkl','rb')) if any('hn' in v for v in variants) else None
    for v in variants:
        y,w=labels(v,EV,m,fid,thr_hi,thr_orph,teacher); keep=(y==1)|((y==0)&((r<0.25)|(w>2)))
        sc=np.zeros(len(fid),np.float32)
        for k in range(5):
            ii=np.where(keep&(fold!=k))[0]
            mdl=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.1,num_leaves=15,min_child_samples=100,colsample_bytree=0.8,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
            mdl.fit(np.ascontiguousarray(X[ii]),y[ii],sample_weight=w[ii]); te=np.where(fold==k)[0]
            for a in range(0,len(te),400000): sc[te[a:a+400000]]=mdl.predict_proba(np.ascontiguousarray(X[te[a:a+400000]]))[:,1]
        np.save(f'st_score_{tag}_F126_{v}.npy',sc); print(v,'pos',int((y==1).sum()),'neg',int((y==0).sum()),'unk',int((y==-1).sum()),flush=True)
