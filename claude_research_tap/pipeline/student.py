"""Per-frame firmware-feature students trained on different label variants; out-of-fold scores per variant.
usage: student.py <events.csv> <featset: F36|F126> <variant,variant,...>"""
import sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
def frame_labels(variant,EV,m,fid,t,ptr,thr_hi,thr_orph):
    """returns y (1/0/-1) and weight per frame"""
    y=np.zeros(len(fid),np.int8); w=np.ones(len(fid),np.float32)
    start=pd.Series(np.arange(len(fid))).groupby(fid).agg(['first','last'])
    posfiles=set(m[m.pos==1].i)
    if variant=='P_raw':
        y[:]=ptr; return y,w
    evg=dict(tuple(EV.groupby('i')))
    def span(i,a,b,val,wt=None,only0=False):
        s,e=start.loc[i,'first'],start.loc[i,'last']; lo=s+max(0,int(np.ceil(a/10))); hi=min(e,s+int(np.floor(b/10)))
        if hi<lo: return
        if only0:
            q=y[lo:hi+1]!=1; y[lo:hi+1][q]=val
        else:
            y[lo:hi+1]=val
            if wt is not None: w[lo:hi+1]=wt
    for i in posfiles:
        if i not in evg or i not in start.index: continue
        e=evg[i]; P=e[e.tier.isin(['PRESS','EDGE'])]
        if variant=='P_on':
            for x in P.itertuples(): span(i,x.t_ptr-100,x.t_rel+150,-1)
            for x in P.itertuples(): span(i,x.t_ptr,x.t_ptr+80,1)
            continue
        win={'F_on':80,'F_on_strong':80,'F_on150':150,'F_ext':None}[variant]
        for x in P.itertuples(): span(i,min(x.t_ptr,x.t_on)-150,max(x.t_rel,x.t_off)+150,-1)
        for x in e[(e.tier=='ORPHAN')&(e.score>=thr_orph)].itertuples(): span(i,x.t_on-200,x.t_on+250,-1)
        for x in e[(e.tier=='INPRESS')&(e.score>=thr_orph)].itertuples(): span(i,x.t_on-150,x.t_on+200,-1)
        for x in P[P.tier=='PRESS'].itertuples():
            ok=(x.score>=thr_hi) if variant=='F_on_strong' else (x.p_real>=0.5)
            if not ok: continue
            span(i,x.t_on,x.t_off if win is None else x.t_on+win,1,wt=1.0 if variant=='F_on_strong' else x.p_real)
    return y,w
if __name__=='__main__':
    evf,fs,variants=sys.argv[1],sys.argv[2],sys.argv[3].split(',')
    m=pd.read_pickle('meta.pkl'); m=m[m.fold.notna()]; EV=pd.read_csv(evf); summ=json.load(open(os.path.join(os.path.dirname(evf),'summary.json')))
    thr_hi=summ['thr_at_neg_false_per_min']['1']; thr_orph=summ['thr_at_neg_false_per_min']['5']
    cols=json.load(open('g10/cols.json')); X=np.load('st_X.npy',mmap_mode='r'); d=np.load('st_idx.npz'); fid,t,ptr=d['fid'],d['t'],d['ptr']
    F36=[k for k,c in enumerate(cols) if ('emg_psd[0]' in c) or ('acc_psd[0]' in c) or c.startswith('p_core->wl[')]; assert len(F36)==36
    fc=F36 if fs=='F36' else list(range(126))
    fold=pd.Series(m.fold.values,index=m.i).reindex(fid).values.astype(int)
    rng=np.random.default_rng(0); r=rng.random(len(fid)); tag=os.path.basename(os.path.dirname(evf))
    for v in variants:
        y,w=frame_labels(v,EV,m,fid,t,ptr,thr_hi,thr_orph)
        keep=(y==1)|((y==0)&(r<0.25))
        sc=np.zeros(len(fid),np.float32)
        for k in range(5):
            ii=np.where(keep&(fold!=k))[0]
            mdl=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.1,num_leaves=15,min_child_samples=100,colsample_bytree=0.8,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
            mdl.fit(np.ascontiguousarray(X[ii][:,fc]),y[ii],sample_weight=w[ii])
            te=np.where(fold==k)[0]
            for a in range(0,len(te),400000): sc[te[a:a+400000]]=mdl.predict_proba(np.ascontiguousarray(X[te[a:a+400000]][:,fc]))[:,1]
        np.save(f'st_score_{tag}_{fs}_{v}.npy',sc); print(v,fs,'pos frames',int((y==1).sum()),'neg',int((y==0).sum()),'unk',int((y==-1).sum()),flush=True)
