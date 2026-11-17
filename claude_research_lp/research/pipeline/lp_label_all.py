"""Label every DATA_LONG_PRESS recording, row by row, with models that never saw that person.
Per channel: teacher label (0 REST, 1 ACTIVE, 2 OFF, -1 unknown; offline, uses the whole recording) and the causal student's
probabilities and debounced state (what a deployed system would have said)."""
import sys, os, pickle; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','state_tool'))
import numpy as np, pandas as pd
from lp_state import channel_features, decide
from lpc import meta
from student_lp import SETS
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
m=meta(); T=pd.read_pickle('teacher.pkl'); fold=T.groupby('i').fold.first()
offm=pickle.load(open('off3_models.pkl','rb')); stum=pickle.load(open('stu_aug27_models.pkl','rb')); FS=SETS['own27']
tot=np.zeros((4,),int); pred_tot={}
for r in m.itertuples():
    if r.i not in fold.index: continue
    k=int(fold[r.i]); df=pd.read_csv(r.path,low_memory=False); F,g,t=channel_features(df); n=len(g); row=np.clip(np.searchsorted(g,t,side='right')-1,0,n-1)
    o=pd.DataFrame({'Timestamp':df['Timestamp'].values})
    for c in range(3):
        X=np.stack([F[f][:,c] for f in FS],1); s=offm[k].predict_proba(X)[:,1]; lw=F['lw'][:,c]; A=20*lw
        off=(s>=0.5)|(lw<2.2); on=(s<0.2)&(lw>=2.35); on[:100]=False
        lab=np.full(n,-1,np.int8)
        if on.sum()>=200:
            fl=np.percentile(A[on],5); lab[on&(A<=fl+3)]=0; lab[on&(A>=fl+6)]=1
        lab[off]=2; lab[:100]=-1
        P=stum[k].predict_proba(X); st=decide(P,100)
        o[f'ch{c+1}_label']=lab[row]; o[f'ch{c+1}_state']=st[row]; o[f'ch{c+1}_p_rest']=P[row,0].round(3); o[f'ch{c+1}_p_active']=P[row,1].round(3); o[f'ch{c+1}_p_off']=P[row,2].round(3)
        tot+=np.bincount(lab[row]+1,minlength=4)
    o.to_csv(f'{out}/{r.cat}__{r.name.replace(".csv.gz","")}.states.csv.gz',index=False)
print('rows x channels by teacher label: unknown %d REST %d ACTIVE %d OFF %d'%tuple(tot))
