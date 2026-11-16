"""Score every 10 ms frame of every recording (edge-padded windows) with the round-2 fold models."""
import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from v2 import *
m=pd.read_pickle('meta.pkl'); models=pickle.load(open('models3.pkl','rb')); S={}
for r in m.itertuples():
    if pd.isna(r.fold): continue
    g,TA,TR,TE,on=traces2(r.i); n=len(g)
    if n<80: continue
    c=np.arange(0,n); S[r.i]=dict(t=c*G,ALL3=models[int(r.fold)].predict_proba(feats(TA,TR,TE,c))[:,1].astype(np.float32))
pickle.dump(S,open('scores3f.pkl','wb')); print(len(S))
