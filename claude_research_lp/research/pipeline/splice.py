"""Latency test by splicing: CONNECTED recording followed by the same person's DISCONNECTED recording (and the reverse).
The held-out fold model for that person scores the spliced stream with causal features."""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
from student_lp import SETS
import pickle
tag=sys.argv[1]; fs=sys.argv[2]; J=[FEATS.index(f) for f in SETS[fs]]
m=meta(); T=pd.read_pickle('teacher.pkl'); PO=np.load(f'stu_{tag}.npy'); T['off']=PO[:,2]>=0.5; steady=T[T.t>=6000].groupby(['i','ch']).off.mean(); conn_ok=T.groupby(['i','ch']).off.mean(); fold=T.groupby('i').fold.first(); models=pickle.load(open(f'stu_{tag}_models.pkl','rb'))
def run(Xa,Xb,k):
    X=np.concatenate([Xa,Xb]); F=channel_features(X); n0=len(Xa)
    return np.stack([models[k].predict_proba(F[:,c,:][:,J]) for c in range(3)],1), n0     # (n, ch, 3)
def first_sustained(x,n=20):
    r=np.convolve(x.astype(int),np.ones(n,int),'valid'); w=np.where(r==n)[0]; return w[0] if len(w) else None
rows=[]
C=m[m.cat=='CONNECTED']; D=m[m.cat=='DISCONNECTED']
for a in C.itertuples():
    b=D[(D.person==a.person)&(D.orient==a.orient)]
    if not len(b): continue
    b=b.iloc[0]; Xa,_=load(a.i); Xb,_=load(b.i); k=int(fold[a.i])
    P,n0=run(Xa,Xb,k); off=P[:,:,2]>=0.5; rest_or_act=P[:,:,2]<0.5
    Pr,n1=run(Xb,Xa,k); offr=Pr[:,:,2]>=0.5
    for c in range(3):
        steady_off=steady.get((b.i,c),0)>0.8; before_ok=conn_ok.get((a.i,c),1)<0.1
        if steady_off and before_ok:
            j=first_sustained(off[n0:,c]); kind='quiet' if np.median(np.log10(Xb[-300:,WL[c]]+1))<2.4 else 'noise'
            rows.append(dict(person=a.person,ori=a.orient,ch=c,dir='disconnect',kind=kind,lat=None if j is None else j*0.01))
            # recovery: reverse stream, channel OFF at end of first part and should become not-OFF
            if True:
                j2=first_sustained(~offr[n1:,c]); rows.append(dict(person=a.person,ori=a.orient,ch=c,dir='reconnect',kind=kind,lat=None if j2 is None else j2*0.01))
R=pd.DataFrame(rows); R.to_csv(f'splice_{tag}.csv',index=False); pd.set_option('display.width',200)
R['lat']=R.lat.astype(float); print(R.groupby(['dir','kind']).lat.describe(percentiles=[.25,.5,.75,.9]).round(2).to_string()); print('never within recording:',R.lat.isna().groupby([R.dir,R.kind]).sum().to_dict())
