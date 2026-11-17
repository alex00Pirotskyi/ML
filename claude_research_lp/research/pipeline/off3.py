"""OFF detector round 2: retrain after removing label noise found by round 1 (person-grouped out-of-fold).
 - OFF-folder channels whose round-1 median score < 0.5 and that are not dead-quiet are treated as unknown (probably still connected)
 - non-protocol channels whose round-1 median score > 0.5 are treated as unknown (probably real contact problems)
 - CONNECTED / NOTHING stay negative regardless."""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
import lightgbm as lgb, pickle
m=meta(); m['g4']=m.person.str.extract(r'^([a-z]+)')[0].str[:4]
X=np.load('fr_X.npy',mmap_mode='r'); M=pd.read_pickle('fr_M.pkl').merge(m[['i','cat','person','g4']],on='i',how='left'); S=pd.read_pickle('off1_scores.pkl'); M['s1']=S.s.values; M['fold']=S.fold.values
OWN=[FEATS.index(f) for f in ['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43','sd1','rng1','min3','max3','sd3','min6','min10','max10','sd10','rise10','Pmin10','Prise10']]
lw=np.asarray(X[:,0]); offf=M.cat.isin(['DISCONNECTED','LOOSE','TABLE']).values; conn=M.cat.isin(['CONNECTED','NOTHING']).values
med=M.groupby(['i','ch']).s1.transform('median').values; lwm=pd.Series(lw).groupby([M.i,M.ch]).transform('median').values
Pm=np.asarray(X[:,4]); lp=M.cat.str.contains('LONG|DRAG').values
g=pd.DataFrame(dict(i=M.i.values,ch=M.ch.values,ptr=M.ptr.values,Pm=Pm))[lp].groupby(['i','ch','ptr']).Pm.median().unstack(); g['c']=g[1]-g[0]
locked=pd.MultiIndex.from_arrays([M.i,M.ch]).isin(g[g.c>6].index)          # channel clearly follows the pointer: activity, not noise
pos=offf&((med>=0.5)|(lwm<2.3)); unk=(offf&~pos)|(~offf&~conn&(med>0.5)&~locked); neg=~pos&~unk; print('pointer-locked file-channels forced negative:',int((g.c>6).sum()))
print('channel-frames: pos %d neg %d unknown %d'%(pos.sum(),neg.sum(),unk.sum()))
r=np.random.default_rng(0).random(len(M)); keep=pos|conn|(neg&(r<0.25)); sc=np.zeros(len(M),np.float32); models=[]
for k in range(5):
    ii=np.where(keep&(M.fold!=k).values)[0]
    mdl=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.08,num_leaves=15,min_child_samples=200,colsample_bytree=0.7,subsample=0.7,subsample_freq=1,verbose=-1,n_jobs=2)
    mdl.fit(np.ascontiguousarray(X[ii][:,OWN]),pos[ii],sample_weight=np.where(lp[ii]&neg[ii],3.0,1.0)); te=np.where((M.fold==k).values)[0]; sc[te]=mdl.predict_proba(np.ascontiguousarray(X[te][:,OWN]))[:,1]; models.append(mdl)
pickle.dump(models,open('off3_models.pkl','wb'))
M['s2']=sc; M['lab_off']=np.where(pos,1,np.where(unk,-1,0)); M[['i','ch','t','s1','s2','fold','lab_off']].to_pickle('off3_scores.pkl')
imp=pd.Series(np.mean([mm.booster_.feature_importance('gain') for mm in models],0),index=[FEATS[j] for j in OWN]); print((imp/imp.sum()).sort_values(ascending=False).round(3).head(10).to_dict())
pd.set_option('display.width',250)
fc=M.groupby(['cat','i','ch']).agg(s=('s2','median'),frac=('s2',lambda x:(x>=0.5).mean())).reset_index()
print('share of channel-frames with OFF score >= 0.5, by folder:'); print(M.groupby('cat').s2.apply(lambda x:(x>=0.5).mean()).round(4).to_string())
rec=fc.groupby(['cat','i']).frac.max().reset_index(); print('recordings with >=1 channel OFF for more than half the time:'); print(rec.groupby('cat').frac.apply(lambda x:'%d/%d'%((x>0.5).sum(),len(x))).to_string())
c=fc[fc.cat=='CONNECTED'].merge(m[['i','person']],on='i'); print('CONNECTED channels flagged >50%:',c[c.frac>0.5].groupby('person').size().to_dict())

L=M[lp]; print('LP/DRAG flagged: in holds %.4f, in gaps %.4f; pointer-locked channels in holds %.4f'%((L[L.ptr==1].s2>=.5).mean(),(L[L.ptr==0].s2>=.5).mean(),(M[lp&locked&(M.ptr==1).values].s2>=.5).mean()))
