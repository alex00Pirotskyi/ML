"""Weakly supervised OFF detector, round 1: channel-frames in DISCONNECTED/LOOSE/TABLE vs all other folders. Person-grouped CV."""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
import lightgbm as lgb
m=meta(); m['g4']=m.person.str.extract(r'^([a-z]+)')[0].str[:4]
X=np.load('fr_X.npy',mmap_mode='r'); M=pd.read_pickle('fr_M.pkl').merge(m[['i','cat','person','g4']],on='i')
OWN=[FEATS.index(f) for f in ['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43','sd1','rng1','min3','max3','sd3']]
ug=np.sort(M.g4.unique()); fo={g:k%5 for k,g in enumerate(np.random.default_rng(1).permutation(ug))}; M['fold']=M.g4.map(fo)
y=M.cat.isin(['DISCONNECTED','LOOSE','TABLE']).values; r=np.random.default_rng(0).random(len(M))
keep=y|M.cat.isin(['CONNECTED','NOTHING']).values|(r<0.25)
sc=np.zeros(len(M),np.float32)
for k in range(5):
    ii=np.where(keep&(M.fold!=k).values)[0]
    mdl=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.08,num_leaves=15,min_child_samples=200,colsample_bytree=0.7,subsample=0.7,subsample_freq=1,verbose=-1,n_jobs=2)
    mdl.fit(np.ascontiguousarray(X[ii][:,OWN]),y[ii]); te=np.where((M.fold==k).values)[0]; sc[te]=mdl.predict_proba(np.ascontiguousarray(X[te][:,OWN]))[:,1]
M['s']=sc; M[['i','ch','t','s','fold']].to_pickle('off1_scores.pkl')
fc=M.groupby(['cat','i','ch']).s.median().reset_index()
pd.set_option('display.width',250)
print('file-channel median score distribution by folder'); print(fc.groupby('cat').s.describe(percentiles=[.05,.25,.5,.75,.95]).round(3).to_string())
c=fc[fc.cat.isin(['CONNECTED','DISCONNECTED','LOOSE','TABLE'])].merge(m[['i','person','orient']],on='i')
P=c.pivot_table(index=['person','cat','orient'],columns='ch',values='s').round(2); print(P.to_string())
