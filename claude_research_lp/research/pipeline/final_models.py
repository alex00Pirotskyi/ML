"""Train the deployable students on all people (with transition augmentation) and save them as LightGBM text models."""
import sys, json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
from student_lp import SETS
from student_aug import splice_rows
import lightgbm as lgb
m=meta(); X=np.load('fr_X.npy',mmap_mode='r'); T=pd.read_pickle('teacher.pkl'); y=T.lab.values
Xa,ya,fa=splice_rows(m,T); r=np.random.default_rng(0).random(len(T)); ii=np.where((y>=0)&((y!=0)|(r<0.5)))[0]
for tag,fs in [('full27','own27'),('lean8','lean8')]:
    J=[FEATS.index(f) for f in SETS[fs]]
    Xt=np.concatenate([np.ascontiguousarray(X[ii][:,J]),Xa[ya>=0][:,J]]); yt=np.concatenate([y[ii],ya[ya>=0]])
    mdl=lgb.LGBMClassifier(objective='multiclass',n_estimators=200,learning_rate=0.08,num_leaves=15,min_child_samples=200,colsample_bytree=0.8,subsample=0.7,subsample_freq=1,verbose=-1,n_jobs=2).fit(Xt,yt)
    mdl.booster_.save_model(f'model_{tag}.txt'); json.dump(dict(features=SETS[fs],classes=['REST','ACTIVE','OFF'],grid_ms=10,off_debounce_s=1.0),open(f'model_{tag}.json','w'),indent=1); print(tag,Xt.shape,os.path.getsize(f'model_{tag}.txt'))
