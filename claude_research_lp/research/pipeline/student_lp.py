"""Causal per-channel 3-state student (REST / ACTIVE / OFF) trained on teacher states. Person-grouped 5-fold, out-of-fold predictions.
usage: student_lp.py <tag> <feature list name>"""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
import lightgbm as lgb, pickle
SETS={'own27':['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43','sd1','rng1','min3','max3','sd3','min6','min10','max10','sd10','rise10','Pmin10','Prise10'],
      'inst15':['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43'],
      'short20':['lw','ht','wlq','Pi','Pm','dPim','f0','f1','f2','f3','f4','r01','r21','r32','r43','sd1','rng1','min3','max3','sd3'],
      'lean8':['lw','Pm','r01','r32','sd3','min10','rise10','max10']}
if __name__=='__main__':
    tag,fs=sys.argv[1],sys.argv[2]; J=[FEATS.index(f) for f in SETS[fs]]
    X=np.load('fr_X.npy',mmap_mode='r'); T=pd.read_pickle('teacher.pkl'); y=T.lab.values; fold=T.fold.values
    r=np.random.default_rng(0).random(len(T)); keep=(y>=0)&((y!=0)|(r<0.5))
    P=np.zeros((len(T),3),np.float32); models=[]
    for k in range(5):
        ii=np.where(keep&(fold!=k))[0]
        mdl=lgb.LGBMClassifier(objective='multiclass',n_estimators=int(sys.argv[3]) if len(sys.argv)>3 else 200,learning_rate=0.08,num_leaves=int(sys.argv[4]) if len(sys.argv)>4 else 15,min_child_samples=200,colsample_bytree=0.8,subsample=0.7,subsample_freq=1,verbose=-1,n_jobs=2)
        mdl.fit(np.ascontiguousarray(X[ii][:,J]),y[ii]); te=np.where(fold==k)[0]
        for a in range(0,len(te),500000): P[te[a:a+500000]]=mdl.predict_proba(np.ascontiguousarray(X[te[a:a+500000]][:,J]))
        models.append(mdl)
    np.save(f'stu_{tag}.npy',P); pickle.dump(models,open(f'stu_{tag}_models.pkl','wb'))
    pred=P.argmax(1); ok=y>=0
    from sklearn.metrics import confusion_matrix
    C=confusion_matrix(y[ok],pred[ok]); rec=C.diagonal()/C.sum(1); prec=C.diagonal()/C.sum(0)
    print(tag,fs,'agreement with teacher (unknown excluded): recall REST %.3f ACTIVE %.3f OFF %.3f | precision %.3f %.3f %.3f | balanced acc %.3f'%(*rec,*prec,rec.mean()))
    imp=pd.Series(np.mean([m_.booster_.feature_importance('gain') for m_ in models],0),index=SETS[fs]); print((imp/imp.sum()).sort_values(ascending=False).round(3).head(8).to_dict())
