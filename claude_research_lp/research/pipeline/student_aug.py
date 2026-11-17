"""Student with transition augmentation: training also sees spliced streams (CONNECTED<->DISCONNECTED/LOOSE of the same person)
so the model learns what the causal lookback features look like right after a contact change.
usage: student_aug.py <tag> <feature set> [n_trees] [leaves]"""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
from student_lp import SETS
import lightgbm as lgb, pickle
from sklearn.metrics import confusion_matrix
def splice_rows(m,T):
    lab={k:(g.t.values,g.lab.values) for k,g in T.groupby(['i','ch'])}; fold=T.groupby('i').fold.first()
    XS=[];YS=[];FS=[]
    C=m[m.cat=='CONNECTED']
    for a in C.itertuples():
        for b in m[m.cat.isin(['DISCONNECTED','LOOSE'])&(m.person==a.person)&(m.orient==a.orient)].itertuples():
            Xa,_=load(a.i); Xb,_=load(b.i)
            for (i1,X1),(i2,X2) in (((a.i,Xa),(b.i,Xb)),((b.i,Xb),(a.i,Xa))):
                F=channel_features(np.concatenate([X1,X2])); n1=len(X1); idx=np.arange(100,len(F),2)
                src=np.where(idx<n1,i1,i2); tt=np.where(idx<n1,idx,idx-n1)*10
                for c in range(3):
                    y=np.empty(len(idx),np.int8)
                    for i_ in (i1,i2):
                        q=src==i_; t_,l_=lab[(i_,c)]; j=np.clip(np.searchsorted(t_,tt[q]),0,len(t_)-1); y[q]=l_[j]
                    XS.append(F[idx,c,:]); YS.append(y); FS.append(np.full(len(idx),int(fold[a.i])))
    return np.concatenate(XS),np.concatenate(YS),np.concatenate(FS)
if __name__=='__main__':
    tag,fs=sys.argv[1],sys.argv[2]; J=[FEATS.index(f) for f in SETS[fs]]; nt=int(sys.argv[3]) if len(sys.argv)>3 else 200; nl=int(sys.argv[4]) if len(sys.argv)>4 else 15
    m=meta(); X=np.load('fr_X.npy',mmap_mode='r'); T=pd.read_pickle('teacher.pkl'); y=T.lab.values; fold=T.fold.values
    Xa,ya,fa=splice_rows(m,T); print('augmentation rows',Xa.shape,np.bincount(ya[ya>=0]))
    r=np.random.default_rng(0).random(len(T)); keep=(y>=0)&((y!=0)|(r<0.5)); P=np.zeros((len(T),3),np.float32); models=[]
    for k in range(5):
        ii=np.where(keep&(fold!=k))[0]; qa=(fa!=k)&(ya>=0)
        Xt=np.concatenate([np.ascontiguousarray(X[ii][:,J]),Xa[qa][:,J]]); yt=np.concatenate([y[ii],ya[qa]])
        mdl=lgb.LGBMClassifier(objective='multiclass',n_estimators=nt,learning_rate=0.08,num_leaves=nl,min_child_samples=200,colsample_bytree=0.8,subsample=0.7,subsample_freq=1,verbose=-1,n_jobs=2).fit(Xt,yt)
        te=np.where(fold==k)[0]
        for a in range(0,len(te),500000): P[te[a:a+500000]]=mdl.predict_proba(np.ascontiguousarray(X[te[a:a+500000]][:,J]))
        models.append(mdl)
    np.save(f'stu_{tag}.npy',P); pickle.dump(models,open(f'stu_{tag}_models.pkl','wb'))
    pred=P.argmax(1); ok=y>=0; C=confusion_matrix(y[ok],pred[ok]); rec=C.diagonal()/C.sum(1); prec=C.diagonal()/C.sum(0)
    print(tag,fs,'agreement with teacher: recall REST %.3f ACTIVE %.3f OFF %.3f | precision %.3f %.3f %.3f | balanced acc %.3f'%(*rec,*prec,rec.mean()))
