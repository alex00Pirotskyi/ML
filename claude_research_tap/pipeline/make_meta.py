"""Recording metadata, person groups and cross-validation folds -> meta.pkl
Run after build_cache.py.  Person group = first 4 letters of the normalised User string (merges aliases; not verified identity).
If folds.csv (name,fold) exists in the working directory it is used, otherwise folds are drawn with a fixed seed."""
import sys, os, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
m=meta()
x=m.name.str.extract(r'_(?P<bpm>\d+)bpm_(?P<date>\d{4}-\d\d-\d\d)_')
m['date']=x.date; m['bpm']=x.bpm.astype(float); m['month']=m.date.str[:7]
m['u']=m.user.map(norm_user); m['grp']=m.u.str[:4]
rows=[]
for r in m.itertuples():
    X,A=load(r.i); t=A[:,0]; dt=np.diff(t); ps=A[:,2]; me=A[:,1]
    rows.append(dict(i=r.i,dur=(t[-1]-t[0])/1000,dtmed=np.median(dt) if len(dt) else 0,p_edges=int((np.diff(ps)>0).sum()),m_edges=int((np.diff(me)>0).sum())))
m=m.merge(pd.DataFrame(rows),on='i')
if os.path.exists('folds.csv'):
    f=pd.read_csv('folds.csv'); m=m.merge(f[['name','fold']].drop_duplicates('name'),on='name',how='left')
else:
    ok=m[m.dur>=0.8]; ug=np.unique(ok.grp.values); perm=np.random.default_rng(0).permutation(ug); fo={g:k%5 for k,g in enumerate(perm)}
    m['fold']=m.grp.map(fo); m.loc[m.dur<0.8,'fold']=np.nan
m.to_pickle('meta.pkl'); print(len(m),m.fold.value_counts().to_dict())
