"""Recording metadata and cross-validation folds -> meta.pkl
Run after build_cache.py and make_file_folds.py: the folds come from folds.csv (name, fold) in the working directory,
gesture-aware FILE folds (never person folds). The 'grp' column (first 4 letters of the User string) is kept for reports."""
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
if not os.path.exists('folds.csv'):
    raise SystemExit('folds.csv missing: run make_file_folds.py first (file folds only, never person folds)')
f=pd.read_csv('folds.csv'); m=m.merge(f[['name','fold']].drop_duplicates('name'),on='name',how='left')
m.to_pickle('meta.pkl'); print(len(m),m.fold.value_counts().to_dict())
