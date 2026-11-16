import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sig import *
m=pd.read_pickle('meta.pkl'); pr=pd.read_pickle('presses.pkl'); prg=dict(tuple(pr.groupby('i')))
rows=[]
for r in m[m.pos==1].itertuples():
    g,X,A=grid(r.i); n=len(g)
    a=np.log1p(X[:,ACC].sum(1)); s=gaussian_filter1d(a,1.0); on=np.r_[np.zeros(4),s[4:]-s[:-4]]
    e=np.log1p(X[:,EMG]).mean(1); se=gaussian_filter1d(e,3.0); one=np.r_[np.zeros(8),se[8:]-se[:-8]]
    p=prg[r.i]
    for k,(u,d) in enumerate(zip(p.t_up.values,p.t_dn.values)):
        c=int(u//DT); 
        if c<60 or c>n-60: continue
        w=on[c-50:c+51]; j=w.argmax(); we=one[c-50:c+51]; je=we.argmax()
        rows.append(dict(i=r.i,k=k,t=u,dacc=(j-50)*DT,hacc=w[j],demg=(je-50)*DT,hemg=we[je],base=np.median(on[max(0,c-200):c+200])))
T=pd.DataFrame(rows); T.to_pickle('timing.pkl')
T=T.merge(m[['i','cat','month','grp','date']],on='i')
pd.set_option('display.width',250)
q=[.01,.05,.25,.5,.75,.95,.99]
S_=T[T.hacc>1.0]   # clear acc impulse
print('events',len(T),'with clear ACC impulse',len(S_))
print('ACC onset time minus pointer-down (ms):'); print(S_.groupby('cat').dacc.describe(percentiles=q).round(0).to_string())
print('EMG onset minus pointer-down:'); print(T[T.hemg>0.3].groupby('cat').demg.describe(percentiles=q).round(0).to_string())
f=S_.groupby('i').dacc.agg(['median','size',lambda x:(x-x.median()).abs().median()]); f.columns=['med','n','mad']; f=f[f.n>=5]
print('per-file median offset:'); print(f.med.describe(percentiles=q).round(0).to_string()); print('within-file MAD:'); print(f['mad'].describe(percentiles=q).round(0).to_string())
S_=S_.merge(f[['med']],left_on='i',right_index=True); S_['res']=S_.dacc-S_.med
print('residual after per-file offset:'); print(S_.res.describe(percentiles=q).round(0).to_string())
print('frac |res|<=30ms',(S_.res.abs()<=30).mean().round(3),'<=50',(S_.res.abs()<=50).mean().round(3),'<=100',(S_.res.abs()<=100).mean().round(3))
print('raw: frac |dacc|<=30',(S_.dacc.abs()<=30).mean().round(3),'<=50',(S_.dacc.abs()<=50).mean().round(3),'<=100',(S_.dacc.abs()<=100).mean().round(3))
print(f.merge(m[['i','month']],left_index=True,right_on='i').groupby('month').med.describe().round(0).to_string())
ss=S_.groupby(['grp','date']).med.agg(['median','std','size']); print('per session offsets spread (std of file medians):'); print(ss['std'].describe().round(1).to_string())
