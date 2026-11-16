import sys; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sig import *
m=pd.read_pickle('meta.pkl'); P=m[m.pos==1]
rows=[]
for r in P.itertuples():
    X,A=load(r.i); t=A[:,0]-A[0,0]; p=A[:,2]; up,dn=edges(p)
    T=t[-1]
    mt=t[np.where(np.diff(A[:,1])>0)[0]+1]
    for k,u in enumerate(up):
        d=dn[dn>u]; d=d[0] if len(d) else len(t)-1
        seg=A[u:d+1,4:6]; 
        trav=np.abs(np.diff(seg,axis=0)).sum(0) if len(seg)>1 else np.zeros(2)
        net=seg[-1]-seg[0]
        rows.append(dict(i=r.i,k=k,t_up=t[u],t_dn=t[d],dur=t[d]-t[u],open_end=(d==len(t)-1 and p[-1]>0),start0=u<3,travx=trav[0],travy=trav[1],netx=net[0],nety=net[1],
            x0=seg[0,0],y0=seg[0,1],gap_prev=t[u]-t[dn[dn<u][-1]] if (dn<u).any() else np.nan,
            dmet=(mt[np.abs(mt-t[u]).argmin()]-t[u]) if len(mt) else np.nan,T=T))
E_=pd.DataFrame(rows); E_.to_pickle('presses.pkl')
d=E_.merge(m[['i','cat','gest','bpm','u','name']],on='i')
d['sub']=d.name.str.extract(r'(SoftTap|MidTap|StrongTap|DoubleTaps|SlowPinch|Tapincontraction|[Tt]aponmotion|SwipeSoft|SwipeMID|SwipeStrong|SwipePushPull)')[0].fillna('other')
pd.set_option('display.width',250)
d['trav']=np.hypot(d.travx,d.travy); d['net']=np.hypot(d.netx,d.nety)
print(d.groupby(['cat','sub'])[['dur','gap_prev','trav','net']].describe(percentiles=[.05,.5,.95]).round(2).T.to_string())
print('short<30ms',(d.dur<30).groupby(d.cat).mean(), 'gap<60',(d.gap_prev<60).groupby(d.cat).mean())
print(d.groupby(['cat','bpm']).dmet.describe(percentiles=[.05,.25,.5,.75,.95]).round(0).to_string())
