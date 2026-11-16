"""Event-level evaluation of student out-of-fold frame scores.
usage: student_eval.py <events.csv> <score.npy> [<score.npy> ...]
Event rule: 30 ms smoothing, score peaks at least 150 ms apart, above a threshold set by the false-event rate in negative folders."""
import sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
REFR=15
from scipy.signal import find_peaks
def events(s):
    """peaks of the smoothed score at least 150 ms apart -> (frame index, height)"""
    pk,_=find_peaks(np.r_[0,s,0],distance=REFR,height=0.01); pk=pk-1; return pk,s[pk]
def evaluate(scf,EV,m,fid,RATES=(0.2,0.5,1,2,5,10)):
    sc=np.load(scf).astype(np.float32)
    bounds=pd.Series(np.arange(len(fid))).groupby(fid).agg(['first','last'])
    tr={i:uniform_filter1d(sc[a:b+1],3) for i,(a,b) in zip(bounds.index,bounds.values)}
    PK={i:events(tr[i]) for i in tr}
    neg=m[m.pos==0]; negmin=sum(len(tr[i]) for i in neg.i)/6000
    negh=np.sort(np.concatenate([PK[i][1] for i in neg.i]))[::-1]
    evg=dict(tuple(EV.groupby('i'))); res={}
    for rate in RATES:
        th=negh[int(rate*negmin)]
        hit=[];lat=[];trusted=[];grp=[];sub=[];extra=0;posmin=0;negsub={}
        for r in m.itertuples():
            pk,h=PK[r.i]; q=h>th; det=pk[q]*10.0
            if not r.pos: negsub[r.sub]=negsub.get(r.sub,0)+len(det); continue
            posmin+=len(tr[r.i])/6000
            e=evg.get(r.i)
            if e is None: continue
            P=e[e.tier=='PRESS']; expl=np.zeros(len(det),bool)
            for x in e[e.tier.isin(['PRESS','EDGE'])].itertuples():
                expl|=(det>=min(x.t_ptr,x.t_on)-100)&(det<=max(x.t_rel,x.t_off)+250)
            for x in e[e.tier.isin(['ORPHAN','INPRESS'])&(e.score>=0.5)].itertuples(): expl|=np.abs(det-x.t_on)<=300
            extra+=int((~expl).sum()); s=tr[r.i]
            for x in P.itertuples():
                c=np.where((det>=x.t_ptr-100)&(det<=x.t_ptr+250))[0]
                hit.append(len(c)>0); trusted.append(x.p_real>=0.9); grp.append(r.grp); sub.append(r.sub)
                if len(c):
                    j=int(det[c[0]]//10)
                    while j>0 and s[j-1]>th: j-=1
                    lat.append(j*10-x.t_on)
                else: lat.append(np.nan)
        H=pd.DataFrame(dict(hit=hit,tr=trusted,grp=grp,sub=sub,lat=lat))
        pg=H[H.tr].groupby('grp').hit.mean()
        res[rate]=dict(thr=float(th),recall_all=H.hit.mean(),recall_trusted=H[H.tr].hit.mean(),recall_tap=H[H.tr&~H['sub'].str.startswith('swipe')].hit.mean(),recall_swipe=H[H.tr&H['sub'].str.startswith('swipe')].hit.mean(),
            person_macro=pg.mean(),person_p10=pg.quantile(0.1),latency_med=H[H.tr].lat.median(),latency_p90=H[H.tr].lat.quantile(0.9),unexplained_pos_per_min=extra/posmin,
            by_sub=H[H.tr].groupby('sub').hit.mean().round(3).to_dict(),neg_by_sub={k:round(v/(m[(m.pos==0)&(m['sub']==k)].i.map(lambda i:len(tr[i])).sum()/6000),2) for k,v in negsub.items()})
    return res
if __name__=='__main__':
    from label3 import load_meta
    m=load_meta(); m=m[m.fold.notna()]; EV=pd.read_csv(sys.argv[1]); fid=np.load('st_idx.npz')['fid']
    rows=[]
    for f in sys.argv[2:]:
        R=evaluate(f,EV,m,fid); json.dump(R,open(f.replace('.npy','.eval.json'),'w'),indent=1)
        for rate,d in R.items(): rows.append(dict(model=os.path.basename(f)[9:-4],neg_per_min=rate,**{k:v for k,v in d.items() if not isinstance(v,dict)}))
    pd.set_option('display.width',250); print(pd.DataFrame(rows).round(3).to_string(index=False))
