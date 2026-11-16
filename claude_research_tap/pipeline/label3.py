"""Fusion labeler v3.  usage: label3.py <scores.pkl> <key> <outdir>
Pointer presses x out-of-fold sensor score -> event table with calibrated confidence.
Changes vs v1: one-to-one press/peak assignment, 100 ms peak spacing, mixture-calibrated P(real press | score),
orphan/in-press handling, separate onset time and extent, edge-aware peaks, pointer-bounce merge,
second matching pass for presses whose sensor event is 150-400 ms away."""
import sys, os, json, pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.isotonic import IsotonicRegression
TOL=150; TOL2=400; EDGE_MS=300; BOUNCE=80; H_MIN=0.03
SUBRE=r'(SoftTap|MidTap|StrongTap|DoubleTaps|SlowPinch|Tapincontraction|[Tt]aponmotion|Tapontheband|TaponArm|TaponHand|Thumb|IndexFinger|MidFinger|RingFinger|Pinky|Fist&Stretch|WristCircle|Twist|ArmCircle|SwipeSoft|SwipeMID|SwipeStrong)'
def load_meta():
    m=pd.read_pickle('meta.pkl'); m['sub']=m.name.str.extract(SUBRE)[0].str.lower(); m.loc[m['sub'].isna(),'sub']=m.cat.str.lower()+'_other'; return m
def peaks(S,i,key):
    s=uniform_filter1d(S[i][key],5); pk,_=find_peaks(np.r_[0,s,0],distance=10,height=H_MIN); pk=pk-1; return S[i]['t'][pk].astype(float),s[pk]
def assign(up,t,h,TOL=TOL):
    """one-to-one: maximise score, small penalty for distance"""
    out=np.full(len(up),-1)
    if not len(up) or not len(t): return out
    D=np.abs(t[None,:]-up[:,None]); ok=D<=TOL
    C=np.where(ok,-(h[None,:]-D/1500.0)-1.0,1e3)
    r,c=linear_sum_assignment(C)
    for a,b in zip(r,c):
        if ok[a,b]: out[a]=b
    return out
def run(scores,key,out):
    os.makedirs(out,exist_ok=True)
    m=load_meta(); pr=pd.read_pickle('presses.pkl'); prg=dict(tuple(pr.groupby('i'))); S=pickle.load(open(scores,'rb')); m=m[m.i.isin(S)].copy()
    PK={i:peaks(S,i,key) for i in m.i}
    span=lambda i:(S[i]['t'][-1]-S[i]['t'][0])/60000
    negmin=sum(span(i) for i in m[m.pos==0].i); posmin=sum(span(i) for i in m[m.pos==1].i)
    negh=np.sort(np.concatenate([PK[i][1] for i in m[m.pos==0].i]))[::-1]
    thr={r:float(negh[int(r*negmin)]) for r in (0.5,1,2,5)}
    T_HI=thr[1]
    ev=[]; null=[]; rng=np.random.default_rng(0)
    for r in m[m.pos==1].itertuples():
        t,h=PK[r.i]; p=prg[r.i]; up=p.t_up.values.astype(float); dn=p.t_dn.values.astype(float); t0,t1=S[r.i]['t'][0],S[r.i]['t'][-1]
        # merge pointer bounces (release -> press again within BOUNCE ms)
        keep=np.r_[True,(up[1:]-dn[:-1])>=BOUNCE]; grp_=np.cumsum(keep)-1
        up=up[keep]; dn=np.array([dn[grp_==g].max() for g in range(grp_.max()+1)]) if len(up) else dn
        a=assign(up,t,h); far=np.zeros(len(up),bool)
        used=np.zeros(len(t),bool); used[a[a>=0]]=True
        # second pass: unsupported presses <-> unused strong peaks up to TOL2 away, not inside another press
        un=np.where((a<0)|(np.where(a>=0,h[np.clip(a,0,None)],0)<0.05))[0] if len(t) else np.array([],int)
        fr=np.where(~used&(h>=thr[2]))[0] if len(t) else np.array([],int)
        if len(un) and len(fr):
            fr=np.array([j for j in fr if not any((up[k]<t[j]<dn[k]) for k in range(len(up)) if k not in un)],int)
        if len(un) and len(fr):
            a2=assign(up[un],t[fr],h[fr],TOL2)
            for q,b in zip(un,a2):
                if b>=0:
                    if a[q]>=0: used[a[q]]=False
                    a[q]=fr[b]; used[fr[b]]=True; far[q]=True
        for k,(u,d) in enumerate(zip(up,dn)):
            j=a[k]; s=h[j] if j>=0 else 0.0; on=t[j] if j>=0 else u; edge=(u-t0<EDGE_MS)or(t1-u<EDGE_MS)
            tier='EDGE' if (edge and s<0.5) else 'PRESS'
            ev.append(dict(i=r.i,k=k,tier=tier,t_ptr=u,t_rel=d,t_on=on,t_off=max(on+40,d+(on-u)) if not far[k] else max(on+40,on+(d-u)),score=s,far=bool(far[k]),edge=bool(edge)))
        for j in np.where(~used)[0]:
            du=np.abs(up-t[j]).min() if len(up) else 1e9; inside=((up<t[j])&(dn>t[j])).any()
            if du<=TOL: tier='DUP'          # second peak next to a matched press
            elif inside: tier='INPRESS'
            else: tier='ORPHAN'
            ev.append(dict(i=r.i,k=-1,tier=tier,t_ptr=np.nan,t_rel=np.nan,t_on=t[j],t_off=t[j]+150,score=h[j]))
        # null windows: random times >=500 ms from any press
        for _ in range(12):
            x=rng.uniform(t0+200,t1-200)
            if x-t0<EDGE_MS+300 or t1-x<EDGE_MS+300: continue
            if len(up) and (np.abs(up-x).min()<500 or ((up<x)&(dn+300>x)).any()): continue
            c=np.abs(t-x)<=TOL; null.append(h[c].max() if c.any() else 0.0)
    for r in m[m.pos==0].itertuples():
        t,h=PK[r.i]
        for j in range(len(t)): ev.append(dict(i=r.i,k=-1,tier='NEG',t_ptr=np.nan,t_rel=np.nan,t_on=t[j],t_off=t[j]+150,score=h[j]))
    EV=pd.DataFrame(ev).merge(m[['i','name','cat','sub','grp','fold','date']],on='i')
    # --- calibrated P(real press | score): two-component mixture, null = random background windows of TAP/SWIPE files
    P=EV[EV.tier=='PRESS']; null=np.array(null)
    bins=np.r_[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0001]
    f=np.histogram(P.score,bins)[0]/len(P); f0=np.histogram(null,bins)[0]/len(null)
    pi0=min(1.0,f[0]/f0[0])                      # false-press fraction (upper bound)
    post=np.clip(1-pi0*f0/np.maximum(f,1e-9),0,1)
    iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit((bins[:-1]+bins[1:])/2,post,sample_weight=f*len(P)+1)
    EV['p_real']=np.nan; EV.loc[EV.tier=='PRESS','p_real']=iso.predict(P.score.values)
    EV['strong']=EV.score>=T_HI
    EV['dt']=EV.t_on-EV.t_ptr
    EV.to_csv(f'{out}/events.csv',index=False)
    cal=pd.DataFrame(dict(score_lo=bins[:-1],score_hi=bins[1:],press_frac=f,null_frac=f0,p_real=iso.predict((bins[:-1]+bins[1:])/2),n_press=(f*len(P)).round().astype(int)))
    summ=dict(key=key,neg_minutes=negmin,pos_minutes=posmin,thr_at_neg_false_per_min=thr,false_press_fraction_upper=pi0,
              press_recall={str(r):{c:float((P[P.cat==c].score>=thr[r]).mean()) for c in ('TAP','SWIPE')} for r in thr},
              tiers=EV[EV.tier!='NEG'].tier.value_counts().to_dict(),
              presses=dict(n=len(P),strong=int(P.score.ge(T_HI).sum()),p_ge_0_9=int((EV.p_real>=0.9).sum()),p_lt_0_5=int((EV.p_real<0.5).sum())),
              orphans_strong_per_min=float(((EV.tier=='ORPHAN')&EV.strong).sum()/posmin),inpress_strong=int(((EV.tier=='INPRESS')&EV.strong).sum()),
              far_matches=int(P.far.fillna(False).sum()),edge_unknown=int((EV.tier=='EDGE').sum()),
              shared_onsets=int(P[P.score>0].duplicated(['i','t_on']).sum()))
    json.dump(summ,open(f'{out}/summary.json','w'),indent=1); cal.to_csv(f'{out}/calibration.csv',index=False)
    return EV,summ,cal
if __name__=='__main__':
    EV,summ,cal=run(sys.argv[1],sys.argv[2],sys.argv[3]); print(json.dumps(summ,indent=1)); pd.set_option('display.width',200); print(cal.round(3).to_string())
