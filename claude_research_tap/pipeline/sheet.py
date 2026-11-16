import sys, os; sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
EV=pd.read_csv(sys.argv[1]); out=sys.argv[2]
cols=json.load(open('g10/cols.json')); A_=[k for k,c in enumerate(cols) if 'acc_psd[0]' in c]; E_=[k for k,c in enumerate(cols) if 'emg_psd[0]' in c]
def sheet(sel,title,fn):
    fig,axs=plt.subplots(6,8,figsize=(24,13)); 
    for ax,x in zip(axs.ravel(),sel.itertuples()):
        d=np.load(f'g10/{x.i:05d}.npz'); X=d['X']; p=d['A'][:,1]; c=int((x.t_ptr if x.tier=='PRESS' else x.t_on)//10); lo=max(0,c-100); hi=min(len(p),c+100); t=(np.arange(lo,hi)-c)*10
        a=np.log1p(X[lo:hi][:,A_].sum(1)); e=np.log1p(X[lo:hi][:,E_]).mean(1)
        ax.fill_between(t,0,1,where=p[lo:hi]>0,transform=ax.get_xaxis_transform(),color='gold',alpha=.35,lw=0)
        ax.plot(t,(a-a.min())/(np.ptp(a)+1e-9),'C0',lw=1); ax.plot(t,(e-e.min())/(np.ptp(e)+1e-9)-1.1,'C3',lw=1)
        ax.axvline(0,color='k',lw=.6,ls=':'); ax.set_yticks([]); ax.set_xticks([-1000,0,1000]); ax.tick_params(labelsize=6)
        ax.set_title(f'{x.sub} s={x.score:.2f} accRange={np.ptp(a):.1f}',fontsize=7)
    fig.suptitle(title+'   (blue: log ACC PSD sum, red: mean log EMG PSD, each scaled to its own range; yellow: PointerState high; dotted: the event)',fontsize=11)
    plt.tight_layout(); plt.savefig(fn,dpi=60); plt.close()
P=EV[(EV.tier=='PRESS')&(EV.score<0.05)].sample(48,random_state=3); sheet(P,'Pointer presses with NO sensor support (suspected false presses)',f'{out}/sheet_unsupported_presses.png')
O=EV[(EV.tier=='ORPHAN')&(EV.score>=0.9)].sample(48,random_state=3); sheet(O,'Strong sensor events with NO pointer press nearby (suspected missed presses)',f'{out}/sheet_orphans.png')
Q=EV[(EV.tier=='PRESS')&(EV.score>=0.9)].sample(48,random_state=3); sheet(Q,'Reference: confirmed presses',f'{out}/sheet_confirmed.png')
