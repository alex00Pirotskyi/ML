"""Independent checks of a causal student: contact protocol, rest recordings, confirmed tap/swipe onsets, long-press holds.
usage: eval_lp.py <student tag>"""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
tag=sys.argv[1]; m=meta(); T=pd.read_pickle('teacher.pkl'); P=np.load(f'stu_{tag}.npy'); T['pred']=P.argmax(1).astype(np.int8); T['p_off']=P[:,2]; T['p_act']=P[:,1]
T=T.merge(m[['i','cat','name','person']],on='i',how='left'); pd.set_option('display.width',250)
N={0:'REST',1:'ACTIVE',2:'OFF'}
print('== predicted state shares by folder (channel-frames, held-out people)'); print(pd.crosstab(T.cat,T.pred.map(N),normalize='index').round(4).to_string())
# recording-level contact
fc=T.groupby(['cat','i','ch']).pred.apply(lambda x:(x==2).mean()).reset_index(); rec=fc.groupby(['cat','i']).pred.max().reset_index()
print('== recordings with >=1 channel predicted OFF more than half the time:',rec.groupby('cat').pred.apply(lambda x:'%d/%d'%((x>0.5).sum(),len(x))).to_dict())
c=T[T.cat.isin(['CONNECTED','NOTHING'])]; w=c.pivot_table(index=['i','t'],columns='ch',values='pred',aggfunc='first')
print('== CONNECTED/NOTHING frames: all three channels REST %.4f | any ACTIVE %.4f | any OFF %.4f'%((w==0).all(axis=1).mean(),(w==1).any(axis=1).mean(),(w==2).any(axis=1).mean()))
pc=c.groupby('person').pred.apply(lambda x:(x==0).mean()); print('   REST share per person: min %.3f median %.3f'%(pc.min(),pc.median()),pc.round(3).to_dict())
# gestures: confirmed onsets and quiet windows
EV=pd.read_csv('tap_events/events.csv',low_memory=False); import json; TH=json.load(open('tap_events/summary.json'))['thr_at_neg_false_per_min']['1']
G=T[T.cat.isin(['TAP','SWIPE'])]; evg=dict(tuple(EV[EV.tier!='NEG'].groupby('name'))); hit=[];quiet_any=[];quiet_all=[];per_ch=[]
for (i,name,cat),d in G.groupby(['i','name','cat']):
    e=evg.get(name)
    if e is None: continue
    w=d.pivot_table(index='t',columns='ch',values='pred',aggfunc='first'); t=w.index.values; act=(w==1).values; rest=(w==0).values
    on=e[(e.tier=='PRESS')&(e.score>=TH)]
    for x in on.itertuples():
        q=(t>=x.t_on)&(t<=x.t_on+300)
        if q.any(): hit.append((cat,act[q].any())); per_ch.append(act[q].any(0).sum())
    busy=np.zeros(len(t),bool)
    for x in e[e.tier.isin(['PRESS','EDGE'])].itertuples(): busy|=(t>=min(x.t_ptr,x.t_on)-600)&(t<=max(x.t_rel,x.t_off)+600)
    for x in e[e.tier.isin(['ORPHAN','INPRESS','DUP'])&(e.score>=0.3)].itertuples(): busy|=np.abs(t-x.t_on)<=600
    q=~busy&(t>1500)
    if q.any(): quiet_any+=list(act[q].any(1)); quiet_all+=list(rest[q].all(1))
H=pd.DataFrame(hit,columns=['cat','hit']); print('== confirmed gesture onsets: any channel ACTIVE within 300 ms:',H.groupby('cat').hit.agg(['mean','size']).round(3).to_dict('index'),'| channels active per onset:',np.bincount(per_ch,minlength=4)/len(per_ch))
print('== quiet windows in gesture recordings (>=600 ms from any gesture evidence): any channel ACTIVE %.3f | all REST %.3f | frames %d'%(np.mean(quiet_any),np.mean(quiet_all),len(quiet_any)))
# long press
L=T[T.cat.str.contains('LONG|DRAG')]; w=L.pivot_table(index=['cat','i','t'],columns='ch',values='pred',aggfunc='first'); p=L[L.ch==0].set_index(['cat','i','t']).ptr
d=pd.DataFrame(dict(anyact=(w==1).any(axis=1),allrest=(w==0).all(axis=1),anyoff=(w==2).any(axis=1),ptr=p))
print('== long press: P(any ACTIVE | pointer high):',d[d.ptr==1].groupby('cat').anyact.mean().round(3).to_dict()); print('   P(any ACTIVE | pointer low):',d[d.ptr==0].groupby('cat').anyact.mean().round(3).to_dict())
ok=T.lab>=0; a=(T.pred[ok]==T.lab[ok]); print('== agreement with teacher by folder:',a.groupby(T.cat[ok]).mean().round(3).to_dict())
# stability: state changes per minute per channel (raw argmax)
ch=T.sort_values(['i','ch','t']); flips=(ch.pred!=ch.groupby(['i','ch']).pred.shift()).groupby(ch.cat).mean(); print('== share of frames where the state changes vs previous frame:',flips.round(4).to_dict())
