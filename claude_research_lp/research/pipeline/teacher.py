"""Offline teacher states per channel-frame: 0 REST, 1 ACTIVE, 2 OFF, -1 unknown.
OFF    : out-of-fold contact score >= 0.5, or WL below the amplifier floor region (log10 WL < 2.2)
ON-sure: contact score < 0.2 and log10 WL >= 2.35
floor  : 5th percentile of amplitude level A = 20*log10(WL) over ON-sure frames of the same recording and channel
REST   : ON-sure and A <= floor + 3 dB ;  ACTIVE: ON-sure and A >= floor + 6 dB ; anything else unknown"""
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from frames import *
R_DB,A_DB=3.0,6.0
if __name__=='__main__':
    m=meta(); X=np.load('fr_X.npy',mmap_mode='r'); M=pd.read_pickle('fr_M.pkl'); S=pd.read_pickle('off3_scores.pkl'); M['s']=S.s2.values; M['fold']=S.fold.values
    M['lw']=np.asarray(X[:,0]); A=20*M.lw
    off=(M.s>=0.5)|(M.lw<2.2); on=(M.s<0.2)&(M.lw>=2.35)
    g=A.where(on).groupby([M.i,M.ch]); F=g.transform(lambda x:np.nanpercentile(x,5) if x.notna().sum()>=40 else np.nan)
    lab=np.full(len(M),-1,np.int8); lab[(on&(A<=F+R_DB)).values]=0; lab[(on&(A>=F+A_DB)).values]=1; lab[off.values]=2
    M['floor']=F; M['A']=A; M['lab']=lab; M[['i','ch','t','ptr','s','fold','lw','floor','lab']].to_pickle('teacher.pkl')
    M=M.merge(m[['i','cat','name']],on='i',how='left'); pd.set_option('display.width',250)
    print('teacher state shares by folder (channel-frames)'); print(pd.crosstab(M.cat,M.lab,normalize='index').rename(columns={-1:'unknown',0:'REST',1:'ACTIVE',2:'OFF'}).round(3).to_string())
    L=M[M.cat.str.contains('LONG|DRAG')&(M.lab>=0)]
    print('LP/DRAG: teacher state vs pointer'); print(pd.crosstab(L.ptr,L.lab,normalize='index').round(3).to_string())
    # any-channel activity vs pointer per frame
    w=M[M.cat.str.contains('LONG|DRAG')].pivot_table(index=['cat','i','t'],columns='ch',values='lab',aggfunc='first'); p=M[M.cat.str.contains('LONG|DRAG')&(M.ch==0)].set_index(['cat','i','t']).ptr
    anyact=(w==1).any(axis=1); allrest=(w==0).all(axis=1); d=pd.DataFrame(dict(anyact=anyact,allrest=allrest,ptr=p))
    print('LP/DRAG frames: P(any channel ACTIVE | pointer high) by folder:',d[d.ptr==1].groupby('cat').anyact.mean().round(3).to_dict()); print('  P(any channel ACTIVE | pointer low):',d[d.ptr==0].groupby('cat').anyact.mean().round(3).to_dict()); print('  P(all REST | pointer low):',d[d.ptr==0].groupby('cat').allrest.mean().round(3).to_dict())
