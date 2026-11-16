#!/usr/bin/env python3
"""Subject-disjoint versioned split audit for verified-only TAP V4 sidecars.

An explicit human-confirmed alias map is mandatory for --release. No per-row
random splits and no assumption that hand/PSD/"right" aliases are new people.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd


def normalize(s):
    return re.sub(r'\s+',' ',str(s).strip().casefold())


def subject_of(path):
    d=pd.read_csv(path,usecols=['User'],nrows=1,compression='infer')
    if d.empty:raise ValueError(f'No User field: {path}')
    s=normalize(d.User.iloc[0])
    if not s:raise ValueError(f'Invalid subject: {path}')
    return s


def read_aliases(path,required):
    if path is None:
        if required:raise ValueError('--release requires --aliases mapping reviewed by a human')
        return {}
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig',newline='')))
    if any('subject_raw' not in x or 'subject_group' not in x for x in rows):
        raise ValueError('aliases CSV must provide subject_raw and subject_group')
    mapping={}
    for row in rows:
        raw=normalize(row['subject_raw']);group=normalize(row['subject_group'])
        if not raw or not group:raise ValueError('Blank alias key/group')
        if raw in mapping and mapping[raw]!=group:raise ValueError('Conflicting human alias grouping')
        mapping[raw]=group
    return mapping


def assign_groups(rows,seed=20261116):
    """Deterministic greedy, subject-disjoint category-stratified allocations."""
    grouped=defaultdict(list)
    for r in rows:grouped[r['subject_group']].append(r)
    cats=sorted({r['category'] for r in rows});targets={'train':.70,'val':.15,'test':.15}
    totals={c:sum(r['category']==c for r in rows) for c in cats}
    loads={k:Counter() for k in targets};subject_load=Counter();mapping={}
    names=sorted(grouped,key=lambda g:(-len(grouped[g]),hashlib.sha256(f'{seed}|{g}'.encode()).hexdigest()))
    for name in names:
        cs=Counter(r['category'] for r in grouped[name])
        def loss(split):
            # Compare the GLOBAL distribution after each candidate assignment,
            # not just the local target split. The latter incorrectly sends
            # nearly everything to val/test because they have small targets.
            fit=0.
            for k,target in targets.items():
                future=loads[k]+(cs if k==split else Counter())
                fit+=sum(((future[c]-target*totals[c])/max(1,totals[c]))**2 for c in cats)
                fit+=.10*((subject_load[k]+int(k==split)-target*len(names)) /
                          max(1,len(names)))**2
                fit+=.10*((sum(future.values())-target*len(rows)) /
                          max(1,len(rows)))**2
            return fit
        # Penalize val/test undercoverage early with deterministic priority.
        best=min(targets,key=lambda k:(loss(k),hashlib.sha256(f'{seed}|{name}|{k}'.encode()).hexdigest()))
        mapping[name]=best;loads[best].update(cs);subject_load[best]+=1
    for row in rows:row['split']=mapping[row['subject_group']]
    assert len({(r['subject_group'],r['split']) for r in rows})==len(mapping)
    return loads,subject_load


def generate(manifest_file,root,out,aliases=None,release=False):
    manifest_file=Path(manifest_file).resolve();root=Path(root).resolve();out=Path(out).resolve()
    if root.name!='DATA_TAP':raise ValueError('Source must be DATA_TAP')
    manifest=[json.loads(x) for x in manifest_file.read_text().splitlines()]
    m=[x for x in manifest if x['status']=='OK']
    map_user=read_aliases(aliases,release)
    raw_users={};rows=[]
    for e in m:
        path=root/e['source'];subject=subject_of(path)
        raw_users[subject]=raw_users.get(subject,0)+1
        group=map_user.get(subject,subject)
        rows.append({'source':e['source'],'source_sha256':e['source_sha256'],
                     'category':e['category'],'subject_raw':subject,
                     'subject_group':group,
                     'gold_events':e.get('gold_events',0)})
    if release and set(raw_users)-set(map_user):
        raise ValueError('Unmapped user aliases in production release: '+str(sorted(set(raw_users)-set(map_user))[:8]))
    out.mkdir(parents=True,exist_ok=True)
    template=[{'subject_raw':x,'subject_group':map_user.get(x,x),'recordings':n,
               'human_verified':'no' if not release else 'yes'} for x,n in sorted(raw_users.items())]
    pd.DataFrame(template).to_csv(out/'subject_aliases_review_template.csv',index=False)
    loads,num_subjects=assign_groups(rows)
    pd.DataFrame(rows).to_csv(out/'source_splits.csv',index=False)
    per_split=Counter(r['split'] for r in rows)
    stats={'files':len(rows),'canonical_subjects':len({x['subject_group'] for x in rows}),
           'raw_subject_variants':len(raw_users),
           'subject_disjoint':True,'alias_map_confirmed':release,
           'gold_events_total':sum(int(x['gold_events']) for x in rows),
           'files_per_split':dict(per_split),'subjects_per_split':dict(num_subjects),
           'classes_per_split':{k:dict(v) for k,v in loads.items()},
           'warning':'No true physical-event evaluation set is certified by splitting weak folder labels'}
    (out/'split_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
    return stats


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--input-dir',type=Path,default=Path('DATA/DATA_TAP'))
    a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--aliases',type=Path)
    a.add_argument('--release',action='store_true')
    x=a.parse_args()
    generate(x.manifest,x.input_dir,x.out_dir,x.aliases,x.release)

if __name__=='__main__':main()
