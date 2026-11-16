#!/usr/bin/env python3
"""Tests for subject leakage protections and alias validation."""
from pathlib import Path
import csv,json,tempfile
from split_v4 import assign_groups,read_aliases

def main():
  rows=[{'source':f'{k}/session{i}.csv.gz','category':k,'subject_raw':person,
         'subject_group':person,'gold_events':0}
         for j,person in enumerate(['A','B','C','D','E','F','G','H','I','J','K','L'])
         for i,k in enumerate(['TAP','FINGER','SWIPE','WRIST','NOTHING'])]
  l1,n1=assign_groups(rows,20261116)
  mapping={p:set(r['split'] for r in rows if r['subject_group']==p)
           for p in {r['subject_group'] for r in rows}}
  assert all(len(v)==1 for v in mapping.values())
  assert set(r['split'] for r in rows)=={'train','val','test'}
  snapshot=[(r['source'],r['split']) for r in rows]
  assign_groups(rows,20261116)
  assert snapshot==[(r['source'],r['split']) for r in rows]
  with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'alias.csv'
    with p.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=['subject_raw','subject_group']);w.writeheader()
      w.writerows([{'subject_raw':'ohad left','subject_group':'ohad'},
                   {'subject_raw':'ohadright','subject_group':'ohad'}])
    x=read_aliases(p,True)
    assert x['ohad left']==x['ohadright']=='ohad'
    try:read_aliases(None,True)
    except ValueError:pass
    else:raise AssertionError('Production must require confirmed aliases')
  print('V4_NO_SUBJECT_LEAK_SPLIT_TESTS_OK')

if __name__=='__main__':main()
