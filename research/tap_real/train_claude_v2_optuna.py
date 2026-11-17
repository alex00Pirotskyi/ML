#!/usr/bin/env python3
"""Claude V2 TAP/SWIPE onset labels -> V6 Optuna + export_plus embedded C models.

Run once from the ML repository:
    python research/tap_real/train_claude_v2_optuna.py

Creates strong-only 0-80ms labels if absent; maps the schema without altering
source data; optimizes silver-label F1 and compact C model bytes. Results are
research candidates, not independently verified physical-onset performance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

CPU_FRACTION = 0.80  # Hardcoded; no thread CLI setting.

def cpu_count():
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        if quota != 'max':
            available = min(available, max(1, math.ceil(int(quota) / int(period))))
    except (ValueError, OSError, ZeroDivisionError):
        pass
    return max(1, available)

CPU_THREADS = max(1, int(cpu_count() * CPU_FRACTION))
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var] = str(CPU_THREADS)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'/'tap_real'))
import train_optuna_student_v6 as v6

SOURCE = ROOT/'DATA'/'DATA_TAP'
CLAUDE = ROOT/'claude_research_tap'/'results'
ORIGINAL = ROOT/'DATA'/'DATA_TAP_LABELS_CLAUDE_V2_20261117'
ADAPTED = ROOT/'research'/'tap_real'/'cache_claude_v2_v6_schema'
OUTPUT = ROOT/'research'/'tap_real'/'results_claude_v2_optuna'
POS = {'TAP', 'SWIPE'}
NEG = {'FINGER', 'WRIST', 'ARM', 'BAND', 'NOTHING'}
TRIALS = 40
FOLDS = 3
MAX_ROWS_PER_CLASS_PER_FILE = 45
TOP_K = 10

def source_sha(path):
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def prepare():
    if not ORIGINAL.is_dir():
        print('CLAUDE_EXPORT_LABELS_START', flush=True)
        subprocess.run([sys.executable, str(ROOT/'claude_research_tap'/'pipeline'/'export_labels.py'),
                        str(SOURCE),str(CLAUDE/'events_v3.csv.gz'),str(CLAUDE/'summary.json'),
                        str(ORIGINAL),'--win','0','80'], check=True, cwd=ROOT)
    frozen = pd.read_csv(CLAUDE/'folds.csv', dtype={'sha':str,'name':str,'cat':str})
    if len(frozen) != 2790 or frozen.name.duplicated().any():
        raise ValueError('Claude frozen 2790-file manifest missing or ambiguous')
    if len(list(SOURCE.glob('*/*.csv.gz'))) != 2921:
        raise ValueError('Source archive count changed; verify original dataset before training')
    stamp = hashlib.sha256((CLAUDE/'folds.csv').read_bytes()+
                           (CLAUDE/'events_v3.csv.gz').read_bytes()).hexdigest()
    meta = ADAPTED/'provenance.json'
    if meta.exists():
        old = json.loads(meta.read_text())
        if old.get('source_and_teacher_fingerprint') != stamp:
            raise ValueError('Cached adapted labels from different source/teacher: '+str(ADAPTED))
        print('REUSE_CLAUDE_V2_ADAPTED_LABELS',old['converted_files'],flush=True)
        return
    if ADAPTED.exists() and any(ADAPTED.iterdir()):
        raise ValueError('Incomplete adapted label directory. Inspect/remove: '+str(ADAPTED))
    ADAPTED.mkdir(parents=True, exist_ok=True)
    seen = set()
    statuses = []
    converted = 0
    unsupported = 0
    for r in frozen.itertuples():
        if r.cat not in POS|NEG:
            raise ValueError('Unrecognized source category '+r.cat)
        src = SOURCE/r.cat/r.name
        lbl = ORIGINAL/r.name.replace('.csv.gz','.labels.csv.gz')
        if not src.is_file() or not lbl.is_file():
            raise FileNotFoundError(f'Source or labeled sidecar absent: {src}, {lbl}')
        digest = source_sha(src)
        if digest != r.sha or digest in seen:
            raise ValueError('Source hash changed or duplicate in frozen folds: '+r.name)
        seen.add(digest)
        lab = pd.read_csv(lbl, compression='gzip', usecols=['Timestamp','onset_label','weight',
                          'event_id','active_label','gesture_kind'])
        y = lab.onset_label.to_numpy(np.int8)
        w = lab.weight.to_numpy(np.float32)
        if not np.isin(y, [-1,0,1]).all() or not np.isfinite(w).all():
            raise ValueError('Invalid Claude label values: '+r.name)
        if r.cat in NEG and (y == 1).any():
            raise ValueError('Unexpected positive inside negative folder: '+r.name)
        # No strong confirmed event? Never turn a positive bag into a negative bag.
        if r.cat in POS and not (y == 1).any():
            unsupported += 1
            statuses.append({'source':f'{r.cat}/{r.name}','category':r.cat,
                             'source_sha256':digest,'status':'UNKNOWN_ONLY'})
            continue
        # Use only negative folders for negative training: positive bag backgrounds
        # can contain missed target events and are not independent negatives.
        if r.cat in POS:
            y[y == 0] = -1
        w[y == -1] = 0.0
        if ((y == 1) & (w <= 0)).any():
            raise ValueError('Confirmed onset has zero weight: '+r.name)
        dst = ADAPTED/r.cat/r.name.replace('.csv.gz','.labels.csv.gz')
        dst.parent.mkdir(exist_ok=True)
        pd.DataFrame({'Timestamp':lab.Timestamp,
                      'gesture_label':y,'gesture_weight':w,
                      'gesture_kind':lab.gesture_kind,
                      'gesture_event_id':lab.event_id}).to_csv(dst,index=False,compression='gzip')
        converted += 1
        statuses.append({'source':f'{r.cat}/{r.name}','category':r.cat,
                         'source_sha256':digest,'status':'OK'})
        if converted % 350 == 0:
            print('ADAPT_CLAUDE_V2', converted, flush=True)
    with (ADAPTED/'manifest.jsonl').open('w') as f:
        for row in statuses:
            f.write(json.dumps(row,sort_keys=True)+'\n')
    meta.write_text(json.dumps({'source_and_teacher_fingerprint':stamp,'converted_files':converted,
          'unknown_only_positive_bags':unsupported,'labeling':'Claude V2 strong-onset 0-80ms;'
          ' NEGATIVE folders only contribute negative rows; UNKNOWN excluded'},indent=2)+'\n')
    print('CLAUDE_LABEL_ADAPTER_READY',converted,'skipped_unknown_positive_bags',unsupported,flush=True)

def main():
    print('CPU_80_PERCENT',cpu_count(),CPU_THREADS,flush=True)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Immutable previous study exists; archive/rename '+str(OUTPUT))
    prepare()
    base_load = v6.load_dataset
    base_params = v6.model_params
    def load(*a,**kw):
        data, info = base_load(*a,**kw)
        # ACC is the strongest pattern found in Claude research; test it alone.
        data['feature_sets']['acc'] = list(v6.ACC)
        if set(v6.WL).issubset(data['columns']):
            data['feature_sets']['acc_wl'] = list(v6.ACC+v6.WL)
        info['feature_sets'] = data['feature_sets']
        info['input_provenance'] = 'Claude V2 strong-only sensor-timed onset SILVER labels, not V5'
        info['positive_bag_background'] = 'UNKNOWN not negative'
        v6.save_json(a[2]/'dataset_manifest.json',info)
        return data,info
    def model_params(trial,seed):
        params=base_params(trial,seed)
        params['n_jobs']=CPU_THREADS
        return params
    v6.load_dataset = load
    v6.model_params = model_params
    args = SimpleNamespace(input_dir=SOURCE, labels_dir=ADAPTED, out_dir=OUTPUT,
                           trials=TRIALS, cv_folds=FOLDS, top_k=TOP_K,
                           max_per_class=MAX_ROWS_PER_CLASS_PER_FILE,
                           sample_files=None, seed=20261117, test_fraction=0.22,
                           timeout_minutes=0, verify_hash=True)
    v6.run(args)
    provenance={'label_source':'Claude V2 sensor-timed 0-80ms, strong events only',
                'unknown_excluded':True,'target_bag_background_as_negative':False,
                'objectives':['participant-group CV silver-label F1 (maximize)',
                              'export_plus compact static C model bytes (minimize)'],
                'cpu_fraction':CPU_FRACTION,'cpu_threads':CPU_THREADS,
                'warning':'V6 exporter filenames and some inherited comments say V5: actual input labels are Claude V2.',
                'limitations':['Silver-label agreement is NOT measured physical tap recall or false events/hour',
                               'Only 36 directly addressable firmware EMG/ACC/WL features are used; Claude 126 features need separate firmware validation',
                               'Stub C compilation is not an nRF52832 firmware build']}
    v6.save_json(OUTPUT/'CLAUDE_V2_PROVENANCE.json',provenance)
    for p in (OUTPUT/'models').glob('*.json'):
        report=json.loads(p.read_text())
        if 'optuna_trial' in report:
            report['label_source']='Claude V2 strong onset silver labels (0-80ms)'
            report['limitations']=[str(x).replace('V5 inferred','Claude V2 inferred').replace('V5 ','Claude V2 ')
                                   for x in report.get('limitations',[])]
            v6.save_json(p,report)
    print('CLAUDE_V2_TRAINING_FINISHED',OUTPUT,flush=True)

if __name__=='__main__':
    main()
