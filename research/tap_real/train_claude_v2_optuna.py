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
import shutil
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
from publish_tap_swipe_headers import (publish as publish_stable_headers,
                                       _add_uppercase_predict_alias)

SOURCE = ROOT/'DATA'/'DATA_TAP'
CLAUDE = ROOT/'claude_research_tap'/'results'
ORIGINAL = ROOT/'DATA'/'DATA_TAP_LABELS_CLAUDE_V2_20261117'
ADAPTED = ROOT/'research'/'tap_real'/'cache_claude_v2_v6_schema'
OUTPUT = ROOT/'research'/'tap_real'/'results_claude_v2_optuna_126_thr085'
# Keep original threshold-tuned experiments immutable; this is a new study.
POS = {'TAP', 'SWIPE'}
NEG = {'FINGER', 'WRIST', 'ARM', 'BAND', 'NOTHING'}
TRIALS = 40
FOLDS = 3
MAX_ROWS_PER_CLASS_PER_FILE = 45
TOP_K = 10

# Claude's 126 direct p_core CSV fields. No engineered derivatives.
# EMG 45 (instant/oldest/mean) + ACC 72 (instant/oldest/mean/rate)
# + WL 9 (full/head/tail) = 126.
def _psd(group, channels, n_zones):
    return tuple(f"p_core->{group}[{ch}][{zone}]"
                 for ch in channels for zone in range(n_zones))

EMG_CURRENT = _psd('emg_psd[0]', ('CH1','CH2','CH3'), 5)
EMG_OLDEST = _psd('emg_psd[EMG_PSD_HISTORY_SIZE - 1]', ('CH1','CH2','CH3'), 5)
EMG_MEAN = _psd('emg_psd_mean', ('CH1','CH2','CH3'), 5)
ACC_CURRENT = _psd('acc_psd[0]', ('X','Y','Z'), 6)
ACC_OLDEST = _psd('acc_psd[ACC_PSD_HISTORY_SIZE - 1]', ('X','Y','Z'), 6)
ACC_MEAN = _psd('acc_psd_mean', ('X','Y','Z'), 6)
ACC_RATE = _psd('acc_rate', ('X','Y','Z'), 6)
WL_FULL = tuple(f'p_core->wl[CH{x}]' for x in (1,2,3))
WL_HEAD = tuple(f'p_core->wl_head[CH{x}]' for x in (1,2,3))
WL_TAIL = tuple(f'p_core->wl_tail[CH{x}]' for x in (1,2,3))
EMG_ALL = EMG_CURRENT + EMG_OLDEST + EMG_MEAN
ACC_ALL = ACC_CURRENT + ACC_OLDEST + ACC_MEAN + ACC_RATE
WL_ALL = WL_FULL + WL_HEAD + WL_TAIL
FULL_126 = ACC_ALL + EMG_ALL + WL_ALL
INSTANT_36 = ACC_CURRENT + EMG_CURRENT + WL_FULL
assert len(FULL_126)==126 and len(set(FULL_126))==126
assert len(INSTANT_36)==36

# Full model is an explicit Optuna starting trial, not a random hope.
FEATURE_SETS = {
    'full126': FULL_126,
    'instant36': INSTANT_36,
    'acc72': ACC_ALL,
    'emg45': EMG_ALL,
    'acc_emg117': ACC_ALL + EMG_ALL,
    'acc_wl81': ACC_ALL + WL_ALL,
    'emg_wl54': EMG_ALL + WL_ALL,
    'acc_temporal_emg_current90': ACC_ALL + EMG_CURRENT + WL_FULL,
    'acc_current_emg_temporal_wl72': ACC_CURRENT + EMG_ALL + WL_ALL,
    'acc_current18': ACC_CURRENT,
}

def compile_header_126(header, name, output):
    """C syntax smoke against a 126-field synthetic t_core, NOT real FW ABI."""
    cc = shutil.which('cc') or shutil.which('gcc')
    if not cc:
        return {'status':'not_available','reason':'No C compiler'}
    stub = output/'c_compiler_check'
    stub.mkdir(parents=True,exist_ok=True)
    (stub/'core_type_def.h').write_text("""#ifndef CORE_TYPE_DEF_H
#define CORE_TYPE_DEF_H
#include <stdint.h>
#define X 0
#define Y 1
#define Z 2
#define CH1 0
#define CH2 1
#define CH3 2
#define ACC_PSD_HISTORY_SIZE 10
#define EMG_PSD_HISTORY_SIZE 15
typedef struct {
    float emg_psd[EMG_PSD_HISTORY_SIZE][3][5];
    float emg_psd_mean[3][5];
    float acc_psd[ACC_PSD_HISTORY_SIZE][3][6];
    float acc_psd_mean[3][6];
    float acc_rate[3][6];
    float wl[3], wl_head[3], wl_tail[3];
} t_core;
#endif
""")
    # The compiler probes all 126 expressions even when a small model
    # does not use every feature. This verifies export grammar, not struct ABI.
    probe = stub/'all126_feature_probe.c'
    probe.write_text('#include "core_type_def.h"\n'
        'float probe_all_126(const t_core *p_core) {\n'
        '  return ' + ' +\n    '.join(f'(float)({v})' for v in FULL_126) + ';\n}\n')
    unit = stub/(name+'.c')
    unit.write_text('#include "'+header.name+'"\n'
        'int model_compiles(void) { t_core data={0}; return '+name+'_predict(&data); }\n')
    for src in (probe,unit):
        cmd=[cc,'-std=c11','-O2','-c',str(src),
             '-I',str(stub),'-I',str(header.parent),
             '-o',str(stub/(src.stem+'.o'))]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        if p.returncode:
            raise RuntimeError('126-field synthetic C compile failed: '+
                               (p.stderr or p.stdout)[-3000:])
    return {'status':'passed','compiler':cc,'scope':'synthetic 126-field t_core; not actual firmware ABI or ARM build',
            'object_bytes':(stub/(name+'.o')).stat().st_size,
            'all_feature_accessors_probed':len(FULL_126)}

def enable_full_feature_schema():
    # The V6 source is left unchanged; its module-level feature whitelist is
    # extended only in this process. The source-hash verification stays intact.
    v6.FULL_FEATURES = FULL_126
    v6.export_plus.validate_feature_schema_accessors(FULL_126)
    v6.compile_header = compile_header_126

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
        # Completed results can be republished with a stable C API without
        # rerunning a 40-trial Optuna study. Preserve the trained model bytes.
        summary_file = OUTPUT / 'summary.json'
        if summary_file.is_file():
            summary = json.loads(summary_file.read_text(encoding='utf-8'))
            if (summary.get('status') == 'success'
                    and summary.get('fixed_decision_threshold') == v6.FIXED_DECISION_THRESHOLD
                    and summary.get('threshold_optimized_by_optuna') is False):
                stable = publish_stable_headers(
                    OUTPUT, compile_header=compile_header_126)
                print('CLAUDE_V2_REUSE_EXISTING_STUDY_NO_RETRAIN',
                      str(OUTPUT/'models'/'TAP_SWIPE.h'),
                      'rank', stable['selected_rank'], flush=True)
                return
        raise ValueError('Incomplete or legacy threshold-tuned study exists; inspect or archive '+str(OUTPUT))
    prepare()
    enable_full_feature_schema()
    base_load = v6.load_dataset
    base_params = v6.model_params
    def load(*a,**kw):
        data, info = base_load(*a,**kw)
        if tuple(data['columns']) != FULL_126:
            raise ValueError('Full 126-column p_core schema missing/inconsistent across recordings. '
                             'Need all 126 raw fields for this experiment.')
        data['feature_sets'] = {name:list(cols) for name,cols in FEATURE_SETS.items()}
        info['feature_sets'] = data['feature_sets']
        info['full_raw_p_core_columns'] = len(FULL_126)
        info['feature_schema'] = 'Exactly Claude 126 logged p_core fields, no derived columns'
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
    # Always export one real full126 model, even if a smaller feature set wins
    # Optuna's Pareto/utility ranking. Preserve the other nine ranked choices.
    original_select = v6.pareto_select
    def retain_full126_candidate(study, k):
        selected = original_select(study, k)
        full = [t for t in study.trials
                if t.state == v6.optuna.trial.TrialState.COMPLETE
                and t.values is not None and t.params.get('feature_set') == 'full126']
        if not full:
            raise RuntimeError('No successful full126 trial: check the study failures')
        best = max(full, key=lambda t: float(t.values[0]))
        if best.number not in {t.number for t in selected}:
            selected[-1] = best
        return selected
    v6.pareto_select = retain_full126_candidate
    args = SimpleNamespace(input_dir=SOURCE, labels_dir=ADAPTED, out_dir=OUTPUT,
                           trials=TRIALS, cv_folds=FOLDS, top_k=TOP_K,
                           max_per_class=MAX_ROWS_PER_CLASS_PER_FILE,
                           sample_files=None, seed=20261117, test_fraction=0.22,
                           timeout_minutes=0, verify_hash=True,
                           initial_feature_sets=('full126','instant36'),
                           stable_headers=True,
                           header_postprocess=_add_uppercase_predict_alias)
    v6.run(args)
    provenance={'label_source':'Claude V2 sensor-timed 0-80ms, strong events only',
                'unknown_excluded':True,'target_bag_background_as_negative':False,
                'fixed_decision_threshold':v6.FIXED_DECISION_THRESHOLD,
                'threshold_optimized_by_optuna':False,
                'objectives':['participant-group CV silver-label F1 at fixed 0.85 (maximize)',
                              'export_plus compact static C model bytes (minimize)'],
                'cpu_fraction':CPU_FRACTION,'cpu_threads':CPU_THREADS,
                'firmware_header':'TAP_SWIPE.h',
                'stable_header_names_from_training':True,
                'note':'Only research metadata retains trial identity; firmware C symbols and filenames are stable.',
                'raw_features_available':len(FULL_126),
                'compared_feature_sets':{k:len(v) for k,v in FEATURE_SETS.items()},
                'limitations':['Silver-label agreement is NOT measured physical tap recall or false events/hour',
                               'Legacy 2026-11-17 study with tuned threshold is NOT directly comparable to fixed-0.85 CV F1',
                               '126 logged p_core fields assumed available at runtime; actual firmware declarations and memory/CPU costs must be checked',
                               'Stub C compilation is not an nRF52832 firmware build']}
    v6.save_json(OUTPUT/'CLAUDE_V2_PROVENANCE.json',provenance)
    for p in (OUTPUT/'models').glob('*.json'):
        report=json.loads(p.read_text())
        if 'optuna_trial' in report:
            report['label_source']='Claude V2 strong onset silver labels (0-80ms)'
            report['limitations']=[str(x).replace('V5 inferred','Claude V2 inferred').replace('V5 ','Claude V2 ')
                                   for x in report.get('limitations',[])]
            v6.save_json(p,report)
    stable = publish_stable_headers(OUTPUT, compile_header=compile_header_126)
    print('CLAUDE_V2_TRAINING_FINISHED',OUTPUT,'raw_features',len(FULL_126),
          'stable_header',str(OUTPUT/'models'/'TAP_SWIPE.h'),
          'default_rank',stable['selected_rank'],flush=True)

if __name__=='__main__':
    main()
