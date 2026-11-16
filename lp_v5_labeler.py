#!/usr/bin/env python3
"""Reproducible V5 LP/quality pseudo-label export from DATA/DATA_LONG_PRESS.

The contact proxy is cross-fitted by PERSON for every recording used in its
training; a quality model fitted on the same recording cannot label that file.
No pointer timestamps are used to create labels. Training loader must drop -1.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from lp_v3_labeler import train_contact_quality,normalize_person,EXTRA_READ
from scripts.lp_v3_feature_probe import record
from lp_state_v5 import label_recording,VERSION,Settings
from lp_v5_cpu import CPU_FRACTION, cpu_workers, available_cpus, prepare_process_worker

INPUT_ROOT=Path('DATA/DATA_LONG_PRESS')
OUTPUT_ROOT=Path('DATA/DATA_LONG_PRESS_LABELED_V5')
LP_GROUPS=('SOFT_LONG_PRESS','MID_LONG_PRESS','STRONG_LONG_PRESS','LONG_PRESS')
CONTROL_GROUPS=('CONNECTED','NOTHING','DISCONNECTED','LOOSE','TABLE')
GROUPS=LP_GROUPS+CONTROL_GROUPS

# Change these defaults here, not through command-line arguments.
@dataclass(frozen=True)
class LabelerConfig:
    input_dir: Path = INPUT_ROOT
    output_dir: Path = OUTPUT_ROOT
    dry_run: bool = False
    overwrite: bool = False
    limit_per_group: int = 0
    report: Path | None = None
    strict_contact_prob: float = 0.20
    review_contact_prob: float = 0.40
    min_events: int = 3
    min_event_ms: float = 350.0
    cpu_fraction: float = CPU_FRACTION



def checksum(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_text(path,mode):
    if path.name.endswith('.gz'):
        return gzip.open(path,mode,newline='',encoding='utf-8-sig' if 'r' in mode else 'utf-8',compresslevel=3)
    return open(path,mode,newline='',encoding='utf-8-sig' if 'r' in mode else 'utf-8')


def write_labeled(src:Path,dest:Path,lab:pd.DataFrame):
    """Stream originals with no numerical rewrites; atomically replace output."""
    if set(lab.columns).intersection(pd.read_csv(src,compression='gzip',nrows=0).columns):
        raise ValueError('A source already contains one or more V5 output columns')
    dest.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.v5-',suffix='.csv.gz',dir=dest.parent)
    os.close(fd)
    tmp=Path(name)
    try:
        with read_text(src,'rt') as fin,read_text(tmp,'wt') as fout:
            reader=csv.reader(fin)
            writer=csv.writer(fout)
            names=next(reader)
            writer.writerow(names+list(lab.columns))
            vals=lab.to_numpy()
            total=0
            for line in reader:
                if total>=len(vals):raise ValueError('Source grew during export')
                writer.writerow(line+[str(int(v)) if isinstance(v,(np.integer,int))
                                    else format(float(v),'.7g') if isinstance(v,(np.floating,float))
                                    else str(v) for v in vals[total]])
                total+=1
            if total!=len(vals):raise ValueError('Source truncated during export')
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()


def infer_quality(path,group,df,models,full,known):
    """No arbitrary .35 contact fallback: unknown quality means abstention."""
    if str(path) in known:
        return np.asarray(known[str(path)],float),'cross_fitted_oof'
    rec=record(path,group)
    if rec is None:
        return np.ones(3,float),'missing_contact_profile'
    uid=normalize_person(df['User'].iloc[0] if 'User' in df else None,path.name)
    model=models.get(uid,full)
    return model.predict_proba(np.asarray([ch['feat'] for ch in rec['ch']],float))[:,1], 'heldout_person_or_full'


_LABEL_WORKER_STATE = None


def _init_label_worker(root, out, settings, dry_run, full, models, known):
    """Each subprocess holds one copy of the fitted contact proxy."""
    global _LABEL_WORKER_STATE
    prepare_process_worker()
    _LABEL_WORKER_STATE = (root, out, settings, dry_run, full, models, known)


def _process_label_file(item):
    """One independent capture. Return results for deterministic ordered audit."""
    root, out, cfg, dry_run, full, models, known = _LABEL_WORKER_STATE
    group, path = item
    rel = path.relative_to(root)
    info = {"group": group, "file": rel.as_posix()}
    try:
        df = pd.read_csv(path, compression="gzip",
                         usecols=lambda c: c in EXTRA_READ, low_memory=False)
        cp, quality_src = infer_quality(path, group, df, models, full, known)
        lab, meta = label_recording(df, cp, group=group, settings=cfg)
        if len(lab) != len(df):
            raise RuntimeError("Source/label rows mismatch")
        vals = lab.lp_state.to_numpy()
        if not set(np.unique(vals)).issubset({-1, 0, 1}):
            raise RuntimeError("Non-tristate output")
        if np.any(lab.lp_state_weight[vals < 0].to_numpy() != 0):
            raise RuntimeError("Unknown samples have non-zero weight")
        if not dry_run:
            write_labeled(path, out / rel, lab)
        info.update(meta)
        info.update(quality_source=quality_src, source_sha256=checksum(path))
        if group in CONTROL_GROUPS:
            # A diagnostic only: do not treat controls as physical LP truth.
            raw, _ = label_recording(df, cp, group="", settings=cfg)
            info["context_free_active"] = int((raw.lp_state == 0).sum())
            if group == "DISCONNECTED":
                info["nominally_disconnected_ch1_2_activity"] = [
                    int((raw[f"lp_high_activity_ch{ch}"] > 0).sum())
                    for ch in (1, 2)
                ]
        return info, None
    except Exception as exc:
        return None, {**info, "error": f"{type(exc).__name__}: {exc}"[:250]}


def main(config: LabelerConfig | None = None):
    """Run with editable top-of-file defaults or explicit Python settings."""
    opts = config if config is not None else LabelerConfig()
    root = opts.input_dir.resolve()
    out = opts.output_dir.resolve()
    if not root.is_dir():
        raise ValueError("Missing input root")
    if root != INPUT_ROOT.resolve():
        raise ValueError("V5 only accepts source DATA/DATA_LONG_PRESS")
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Input and output directories must not be ancestors of each other")
    protected = {Path(x).resolve() for x in (
        "DATA/DATA_LONG_PRESS_LABELED",
        "DATA/DATA_LONG_PRESS_LABELED_V2",
        "DATA/DATA_LONG_PRESS_LABELED_V3",
        "DATA/DATA_LONG_PRESS_LP_STATE_V4",
        "DATA/DATA_LONG_PRESS_LABELED_V4",
    )}
    if out in protected:
        raise ValueError("Output points at a protected older labeled dataset")
    if opts.limit_per_group < 0:
        raise ValueError("limit_per_group must be >=0")
    if opts.report and (opts.report.resolve() == root or
                        root in opts.report.resolve().parents):
        raise ValueError("Report must not be inside source tree")
    cfg = Settings(strict_bad_prob=opts.strict_contact_prob,
                   review_bad_prob=opts.review_contact_prob,
                   min_events=opts.min_events, min_event_ms=opts.min_event_ms)
    if not 0 <= cfg.strict_bad_prob < cfg.review_bad_prob <= 1:
        raise ValueError("Invalid quality thresholds")
    if cfg.min_events < 1 or cfg.min_event_ms <= 0:
        raise ValueError("Invalid event requirements")
    workers = cpu_workers(opts.cpu_fraction)
    paths = [(g, p) for g in GROUPS for p in
             sorted((root / g).glob("*.csv.gz"))[:opts.limit_per_group or None]]
    if not opts.dry_run and not opts.overwrite:
        existing = [str(out / p.relative_to(root))
                    for _, p in paths if (out / p.relative_to(root)).exists()]
        if existing:
            raise FileExistsError("Existing V5 outputs; change output_dir or enable overwrite in LabelerConfig")
    started = time.monotonic()
    print("CPU_POLICY", f"{opts.cpu_fraction:.0%}", "available", available_cpus(),
          "labeler_processes", workers, flush=True)
    print("CONTACT_MODEL_TRAINING", len(paths), "selected files", flush=True)
    full, models, known, refs, quality = train_contact_quality(root, .7)
    print("CONTACT_MODEL_WEAK_CV", json.dumps(quality), flush=True)
    records, errors = [], []
    if workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(paths)),
            initializer=_init_label_worker,
            initargs=(root, out, cfg, opts.dry_run, full, models, known),
        ) as pool:
            results = pool.map(_process_label_file, paths, chunksize=1)
            for i, (info, error) in enumerate(results, 1):
                if error is not None:
                    errors.append(error)
                    print("ERROR", error["file"], error["error"], flush=True)
                else:
                    records.append(info)
                if i % 25 == 0 or i == len(paths):
                    print("PROGRESS", i, len(paths), "ok", len(records),
                          "errors", len(errors), "elapsed",
                          round(time.monotonic() - started, 2), flush=True)
    else:
        _init_label_worker(root, out, cfg, opts.dry_run, full, models, known)
        for i, item in enumerate(paths, 1):
            info, error = _process_label_file(item)
            if error is not None:
                errors.append(error)
                print("ERROR", error["file"], error["error"], flush=True)
            else:
                records.append(info)
            if i % 25 == 0 or i == len(paths):
                print("PROGRESS", i, len(paths), "ok", len(records),
                      "errors", len(errors), "elapsed",
                      round(time.monotonic() - started, 2), flush=True)
    by={}
    for g in GROUPS:
        sample=[x for x in records if x['group']==g]
        n=sum(x['n'] for x in sample)
        a=sum(x['active'] for x in sample);r=sum(x['rest'] for x in sample)
        by[g]={'files':len(sample),'rows':n,'active':a,'rest':r,
               'unknown':sum(x['unknown'] for x in sample),
               'coverage':round((a+r)/max(n,1),5),
               'active_fraction':round(a/max(n,1),5),
               'files_any_active':sum(x['active']>0 for x in sample),
               'single_channel_files':sum(x['single_channel'] for x in sample),
               'raw_context_free_active_fraction':round(sum(x.get('context_free_active',0) for x in sample)/max(n,1),5),
               'nominal_bad_CH1_2_activity':sum(sum(x.get('nominally_disconnected_ch1_2_activity',[])) for x in sample)}
        print('V5_GROUP',g,json.dumps(by[g]),flush=True)
    review=[]
    for x in records:
        if (x['group'] in LP_GROUPS and
             (x['strict_accepted_channels']==0 or x['single_channel'] or x['review_candidate_channels']>0)):
            review.append({'file':x['file'],'reason':'lp_review',
                           'accepted_ch':x['strict_accepted_channels'],
                           'candidate_ch':x['review_candidate_channels'],
                           'channel_events':x['channel_events']})
        elif x['group'] in CONTROL_GROUPS and x.get('context_free_active',0)>0:
            review.append({'file':x['file'],'reason':'control_EMG_activity_uncertain',
                           'active_samples_without_context':x['context_free_active'],
                           'channel_events':x['channel_events']})
    report={'version':VERSION,'contract':{'1':'EMG-supported rest', '0':'EMG-supported active',
            '-1':'UNKNOWN, exclude from training'},
            'source_root':str(root),'output_root':str(out),'dry_run':opts.dry_run,
            'configuration':{**cfg.__dict__,'contact_training_weak_weight':0.7, 'cpu_fraction':opts.cpu_fraction, 'cpu_workers':workers},'selected_files':len(paths),
            'processed_files':len(records),'errors':errors,
            'groups':by,'quality_weak_proxy_cv':quality,
            'warning':'Neither pointer nor folder context is physically synchronized hold truth; quality proxy is not measured electrode contact. Student must use causal firmware-only features.',
            'review_queue':review,'files':records,
            'source_code_hashes':{str(n):checksum(n) for n in (Path(__file__),Path(__file__).with_name('lp_state_v5.py'),Path(__file__).with_name('lp_v3_labeler.py'),Path(__file__).parent/'scripts'/'lp_v3_feature_probe.py',Path(__file__).with_name('lp_causal_fw_teacher.py'))},
            'elapsed_s':round(time.monotonic()-started,2)}
    rp=opts.report or (Path('/tmp/lp_v5_dry_run_audit.json') if opts.dry_run else out/'label_summary.json')
    rp.parent.mkdir(parents=True,exist_ok=True)
    rp.write_text(json.dumps(report,indent=2))
    print('V5_REPORT',json.dumps({k:v for k,v in report.items() if k not in ('files','review_queue','errors','quality_weak_proxy_cv')}),flush=True)
    print('V5_REPORT_PATH',rp,flush=True)
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
