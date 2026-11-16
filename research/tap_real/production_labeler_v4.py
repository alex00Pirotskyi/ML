#!/usr/bin/env python3
"""Fail-closed, reproducible production-grade TAP *label export* (not a tap oracle).

Only an independently verified annotation can emit tap_label=1. V3 signal events,
Oculus PointerState, metronome beats and filenames are useful review candidates,
never positive truth. Output is disjoint from DATA/DATA_TAP and atomically published.

This module implements auditable 0/1/-1 label mechanics. It cannot replace actual
synchronized human or hardware annotations of physical tap onset times.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from tap_event_shape_v3 import CATEGORIES, EMG, Config, ShapeEvent, extract_events, extract_signals

try:
    from silver_anchor import AnchorConfig, evaluate as evaluate_silver
except ImportError:
    from .silver_anchor import AnchorConfig, evaluate as evaluate_silver

VERSION = '4.0.0'
GOLD_EVIDENCE = {'synchronized_video', 'physical_contact_sensor', 'synchronized_kinematics'}
ANNOTATION_FIELDS = ('source', 'source_sha256', 'onset_rel_ms', 'decision',
                     'reviewer_id', 'evidence_type', 'sync_error_ms', 'notes')
GOOD_TYPES = {'TAP', 'NOT_TAP', 'UNCERTAIN'}


@dataclass(frozen=True)
class ExportPolicy:
    core_after_ms: float = 80.0
    max_annotation_disagreement_ms: float = 45.0
    max_annotation_sync_error_ms: float = 25.0
    max_gold_gap_ms: float = 500.0
    min_double_gap_ms: float = 80.0
    guard_around_gold_ms: float = 145.0
    gap_guard_before_ms: float = 30.0
    gap_guard_after_ms: float = 120.0
    min_valid_dt_samples: int = 2
    max_gold_onset_emg_distance_ms: float = 210.0
    # GOLD can annotate V3 missed candidates; a frozen/noisy signal is still excluded.
    min_usable_emg_activity: float = 0.025
    min_channel_dynamic_range: float = 5.0
    max_artifact_fraction: float = 0.30


@dataclass(frozen=True)
class EvidenceRow:
    source: str
    source_sha256: str
    onset_rel_ms: float
    decision: str
    reviewer_id: str
    evidence_type: str
    sync_error_ms: float
    notes: str


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or '\\' in value or value.startswith('/'):
        raise ValueError(f'Unsafe or non-POSIX source path: {value!r}')
    posix = PurePosixPath(value)
    if len(posix.parts) < 2 or '..' in posix.parts or '.' in posix.parts or posix.as_posix()!=value:
        raise ValueError(f'Unsafe source path: {value!r}')
    if posix.parts[0] not in CATEGORIES or not value.endswith(('.csv.gz', '.csv')):
        raise ValueError(f'Only DATA_TAP category CSVs may be referenced: {value!r}')
    return posix.as_posix()


def input_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def read_annotations(path: Path | None) -> dict[str, list[EvidenceRow]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f'Annotation CSV does not exist: {path}')
    grouped: dict[str, list[EvidenceRow]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in ANNOTATION_FIELDS):
            raise ValueError(f'Annotation schema must include {ANNOTATION_FIELDS}')
        seen = set()
        for line, row in enumerate(reader, 2):
            source = validate_relative_path((row['source'] or '').strip())
            sha = (row['source_sha256'] or '').strip().lower()
            if not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError(f'Line {line}: invalid SHA256')
            try:
                onset = float(row['onset_rel_ms'])
                error = float(row['sync_error_ms'])
            except (TypeError, ValueError) as e:
                raise ValueError(f'Line {line}: invalid onset/synchronization error') from e
            if not math.isfinite(onset) or onset < 0 or not math.isfinite(error) or error < 0:
                raise ValueError(f'Line {line}: onset and sync_error_ms must be finite nonnegative')
            decision = (row['decision'] or '').strip().upper()
            if decision not in GOOD_TYPES:
                raise ValueError(f'Line {line}: invalid decision {decision!r}')
            reviewer = (row['reviewer_id'] or '').strip()
            if not reviewer or len(reviewer) > 100:
                raise ValueError(f'Line {line}: reviewer identity required')
            evidence = (row['evidence_type'] or '').strip().lower()
            if decision == 'TAP' and evidence not in GOLD_EVIDENCE:
                raise ValueError(f'Line {line}: pointer/metronome/filename are not physical ground truth')
            key = (source, reviewer, round(onset, 3))
            if key in seen:
                raise ValueError(f'Line {line}: duplicate annotation {key}')
            seen.add(key)
            grouped[source].append(EvidenceRow(source, sha, onset, decision,
                                               reviewer, evidence, error,
                                               (row.get('notes') or '').strip()))
    return dict(grouped)


def consensus(rows: list[EvidenceRow], duration_ms: float,
              cfg: ExportPolicy) -> tuple[list[dict], list[dict]]:
    """Group separately reviewed physical events by relative annotated onset.

    A tap is GOLD only with two independent synchronized reviewers, or with
    a calibrated physical contact sensor annotation (single direct instrument).
    Each reviewer is allowed once per event. Conflicts always abstain.
    """
    clustered: list[list[EvidenceRow]] = []
    for row in sorted(rows, key=lambda x: x.onset_rel_ms):
        if row.onset_rel_ms > duration_ms:
            raise ValueError(f'Out-of-bounds physical annotation: {row}')
        if row.sync_error_ms > cfg.max_annotation_sync_error_ms and row.decision == 'TAP':
            raise ValueError(f'Synchronization uncertainty too high for GOLD: {row}')
        # Protect against transitive chain merging across multiple rapid taps.
        if clustered and row.onset_rel_ms - clustered[-1][0].onset_rel_ms <= cfg.max_annotation_disagreement_ms:
            clustered[-1].append(row)
        else:
            clustered.append([row])
    gold: list[dict] = []
    rejected: list[dict] = []
    for cluster in clustered:
        a = np.array([x.onset_rel_ms for x in cluster], dtype=float)
        decisions = {x.decision for x in cluster}
        reviewers = {x.reviewer_id for x in cluster}
        t = float(np.median(a))
        if len(a) >= 2 and len(reviewers) < len(a):
            rejected.append({'onset_rel_ms': t, 'reason': 'repeated_reviewer', 'num_annotations': len(a)})
            continue
        if max(a)-min(a) > cfg.max_annotation_disagreement_ms:
            rejected.append({'onset_rel_ms': t, 'reason': 'onset_disagreement'})
            continue
        if decisions != {'TAP'}:
            rejected.append({'onset_rel_ms': t,
                             'reason': 'conflicting_or_nonpositive_review',
                             'decisions': sorted(decisions)})
            continue
        hardware = any(x.evidence_type == 'physical_contact_sensor' for x in cluster)
        if len(reviewers) < 2 and not hardware:
            rejected.append({'onset_rel_ms': t, 'reason': 'awaiting_independent_second_review'})
            continue
        if hardware:
            # Independent sensor's clock still must be synchronized and measured.
            if not all(x.sync_error_ms <= 10.0 for x in cluster if x.evidence_type == 'physical_contact_sensor'):
                rejected.append({'onset_rel_ms': t, 'reason': 'hardware_clock_uncertain'})
                continue
        gold.append({'onset_rel_ms': t, 'sync_error_ms': max(x.sync_error_ms for x in cluster),
                     'reviewers': sorted(reviewers),
                     'evidence': sorted({x.evidence_type for x in cluster})})
    # Never choose which of two impossible overlapping gold onsets is right.
    safe=[]
    for i,g in enumerate(gold):
        neighbors = [abs(g['onset_rel_ms']-x['onset_rel_ms']) for j,x in enumerate(gold) if i != j]
        if neighbors and min(neighbors) < cfg.min_double_gap_ms:
            rejected.append({'onset_rel_ms':g['onset_rel_ms'], 'reason':'unresolvable_gold_overlap'})
        else:
            safe.append(g)
    return safe, rejected


def event_sensor_quality(signal: dict, rel_ms: float, policy: ExportPolicy) -> tuple[bool, str]:
    ts=signal['ts']; t=ts[0]+rel_ms
    mask=(ts>=t-100)&(ts<=t+100)
    if int(mask.sum()) < policy.min_valid_dt_samples:
        return False,'insufficient_local_samples'
    raw=signal['emg_raw'][mask]
    if np.mean((raw<=0) | (raw>=65000)) > policy.max_artifact_fraction:
        return False,'sensor_saturated_or_zero'
    # 15 spectral bands need not vary; one channel can be enough for soft taps.
    dynamic=np.ptp(raw,axis=0)
    if np.max(dynamic) < policy.min_channel_dynamic_range:
        return False,'frozen_emg'
    local=np.asarray(signal['activity'])[mask]
    if np.max(local)<policy.min_usable_emg_activity:
        return False,'no_observable_emg_evidence'
    dt=np.diff(ts)
    if np.any((ts[:-1]>=t-120)&(ts[:-1]<=t+120)&(dt>140)):
        return False,'timestamp_gap'
    return True,'OK'


def _near(mask_value: np.ndarray, ts: np.ndarray, t: float, before: float, after: float) -> None:
    mask_value[(ts>=t-before) & (ts<=t+after)] = True


def build_labels(df: pd.DataFrame, category: str, annotations: list[EvidenceRow],
                 silver: dict, policy: ExportPolicy,
                 sensor: dict | None = None, candidates: list[ShapeEvent] | None = None) -> tuple[pd.DataFrame, list[dict], dict]:
    """Pure-ish deterministic label semantics: verified positives, explicit negatives, abstention.

    Does not use category labels as evidence for physical tap onsets; folder
    categories ONLY supply instructed non-TAP negatives.
    """
    if category not in CATEGORIES:
        raise ValueError('Unsupported recording category')
    S = sensor if sensor is not None else extract_signals(df)
    events = candidates if candidates is not None else extract_events(S)
    ts = S['ts'];t0=ts[0];n=len(ts)
    labels=np.full(n,-1,dtype=np.int8)
    weights=np.zeros(n,np.float32)
    event_id=np.full(n,-1,np.int32)
    pair_id=np.zeros(n,np.int32)
    provenance=np.full(n,'UNKNOWN',dtype=object)
    cand_id=np.full(n,-1,np.int32)
    cand_pair=np.zeros(n,np.int32)
    candidate_score=np.zeros(n,np.float32)
    for j,e in enumerate(events,1):
        ix=(ts>=e.onset_ms)&(ts<=e.end_ms)
        cand_id[ix]=j
        cand_pair[ix]=getattr(e,'pair_id',0)
        candidate_score[ix]=e.shape_score
    silver_bits=np.zeros(n,np.int8)
    for hit in silver.get('hits',[]):
        _near(silver_bits,ts,hit['candidate_onset_ms'],10,65)
    raw=np.asarray(S['emg_raw'])
    healthy_channels=int(sum(np.ptp(raw[:,ch,:],axis=0).max()>=policy.min_channel_dynamic_range
                             and np.mean((raw[:,ch,:]>0)&(raw[:,ch,:]<65000))>=.50
                             for ch in range(3)))
    record_usable=healthy_channels>=1
    if category!='TAP' and record_usable:
        # Known-instruction negative recording, never infer from class_id (SWIPE
        # often has the same class_id as TAP in the source data).
        labels[:]=0
        # Instructed gesture-class negatives are weak folder supervision, not
        # synchronized proof that no physical tap happened anywhere.
        weights[:]=.70
        provenance[:]='RECORDED_CATEGORY_NEGATIVE'
    elif category!='TAP':
        provenance[:]='UNUSABLE_NEGATIVE_SENSOR'
    gold,rejected=consensus(annotations,float(ts[-1]-t0),policy)
    if category!='TAP' and gold:
        raise ValueError('Positive physical TAP annotations in a NON-TAP category require adjudication')
    # Nothing in a TAP recording is assumed negative unless independently reviewed.
    for ix,g in enumerate(gold,1):
        ok,why=event_sensor_quality(S,g['onset_rel_ms'],policy)
        g['sensor_usable']=ok;g['sensor_note']=why
        if not ok:
            rejected.append({'onset_rel_ms':g['onset_rel_ms'], 'reason':why})
            continue
        event_t=t0+g['onset_rel_ms']
        offset=float(g['sync_error_ms'])
        lo=event_t+offset
        hi=event_t+policy.core_after_ms-offset
        core=(ts>=lo)&(ts<=hi)
        if int(core.sum())<policy.min_valid_dt_samples:
            rejected.append({'onset_rel_ms':g['onset_rel_ms'], 'reason':'too_few_positive_rows'})
            continue
        labels[core]=1
        weights[core]=1.
        event_id[core]=ix
        provenance[core]='VERIFIED_GOLD'
        g['event_id']=ix
        g['positive_rows']=int(core.sum())
    valid_gold=[g for g in gold if 'event_id' in g]
    # Two physically verified events become a double pair only when both exist.
    next_pair=1
    used=set()
    for i in range(len(valid_gold)-1):
        if i in used or i+1 in used:continue
        a=valid_gold[i];b=valid_gold[i+1]
        gap=b['onset_rel_ms']-a['onset_rel_ms']
        if policy.min_double_gap_ms<=gap<=policy.max_gold_gap_ms:
            for g in (a,b):
                g['pair_id']=next_pair
                pair_id[event_id==g['event_id']]=next_pair
            used.add(i);used.add(i+1);next_pair+=1
    # Discard warm-up and sensor timestamp discontinuities for both classes.
    embargo=(ts<=ts[0]+170)
    jumps=np.where(np.diff(ts)>140)[0]
    for j in jumps:
        embargo|=(ts>=ts[j]-policy.gap_guard_before_ms)&(ts<=ts[j+1]+policy.gap_guard_after_ms)
    if np.any(embargo&(labels==1)):
        raise ValueError('Verified positive overlaps warmup or timestamp discontinuity; correct source/annotation')
    labels[embargo]=-1;weights[embargo]=0;event_id[embargo]=-1;pair_id[embargo]=0
    provenance[embargo]='SIGNAL_UNAVAILABLE'
    # Annotated TAPs that did not pass quality are never silently negative.
    for g in gold:
        if 'event_id' not in g:
            zone=(ts>=t0+g['onset_rel_ms']-policy.guard_around_gold_ms)&(ts<=t0+g['onset_rel_ms']+policy.guard_around_gold_ms)
            labels[zone]=-1;weights[zone]=0;event_id[zone]=-1;pair_id[zone]=0
            provenance[zone]='GOLD_SENSOR_REJECTED'
    out=pd.DataFrame({'Timestamp':df['Timestamp'],'tap_label':labels,
                      'tap_label_weight':weights,'tap_event_id':event_id,
                      'tap_pair_id':pair_id,'tap_provenance':provenance,
                      'tap_candidate_id':cand_id,'tap_candidate_pair_id':cand_pair,
                      'tap_candidate_score':candidate_score,
                      'tap_silver_proposal':silver_bits})
    if not set(map(int,out.tap_label.unique())).issubset({-1,0,1}):
        raise AssertionError('Illegal label values')
    if (out.loc[out.tap_label!=1,'tap_event_id']!=-1).any():
        raise AssertionError('Positive event ID leakage outside GOLD')
    if (out.loc[out.tap_label!=1,'tap_pair_id']!=0).any():
        raise AssertionError('Double pair ID outside GOLD')
    if (out.loc[out.tap_label==-1,'tap_label_weight']!=0).any():
        raise AssertionError('Unknown labels must have zero weight')
    stats={'gold_events':len(valid_gold),'gold_rejected':len(rejected),
           'gold_pair_count':next_pair-1,
           'negative_rows':int((labels==0).sum()),
           'positive_rows':int((labels==1).sum()),
           'unknown_rows':int((labels==-1).sum()),
           'silver_proposal_frames':int((silver_bits>0).sum()),
           'candidate_events':len(events),
           'silver_status':silver.get('status','UNKNOWN'),
           'silver_reason':silver.get('reason','UNKNOWN'),
           'silver_supporting_events':len(silver.get('hits',[])),
           'double_pair_candidates':len(set(getattr(e,'pair_id',0) for e in events if getattr(e,'pair_id',0)>0)),
           'timestamp_gaps':int(len(jumps)),
           'healthy_emg_channels':healthy_channels}
    return out,[{'kind':'GOLD',**g} for g in valid_gold]+[{'kind':'REJECTED',**r} for r in rejected],stats


def build_review_queue(source: str, sha: str, category: str, events: list[ShapeEvent],
                       signal: dict, silver: dict) -> list[dict]:
    ts=signal['ts']; t0=ts[0]
    hits=silver.get('hits',[])
    queue=[]
    for number,e in enumerate(events,1):
        approx=e.onset_ms-t0
        supported=any(abs(e.peak_ms-h['candidate_peak_ms'])<=1 for h in hits)
        is_hard_negative=(category=='FINGER' and e.shape_score>=.60)
        priority='HARD_FINGER' if is_hard_negative else ('SOFT_OR_AMBIGUOUS' if e.provisional!='TAP_SHAPE' else 'TAP_SHAPE')
        if category!='TAP' and not is_hard_negative:continue
        queue.append({'source':source,'source_sha256':sha,
                      'candidate_id':number,'onset_rel_ms':round(approx,1),
                      'peak_rel_ms':round(e.peak_ms-t0,1),'category':category,
                      'candidate_type':e.provisional,'shape_score':round(e.shape_score,4),
                      'activity':round(e.activity,4),'rise_time_ms':round(e.rise_time_ms,1),
                      'candidate_pair_id':getattr(e,'pair_id',0),
                      'silver_supported':int(supported), 'review_priority':priority})
    # Candidate miner misses some very soft taps. Oculus is not truth, but
    # isolated pointer edges without any nearby EMG candidate must be shown
    # to an independent reviewer rather than silently omitted.
    if category=='TAP' and 'PointerState' in signal.get('columns',[]):
        raw_pointer=np.asarray(signal.get('pointer',[]),dtype=float)
        if len(raw_pointer)==len(ts) and np.isfinite(raw_pointer).all():
            high=raw_pointer>0.5
            ptimes=ts[high & ~np.r_[False,high[:-1]]] 
            if 3<=len(ptimes)<=32 and high.mean()<.72:
                for ptime in ptimes:
                    # Typical Oculus delay is variable; ±250ms is only a
                    # *review window*, never an inferred positive onset.
                    if any(-40. <= ptime-e.peak_ms <= 250. for e in events):
                        continue
                    queue.append({'source':source,'source_sha256':sha,
                          'candidate_id':-1,'onset_rel_ms':round(max(0.,ptime-t0-100.),1),
                          'peak_rel_ms':round(ptime-t0,1),'category':category,
                          'candidate_type':'POINTER_ONLY_UNVERIFIED',
                          'shape_score':0.,'activity':0.,'rise_time_ms':0.,
                          'candidate_pair_id':0,'silver_supported':0,
                          'review_priority':'MISSING_SOFT_TAP_REVIEW'})
    return queue


def export(root: Path, output: Path, annotation_file: Path | None,
           policy_mode: str='candidate', max_files:int|None=None,
           only_double:bool=False, cfg:ExportPolicy=ExportPolicy()) -> dict:
    root=root.expanduser().resolve();output=output.expanduser().resolve()
    if root.name!='DATA_TAP' or not root.is_dir():
        raise ValueError('Input root must be an existing directory named DATA_TAP')
    if output==root or root in output.parents or output in root.parents:
        raise ValueError('Output must be disjoint from DATA_TAP, even through symlinks')
    if policy_mode not in ('candidate','production'):
        raise ValueError('Policy must be candidate or production')
    if policy_mode=='production' and annotation_file is None:
        raise ValueError('Production mode requires independent physical-onset annotation CSV')
    if output.exists():
        raise ValueError('Output path must not exist; results are immutable/versioned')
    stage=output.with_name(output.name+'.stage')
    if stage.exists():
        raise ValueError('Staging directory exists; inspect before rerunning')
    annotations=read_annotations(annotation_file)
    filelist=sorted(root.rglob('*.csv.gz'))+sorted(root.rglob('*.csv'))
    files=[f for f in filelist if f.parent.name in CATEGORIES]
    if only_double:files=[f for f in files if f.parent.name=='TAP' and 'doubletap' in f.name.lower()]
    if max_files is not None:
        if max_files<1:raise ValueError('max-files must be positive')
        files=files[:max_files]
    names={f.relative_to(root).as_posix() for f in files}
    if set(annotations)-names:
        raise ValueError(f'Annotations referencing missing/unselected source(s): {sorted(set(annotations)-names)[:5]}')
    manifest=[];review=[];duplicate_hashes={};gold_total=0;gold_pairs=0;errors=[]
    stage.mkdir(parents=True)
    try:
        for f in files:
            rel=f.relative_to(root).as_posix();entry={'source':rel,'category':f.parent.name,'labeler_version':VERSION}
            try:
                stat_before=f.stat();sha=input_hash(f);entry['source_sha256']=sha
                if sha in duplicate_hashes:
                    if rel in annotations:
                        raise ValueError('Annotations on duplicate recording; annotate canonical recording only')
                    entry.update(status='DUPLICATE',canonical=duplicate_hashes[sha])
                    manifest.append(entry);continue
                duplicate_hashes[sha]=rel
                for row in annotations.get(rel,[]):
                    if row.source_sha256!=sha:raise ValueError('Stale/tampered source checksum in annotation CSV')
                # Source has 134 columns in some revisions; only read the
                # explicitly supported sensor schema (input audit still hashes
                # every decompressed byte of the original source).
                keep={'Timestamp','PointerState',*sum(EMG,[]),
                      *[f'p_core->wl[CH{ch}]' for ch in range(1,4)],
                      *[f'p_core->acc_psd[0][{ax}][{z}]'
                        for ax in 'XYZ' for z in range(6)]}
                df=pd.read_csv(f,compression='infer',low_memory=False,
                               usecols=lambda col: col in keep)
                S=extract_signals(df)
                events=extract_events(S)
                S['columns']=list(df.columns)
                ptr=pd.to_numeric(df.PointerState,errors='coerce').to_numpy(float) if 'PointerState' in df else np.zeros(len(df))
                S['pointer']=ptr
                silver=evaluate_silver(events,S['ts'],ptr,f.parent.name)
                label,gold_detail,qc=build_labels(df,f.parent.name,annotations.get(rel,[]),silver,cfg,S,events)
                if stat_before.st_size!=f.stat().st_size or stat_before.st_mtime_ns!=f.stat().st_mtime_ns:
                    raise ValueError('Source changed during read')
                if int(qc['positive_rows'])>0 and not annotations.get(rel):
                    raise AssertionError('Unannotated positive label created')
                folder=stage/f.parent.name;folder.mkdir(exist_ok=True)
                name=f.name.removesuffix('.gz').removesuffix('.csv')
                label.to_csv(folder/(name+'.labels.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
                rows=[{'candidate_id':i,'source':rel,**asdict(e)} for i,e in enumerate(events,1)]
                pd.DataFrame(rows).to_csv(folder/(name+'.events.csv'),index=False)
                review.extend(build_review_queue(rel,sha,f.parent.name,events,S,silver))
                gold_total+=qc['gold_events'];gold_pairs+=qc['gold_pair_count']
                entry.update(status='OK',rows=len(df),**qc,
                             gold_detail=gold_detail)
            except Exception as exc:
                entry.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                errors.append(entry)
            manifest.append(entry)
        if errors:
            raise RuntimeError(f'{len(errors)} labeling failures; first: {errors[0]}')
        if policy_mode=='production' and gold_total==0:
            raise ValueError('Production release blocked: ZERO independently verified usable GOLD tap events')
        for group in review:
            group['reviewer_id']='';group['evidence_type']='';group['sync_error_ms']='';group['decision']=''
        pd.DataFrame(review).to_csv(stage/'review_queue.csv',index=False)
        pd.DataFrame(columns=ANNOTATION_FIELDS).to_csv(stage/'annotations_template.csv',index=False)
        (stage/'manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in manifest),encoding='utf-8')
        summary={'version':VERSION,'policy_mode':policy_mode,
                 'source_root':str(root),'files_seen':len(files),
                 'files_ok':sum(x['status']=='OK' for x in manifest),
                 'duplicates':sum(x['status']=='DUPLICATE' for x in manifest),
                 'failed':len(errors),'gold_events':gold_total,'gold_pairs':gold_pairs,
                 'silver_supported_files':sum(x.get('silver_status')=='SUPPORTED' for x in manifest),
                 'silver_proposals':sum(x.get('silver_supporting_events',0) for x in manifest),
                 'candidate_events':sum(x.get('candidate_events',0) for x in manifest),
                 'review_queue_rows':len(review),
                 'total_positive_rows':sum(x.get('positive_rows',0) for x in manifest),
                 'total_negative_rows':sum(x.get('negative_rows',0) for x in manifest),
                 'total_unknown_rows':sum(x.get('unknown_rows',0) for x in manifest),
                 'annotation_file_sha256':hashlib.sha256(annotation_file.read_bytes()).hexdigest() if annotation_file else None,
                 'interpretation':'Only independently verified physical onset events may be positive. Silver candidates are NOT ground truth.'}
        (stage/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
        os.replace(stage,output)
        return summary
    except Exception:
        # FAIL CLOSED: never expose an incomplete output as a production dataset.
        if stage.exists():shutil.rmtree(stage)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('DATA/DATA_TAP'))
    parser.add_argument('--out-dir',type=Path,default=Path('DATA/DATA_TAP_V4_CANDIDATES'))
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--policy',choices=('candidate','production'),default='candidate')
    parser.add_argument('--max-files',type=int)
    parser.add_argument('--only-double-taps',action='store_true')
    a=parser.parse_args()
    summary=export(a.input_dir,a.out_dir,a.annotations,a.policy,a.max_files,a.only_double_taps)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
