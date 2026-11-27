#!/usr/bin/env python3
"""Event-level check of the TAP F1 trainer's models (train_tap_f1.py): what a user sees on the device.

Every saved model of TRAIN_FOLDER (<tag>_model.pkl + <tag>_info.json) runs over every row of the held-out fold's
recordings (DATA/DATA_TAP_F1, the trainer's test fold) and fires like the firmware: an event on a rising edge of
score >= the model's decision threshold (0.85), then 150 ms refractory. Measured without any label:

  taps   every pointer press of the TAP / SWIPE test recordings, per kind (normal, soft, double 1st, double 2nd,
         240 bpm, slow pinch, swipe); hit = an event -100..+250 ms around the press (replay.py's rule)
  false  events per minute of every no-tap test recording, per kind (finger, fist, wrist, arm, band, nothing)

So models trained on different labels or inputs are measured the same way.

    python evaluate_tap_f1.py [TRAIN_FOLDER] [--data DATA/DATA_TAP_F1] [--fold 0]
    -> TRAIN_FOLDER/event_eval.md, event_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REFRACTORY_MS, HIT_BEFORE_MS, HIT_AFTER_MS, BOUNCE_MS, PAIR_MS = 150.0, 100.0, 250.0, 40.0, 500.0
POS_KINDS = ("normal", "soft", "double_1st", "double_2nd", "fast240", "slowpinch", "swipe")
NEG_KINDS = ("finger", "fist", "wrist", "arm", "band", "nothing")
TH_GRID = np.round(np.r_[np.arange(0.30, 0.95, 0.05), 0.97, 0.98, 0.99], 4)


def rec_kind(folder, rel):
    f = str(rel).split("/")[-1]
    if folder == "TAP":
        for key, kind in (("DoubleTap", "double"), ("240bpm", "fast240"), ("SlowPinch", "slowpinch"), ("SoftTap", "soft")):
            if key in f:
                return kind
        return "normal"
    if folder == "SWIPE":
        return "swipe"
    if folder == "FINGER":
        return "fist" if "fist" in f.lower() else "finger"
    return folder.lower()


def events(t, s, th):
    """firmware events: rising edge of s >= th, 150 ms refractory"""
    up = np.flatnonzero((s >= th) & ~np.r_[False, s[:-1] >= th])
    out, last = [], -1e18
    for r in up:
        if t[r] - last >= REFRACTORY_MS:
            out.append(t[r]); last = t[r]
    return np.asarray(out)


def presses(t, ptr):
    """pointer press-down times; a gap under 40 ms is a bounce; a press already down at the start is not counted"""
    e = np.diff(np.r_[0, np.asarray(ptr, np.int8), 0])
    dn = t[np.flatnonzero(e > 0)].astype(float)
    up = t[np.minimum(np.flatnonzero(e < 0) - 1, len(t) - 1)].astype(float)
    if len(dn) > 1:
        dn = dn[np.r_[True, (dn[1:] - up[:-1]) >= BOUNCE_MS]]
    if len(ptr) and ptr[0]:
        dn = dn[1:]
    return dn


def match(ev, pr):
    used = np.zeros(len(ev), bool); hit = np.zeros(len(pr), bool)
    for k, o in enumerate(pr):
        i = int(np.searchsorted(ev, o - HIT_BEFORE_MS))
        while i < len(ev) and used[i]:
            i += 1
        if i < len(ev) and ev[i] <= o + HIT_AFTER_MS:
            used[i] = True; hit[k] = True
    return hit, used


class Counts:
    """hits / presses per tap kind and false events / minutes per no-tap kind, on the threshold grid"""

    def __init__(self, ths=TH_GRID):
        self.ths = np.asarray(ths)
        self.a = {f"hit|{k}": np.zeros(len(ths)) for k in POS_KINDS}
        self.a.update({f"n|{k}": 0 for k in POS_KINDS})
        self.a.update({f"false|{k}": np.zeros(len(ths)) for k in NEG_KINDS})
        self.a.update({f"min|{k}": 0.0 for k in NEG_KINDS})
        self.a["extra|taprec"] = np.zeros(len(ths)); self.a["min|taprec"] = 0.0

    def add(self, folder, rel, t, ptr, s):
        t = np.asarray(t, float); kind = rec_kind(folder, rel)
        minutes = float(t[-1] - t[0]) / 6e4 if len(t) > 1 else 0.0
        if folder in ("TAP", "SWIPE"):
            pr = presses(t, ptr)
            pr = pr[(pr >= t[0] + HIT_BEFORE_MS) & (pr <= t[-1] - HIT_AFTER_MS)]
            gap = np.diff(np.r_[-1e18, pr])
            kinds = np.array([(("double_2nd" if g < PAIR_MS else "double_1st") if kind == "double" else kind) for g in gap])
            for k in set(kinds):
                self.a[f"n|{k}"] += int((kinds == k).sum())
            self.a["min|taprec"] += minutes
            for j, th in enumerate(self.ths):
                hit, used = match(events(t, s, th), pr)
                for k in set(kinds):
                    self.a[f"hit|{k}"][j] += hit[kinds == k].sum()
                self.a["extra|taprec"][j] += int((~used).sum())
        elif kind in NEG_KINDS:
            self.a[f"min|{kind}"] += minutes
            for j, th in enumerate(self.ths):
                self.a[f"false|{kind}"][j] += len(events(t, s, th))

    def merge(self, other):
        for k in self.a:
            self.a[k] = self.a[k] + other.a[k]
        return self

    def at(self, th):
        j = int(np.argmin(np.abs(self.ths - th))); a = self.a
        tap_n = sum(a[f"n|{k}"] for k in POS_KINDS if k != "swipe")
        out = dict(threshold=float(self.ths[j]),
                   taps=sum(a[f"hit|{k}"][j] for k in POS_KINDS if k != "swipe") / max(tap_n, 1))
        for k in POS_KINDS:
            out[k] = a[f"hit|{k}"][j] / max(a[f"n|{k}"], 1)
        mins = sum(a[f"min|{k}"] for k in NEG_KINDS)
        out["fpm_move"] = sum(a[f"false|{k}"][j] for k in NEG_KINDS) / max(mins, 1e-9)
        for k in NEG_KINDS:
            out[f"fpm_{k}"] = a[f"false|{k}"][j] / max(a[f"min|{k}"], 1e-9)
        out["extra_taprec_pm"] = a["extra|taprec"][j] / max(a["min|taprec"], 1e-9)
        return out

    def at_false_rate(self, target):
        """metrics at the threshold where the movement false-tap rate equals target (interpolated), or None"""
        a = self.a
        mins = sum(a[f"min|{k}"] for k in NEG_KINDS)
        curve = sum(a[f"false|{k}"] for k in NEG_KINDS) / max(mins, 1e-9)
        for j in range(len(self.ths) - 1):
            if curve[j] >= target >= curve[j + 1]:
                w = 0.0 if curve[j] == curve[j + 1] else (curve[j] - target) / (curve[j] - curve[j + 1])
                lerp = lambda v: v[j] * (1 - w) + v[j + 1] * w
                tap_n = sum(a[f"n|{k}"] for k in POS_KINDS if k != "swipe")
                out = dict(threshold=float(lerp(self.ths)), fpm_move=float(target),
                           taps=lerp(sum(a[f"hit|{k}"] for k in POS_KINDS if k != "swipe")) / max(tap_n, 1))
                for k in POS_KINDS:
                    out[k] = lerp(a[f"hit|{k}"]) / max(a[f"n|{k}"], 1)
                for k in NEG_KINDS:
                    out[f"fpm_{k}"] = lerp(a[f"false|{k}"]) / max(a[f"min|{k}"], 1e-9)
                return out
        return None

    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.a.items()}


SHOW = ["taps", "normal", "soft", "double_1st", "double_2nd", "fast240", "slowpinch", "swipe", "fpm_move", "fpm_fist",
        "fpm_finger", "fpm_wrist", "fpm_arm", "fpm_band", "fpm_nothing"]


def _predict(container, X):
    import lightgbm as lgb
    Xs = np.ascontiguousarray(X[:, np.asarray(container["indices"], int)], dtype=np.float32)
    pre = container.get("preprocessor")
    if pre is not None:
        Xs = np.asarray(pre.transform(Xs), np.float32)
    m = container["model"]
    if isinstance(m, lgb.LGBMModel):
        return np.asarray(m.booster_.predict(Xs), np.float64)
    return np.asarray(m.predict_proba(Xs)[:, 1], np.float64)


def evaluate_folder(train_folder, data=ROOT / "DATA" / "DATA_TAP_F1", fold=0,
                    roadmap=ROOT / "DATA" / "fold_roadmap_files.json", log=print):
    import joblib
    from fold_split import load_roadmap
    train_folder, data = Path(train_folder), Path(data)
    models = []
    for p in sorted(train_folder.glob("*_model.pkl")):
        info_p = p.with_name(p.name.replace("_model.pkl", "_info.json"))
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        models.append(dict(tag=p.name.replace("_model.pkl", ""), container=joblib.load(p), info=info))
    if not models:
        raise SystemExit(f"no *_model.pkl in {train_folder}")
    sp = Path(models[0]["info"].get("schema_file") or "")
    sp = sp if sp.is_file() else train_folder / "TAP_feature_schema.json"      # the folder may have been moved
    schema = json.loads(sp.read_text())
    names = list(schema["feature_names"])
    client = load_roadmap(roadmap, ROOT / "DATA")
    files = [p for p in sorted(data.rglob("*.csv")) if not any(x.startswith("_") for x in p.relative_to(data).parts[:-1])]
    test = []
    for p in files:
        info = client.get_file_info(p, "DATA_TAP")
        if info.fold_id == int(fold) and not info.exclude_from_training:
            test.append(p)
    counts = [Counts() for _ in models]
    for i, p in enumerate(test, 1):
        df = pd.read_csv(p, usecols=names + ["timestamp", "PointerState"])
        t = df["timestamp"].to_numpy(np.float64); t = t - t[0]
        X = df[names].to_numpy(np.float32)
        for m, c in zip(models, counts):
            c.add(p.parent.name, p.name, t, df["PointerState"].to_numpy() > 0, _predict(m["container"], X))
        if i % 200 == 0:
            log(f"  {i}/{len(test)} test recordings")
    rows, out = [], {}
    for m, c in zip(models, counts):
        th = float(m["container"].get("decision_threshold", 0.85))
        r = dict(model=m["tag"], size=m["info"].get("model_size_bytes"), f1_rows=m["info"].get("f1_score"), **c.at(th))
        rows.append(r)
        out[m["tag"]] = dict(at_decision_threshold=r, curve={f"{x:g}": c.at(x) for x in c.ths}, counts=c.to_json())
    df = pd.DataFrame(rows)
    (train_folder / "event_eval.json").write_text(json.dumps(out, indent=1, default=float))
    md = ["# Event check of the saved models (held-out fold %d, %d recordings)" % (int(fold), len(test)), "",
          "taps = share of pointer presses with an event -100..+250 ms; fpm = false events per minute of no-tap "
          "recordings; at each model's decision threshold, 150 ms refractory.", "",
          "| model | bytes | row F1 | thr | " + " | ".join(SHOW) + " |",
          "|---|---|---|---|" + "---|" * len(SHOW)]
    for r in rows:
        md.append(f"| {r['model']} | {r['size']} | {(r['f1_rows'] or 0):.4f} | {r['threshold']:.2f} | "
                  + " | ".join(f"{r[k]:.3f}" if not k.startswith("fpm") else f"{r[k]:.2f}" for k in SHOW) + " |")
    (train_folder / "event_eval.md").write_text("\n".join(md) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        log(df[["model", "size", "f1_rows", "threshold"] + SHOW].to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_folder", nargs="?", default="TAP_MODELS_F1_FILEFOLDS")
    ap.add_argument("--data", type=Path, default=ROOT / "DATA" / "DATA_TAP_F1")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--roadmap", type=Path, default=ROOT / "DATA" / "fold_roadmap_files.json")
    a = ap.parse_args(argv)
    evaluate_folder(a.train_folder, a.data, a.fold, a.roadmap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
