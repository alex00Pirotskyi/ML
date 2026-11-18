#!/usr/bin/env python3
"""Fold-local, person-grouped feature ranking for the unified TAP / LONG_PRESS trainer (v3).

Implements FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md:

* Candidate pool = every available firmware field (150 today). No family allow/deny lists.
* Everything is fitted on ONE outer fold's training people only:
  - correlation clusters (Spearman |rho| on their training rows, complete linkage, cut chosen by silhouette
    among strong-correlation cuts, |rho| >= 0.7);
  - an inner person-grouped cross-fitting (INNER_FOLDS): a screening LightGBM on all candidates is trained
    on inner-training people and replayed through the firmware metric on inner-validation people.
* Relevance = drop in that replay score when a feature (or a whole correlated cluster, jointly) is
  perturbed by a WITHIN-RECORDING CIRCULAR SHIFT. This keeps recording boundaries and each column's
  autocorrelation and breaks only its alignment with the labels and the other columns. It is block
  permutation importance (marginal), NOT conditional sampling; the correlated-group measure is the
  joint shift of the cluster. Reports say so.
* The replay score is evaluated at one operating threshold per inner split, chosen on the UNPERTURBED
  screening model (the screening model has a fixed class weight, so the trainer's fixed firmware
  threshold would be an arbitrary point on its curve). The same threshold scores every perturbation.
* Ordering: clusters by joint importance; inside a cluster, members by their own importance; round-robin
  over clusters whose joint importance is positive, then the remaining members, then everything else.
  The ordering is only a search path: the trainer's integer K (1..150) + retraining + replay decide.
* Rankings are cached by a key over data content, labels, columns, gate, task, outer-fold people,
  ranker version and settings; any change invalidates the cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold

RANKER_VERSION = "fs-2"
INNER_FOLDS = 3
REPEATS_FEATURE = 1
REPEATS_GROUP = 2
CLUSTER_SAMPLE_ROWS = 60_000
# Candidate distance cuts (1 - |rho|). Complete linkage: EVERY pair inside a cluster has |rho| >= 1 - cut, so a
# cluster is a set of interchangeable predictors. Cuts stop at strong correlation (|rho| >= 0.7): with looser
# cuts the silhouette prefers splitting whole sensor blocks (e.g. 54 ACC fields at |rho| >= 0.51 on the full
# TAP data), which is a family split, not a substitute group (measured, see FEATURE_SELECTION_RESULTS.md).
CLUSTER_CUTS = tuple(np.round(np.arange(0.05, 0.31, 0.05), 2))
CLUSTER_LINKAGE = "complete"
SCREEN_PARAMS = dict(objective="binary", n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=8,
                     min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.8, subsample=0.8,
                     subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
SCREEN_POSITIVE_WEIGHT = dict(tap=1.0, lp=0.02)
TAP_GRID = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999, 0.9995)
LP_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


# ============================================================================ correlation structure
def correlation_clusters(X: np.ndarray, seed: int = 0):
    """Clusters of mutually correlated columns from TRAINING rows: Spearman |rho|, complete linkage, distance
    cut chosen by the silhouette score among strong-correlation cuts (data-driven, no feature-family rules)."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=min(CLUSTER_SAMPLE_ROWS, len(X)), replace=False) if len(X) else np.zeros(0, int)
    R = np.apply_along_axis(rankdata, 0, np.asarray(X[np.sort(rows)], np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(np.abs(rho), nan=0.0)
    np.fill_diagonal(rho, 1.0)
    dist = np.clip(1.0 - rho, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method=CLUSTER_LINKAGE)
    best = None
    for cut in CLUSTER_CUTS:
        lab = fcluster(Z, t=cut, criterion="distance")
        k = len(set(lab))
        if 2 <= k <= len(lab) - 1:
            sil = float(silhouette_score(dist, lab, metric="precomputed"))
            if best is None or sil > best[1]:
                best = (float(cut), sil, lab)
    if best is None:
        lab = np.arange(1, X.shape[1] + 1); best = (0.0, float("nan"), lab)
    cut, sil, lab = best
    clusters = [sorted(np.flatnonzero(lab == c).tolist()) for c in sorted(set(lab))]
    return dict(clusters=clusters, cut=cut, silhouette=sil, abs_rho=rho)


# ============================================================================ replay scoring of one model
class InnerReplay:
    """Contiguous all-column replay matrix of the inner-validation people + the firmware replay rules."""

    def __init__(self, g, R, rpl, people, task, tap_target):
        self.g, self.R, self.task, self.target = g, R, task, tap_target
        prep = g.prepare_tap_replay if task == "tap" else g.prepare_lp_replay
        self.items = prep(rpl, people)
        ranges = [(it[0], it[1]) for it in self.items]
        self.X = g._gather(rpl["X"], ranges, list(range(rpl["X"].shape[1]))).copy() if ranges else None
        self.seg = []
        pos = 0
        for a, b in ranges:
            self.seg.append((pos, pos + b - a)); pos += b - a
        self.t = [np.asarray(rpl["t"][a:b], np.float64) for a, b in ranges]
        self.steady = (np.concatenate([np.asarray(rpl["steady"][a:b], bool) for a, b in ranges])
                       if ranges and "steady" in rpl else None)
        self.label = [np.asarray(rpl["label"][a:b]) for a, b in ranges] if task == "tap" else None

    def probabilities(self, model, sel=None, num_iteration=None):
        X = self.X if sel is None else np.ascontiguousarray(self.X[:, sel])
        return model.booster_.predict(X, num_iteration=num_iteration)

    def report(self, p, threshold, rows=None):
        """Full firmware replay report at one threshold; rows: optional boolean mask of countable rows
        per position (e.g. steady state) -- each recording then starts at its first countable row."""
        R = self.R
        if self.task == "tap":
            c = R.TapCounts()
            for (s, e), t, lab, it in zip(self.seg, self.t, self.label, self.items):
                a = 0
                if rows is not None:
                    nz = np.flatnonzero(rows[s:e])
                    if not len(nz):
                        continue
                    a = int(nz[0])
                ons = it[3] if a == 0 else it[3][it[3] >= t[a] + R.TAP_HIT_BEFORE_MS]
                R.tap_replay_recording(p[s + a:e] >= threshold, t[a:], lab[a:], ons, it[2], c)
            return c.report(target=self.target) if c.onsets else None
        c = R.LpCounts()
        q = R.q20(p)
        for (s, e), t, it in zip(self.seg, self.t, self.items):
            if rows is not None and not rows[s]:
                continue
            R.lp_replay_press(q[s:e], t, it[2], it[3], it[4], it[5], c, threshold=threshold)
        return c.report() if (c.holds + c.taps) else None

    def score(self, p, threshold):
        r = self.report(p, threshold)
        return float(r["objective_score"]) if r else float("nan")

    def operating_point(self, p):
        grid = TAP_GRID if self.task == "tap" else LP_GRID
        scores = {thr: self.score(p, thr) for thr in grid}
        thr = max(scores, key=lambda k: (-1 if np.isnan(scores[k]) else scores[k]))
        return thr, scores[thr], scores

    def shift(self, cols, rng):
        """Within-recording circular shift of the given columns (the same shift for all of them)."""
        saved = self.X[:, cols].copy()
        for s, e in self.seg:
            n = e - s
            if n >= 8:
                k = int(rng.integers(n // 4, 3 * n // 4 + 1))
                self.X[s:e, cols] = np.roll(self.X[s:e, cols], k, axis=0)
        return saved

    def restore(self, cols, saved):
        self.X[:, cols] = saved


# ============================================================================ ranking
def ranking_key(data, task, outer_train_ids, columns, gate, tap_target, seed):
    h = hashlib.sha256()
    h.update(json.dumps(dict(v=RANKER_VERSION, task=task, columns=list(columns), gate=gate, target=tap_target,
                             seed=seed, inner=INNER_FOLDS, rf=REPEATS_FEATURE, rg=REPEATS_GROUP,
                             screen=SCREEN_PARAMS, spw=SCREEN_POSITIVE_WEIGHT, cuts=list(CLUSTER_CUTS),
                             sample=CLUSTER_SAMPLE_ROWS, link=CLUSTER_LINKAGE, tg=TAP_GRID, lg=LP_GRID),
                             sort_keys=True).encode())
    h.update(str(data.get("content_sha256")).encode())
    h.update(np.ascontiguousarray(np.sort(np.unique(data["group"][outer_train_ids]).astype(str))).tobytes())
    return h.hexdigest()


def fit_fold_ranker(g, R, data, rpl, outer_train_ids, task, *, seed=0, cache_dir: Path | None = None,
                    tap_target=None, log=print):
    """Feature ranking from the outer fold's TRAINING people only (see module docstring)."""
    columns = list(data["columns"])
    tap_target = R.TAP_FALSE_PER_MIN_TARGET if tap_target is None else float(tap_target)
    gate = dict(train_min_frames=int(getattr(g, "TRAIN_MIN_FRAMES", getattr(g, "LEAN8_MIN_FRAMES", 0))),
                eval_min_frames=int(getattr(g, "EVAL_MIN_FRAMES", getattr(g, "LEAN8_MIN_FRAMES", 0))),
                eval_mask=getattr(g, "RANKER_REPLAY_MASK", "ready"))
    key = ranking_key(data, task, outer_train_ids, columns, gate, tap_target, seed)
    if cache_dir is not None:
        f = Path(cache_dir) / f"ranking_{task}_{key[:20]}.json"
        if f.is_file():
            return json.loads(f.read_text())
    t0 = time.perf_counter()
    groups = data["group"]
    outer_people = sorted(set(groups[outer_train_ids]))
    clus = correlation_clusters(data["X"][outer_train_ids], seed)
    clusters = clus["clusters"]
    member_of = {j: ci for ci, c in enumerate(clusters) for j in c}
    rpl_eval = dict(rpl, ready=rpl[gate["eval_mask"]])
    inner = []
    for i, (a, b) in enumerate(GroupKFold(n_splits=INNER_FOLDS).split(
            outer_train_ids, data["y"][outer_train_ids], groups[outer_train_ids])):
        tr = outer_train_ids[a]; val_people = set(groups[outer_train_ids[b]])
        assert not (val_people & set(groups[tr])), "inner split shares people"
        params = dict(SCREEN_PARAMS, n_jobs=g.CPU_THREADS, random_state=g.SEED + i)
        model = g.lgb.LGBMClassifier(**params)
        model.fit(data["X"][tr], data["y"][tr], sample_weight=g.weights_for_task(data, tr, task, SCREEN_POSITIVE_WEIGHT[task]))
        ev = InnerReplay(g, R, rpl_eval, val_people, task, tap_target)
        if ev.X is None:
            continue
        p0 = ev.probabilities(model)
        thr, s0, curve = ev.operating_point(p0)
        if not np.isfinite(s0):
            log(f"FS_RANKER {task} inner {i}: no countable events for these people at this gate; skipped")
            del ev
            continue
        rng = np.random.default_rng(seed * 1000 + i)
        imp = np.zeros(len(columns)); gimp = np.zeros(len(clusters))
        for j in range(len(columns)):
            vals = []
            for _ in range(REPEATS_FEATURE):
                saved = ev.shift([j], rng); vals.append(ev.score(ev.probabilities(model), thr)); ev.restore([j], saved)
            imp[j] = s0 - float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        for ci, c in enumerate(clusters):
            if len(c) == 1:
                gimp[ci] = imp[c[0]]; continue
            vals = []
            for _ in range(REPEATS_GROUP):
                saved = ev.shift(c, rng); vals.append(ev.score(ev.probabilities(model), thr)); ev.restore(c, saved)
            gimp[ci] = s0 - float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        gain = model.booster_.feature_importance("gain")
        inner.append(dict(split=i, people=len(val_people), threshold=thr, baseline_score=s0,
                          threshold_curve={str(k): v for k, v in curve.items()},
                          importance=imp.tolist(), group_importance=gimp.tolist(),
                          gain=(gain / max(gain.sum(), 1e-12)).tolist()))
        log(f"FS_RANKER {task} inner {i}: people={len(val_people)} thr={thr} S0={s0:.4f} "
            f"top={[columns[k] for k in np.argsort(-imp)[:5]]}")
        del ev
    if inner:
        imp = np.array([x["importance"] for x in inner]); gimp = np.array([x["group_importance"] for x in inner])
        mean_imp = np.nan_to_num(np.nanmean(imp, 0)); mean_g = np.nan_to_num(np.nanmean(gimp, 0))
        se_g = (np.nan_to_num(np.nanstd(gimp, 0, ddof=1)) / np.sqrt(len(inner)) if len(inner) > 1
                else np.zeros(len(clusters)))
    else:                                      # nothing countable: no evidence, keep the original column order
        log(f"FS_RANKER {task}: no usable inner split; ranking falls back to the original column order")
        mean_imp = np.zeros(len(columns)); mean_g = np.zeros(len(clusters)); se_g = np.zeros(len(clusters))
    order = group_aware_order(mean_imp, mean_g, se_g, clusters)
    out = dict(version=RANKER_VERSION, key=key, task=task, columns=columns, order=order,
               ordered_features=[columns[j] for j in order], mean_importance=mean_imp.tolist(),
               mean_group_importance=mean_g.tolist(), group_se=se_g.tolist(), clusters=clusters,
               cluster_of=[member_of[j] for j in range(len(columns))], cluster_cut=clus["cut"],
               silhouette=clus["silhouette"], inner=inner, outer_people=outer_people, gate=gate,
               tap_target=tap_target, method="within-recording circular-shift (block) permutation importance on "
               "firmware replay; correlated clusters shifted jointly; marginal, not conditional sampling",
               seconds=round(time.perf_counter() - t0, 1))
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"ranking_{task}_{key[:20]}.json").write_text(json.dumps(out))
    return out


def group_aware_order(imp, gimp, gse, clusters):
    """Search path for K. Clusters whose joint importance is positive (mean - 1 SE > 0), strongest first:
    round 1 takes each cluster's best member; later rounds take further members only while their own
    importance is positive. Then every remaining column by its own importance. Each column exactly once."""
    imp = np.asarray(imp, float); gimp = np.asarray(gimp, float); gse = np.asarray(gse, float)
    strong = [int(ci) for ci in np.argsort(-gimp) if gimp[ci] - gse[ci] > 0]
    queues = {ci: sorted(clusters[ci], key=lambda j: -imp[j]) for ci in strong}
    order = []
    first = True
    while True:
        added = False
        for ci in strong:
            q = queues[ci]
            if q and (first or imp[q[0]] > 0):
                order.append(int(q.pop(0))); added = True
        first = False
        if not added:
            break
    used = set(order)
    order += [int(j) for j in np.argsort(-imp, kind="stable") if int(j) not in used]
    assert sorted(order) == list(range(len(imp)))
    return order


def select_top_k(ranking, k):
    """First k features of the ranking, returned in the ORIGINAL column order (model matrix / C schema)."""
    k = max(1, min(int(k), len(ranking["order"])))
    return sorted(ranking["order"][:k])


# ============================================================================ stability
def nogueira_stability(subsets, d):
    """Nogueira, Sechidis & Brown (JMLR 2018) stability of M selected subsets of a d-feature space."""
    Z = np.zeros((len(subsets), d))
    for i, s in enumerate(subsets):
        Z[i, list(s)] = 1
    M = len(subsets)
    if M < 2:
        return float("nan")
    p = Z.mean(0)
    kbar = Z.sum(1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom <= 0:
        return 1.0
    s2 = M / (M - 1) * p * (1 - p)
    return float(1 - s2.mean() / denom)


def report_stability(rankings, ks=(5, 10, 15, 20, 30, 40, 60, 80, 100, 125)):
    """Stability of the top-K subsets across outer-fold rankings and across all inner-split orders."""
    d = len(rankings[0]["columns"])
    out = {}
    for k in ks:
        outer = [r["order"][:k] for r in rankings]
        inner = [list(np.argsort(-np.asarray(x["importance"]))[:k]) for r in rankings for x in r["inner"]]
        out[str(k)] = dict(outer_folds=nogueira_stability(outer, d), inner_splits=nogueira_stability(inner, d))
    freq = np.zeros(d)
    for r in rankings:
        for j in r["order"][:20]:
            freq[j] += 1
    out["top20_frequency"] = {rankings[0]["columns"][j]: int(freq[j]) for j in np.argsort(-freq) if freq[j] > 0}
    return out


def validate_fold_local(ranking, validation_people):
    """The ranking must not have used any validation person."""
    leak = set(ranking["outer_people"]) & set(validation_people)
    if leak:
        raise AssertionError(f"feature ranking used validation people: {sorted(leak)[:5]}")
