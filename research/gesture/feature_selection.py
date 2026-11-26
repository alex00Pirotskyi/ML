#!/usr/bin/env python3
"""Fold-local, group-wise feature ranking for the unified TAP / LONG_PRESS trainer (v3).

The groups are the trainer's: one group per recording FILE (MUDRA_SPLIT=file; a person split is not allowed), and
"people" below means those groups. The inner folds are gesture-aware like the trainer's CV folds: StratifiedKFold over
the files by recording type (folder | gesture | bpm, rare types pooled), so every inner validation part holds its
share of every gesture (inner_splits).

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
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

RANKER_VERSION = "fs-3"          # fs-3: gesture-aware file inner folds (fs-2: GroupKFold over the groups)
INNER_FOLDS = 3
INNER_STRATUM_MIN_FILES = 5      # as the trainer's STRATUM_MIN_FILES: rarer recording types are pooled one level up
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
                extra = it[4] if len(it) > 4 else None
                if extra and a:
                    extra = {k: (lo[lo >= t[a]], hi[lo >= t[a]]) for k, (lo, hi) in extra.items()}
                R.tap_replay_recording(p[s + a:e] >= threshold, t[a:], lab[a:], ons, it[2], c, extra=extra,
                                       kind=it[5] if len(it) > 5 else None)
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


# ============================================================================ exact fast shifts (TAP)
FAST_SHIFT = os.environ.get("MUDRA_FS_FAST", "1").strip() != "0"   # 0 = the original full re-prediction everywhere
FAST_MAX_ROW_FRACTION = 0.30     # above this share of rows in reach: the full model on the shifted matrix (as before)
FAST_MARGIN = 1e-9               # raw-score margin around the threshold for the reach test


class FastTapShift:
    """The same scores as InnerReplay.shift -> score(probabilities) -> restore, computed without re-predicting the
    whole model on every row. Exact by construction, and checked when it is built:

    * Only the trees that split on a shifted column can move a row's raw score, and each of them by at most its
      leaf range. A row whose raw-score distance to the threshold exceeds the sum of those ranges keeps its
      decision. Rows sorted by that distance once, so the rows in reach are a prefix.
    * Rows in reach are re-predicted with a sub-model of just those trees, built from the model text: its leaf
      sum on the shifted values minus the same trees' leaf sum on the original values (from the leaf table) is the
      change of the raw score. Rows whose leaves did not change keep their decision bit for bit.
    * If more than FAST_MAX_ROW_FRACTION of the rows are in reach, the full model is run on the shifted matrix,
      exactly as before.
    * The TAP counts are additive per recording, so only recordings whose decisions changed are counted again.

    The random shifts are drawn from the same generator in the same order as InnerReplay.shift. Built only when the
    leaf table reproduces the model's raw scores bit for bit and the baseline counts reproduce s0; otherwise None
    (the caller keeps the original path)."""

    def __init__(self, ev, model, thr, s0, p0):
        self.ev, self.R, self.thr, self.s0 = ev, ev.R, float(thr), float(s0)
        self.booster = model.booster_
        X = ev.X
        self.n = len(X)
        text = self.booster.model_to_string()
        head, rest = text.split("\nTree=0\n", 1)
        trees_part, tail = rest.split("\nend of trees\n", 1)
        self.blocks = [b.strip("\n") for b in ("Tree=0\n" + trees_part).split("\n\n\n") if b.strip()]
        self.head_lines = [line for line in head.split("\n") if not line.startswith("tree_sizes=")]
        kv_head = dict(line.split("=", 1) for line in self.head_lines if "=" in line)
        self.feature_names = kv_head["feature_names"].split()
        self.feature_infos = kv_head["feature_infos"].split()
        self.tail = tail[tail.index("parameters:"):] if "parameters:" in tail else ""
        self.leaf_values, feats = [], []
        for b in self.blocks:
            kv = dict(line.split("=", 1) for line in b.split("\n") if "=" in line)
            self.leaf_values.append(np.array([float(v) for v in kv["leaf_value"].split()], np.float64))
            feats.append(set(int(v) for v in kv["split_feature"].split()) if "split_feature" in kv else set())
        self.tree_feats = feats
        self.trees_of = {}
        for t, fs in enumerate(feats):
            for f in fs:
                self.trees_of.setdefault(f, []).append(t)
        self.leaf_range = np.array([lv.max() - lv.min() for lv in self.leaf_values], np.float64)
        leaf = self.booster.predict(X, pred_leaf=True)
        self.leaf = np.asfortranarray(leaf, dtype=np.uint8 if leaf.max() < 256 else np.int32)  # tree-major
        del leaf
        raw = np.zeros(self.n, np.float64)
        for t, lv in enumerate(self.leaf_values):          # LightGBM's own order: tree after tree, in double
            raw += lv[self.leaf[:, t]]
        self.exact = bool(np.array_equal(raw, self.booster.predict(X, raw_score=True)))
        self.raw0 = raw
        self.d0 = np.asarray(p0) >= self.thr
        logit = np.log(self.thr / (1.0 - self.thr)) if self.thr < 1.0 else np.inf
        dist = np.abs(raw - logit)
        self.order = np.argsort(dist, kind="stable")
        self.dist_sorted = dist[self.order]
        self.seg_start = np.array([s for s, _ in ev.seg], np.int64)
        self.seg_len = np.array([e - s for s, e in ev.seg], np.int64)
        self.seg_id = np.repeat(np.arange(len(ev.seg)), self.seg_len)
        # per-recording baseline counts (additive) and the totals, accumulated exactly as InnerReplay.report does
        R = self.R
        self.base = R.TapCounts()
        self.rec_hits = np.zeros(len(ev.seg), np.int64); self.rec_false = np.zeros(len(ev.seg), np.int64)
        for s, ((a, b), t, lab, it) in enumerate(zip(ev.seg, ev.t, ev.label, ev.items)):
            c = self._count(s, self.d0[a:b])
            self.rec_hits[s], self.rec_false[s] = c.hits, c.false
            self.base.add(c)
        self.kind_of = [it[5] if len(it) > 5 else None for it in ev.items]
        self.consistent = self._score(self.base.hits, self.base.false, {}, {}) == self.s0
        self.sub = {}
        self.stats = Counter()
        self.sub_ok = True
        if self.trees_of:                            # the sub-model text parses and sums the leaves exactly
            busiest = max(self.trees_of, key=lambda f: len(self.trees_of[f]))
            trees = self.trees_of[busiest]
            rows = np.arange(0, self.n, max(1, self.n // 5000))
            want = np.zeros(len(rows), np.float64)
            for t in trees:
                want += self.leaf_values[t][self.leaf[rows, t]]
            sub, cols, _ = self._sub_model(trees)
            got = sub.predict(np.ascontiguousarray(X[np.ix_(rows, cols)]), raw_score=True)
            self.sub_ok = bool(np.array_equal(got, want))

    @classmethod
    def build(cls, ev, model, thr, s0, p0, log=print):
        try:
            f = cls(ev, model, thr, s0, p0)
        except Exception as e:                       # unexpected model text etc.: keep the original path
            log(f"FS_RANKER fast shifts off: {type(e).__name__}: {e}")
            return None
        if not (f.exact and f.consistent and f.sub_ok):
            log(f"FS_RANKER fast shifts off: leaf table exact={f.exact}, baseline counts reproduce S0={f.consistent}, "
                f"sub-model exact={f.sub_ok}")
            return None
        return f

    def _count(self, s, decisions):
        ev, R = self.ev, self.R
        it = ev.items[s]
        c = R.TapCounts()
        R.tap_replay_recording(decisions, ev.t[s], ev.label[s], it[3], it[2], c, extra=None,
                               kind=it[5] if len(it) > 5 else None)
        return c

    def _score(self, hits, false, d_hits, d_false):
        c = self.R.TapCounts()
        c.onsets, c.hits, c.false, c.neg_ms = self.base.onsets, int(hits), int(false), self.base.neg_ms
        c.kind = {k: dict(onsets=v["onsets"], hits=v["hits"] + d_hits.get(k, 0), false=v["false"] + d_false.get(k, 0),
                          neg_ms=v["neg_ms"]) for k, v in self.base.kind.items()}
        return float(c.report(target=self.ev.target)["objective_score"]) if c.onsets else float("nan")

    def _draw(self, rng):
        """The shifts InnerReplay.shift would draw: one per recording of 8 rows or more, in order."""
        ks = np.zeros(len(self.seg_len), np.int64)
        for i, n in enumerate(self.seg_len.tolist()):
            if n >= 8:
                ks[i] = int(rng.integers(n // 4, 3 * n // 4 + 1))
        return ks

    def _sub_model(self, trees):
        """(booster of just these trees on the compact columns they split on, those columns)."""
        key = tuple(trees)
        if key not in self.sub:
            import lightgbm as lgb
            cols = sorted(set(f for t in trees for f in self.tree_feats[t]))
            pos = {f: i for i, f in enumerate(cols)}
            head = []
            for line in self.head_lines:
                if line.startswith("max_feature_idx="):
                    line = f"max_feature_idx={len(cols) - 1}"
                elif line.startswith("feature_names="):
                    line = "feature_names=" + " ".join(self.feature_names[f] for f in cols)
                elif line.startswith("feature_infos="):
                    line = "feature_infos=" + " ".join(self.feature_infos[f] for f in cols)
                head.append(line)
            body = []
            for i, t in enumerate(trees):
                out = []
                for line in self.blocks[t].split("\n"):
                    if line.startswith("Tree="):
                        line = f"Tree={i}"
                    elif line.startswith("split_feature="):
                        line = "split_feature=" + " ".join(str(pos[int(v)]) for v in line.split("=", 1)[1].split())
                    out.append(line)
                body.append("\n".join(out))
            text = "\n".join(head) + "\n" + "\n\n".join(body) + "\n\n\nend of trees\n\n" + self.tail
            self.sub[key] = (lgb.Booster(model_str=text), np.asarray(cols, np.int64), pos)
        return self.sub[key]

    def score(self, cols, rng):
        ks = self._draw(rng)
        trees = sorted(set(t for c in cols for t in self.trees_of.get(int(c), ())))
        if not trees:                                # no tree uses these columns: the scores cannot move
            self.stats["unused"] += 1
            return self.s0
        reach = float(self.leaf_range[trees].sum()) + FAST_MARGIN
        count = int(np.searchsorted(self.dist_sorted, reach, side="right"))
        if count == 0:                               # no decision within reach of these trees
            self.stats["out_of_reach"] += 1
            return self.s0
        if count <= FAST_MAX_ROW_FRACTION * self.n:
            flip = self._flips_in_reach(cols, ks, trees, count)
            self.stats["sub_model"] += 1; self.stats["rows"] += count
        else:
            flip = self._flips_full(cols, ks)
            self.stats["full_model"] += 1
        if not len(flip):
            return self.s0
        return self._rescore(flip)

    def _flips_in_reach(self, cols, ks, trees, count):
        X = self.ev.X
        rows = np.sort(self.order[:count])
        seg = self.seg_id[rows]
        a = self.seg_start[seg]; n = self.seg_len[seg]
        src = a + (rows - a - ks[seg]) % n          # np.roll(x, k)[i] == x[(i - k) mod n]; k = 0: unchanged
        sub, used, pos = self._sub_model(trees)
        Xu = X[np.ix_(rows, used)]                   # only the columns these trees split on
        for c in cols:
            if int(c) in pos:
                Xu[:, pos[int(c)]] = X[src, c]
        new = sub.predict(Xu, raw_score=True)
        old = np.zeros(count, np.float64)
        for t in trees:                              # the sub-model's own order and precision
            old += self.leaf_values[t][self.leaf[rows, t]]
        delta = new - old
        moved = delta != 0
        if not moved.any():
            return np.zeros(0, np.int64)
        r = rows[moved]
        p = 1.0 / (1.0 + np.exp(-(self.raw0[r] + delta[moved])))
        return r[(p >= self.thr) != self.d0[r]]

    def _flips_full(self, cols, ks):
        ev = self.ev
        saved = ev.X[:, cols].copy()
        for (s, e), k in zip(ev.seg, ks.tolist()):
            if e - s >= 8:
                ev.X[s:e, cols] = np.roll(ev.X[s:e, cols], k, axis=0)
        try:
            p = self.booster.predict(ev.X)
        finally:
            ev.restore(cols, saved)
        return np.flatnonzero((p >= self.thr) != self.d0)

    def _rescore(self, flip):
        flip = np.sort(flip)
        seg = self.seg_id[flip]
        hits, false = self.base.hits, self.base.false
        d_hits, d_false = {}, {}
        cut = np.flatnonzero(np.diff(seg)) + 1
        for f in np.split(flip, cut):                # the flipped rows of one recording at a time
            s = int(self.seg_id[f[0]])
            a, b = self.ev.seg[s]
            d = self.d0[a:b].copy()
            f = f - a
            d[f] = ~d[f]
            c = self._count(s, d)
            dh, df = c.hits - int(self.rec_hits[s]), c.false - int(self.rec_false[s])
            hits += dh; false += df
            k = self.kind_of[s]
            if k is not None:
                d_hits[k] = d_hits.get(k, 0) + dh; d_false[k] = d_false.get(k, 0) + df
        self.stats["recounted"] += len(cut) + 1
        return self._score(hits, false, d_hits, d_false)

    def stats_line(self):
        n = sum(self.stats[k] for k in ("unused", "out_of_reach", "sub_model", "full_model"))
        return (f"{n} shifts: {self.stats['unused']} unused, {self.stats['out_of_reach']} out of reach, "
                f"{self.stats['sub_model']} sub-model ({self.stats['rows']} rows), {self.stats['full_model']} full model; "
                f"{self.stats['recounted']} recordings recounted")


# ============================================================================ ranking
def ranking_key(data, task, outer_train_ids, columns, gate, tap_target, seed):
    h = hashlib.sha256()
    h.update(json.dumps(dict(v=RANKER_VERSION, task=task, columns=list(columns), gate=gate, target=tap_target,
                             seed=seed, inner=INNER_FOLDS, rf=REPEATS_FEATURE, rg=REPEATS_GROUP,
                             inner_split="file-gesture" if "stratum" in data else "group",
                             inner_min=INNER_STRATUM_MIN_FILES,
                             screen=SCREEN_PARAMS, spw=SCREEN_POSITIVE_WEIGHT, cuts=list(CLUSTER_CUTS),
                             sample=CLUSTER_SAMPLE_ROWS, link=CLUSTER_LINKAGE, tg=TAP_GRID, lg=LP_GRID),
                             sort_keys=True).encode())
    h.update(str(data.get("content_sha256")).encode())
    h.update(np.ascontiguousarray(np.sort(np.unique(data["group"][outer_train_ids]).astype(str))).tobytes())
    return h.hexdigest()


def merge_rare(labels, min_count: int):
    """Pool strata with fewer than min_count members one level up (folder|gesture|bpm -> folder|gesture -> folder
    -> other); a stratum still below 2 members joins the largest one. Same rule as the trainer's merge_rare."""
    lab = np.asarray(labels, object).copy()
    for level in (2, 1, 0):
        counts = Counter(lab.tolist())
        rare = np.array([counts[x] < min_count for x in lab], bool)
        if not rare.any():
            break
        lab[rare] = ["|".join(str(x).split("|")[:level]) or "other" for x in lab[rare]]
    counts = Counter(lab.tolist())
    if counts and min(counts.values()) < 2:
        big = counts.most_common(1)[0][0]
        lab = np.array([big if counts[x] < 2 else x for x in lab], object)
    return lab


def inner_splits(data, ids, seed=0):
    """Inner folds over the outer fold's training rows `ids` (positions into ids), by recording file: no group is on
    both sides. Gesture-aware when the trainer gives every row its recording type (data["stratum"], file split):
    StratifiedKFold over the files by type, as the trainer's CV folds. GroupKFold over the groups otherwise."""
    ids = np.asarray(ids)
    groups = np.asarray(data["group"][ids]).astype(str)
    if "stratum" not in data:
        return list(GroupKFold(n_splits=INNER_FOLDS).split(ids, data["y"][ids], groups))
    uniq, codes = np.unique(groups, return_inverse=True)
    _, first = np.unique(codes, return_index=True)
    strata = merge_rare(np.asarray(data["stratum"][ids])[first], INNER_STRATUM_MIN_FILES)
    out = []
    for _, b_g in StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=7919 + int(seed)).split(uniq, strata):
        val = np.zeros(len(uniq), bool); val[b_g] = True
        out.append((np.flatnonzero(~val[codes]), np.flatnonzero(val[codes])))
    return out


def shifted_score(ev, fast, model, thr, cols, rng):
    """Replay score with `cols` circularly shifted inside every recording (InnerReplay.shift draws from rng);
    the exact fast path when there is one."""
    if fast is not None:
        return fast.score(cols, rng)
    saved = ev.shift(cols, rng)
    try:
        return ev.score(ev.probabilities(model), thr)
    finally:
        ev.restore(cols, saved)


def fit_fold_ranker(g, R, data, rpl, outer_train_ids, task, *, seed=0, cache_dir: Path | None = None,
                    tap_target=None, log=print, threads=None):
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
    for i, (a, b) in enumerate(inner_splits(data, outer_train_ids, seed)):
        tr = outer_train_ids[a]; val_people = set(groups[outer_train_ids[b]])
        assert not (val_people & set(groups[tr])), "inner split shares people"
        params = dict(SCREEN_PARAMS, n_jobs=threads or g.CPU_THREADS, random_state=g.SEED + i)
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
        fast = FastTapShift.build(ev, model, thr, s0, p0, log) if (task == "tap" and FAST_SHIFT) else None
        imp = np.zeros(len(columns)); gimp = np.zeros(len(clusters))
        for j in range(len(columns)):
            vals = [shifted_score(ev, fast, model, thr, [j], rng) for _ in range(REPEATS_FEATURE)]
            imp[j] = s0 - float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        for ci, c in enumerate(clusters):
            if len(c) == 1:
                gimp[ci] = imp[c[0]]; continue
            vals = [shifted_score(ev, fast, model, thr, c, rng) for _ in range(REPEATS_GROUP)]
            gimp[ci] = s0 - float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        if fast is not None:
            log(f"FS_RANKER {task} inner {i}: fast shifts {fast.stats_line()}")
            fast = None
        gain = model.booster_.feature_importance("gain")
        inner.append(dict(split=i, people=len(val_people), threshold=thr, baseline_score=s0,
                          threshold_curve={str(k): v for k, v in curve.items()},
                          importance=imp.tolist(), group_importance=gimp.tolist(),
                          gain=(gain / max(gain.sum(), 1e-12)).tolist()))
        log(f"FS_RANKER {task} inner {i}: groups={len(val_people)} thr={thr} S0={s0:.4f} "
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
