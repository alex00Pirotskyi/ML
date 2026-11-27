#!/usr/bin/env python3
"""Fixed fold roadmap: read it (the trainer's API) and build the file-level, gesture-aware version.

Folds are made of recording FILES, never of people. Every recording is one unit: all its rows, and all its copies
(the same recording in DATA_TAP and DATA_LONG_PRESS, byte-identical duplicates) are in one fold. Inside every
stratum (dataset category, gesture, and the tap rhythm for TAP / SWIPE) the files are dealt to the folds largest
first, so every fold holds the same mix of gestures (DoubleTaps, MidTap, SoftTap, swipes, fist, each finger, wrist,
arm, band ...) and about the same number of rows. Deterministic: the same files always give the same folds.

Build (reads DATA/fold_roadmap.json for the file metadata and duplicate marks, adds recordings found on disk that
it does not list, writes a new roadmap; the old one is not changed):

    python fold_split.py build                       # -> DATA/fold_roadmap_files.json
    python fold_split.py check DATA/fold_roadmap_files.json

Read (the API train_tap_f1.py uses):

    client = load_roadmap(roadmap_path=Path("DATA/fold_roadmap_files.json"), data_root=Path("DATA"))
    client.get_records(dataset="DATA_TAP")           # trainable files: .path .fold_id .class_id .file_id
    client.get_file_info(path, dataset="DATA_TAP")   # a file of a derived folder, matched by its original name
    client.get_split(test_fold=0, dataset="DATA_TAP")
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

FILE_PLAN = "file_level_gesture_aware"
DEFAULT_BASE = Path("DATA/fold_roadmap.json")
DEFAULT_OUT = Path("DATA/fold_roadmap_files.json")
DEFAULT_SEED = 20261127
RHYTHM_CATEGORIES = ("TAP", "SWIPE")          # the tap rhythm (bpm) is part of the stratum here (240 bpm, ...)
ORIENTATIONS = ("horizontal", "vertical", "reverse", "flat", "up", "down", "side")


class RoadmapError(RuntimeError):
    """The roadmap does not describe the input: unknown, ambiguous or leaking files. Training must stop."""


@dataclass(frozen=True)
class FileRecord:
    path: Path
    fold_id: int
    class_id: int
    file_id: str
    dataset: str
    category: str
    gesture: str
    sample_rows: int


@dataclass(frozen=True)
class FileInfo:
    path: Path
    fold_id: int
    class_id: int
    file_id: str
    dataset: str
    category: str
    gesture: str
    exclude_from_training: bool
    duplicate_of_file_id: str | None
    organized_content_sha256: str | None
    recording_id: str | None


def _plain_name(name: str) -> str:
    return name[:-3] if name.endswith(".gz") else name


def _existing(path: Path) -> Path:
    """The roadmap lists .csv names; the data may be stored gzipped next to them."""
    if path.is_file():
        return path
    gz = path.with_name(path.name + ".gz")
    return gz if gz.is_file() else path


class RoadmapClient:
    def __init__(self, doc: Mapping[str, Any], data_root: Path, source: Path):
        self.doc = doc
        self.source = Path(source)
        self.data_root = Path(data_root)
        self.n_folds = int(doc["n_folds"])
        self.plan = str(doc.get("default_fold_plan", ""))
        self.data_fingerprint = str(doc.get("data_fingerprint_sha256", ""))
        review = doc.get("identity_review_required") or {}
        self.identity_review_required = list(review.keys() if isinstance(review, Mapping) else review)
        self._files = list(doc["files"])
        by_id = {f["file_id"]: f for f in self._files}
        if len(by_id) != len(self._files):
            raise RoadmapError(f"{source}: duplicate file_id entries")
        self._by_name: Dict[tuple, List[tuple]] = collections.defaultdict(list)
        datasets = set()
        for f in self._files:
            fold = int(f["fold_id"])
            if not 0 <= fold < self.n_folds:
                raise RoadmapError(f"{source}: {f['file_id']} has fold {fold} outside 0..{self.n_folds - 1}")
            for d in f["destinations"]:
                datasets.add(d["dataset"])
                rel = Path(d["relative_csv_path"])
                self._by_name[(d["dataset"], _plain_name(rel.name))].append((f, d))
        self.datasets = sorted(datasets)

    def _info(self, f: Mapping[str, Any], d: Mapping[str, Any], path: Path | None = None) -> FileInfo:
        return FileInfo(path=Path(path) if path is not None else _existing(self.data_root / d["relative_csv_path"]),
                        fold_id=int(f["fold_id"]), class_id=int(d["class_id"]), file_id=str(f["file_id"]),
                        dataset=str(d["dataset"]), category=str(d["category"]), gesture=str(f.get("gesture", "")),
                        exclude_from_training=bool(f.get("exclude_from_training", False)),
                        duplicate_of_file_id=f.get("duplicate_of_file_id"),
                        organized_content_sha256=f.get("organized_content_sha256"),
                        recording_id=f.get("recording_id"))

    def get_records(self, dataset: str) -> List[FileRecord]:
        """Trainable files of a dataset (duplicates marked excluded are left out), sorted by path."""
        out = []
        for f in self._files:
            if f.get("exclude_from_training"):
                continue
            for d in f["destinations"]:
                if d["dataset"] == dataset:
                    out.append(FileRecord(path=_existing(self.data_root / d["relative_csv_path"]),
                                          fold_id=int(f["fold_id"]), class_id=int(d["class_id"]),
                                          file_id=str(f["file_id"]), dataset=dataset, category=str(d["category"]),
                                          gesture=str(f.get("gesture", "")), sample_rows=int(f.get("sample_rows", 0))))
        return sorted(out, key=lambda r: r.path.as_posix())

    def get_file_info(self, path: str | Path, dataset: str) -> FileInfo:
        """The roadmap entry of a CSV anywhere (a derived folder keeps the original file name; .gz is ignored)."""
        p = Path(path)
        hits = self._by_name.get((dataset, _plain_name(p.name)), [])
        if len(hits) > 1:
            hits = [h for h in hits if h[1]["category"] == p.parent.name]
        if not hits:
            raise RoadmapError(f"{p} is not listed in {self.source} for dataset {dataset}")
        if len(hits) > 1:
            raise RoadmapError(f"{p} matches {len(hits)} roadmap entries of {dataset}")
        f, d = hits[0]
        return self._info(f, d, p)

    def get_split(self, test_fold: int, dataset: str) -> Dict[str, List[FileRecord]]:
        """Test = the files of one fold, train = the rest. Fails if one recording (same id or same content) is on
        both sides."""
        if not 0 <= int(test_fold) < self.n_folds:
            raise RoadmapError(f"test_fold {test_fold} outside 0..{self.n_folds - 1}")
        recs = self.get_records(dataset)
        meta = {f["file_id"]: f for f in self._files}
        side = collections.defaultdict(set)
        for r in recs:
            f = meta[r.file_id]
            s = "test" if r.fold_id == int(test_fold) else "train"
            for key in (f.get("recording_id"), f.get("organized_content_sha256"), f.get("content_sha256")):
                if key:
                    side[key].add(s)
        leaks = [k for k, v in side.items() if len(v) > 1]
        if leaks:
            raise RoadmapError(f"{len(leaks)} recordings are in train AND test fold {test_fold}: {leaks[:3]}")
        return {"train": [r for r in recs if r.fold_id != int(test_fold)],
                "test": [r for r in recs if r.fold_id == int(test_fold)]}


def load_roadmap(roadmap_path: str | Path = DEFAULT_OUT, data_root: str | Path = "DATA",
                 require_verified_identities: bool = False) -> RoadmapClient:
    path = Path(roadmap_path)
    if not path.is_file():
        raise RoadmapError(f"No fold roadmap at {path}. Build it with: python fold_split.py build")
    doc = json.loads(path.read_text(encoding="utf-8"))
    client = RoadmapClient(doc, Path(data_root), path)
    if require_verified_identities and client.plan != FILE_PLAN and client.identity_review_required:
        raise RoadmapError(f"{path}: {len(client.identity_review_required)} person identities need review")
    return client


# ============================================================================================ building the roadmap
def _sha256_text_file(path: Path) -> str:
    opener = gzip.open if path.name.endswith(".gz") else open
    h = hashlib.sha256()
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows_and_class(path: Path) -> tuple[int, int | None]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        ci = header.index("class_id") if "class_id" in header else None
        first = fh.readline()
        n = 1 if first.strip() else 0
        n += sum(1 for line in fh if line.strip())
    cls = int(float(first.split(",")[ci])) if (ci is not None and first.strip()) else None
    return n, cls


def parse_name(name: str) -> Dict[str, Any]:
    """person / gesture / orientation / hand / bpm from a recording file name."""
    stem = _plain_name(name)
    stem = stem[:-4] if stem.endswith(".csv") else stem
    tok = [t for t in stem.split("_") if t]
    person = tok[0] if tok else ""
    hand = "UNSPECIFIED"
    body = []
    for t in tok[1:]:
        low = t.lower()
        if low in ORIENTATIONS:
            break
        if low == "new":
            continue
        if low in ("lefthand", "righthand"):
            hand = low[:-4].upper()
            continue
        body.append(low)
    orient = next((t.upper() for t in tok[1:] if t.lower() in ORIENTATIONS), "UNSPECIFIED")
    m = re.search(r"_(\d+)bpm", stem)
    return dict(filename_person=person, gesture="".join(body) or (tok[1].lower() if len(tok) > 1 else ""),
                orientation=orient, hand=hand, bpm=int(m.group(1)) if m else 0)


def _stratum(f: Mapping[str, Any]) -> tuple:
    dests = f["destinations"]
    d = next((x for x in dests if x["dataset"] == "DATA_TAP"), dests[0])
    gesture = re.sub(r"[^a-z0-9]", "", str(f.get("gesture", "")).lower())
    bpm = int(f.get("bpm") or 0) if d["category"] in RHYTHM_CATEGORIES else 0
    return (d["dataset"], d["category"], gesture, bpm)


def _order_key(file_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{file_id}".encode()).hexdigest()


def assign_folds(files: List[Dict[str, Any]], n_folds: int, seed: int, fixed: Mapping[str, int] | None = None) -> None:
    """In place: fold_id of every file. Files in `fixed` (file_id -> fold, from the previous file roadmap) keep their
    fold, so new recordings never move old ones between train and test. The others are dealt per stratum (largest
    first, to the fold with the fewest files, then rows, of that stratum, then the fewest rows overall); a duplicate
    follows its canonical file."""
    fixed = {k: int(v) for k, v in (fixed or {}).items() if 0 <= int(v) < n_folds}
    canon = [f for f in files if not f.get("exclude_from_training")]
    strata = collections.defaultdict(list)
    for f in canon:
        strata[_stratum(f)].append(f)
    total = [0] * n_folds
    for f in canon:
        if f["file_id"] in fixed:
            total[fixed[f["file_id"]]] += int(f.get("sample_rows") or 0)
    for key in sorted(strata, key=lambda k: tuple(str(x) for x in k)):
        group = sorted(strata[key], key=lambda f: (-int(f.get("sample_rows") or 0), _order_key(f["file_id"], seed)))
        n_in = [0] * n_folds
        r_in = [0] * n_folds
        for f in group:
            if f["file_id"] in fixed:
                k = fixed[f["file_id"]]
                f["fold_id"] = k; n_in[k] += 1; r_in[k] += int(f.get("sample_rows") or 0)
        for f in group:
            if f["file_id"] in fixed:
                continue
            k = min(range(n_folds), key=lambda i: (n_in[i], r_in[i], total[i], i))
            f["fold_id"] = k
            rows = int(f.get("sample_rows") or 0)
            n_in[k] += 1; r_in[k] += rows; total[k] += rows
    by_id = {f["file_id"]: f for f in files}
    for f in files:
        if f.get("exclude_from_training"):
            src = by_id.get(f.get("duplicate_of_file_id") or "") or by_id.get(f.get("canonical_file_id") or "")
            if src is not None and not src.get("exclude_from_training"):
                f["fold_id"] = src["fold_id"]
            else:
                f["fold_id"] = int(_order_key(f["file_id"], seed), 16) % n_folds


def _scan(data_root: Path, datasets: Iterable[str]) -> Dict[tuple, Path]:
    out = {}
    for ds in datasets:
        root = data_root / ds
        if not root.is_dir():
            continue
        for cat in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
            for p in sorted(cat.iterdir()):
                if p.is_file() and (p.name.endswith(".csv") or p.name.endswith(".csv.gz")):
                    out[(ds, f"{ds}/{cat.name}/{_plain_name(p.name)}")] = p
    return out


def fold_summary(files: Sequence[Mapping[str, Any]], n_folds: int) -> List[Dict[str, Any]]:
    folds = []
    for k in range(n_folds):
        sel = [f for f in files if int(f["fold_id"]) == k and not f.get("exclude_from_training")]
        dsets: Dict[str, Any] = {}
        for f in sel:
            for d in f["destinations"]:
                e = dsets.setdefault(d["dataset"], dict(file_count=0, sample_rows=0, by_class={}, by_gesture={}))
                rows = int(f.get("sample_rows") or 0)
                e["file_count"] += 1; e["sample_rows"] += rows
                c = e["by_class"].setdefault(str(d["class_id"]), dict(files=0, rows=0)); c["files"] += 1; c["rows"] += rows
                g = e["by_gesture"].setdefault(f"{d['category']}/{f.get('gesture', '')}", dict(files=0, rows=0))
                g["files"] += 1; g["rows"] += rows
        folds.append(dict(fold_id=k, unique_source_files=len(sel),
                          unique_sample_rows=int(sum(int(f.get("sample_rows") or 0) for f in sel)),
                          excluded_duplicate_or_empty_files=sum(1 for f in files if int(f["fold_id"]) == k
                                                                and f.get("exclude_from_training")),
                          datasets=dsets))
    return folds


def build_file_roadmap(base_path: Path = DEFAULT_BASE, data_root: Path = Path("DATA"), out_path: Path = DEFAULT_OUT,
                       n_folds: int = 5, seed: int = DEFAULT_SEED, reshuffle: bool = False, log=print) -> Dict[str, Any]:
    """Keeps the folds of the recordings already in out_path (unless reshuffle) and deals only new ones."""
    base = json.loads(Path(base_path).read_text(encoding="utf-8")) if Path(base_path).is_file() else {}
    fixed = {}
    if not reshuffle and Path(out_path).is_file():
        prev = json.loads(Path(out_path).read_text(encoding="utf-8"))
        if int(prev.get("n_folds", -1)) == int(n_folds) and prev.get("default_fold_plan") == FILE_PLAN:
            fixed = {f["file_id"]: int(f["fold_id"]) for f in prev.get("files", [])}
    files = [dict(f) for f in base.get("files", [])]
    datasets = sorted({d["dataset"] for f in files for d in f["destinations"]} or {"DATA_TAP"})
    known = {(d["dataset"], d["relative_csv_path"]) for f in files for d in f["destinations"]}
    on_disk = _scan(Path(data_root), datasets)
    # recordings on disk that the base roadmap does not list: one record per file name (all its datasets)
    new = collections.defaultdict(list)
    for (ds, rel), p in on_disk.items():
        if (ds, rel) not in known:
            new[Path(rel).name].append((ds, rel, p))
    by_hash = {}
    for f in files:
        for key in (f.get("organized_content_sha256"), f.get("content_sha256")):
            if key and not f.get("exclude_from_training"):
                by_hash.setdefault(key, f["file_id"])
    added = 0
    for name in sorted(new):
        dests = []
        sha = rows = None
        for ds, rel, p in sorted(new[name]):
            n_, cls = _rows_and_class(p)                 # class per destination (a dataset may map it differently)
            if sha is None:
                sha, rows = _sha256_text_file(p), n_
            dests.append(dict(dataset=ds, category=Path(rel).parent.name, class_id=int(cls if cls is not None else -1),
                              relative_csv_path=rel))
        fid = "file_" + hashlib.sha256(f"{name}:{sha}".encode()).hexdigest()[:24]
        dup = by_hash.get(sha)
        meta = parse_name(name)
        files.append(dict(file_id=fid, source_relative_path=name, filename=name, user_raw=meta["filename_person"],
                          user_values=None, filename_person=meta["filename_person"],
                          subject_id=meta["filename_person"].lower(), subject_full_name_hint=meta["filename_person"],
                          identity_low_quality_user=False, gesture=meta["gesture"], gesture_raw=meta["gesture"],
                          orientation=meta["orientation"], direction="UNSPECIFIED", hand=meta["hand"], bpm=meta["bpm"],
                          recording_date=(re.search(r"(\d{4}-\d{2}-\d{2})", name) or [None])[0], sample_rows=int(rows or 0),
                          destinations=dests, content_sha256=sha, organized_content_sha256=sha,
                          recording_id="rec_" + sha[:24], fold_id=0, canonical_file_id=dup or fid,
                          exclude_from_training=bool(dup) or not rows, duplicate_of_file_id=dup,
                          added_by="fold_split.build (on disk, not in the base roadmap)"))
        if not dup:
            by_hash[sha] = fid
        added += 1
    missing = [(d["dataset"], d["relative_csv_path"]) for f in files for d in f["destinations"]
               if (d["dataset"], d["relative_csv_path"]) not in on_disk and not f.get("exclude_from_training")]
    assign_folds(files, n_folds, seed, fixed)
    kept = sum(1 for f in files if f["file_id"] in fixed)
    files.sort(key=lambda f: f["file_id"])
    fp = hashlib.sha256()
    for f in files:
        fp.update(f"{f['file_id']}|{f.get('organized_content_sha256')}|{f['fold_id']}|{int(bool(f.get('exclude_from_training')))}\n".encode())
    strata = collections.Counter(_stratum(f) for f in files if not f.get("exclude_from_training"))
    coverage = [dict(stratum="/".join(str(x) for x in k), files=v, folds_represented=min(v, n_folds),
                     reason="fewer files than folds") for k, v in sorted(strata.items(), key=lambda kv: str(kv[0])) if v < n_folds]
    base_sha = hashlib.sha256(Path(base_path).read_bytes()).hexdigest() if Path(base_path).is_file() else None
    doc = dict(
        schema_version=2, status="FILE_LEVEL_GESTURE_AWARE", default_fold_plan=FILE_PLAN, fold_unit="file",
        n_folds=int(n_folds), fold_index_base=0, seed=int(seed),
        stratification="dataset category + gesture (+ bpm for TAP / SWIPE); largest files first; duplicates follow "
                       "their canonical recording",
        source_folder=base.get("source_folder"), output_folder=base.get("output_folder", "DATA"),
        built_from=dict(path=str(base_path), sha256=base_sha), data_fingerprint_sha256=fp.hexdigest(),
        person_grouping_mode="none: folds are recording files (people are never a split unit)",
        reuse_existing_roadmap=False,
        summary=dict(base.get("summary", {}), files_in_roadmap=len(files), files_added_from_disk=added,
                     listed_but_missing_on_disk=len(missing),
                     unique_usable_files=sum(1 for f in files if not f.get("exclude_from_training")),
                     unique_usable_rows=int(sum(int(f.get("sample_rows") or 0) for f in files
                                                if not f.get("exclude_from_training")))),
        folds=fold_summary(files, n_folds), identity_review_required={}, coverage_warnings=coverage,
        warnings=list(base.get("warnings", [])) + ([f"{len(missing)} listed files are not on disk: {missing[:3]}"] if missing else []),
        files=files)
    Path(out_path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(f"[fold_split] {out_path}: {doc['summary']['unique_usable_files']} usable files in {n_folds} file folds "
        f"({added} added from disk, {len(missing)} listed but missing, {kept} kept their fold)")
    for fo in doc["folds"]:
        t = fo["datasets"].get("DATA_TAP", {})
        log(f"  fold {fo['fold_id']}: DATA_TAP files={t.get('file_count', 0)} rows={t.get('sample_rows', 0):,} "
            f"classes={ {k: v['files'] for k, v in sorted(t.get('by_class', {}).items())} }")
    return doc


def check(path: Path, data_root: Path = Path("DATA"), dataset: str = "DATA_TAP") -> int:
    c = load_roadmap(path, data_root)
    recs = c.get_records(dataset)
    missing = [r.path for r in recs if not r.path.is_file()]
    for k in range(c.n_folds):
        c.get_split(k, dataset)
    per = collections.Counter(r.fold_id for r in recs)
    print(f"{path}: plan={c.plan} folds={c.n_folds} {dataset} files={len(recs)} per fold={dict(sorted(per.items()))} "
          f"missing={len(missing)}")
    return 1 if missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", type=Path, default=DEFAULT_BASE)
    b.add_argument("--data-root", type=Path, default=Path("DATA"))
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    b.add_argument("--folds", type=int, default=5)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--reshuffle", action="store_true", help="deal every file again (old folds are not kept)")
    k = sub.add_parser("check")
    k.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUT)
    k.add_argument("--data-root", type=Path, default=Path("DATA"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_file_roadmap(a.base, a.data_root, a.out, a.folds, a.seed, a.reshuffle)
        return check(a.out, a.data_root)
    return check(a.path, a.data_root)


if __name__ == "__main__":
    sys.exit(main())
