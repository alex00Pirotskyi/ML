"""Get the repository ready for a fresh `make train` (run with the same MUDRA_* settings as the training).

1. Labels: when recordings were added or changed since the labels were made, the trainer refuses to reuse them
   ("Labels exist but are incomplete"). Out-of-date label folders are moved aside (never deleted), so the trainer
   rebuilds them. A partial label build left by a stopped run is moved aside too.
2. Results: when the results folder of these settings holds a FINISHED run, the trainer would only reuse it (same
   recordings) or refuse (new recordings). It is renamed <folder>.done_<time> (headers and reports kept), so the new
   run trains from scratch. An unfinished run is left in place and resumes.

Nothing is deleted. Prints one PREPARE line per action.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unified as g  # noqa: E402

STAMP = time.strftime("%Y%m%d_%H%M%S")


def aside(path: Path, tag: str) -> Path:
    dest = path.with_name(f"{path.name}.{tag}_{STAMP}")
    path.rename(dest)
    return dest


def labels():
    for raw, lab in ((g.RAW_SOURCE_TAP, g.LABEL_TAP), (g.RAW_SOURCE_LP, g.LABEL_LP)):
        staging = lab.with_name(lab.name + ".building")
        if staging.exists():
            print("PREPARE partial label build from a stopped run moved aside:", aside(staging, "partial"))
        if not lab.exists():
            print("PREPARE labels for", raw.name, "will be built")
            continue
        files = g.canonical_files(raw)
        if g.labels_complete(raw, lab, files):
            print("PREPARE labels for", raw.name, "are up to date:", len(files), "recordings")
        else:
            print("PREPARE recordings in", raw.name, "changed since the labels were made: old labels moved to",
                  aside(lab, "old"), "- they will be rebuilt")


def results():
    out = g.OUTPUT
    if not out.exists():
        print("PREPARE new results folder:", out)
        return
    done = []
    for task in g.TASKS_TO_RUN:
        f = out / task / "summary.json"
        try:
            done.append(f.is_file() and json.loads(f.read_text()).get("status") == "success")
        except (OSError, ValueError):
            done.append(False)
    if done and all(done):
        print("PREPARE finished run moved to", aside(out, "done"), "- a new run starts from scratch")
    else:
        print("PREPARE unfinished run found in", out, "- it resumes")


if __name__ == "__main__":
    labels()
    results()
