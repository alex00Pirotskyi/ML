# ML

Python utilities for gesture dataset validation, organization, PSD processing, and model export.

## Automatically commit CSV files as `.csv.gz`

Enable the repository's Git hook **once per local clone**:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

Then use ordinary Git commands:

```bash
git add .
git commit -m "Add recordings"
git push
```

The pre-commit hook automatically replaces **staged** `*.csv` entries with
losslessly compressed `*.csv.gz` files in the commit. Original `.csv` files
remain on your disk; no uncompressed CSV data is included in the new commit.
It also works for CSVs in nested folders and files with spaces in their names.
The archives have deterministic contents (gzip timestamp set to zero).
Already compressed `.csv.gz` files are left as-is. Python 3 is required.

To read compressed CSVs directly:

```python
import pandas as pd

df = pd.read_csv("DATA/recording.csv.gz")
```

**Important:** Git does not automatically commit new, untracked files or push
commits. You still run `git add`, `git commit`, and `git push`; the hook takes
care of compression during `git commit`. Hooks are local Git configuration,
so the one-time setup must be done separately on each clone. If you bypass or
do not enable the hook, raw CSV files can be committed normally.

The existing `valid.py`, `org.py`, and `psd_mean.py` currently expect plain
`.csv` files, so keep or decompress local CSVs when using those scripts.
For archives approaching 100 MB each, use Git LFS instead of regular Git.
