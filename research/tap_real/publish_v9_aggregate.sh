#!/usr/bin/env bash
# Publish ONLY aggregate reports in an isolated orphan worktree, leaving the
# source repo's working tree and its very large tracked data untouched.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/experiment-dir [run-label]" >&2
  exit 2
fi
EXPERIMENT="$(cd "$1" && pwd)"
RUN_LABEL="${2:-$(date -u +%Y%m%d-%H%M%S)}"
if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run label must use letters/numbers/underscores/hyphens only" >&2
  exit 2
fi
if [ ! -f "$EXPERIMENT/results_v9/SUMMARY_FOR_CHATGPT.json" ]; then
  echo "Missing summary: run analysis first" >&2
  exit 1
fi
REPO_ROOT="$(git rev-parse --show-toplevel)"
BRANCH="research/tap-v9-aggregate-$RUN_LABEL"
TEMP="$(mktemp -d)"
rmdir "$TEMP"
cleanup() {
  git -C "$REPO_ROOT" worktree remove --force "$TEMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# No checkout of the 2+ GB DATA files, no branch switch in the source tree.
git -C "$REPO_ROOT" worktree add --detach --no-checkout "$TEMP" HEAD
git -C "$TEMP" switch --orphan "$BRANCH"
DEST="$TEMP/research/tap_v9_aggregate_runs/$RUN_LABEL"
mkdir -p "$DEST"
for name in SUMMARY_FOR_CHATGPT.json REPORT_FOR_CHATGPT.md leaderboard.csv motif_summary.json motif_cluster_scan.csv SIMULATED_CHANNEL_DROPOUT.json; do
  if [ -f "$EXPERIMENT/results_v9/$name" ]; then
    cp "$EXPERIMENT/results_v9/$name" "$DEST/$name"
  fi
done
printf '%s\n' "Aggregate, research-only. No independently timed physical tap labels were used." > "$DEST/README.txt"
git -C "$TEMP" add -- "research/tap_v9_aggregate_runs/$RUN_LABEL"
git -C "$TEMP" commit -m "research(tap): aggregate V9 report $RUN_LABEL"
git -C "$TEMP" push -u origin "$BRANCH"
echo "Published: https://github.com/alex00Pirotskyi/ML/tree/$BRANCH/research/tap_v9_aggregate_runs/$RUN_LABEL"
