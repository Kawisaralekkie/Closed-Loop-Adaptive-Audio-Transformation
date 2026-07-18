#!/usr/bin/env bash
#
# git_save_onedrive.sh — commit + push specific files WITHOUT scanning the
# whole working tree.
#
# WHY THIS EXISTS
#   This repository lives inside OneDrive (Files On-Demand). Many tracked
#   files are cloud-only placeholders, so a normal `git status` / `git commit`
#   tries to mmap them and fails with "mmap failed: Operation timed out".
#
#   This script sidesteps that by:
#     1. `git add <paths>`                 — stage only the files you name
#     2. `git write-tree`                  — build a tree from the INDEX only
#                                            (no working-tree scan)
#     3. `git commit-tree ... -p HEAD`     — create the commit object
#     4. `git update-ref HEAD <commit>`    — advance the current branch
#     5. `git push origin <branch>`        — publish
#
# USAGE
#   scripts/git_save_onedrive.sh "commit message" <path> [<path> ...]
#
# EXAMPLES
#   scripts/git_save_onedrive.sh "Add CSV exporter" scripts/export_run_metrics_csv.py
#   scripts/git_save_onedrive.sh "Update Dockerfile + config" Dockerfile src/config.py
#
# NOTES
#   * Only the files you list are committed. Nothing else is touched.
#   * Run from anywhere inside the repo.
#   * To skip the push (commit locally only), set NO_PUSH=1:
#       NO_PUSH=1 scripts/git_save_onedrive.sh "msg" path/to/file
#   * To commit a file that is .gitignore'd (e.g. an analysis CSV under
#     logs/), set FORCE=1 so it is force-added:
#       FORCE=1 scripts/git_save_onedrive.sh "msg" logs/.../metrics.csv

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 \"commit message\" <path> [<path> ...]" >&2
    exit 1
fi

MSG="$1"
shift

# Move to the repo root so relative paths behave predictably.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Repo:   $REPO_ROOT"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Branch: $BRANCH"

# 1. Stage only the named paths (git add on specific files does NOT scan
#    the entire working tree, so it stays fast on OneDrive).
#    FORCE=1 allows staging files that are excluded by .gitignore.
echo "Staging: $*"
if [ "${FORCE:-0}" = "1" ]; then
    git add -f -- "$@"
else
    git add -- "$@"
fi

# 2. Build a tree object straight from the index (no working-tree scan).
TREE="$(git write-tree)"
echo "tree:   $TREE"

# 3. Create the commit object with the current HEAD as parent.
COMMIT="$(printf '%s\n' "$MSG" | git commit-tree "$TREE" -p HEAD)"
echo "commit: $COMMIT"

# 4. Advance the branch to the new commit.
git update-ref HEAD "$COMMIT"
echo "HEAD -> $COMMIT ($BRANCH)"

# 5. Push (unless NO_PUSH is set).
if [ "${NO_PUSH:-0}" = "1" ]; then
    echo "NO_PUSH=1 set — skipping push. Run 'git push origin $BRANCH' when ready."
else
    echo "Pushing to origin/$BRANCH ..."
    git push origin "$BRANCH"
    echo "Done. Pushed $COMMIT to origin/$BRANCH."
fi
