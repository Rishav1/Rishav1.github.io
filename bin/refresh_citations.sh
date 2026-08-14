#!/bin/bash
# Refresh _data/citations.yml and push the result, for unattended use.
#
# Google Scholar blocks datacenter IPs, so this cannot run on GitHub Actions
# (see .github/workflows/update-citations.yml). Running it from a home network
# works, so this wrapper is what the scheduled local job invokes.
#
# It is deliberately conservative: it does nothing at all unless the checkout
# is on the main branch with a clean working tree, so it can never disturb
# work in progress or push something unrelated.
#
# Usage: bin/refresh_citations.sh [--dry-run]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CITATIONS_PYTHON:-$HOME/.venvs/al-folio-citations/bin/python}"
BRANCH="main"
FILE="_data/citations.yml"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log() { printf '[%s] %b\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
skip() { log "SKIP: $*"; exit 0; }

cd "$REPO_DIR"

[[ -x "$PYTHON" ]] || skip "no interpreter at $PYTHON (run bin/install_citation_schedule.sh)"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$current_branch" == "$BRANCH" ]] || skip "on branch '$current_branch', not '$BRANCH'"

# Any pre-existing edit other than the citations file means someone is working
# in here; leave it alone rather than sweeping changes into an automated commit.
dirty="$(git status --porcelain --untracked-files=no | grep -v " $FILE\$" || true)"
[[ -z "$dirty" ]] || skip "working tree has other changes:\n$dirty"

log "Updating $BRANCH..."
git fetch --quiet origin "$BRANCH"
if ! git merge --ff-only --quiet "origin/$BRANCH" 2>/dev/null; then
  skip "local $BRANCH has diverged from origin; resolve manually"
fi

log "Fetching citations from Google Scholar..."
if ! "$PYTHON" bin/update_scholar_citations.py; then
  log "ERROR: citation update failed; leaving the repository untouched."
  exit 1
fi

if git diff --quiet -- "$FILE"; then
  log "No citation changes."
  exit 0
fi

git --no-pager diff --stat -- "$FILE"

if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry run: restoring $FILE and exiting without committing."
  git checkout -- "$FILE"
  exit 0
fi

log "Committing and pushing..."
git add -- "$FILE"
git -c user.name="Citations Bot" -c user.email="admin@betterdata.ai" \
  commit --quiet -m "Update Google Scholar citations"
git push --quiet origin "$BRANCH"
log "Pushed. Deploy workflow will publish the new counts."
