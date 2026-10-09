#!/usr/bin/env bash
# Commit whatever the pipeline changed and push, retrying if another push landed first.
set -euo pipefail
git config user.name "blue-line-bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for p in public state tape data; do
  if [ -e "$p" ]; then git add -A -- "$p"; fi   # a folder can be missing before its first file
done
if git diff --cached --quiet; then
  echo "Nothing changed, nothing to commit."
  exit 0
fi
git commit -q -m "$1"
for i in 1 2 3 4 5; do
  if git push -q; then echo "Pushed: $1"; exit 0; fi
  git pull -q --rebase -X theirs || { git rebase --abort || true; git pull -q --no-rebase -X ours; }
  sleep $((i * 3))
done
echo "Push failed after 5 tries" >&2
exit 1
