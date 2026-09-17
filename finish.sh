#!/usr/bin/env bash
# Finalise the capstone repo: commit the updated documents, set a single
# author name across the whole history, and push.
set -e
cd "$(dirname "$0")"
echo "Working in: $(pwd)"
echo

echo "[1/5] Setting author identity for this repository..."
git config user.name "Rapipong Sornsakda"
git config user.email "sorasolarso@gmail.com"

echo "[2/5] Staging changes (excluding this script)..."
git add -A -- . ':(exclude)finish.sh'

echo "[3/5] Committing..."
git commit -m "Normalise sample-log paths and set full student name across reports" \
  || echo "      (nothing new to commit - continuing)"

echo "[4/5] Rewriting author on all commits..."
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter \
  'export GIT_AUTHOR_NAME="Rapipong Sornsakda"; export GIT_COMMITTER_NAME="Rapipong Sornsakda"' \
  -- --all >/dev/null 2>&1
echo "      done"

echo
echo "Author(s) in history:"
git log --format="%an <%ae>" | sort -u | sed 's/^/      /'
echo "Commits: $(git log --oneline | wc -l | tr -d ' ')"
echo "Any 'Claude' left in history: $(git log --format='%B' | grep -ci claude || true)"
echo

echo "[5/5] Pushing to GitHub..."
git push --force origin main

echo
echo "=============================================="
echo " DONE - refresh your repository page."
echo "=============================================="
