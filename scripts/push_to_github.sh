#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_URL="${1:-}"
if [ -z "${REMOTE_URL}" ]; then
  echo "Usage: bash scripts/push_to_github.sh git@github.com:<user>/production-network-alpha.git"
  exit 2
fi

python scripts/release_git_preflight.py --check-only

git init
git branch -M main
git add -A
git commit -m "Public release: production-network alpha research pipeline" || true
git remote remove origin 2>/dev/null || true
git remote add origin "${REMOTE_URL}"
git push -u origin main
