#!/usr/bin/env bash
# Create the public GitHub repo and push (requires: gh auth login)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run:"
  echo "  gh auth login -h github.com"
  exit 1
fi

USER=$(gh api user -q .login)
echo "Creating public repo dust-storm-forecast-saudi under ${USER}..."

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "git@github.com:${USER}/dust-storm-forecast-saudi.git"
else
  git remote add origin "git@github.com:${USER}/dust-storm-forecast-saudi.git"
fi

gh repo create dust-storm-forecast-saudi \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "24h dust-storm onset prediction for Saudi Arabia using XGBoost and MODIS albedo anomaly"

echo "Done: https://github.com/${USER}/dust-storm-forecast-saudi"
