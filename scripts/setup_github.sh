#!/usr/bin/env bash
# Create the public GitHub repo and push (requires: gh auth login)
set -euo pipefail
cd "$(dirname "$0")/.."

# gh auth status exits 1 if ANY stored account has a bad token, even when
# the active account is fine — so verify with an API call instead.
if ! USER=$(gh api user -q .login 2>/dev/null); then
  echo "GitHub CLI is not authenticated. Run:"
  echo "  gh auth login -h github.com"
  exit 1
fi
echo "Creating public repo dust-storm-forecast-saudi under ${USER}..."

PROTO=$(gh config get git_protocol -h github.com 2>/dev/null || echo "https")
if [[ "${PROTO}" == "ssh" ]]; then
  REMOTE_URL="git@github.com:${USER}/dust-storm-forecast-saudi.git"
else
  REMOTE_URL="https://github.com/${USER}/dust-storm-forecast-saudi.git"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

gh repo create dust-storm-forecast-saudi \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "24h dust-storm onset prediction for Saudi Arabia using XGBoost and MODIS albedo anomaly"

echo "Done: https://github.com/${USER}/dust-storm-forecast-saudi"
