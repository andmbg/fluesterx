#!/usr/bin/env bash
set -euo pipefail

missing=0
for v in API_TOKEN HF_TOKEN; do
  if [ -z "${!v:-}" ]; then
    echo "ERROR: required environment variable '$v' is not set" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "Container exiting due to missing required environment variables." >&2
  exit 1
fi

exec "$@"
