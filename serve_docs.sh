#!/usr/bin/env bash
set -euo pipefail

# Launch the mkdocs site locally. Usage: ./serve_docs.sh [PORT]   (default 15000)
# mkdocs-material is in the `dev` dependency group, so `uv run` installs it.

PORT=${1:-15000}
LOG=/tmp/mkdocs_serve.log

# Run from the repo root (where mkdocs.yml lives), regardless of CWD.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXISTING=$(pgrep -f "mkdocs serve" || true)
if [ -n "${EXISTING}" ]; then
    read -r -p "Existing mkdocs server found (PIDs: ${EXISTING}). Kill it? [y/N] " REPLY
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        kill ${EXISTING}
        echo "Killed."
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "Building docs..."
uv run mkdocs build

echo "Serving at http://0.0.0.0:${PORT} (log: ${LOG})"
nohup uv run mkdocs serve --dev-addr "0.0.0.0:${PORT}" > "${LOG}" 2>&1 &
echo "PID $!"
