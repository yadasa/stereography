#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r worker/requirements.txt
fi

echo "Stereo Depth Lab worker"
echo "Local endpoint: http://127.0.0.1:8765"
echo "Leave this terminal open while using the private site."
exec .venv/bin/python -m uvicorn worker.main:app --host 127.0.0.1 --port 8765

