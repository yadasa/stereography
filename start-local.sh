#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
./start-worker.sh &
stereo_worker_pid=$!
trap 'kill "$stereo_worker_pid" 2>/dev/null || true' EXIT INT TERM

echo "Opening Stereo Depth Lab at http://127.0.0.1:4173"
npm run dev

