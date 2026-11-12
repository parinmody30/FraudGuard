#!/usr/bin/env bash
# Start the FraudLab backend (FastAPI :8000) and frontend (Vite :5173) together.
# First run:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#             (cd frontend && npm install) && cp .env.example .env
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

# Backend must run from the repo root (model artifacts are loaded from ./models).
"$PY" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT INT TERM

echo "Waiting for backend..."
for _ in $(seq 1 30); do
  curl -sf http://localhost:8000/api/health >/dev/null && break
  sleep 1
done
curl -sf http://localhost:8000/api/health >/dev/null || { echo "Backend failed to start"; exit 1; }
echo "Backend ready: http://localhost:8000/docs"

cd frontend
npm run dev
