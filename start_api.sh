#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da pasta do proprio script (portavel entre maquinas).
cd "$(dirname "$0")"
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
