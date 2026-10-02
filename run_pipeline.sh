#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da pasta do proprio script (portavel entre maquinas).
cd "$(dirname "$0")"

echo "[1/2] Gerando dataset real..."
python3 generate_data.py

echo "[2/2] Subindo API em :8000..."
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
