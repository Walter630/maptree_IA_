#!/usr/bin/env bash
set -euo pipefail

cd /home/walter/maptree_IA

echo "[1/3] Gerando dataset real..."
python3 generate_data.py

echo "[2/3] Treinando modelos..."
python3 train_models.py

echo "[3/3] Subindo API em :8000..."
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
