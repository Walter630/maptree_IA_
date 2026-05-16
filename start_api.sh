#!/bin/bash
cd /home/walter/maptree_IA
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8000