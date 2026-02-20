#!/bin/bash
cd "$(dirname "$0")/backend"

source .venv/bin/activate

# 固定用 8001：如果被占用就先關掉
lsof -ti :8001 | xargs -r kill -9

python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001