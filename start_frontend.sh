#!/bin/bash
cd "$(dirname "$0")/frontend"

# 如果 5173 被占用，先關掉
lsof -ti :5173 | xargs -r kill -9

# 用最穩的方式開前端
python3 -m http.server 5173