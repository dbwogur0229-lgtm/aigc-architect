#!/usr/bin/env bash
# AIGC Architect 실행 스크립트
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# .env 가 있으면 로드
[ -f .env ] && export $(grep -v '^#' .env | xargs)

streamlit run src/app.py
