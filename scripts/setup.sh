#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v llama-server >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "正在通过 Homebrew 安装 llama.cpp…"
    brew install llama.cpp
  else
    echo "未找到 llama-server。请先按照 llama.cpp 官方说明安装，再重新运行此脚本。" >&2
    exit 1
  fi
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

echo "正在安装 Python 依赖…"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已从 .env.example 创建 .env"
fi

echo
echo "准备完成。运行 ./scripts/start.sh 启动 Novel-factory。"
