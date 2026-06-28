#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "尚未安装项目依赖，正在运行 setup.sh…"
  "$ROOT/scripts/setup.sh"
fi

PYTHON="$ROOT/.venv/bin/python"
RESOLVED_HOST="$($PYTHON -c 'from backend.config import get_settings; print(get_settings().app_host)')"
RESOLVED_PORT="$($PYTHON -c 'from backend.config import get_settings; print(get_settings().app_port)')"
OPEN_BROWSER="${NOVEL_FACTORY_OPEN_BROWSER:-true}"

set +e
PORT_STATE="$($PYTHON "$ROOT/scripts/check_app_port.py" "$RESOLVED_HOST" "$RESOLVED_PORT")"
PORT_CODE=$?
set -e

if [[ $PORT_CODE -eq 10 ]]; then
  echo "Novel-factory 已经在运行：http://$RESOLVED_HOST:$RESOLVED_PORT"
  echo "无需重复启动，直接在浏览器中打开上面的地址即可。"
  if [[ "$OPEN_BROWSER" != "false" && "$OPEN_BROWSER" != "0" ]]; then
    "$PYTHON" "$ROOT/scripts/open_browser.py" "$RESOLVED_HOST" "$RESOLVED_PORT" >/dev/null 2>&1 || true
  fi
  exit 0
fi

if [[ $PORT_CODE -eq 11 ]]; then
  echo "无法启动：$RESOLVED_HOST:$RESOLVED_PORT 已被其他程序占用。" >&2
  echo "可以查看占用程序：lsof -nP -iTCP:$RESOLVED_PORT -sTCP:LISTEN" >&2
  echo "或临时换一个端口：APP_PORT=$((RESOLVED_PORT + 1)) ./scripts/start.sh" >&2
  exit 1
fi

if [[ $PORT_CODE -ne 0 ]]; then
  echo "检查应用端口失败：$PORT_STATE" >&2
  exit "$PORT_CODE"
fi

if [[ "$OPEN_BROWSER" != "false" && "$OPEN_BROWSER" != "0" ]]; then
  "$PYTHON" "$ROOT/scripts/open_browser.py" "$RESOLVED_HOST" "$RESOLVED_PORT" >/dev/null 2>&1 &
fi

exec "$ROOT/.venv/bin/python" -m uvicorn backend.app:app \
  --host "$RESOLVED_HOST" \
  --port "$RESOLVED_PORT"
