#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_MODE="${MODEL_MODE:-local}"
if [[ "${1:-}" == "--api" ]]; then
  MODEL_MODE="deepseek"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "用法：novel [--api]" >&2
  exit 2
fi
export MODEL_MODE

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
  if [[ "$MODEL_MODE" == "deepseek" ]]; then
    echo "若现有实例不是 API 模式，请先停止后再运行 novel --api。"
  fi
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

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT TERM

"$ROOT/.venv/bin/python" -m uvicorn backend.app:app \
  --host "$RESOLVED_HOST" \
  --port "$RESOLVED_PORT" \
  --no-access-log &
SERVER_PID=$!

echo "Novel-factory 已启动：http://$RESOLVED_HOST:$RESOLVED_PORT"
if [[ -t 0 ]]; then
  echo "需要关闭服务时，请在这里输入 exit 后回车。"
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    if IFS= read -r -t 1 COMMAND; then
      case "${COMMAND//[[:space:]]/}" in
        exit|quit)
          echo "正在关闭 Novel-factory…"
          stop_server
          exit 0
          ;;
        "") ;;
        *) echo "可输入 exit 关闭服务。" ;;
      esac
    fi
  done
fi

wait "$SERVER_PID"
EXIT_CODE=$?
SERVER_PID=""
exit "$EXIT_CODE"
