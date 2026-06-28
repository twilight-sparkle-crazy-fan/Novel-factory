#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
LAUNCHER="$INSTALL_DIR/novel"

mkdir -p "$INSTALL_DIR"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf 'exec %q "$@"\n' "$ROOT/scripts/start.sh"
} > "$LAUNCHER"

chmod +x "$LAUNCHER"

echo "已安装 Novel-factory 启动命令：$LAUNCHER"

case ":$PATH:" in
  *":$INSTALL_DIR:"*)
    echo "现在可以直接输入：novel"
    ;;
  *)
    echo
    echo "注意：$INSTALL_DIR 还不在 PATH 中。"
    echo "请把下面这行加入 ~/.zshrc 或 ~/.bashrc 后重新打开终端："
    echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    echo
    echo "临时使用可以先运行："
    echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac
