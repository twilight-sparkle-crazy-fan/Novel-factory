from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shell_launcher_accepts_exit_and_suppresses_access_log() -> None:
    launcher = (ROOT / "scripts/start.sh").read_text(encoding="utf-8")

    assert "--no-access-log" in launcher
    assert "exit|quit" in launcher
    assert "输入 exit" in launcher


def test_powershell_launcher_accepts_exit_and_suppresses_access_log() -> None:
    launcher = (ROOT / "scripts/start.ps1").read_text(encoding="utf-8")

    assert '"--no-access-log"' in launcher
    assert '@("exit", "quit")' in launcher
    assert "输入 exit" in launcher
