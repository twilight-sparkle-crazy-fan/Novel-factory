from __future__ import annotations

from backend.llama_process import RotatingTextLogWriter


def test_rotating_log_writer_caps_current_and_backup_files(tmp_path) -> None:
    log_path = tmp_path / "llama-server.log"
    max_bytes = 1024
    writer = RotatingTextLogWriter(log_path, max_bytes=max_bytes, backup_count=2)
    try:
        for index in range(120):
            writer.write(f"line {index:03d} " + "x" * 24 + "\n")
    finally:
        writer.close()

    log_files = sorted(tmp_path.glob("llama-server.log*"))
    assert log_path in log_files
    assert tmp_path / "llama-server.log.1" in log_files
    assert len(log_files) <= 3
    assert all(path.stat().st_size <= max_bytes for path in log_files)
