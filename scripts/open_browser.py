#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def display_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def wait_until_ready(url: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    health_url = f"{url}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return True
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(0.75)
    return False


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print("usage: open_browser.py HOST PORT [PATH]", file=sys.stderr)
        return 2
    host = display_host(sys.argv[1])
    port = int(sys.argv[2])
    path = sys.argv[3] if len(sys.argv) == 4 else ""
    if path and not path.startswith("/"):
        path = "/" + path
    url = f"http://{host}:{port}{path}"
    wait_until_ready(f"http://{host}:{port}")
    webbrowser.open(url, new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
