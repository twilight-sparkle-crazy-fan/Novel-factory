#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request


FREE = 0
NOVEL_FACTORY_RUNNING = 10
OCCUPIED_BY_OTHER = 11


def check(host: str, port: int) -> int:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=0.8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("app") in {"Novel-factory", "LLM4chat"} and payload.get("status") == "ok":
            print("novel-factory")
            return NOVEL_FACTORY_RUNNING
    except (OSError, ValueError, urllib.error.URLError):
        pass

    try:
        with socket.create_connection((display_host, port), timeout=0.5):
            print("occupied")
            return OCCUPIED_BY_OTHER
    except OSError:
        print("free")
        return FREE


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_app_port.py HOST PORT", file=sys.stderr)
        return 2
    return check(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
