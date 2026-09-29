"""Tiny JSON-over-HTTP client for fleet_coordinator.py, shared by the agent and the CLI."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class CoordinatorDown(Exception):
    """Could not reach the coordinator, or it answered 5xx. Transient: callers retry."""


class CoordinatorError(Exception):
    """The coordinator understood and refused (4xx). Not retryable as-is."""

    def __init__(self, code: int, message: str):
        super().__init__(f"HTTP {code}: {message}")
        self.code = code
        self.message = message


class FleetClient:
    def __init__(self, url: str, token: str, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def call(self, method: str, path: str, body: "dict | None" = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except (ValueError, OSError):
                msg = ""
            if e.code >= 500:
                raise CoordinatorDown(f"HTTP {e.code}: {msg}") from e
            raise CoordinatorError(e.code, msg) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise CoordinatorDown(str(e)) from e

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body or {})
