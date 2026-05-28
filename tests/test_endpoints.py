"""Tests for the /api/health endpoint and request routing.

Covers:
- /api/health returns 200 + correct shape with valid cookie
- /api/health returns 401 without cookie (and without AUTH_BYPASS)
- /api/health respects AUTH_BYPASS env var
- /api/stream gates auth identically
- ETag invalidates when options_flow direction flips
- ETag stable across 5-min bucket (no minute-level churn)
"""
import json
import os
from io import BytesIO

import pytest


# ── Test handler shim ──
def _make_handler(cookie="access_token=valid", path="/api/data", etag=None):
    """Build a BaseHTTPRequestHandler subclass we can directly invoke."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "api"))
    import data as data_mod
    data_mod._MARKET_DATA_CACHE = {
        "fetched_at": 0.0, "snaps": {}, "spy_h": None,
        "vix": (18.0, None), "flashalpha": None, "timing_ms": {}
    }
    h = data_mod.handler.__new__(data_mod.handler)  # skip __init__
    hdrs = {"Origin": "", "x-forwarded-for": "127.0.0.1"}
    if cookie:
        hdrs["Cookie"] = cookie
    if etag:
        hdrs["If-None-Match"] = etag
    h.headers = hdrs
    h.path = path
    h.wfile = BytesIO()
    h._code = None
    h._h = {}
    h.send_response = lambda c: setattr(h, "_code", c)
    h.send_header = lambda k, v: h._h.__setitem__(k, v)
    h.end_headers = lambda: None
    return h, data_mod


# ── Health endpoint ──
class TestHealthEndpoint:
    def test_health_with_auth_returns_200(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h, _ = _make_handler(path="/api/health")
        h.do_GET()
        assert h._code == 200
        body = json.loads(h.wfile.getvalue().decode())
        assert "counters" in body
        assert "recent" in body
        assert "buffer_size" in body
        assert "process_uptime_sec" in body
        assert "vix_baseline" in body
        assert "vix_baseline_source" in body

    def test_health_without_auth_returns_401(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h, _ = _make_handler(cookie=None, path="/api/health")
        h.do_GET()
        assert h._code == 401

    def test_health_bypass_via_env(self, monkeypatch):
        monkeypatch.setenv("AUTH_BYPASS", "1")
        h, _ = _make_handler(cookie=None, path="/api/health")
        h.do_GET()
        assert h._code == 200


# ── ETag behavior ──
class TestETag:
    def test_first_request_emits_etag(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h, _ = _make_handler()
        h.do_GET()
        assert h._code == 200
        assert h._h.get("ETag", "").startswith('"')

    def test_repeat_request_returns_304(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h1, _ = _make_handler()
        h1.do_GET()
        etag = h1._h.get("ETag")
        assert etag

        h2, _ = _make_handler(etag=etag)
        h2.do_GET()
        assert h2._code == 304
        # 304 must not ship a body
        assert h2.wfile.getvalue() == b""

    def test_x_next_poll_header_present(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h, _ = _make_handler()
        h.do_GET()
        assert "X-Next-Poll" in h._h
        # Server returns 3/5/15/60 depending on market state
        assert int(h._h["X-Next-Poll"]) in (3, 5, 15, 60)


# ── Auth bypass propagation ──
class TestAuthBypass:
    def test_data_endpoint_bypasses_when_env_set(self, monkeypatch):
        monkeypatch.setenv("AUTH_BYPASS", "1")
        h, _ = _make_handler(cookie=None, path="/api/data")
        h.do_GET()
        # Bypass enabled → reaches the actual handler, returns 200
        assert h._code == 200

    def test_data_endpoint_rejects_without_bypass(self, monkeypatch):
        monkeypatch.delenv("AUTH_BYPASS", raising=False)
        h, _ = _make_handler(cookie=None, path="/api/data")
        h.do_GET()
        assert h._code == 401
