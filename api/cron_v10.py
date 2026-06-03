"""
Vercel Cron endpoint — /api/cron_v10

Runs one v10 single-tick (entry or flatten) per invocation and persists state
to Upstash KV (Vercel functions are ephemeral, so the local FileStore won't
survive between cron runs). Wired from vercel.json `crons`:

    entry   — 15:30 UTC weekdays → 11:30 ET (EDT) / 10:30 ET (EST). Chosen so the
              10:30 ET PRIME bar is COMPLETE in both seasons (the decision always
              slices morning bars to 10:30, matching the backtest exactly).
    flatten — 19:35 UTC weekdays → 15:35 ET (EDT) / 14:35 ET (EST), at/after the
              backtest's 15:30 ET EOD in EDT (bracket TP/SL cap intraday exits).

Mode is chosen from the current UTC hour unless overridden by ?mode=entry|flatten.

Security: if CRON_SECRET is set, requests must carry `Authorization: Bearer
<CRON_SECRET>` (Vercel Cron sends this automatically). Manual calls without it
are rejected when the secret is configured.
"""
import os
import sys
import json
import traceback
from http.server import BaseHTTPRequestHandler
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytz

NY = pytz.timezone("America/New_York")


def _authorized(headers) -> bool:
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        return True  # no secret configured → allow (best-effort)
    auth = headers.get("Authorization", "")
    return auth == f"Bearer {secret}"


def _pick_mode(query: str) -> str:
    # explicit override: ?mode=entry|flatten
    for part in query.split("&"):
        if part.startswith("mode="):
            v = part.split("=", 1)[1].strip().lower()
            if v in ("entry", "flatten"):
                return v
    # else decide by UTC hour: entry block 14-16 UTC, flatten otherwise
    h = datetime.utcnow().hour
    return "entry" if h in (14, 15, 16) else "flatten"


def _run(mode: str) -> dict:
    # Imported lazily so an import error surfaces as JSON, not a cold-start crash.
    from v10_runner import run_once_entry, run_once_flatten, KVStore
    store = KVStore()
    now = datetime.now(NY)
    if mode == "entry":
        result = run_once_entry(now=now, store=store)
    else:
        result = run_once_flatten(now=now, store=store)
    return result or {"action": "NOOP"}


class handler(BaseHTTPRequestHandler):
    def _respond(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not _authorized(self.headers):
            self._respond(401, {"ok": False, "error": "unauthorized"})
            return
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        mode = _pick_mode(query)
        try:
            result = _run(mode)
            self._respond(200, {"ok": True, "mode": mode, "result": result})
        except Exception as e:
            self._respond(500, {"ok": False, "mode": mode, "error": str(e),
                                "trace": traceback.format_exc()})

    # Vercel Cron uses GET; allow POST too for manual triggering.
    do_POST = do_GET
