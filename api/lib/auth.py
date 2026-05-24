"""
인증/CORS/레이트리밋 헬퍼.

api/data.py 본체에서 분리된 모듈. KV(Upstash) 자격증명은 호출자가
주입하도록 inversion 처리 — 순환 의존을 피한다.
"""
import os
import requests
from datetime import datetime
import pytz

NY = pytz.timezone("America/New_York")


# ── CORS ──────────────────────────────────────────────────────────
ALLOWED_ORIGINS = {
    "https://hannaealgo.vercel.app",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}


def is_origin_allowed(origin):
    if not origin:
        return False
    if origin in ALLOWED_ORIGINS:
        return True
    if origin.endswith(".vercel.app"):
        return True
    if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
        return True
    if origin == "http://localhost" or origin == "http://127.0.0.1":
        return True
    return False


# ── Google Sign-In Token Verification ──────────────────────────────
def verify_google_token(id_token):
    """Verify Google Sign-In JWT via Google tokeninfo endpoint."""
    if not id_token:
        return None
    try:
        r = requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": id_token},
            timeout=8,
        )
        if r.status_code != 200:
            print(f"[Google Token] tokeninfo HTTP {r.status_code}: {r.text[:200]}")
            return None
        payload = r.json()
        # Must be email-verified
        email_ok = payload.get("email_verified") in ("true", True)
        if not email_ok:
            print("[Google Token] email not verified")
            return None
        # Must be issued for our client_id
        expected_aud = os.getenv(
            "GOOGLE_CLIENT_ID",
            "729700534302-3eaf1oulfa91mt75ootm5m2lohvibk5p.apps.googleusercontent.com",
        )
        if payload.get("aud") != expected_aud:
            print(f"[Google Token] aud mismatch: {payload.get('aud')} vs {expected_aud}")
            return None
        return payload
    except Exception as e:
        print(f"[Google Token Error] {e}")
    return None


# ── Rate Limit (Upstash KV with in-memory fallback) ────────────────
_IN_MEM_LIMITS = {}


def check_rate_limit(ip, limit=15, kv_creds=None):
    """Client IP rate limiter. Defaults to 15 requests / minute.

    kv_creds: optional (base_url, token) tuple from caller's KV helper.
              If None or invalid, falls back to in-memory bucket.
    """
    global _IN_MEM_LIMITS
    minute_str = datetime.now(NY).strftime("%Y%m%d%H%M")

    # 1. Try Upstash Redis Rate Limiting if available
    base, token = (kv_creds or (None, None))
    if base and token:
        try:
            key = f"rate_limit:{ip}:{minute_str}"
            url = f"{base}/pipeline"
            commands = [
                ["INCR", key],
                ["EXPIRE", key, "60"],
            ]
            r = requests.post(
                url,
                json=commands,
                headers={"Authorization": f"Bearer {token}"},
                timeout=2,
            )
            if r.status_code == 200:
                res = r.json()
                if isinstance(res, list) and len(res) > 0:
                    count = res[0].get("result", 1)
                    if isinstance(count, int) and count > limit:
                        return False
                    return True
        except Exception as e:
            print(f"[Rate Limit KV Error] {e}")
            # Fall back to in-memory on KV error

    # 2. In-Memory Fallback
    for client in list(_IN_MEM_LIMITS.keys()):
        _IN_MEM_LIMITS[client] = {k: v for k, v in _IN_MEM_LIMITS[client].items() if k == minute_str}
        if not _IN_MEM_LIMITS[client]:
            del _IN_MEM_LIMITS[client]

    if ip not in _IN_MEM_LIMITS:
        _IN_MEM_LIMITS[ip] = {}

    current_count = _IN_MEM_LIMITS[ip].get(minute_str, 0)
    if current_count >= limit:
        return False

    _IN_MEM_LIMITS[ip][minute_str] = current_count + 1
    return True
