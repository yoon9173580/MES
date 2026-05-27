"""
LAYER 3 — Options Flow Bias

SPY 옵션 체인의 Put/Call 비율 + Open Interest 분포로 시장 의향 파악.

데이터 소스:
  1순위: ENABLE_OPTIONS_API=true이고 ALPACA 옵션 데이터 권한 있을 때 → Alpaca
  2순위: Yahoo Finance 무료 옵션 체인 (rate limited, 정확도 떨어짐)
  fallback: 데이터 없으면 score 0 + status NO_DATA

Vercel 서버리스에서 매 요청마다 외부 호출은 비싸므로 ttl=120초 캐시.
"""
import os
import time
import requests
from typing import Optional

_CACHE = {"at": 0.0, "data": None}
CACHE_TTL_SEC = 120

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")


def _polygon_options_chain(symbol: str = "SPY") -> Optional[dict]:
    """Polygon.io 옵션 체인 — 유료 (Starter $29/mo부터). POLYGON_API_KEY 필요.

    /v3/snapshot/options/SPY 응답을 yahoo와 같은 모양으로 정규화.
    """
    if not POLYGON_API_KEY:
        return None
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY, "limit": 250}, timeout=5)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None
        # Normalize to Yahoo shape so the downstream parser stays simple.
        calls, puts = [], []
        for o in results:
            day = o.get("day", {}) or {}
            details = o.get("details", {}) or {}
            row = {
                "strike": details.get("strike_price"),
                "volume": day.get("volume", 0),
                "openInterest": o.get("open_interest", 0),
            }
            if details.get("contract_type") == "call":
                calls.append(row)
            elif details.get("contract_type") == "put":
                puts.append(row)
        return {"optionChain": {"result": [{"options": [{"calls": calls, "puts": puts}]}]}}
    except Exception:
        return None


def _tradier_options_chain(symbol: str = "SPY") -> Optional[dict]:
    """Tradier 옵션 체인 — 유료 ($10/mo). TRADIER_API_KEY 필요."""
    if not TRADIER_API_KEY:
        return None
    try:
        # Get nearest expiry first
        exp_r = requests.get(
            "https://api.tradier.com/v1/markets/options/expirations",
            params={"symbol": symbol},
            headers={"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"},
            timeout=4,
        )
        if exp_r.status_code != 200:
            return None
        expirations = exp_r.json().get("expirations", {}).get("date", [])
        if not expirations:
            return None
        nearest = expirations[0] if isinstance(expirations, list) else expirations

        r = requests.get(
            "https://api.tradier.com/v1/markets/options/chains",
            params={"symbol": symbol, "expiration": nearest, "greeks": "false"},
            headers={"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        opts = r.json().get("options", {}).get("option", [])
        if not opts:
            return None
        calls = [{"strike": o["strike"], "volume": o.get("volume", 0),
                  "openInterest": o.get("open_interest", 0)}
                 for o in opts if o.get("option_type") == "call"]
        puts  = [{"strike": o["strike"], "volume": o.get("volume", 0),
                  "openInterest": o.get("open_interest", 0)}
                 for o in opts if o.get("option_type") == "put"]
        return {"optionChain": {"result": [{"options": [{"calls": calls, "puts": puts}]}]}}
    except Exception:
        return None


def _yahoo_options_chain(symbol: str = "SPY") -> Optional[dict]:
    """무료 Yahoo 옵션 체인 — 폴백. rate limited, 신뢰도 낮음."""
    try:
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (ESFutures/2.0)"},
            timeout=4,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _fetch_chain(symbol: str = "SPY"):
    # type: (str) -> tuple  # (chain_dict_or_None, source_str)
    """Try premium feeds first, fall back to Yahoo. Returns (chain, source)."""
    chain = _polygon_options_chain(symbol)
    if chain:
        return chain, "POLYGON"
    chain = _tradier_options_chain(symbol)
    if chain:
        return chain, "TRADIER"
    chain = _yahoo_options_chain(symbol)
    if chain:
        return chain, "YAHOO"
    return None, "NO_DATA"


def _parse_flow_metrics(chain_resp: dict) -> dict:
    """체인 응답에서 P/C 비율, OI 비율, 비정상 거래량 계산."""
    if not chain_resp:
        return None
    try:
        result = chain_resp.get("optionChain", {}).get("result", [])
        if not result:
            return None
        opt = result[0].get("options", [])
        if not opt:
            return None
        calls = opt[0].get("calls", [])
        puts  = opt[0].get("puts", [])
        if not calls or not puts:
            return None

        call_vol  = sum((c.get("volume") or 0) for c in calls)
        put_vol   = sum((p.get("volume") or 0) for p in puts)
        call_oi   = sum((c.get("openInterest") or 0) for c in calls)
        put_oi    = sum((p.get("openInterest") or 0) for p in puts)

        if call_vol == 0 and put_vol == 0:
            return None

        # 0으로 나누기 방어
        pc_vol_ratio = put_vol / call_vol if call_vol > 0 else (10.0 if put_vol > 0 else 1.0)
        pc_oi_ratio  = put_oi  / call_oi  if call_oi  > 0 else (10.0 if put_oi  > 0 else 1.0)

        # 비정상 거래량: 단일 옵션 volume > 3x OI = unusual activity
        unusual = []
        for c in calls + puts:
            v = c.get("volume") or 0
            oi = c.get("openInterest") or 0
            if oi > 0 and v >= 3 * oi and v >= 500:
                unusual.append({
                    "type": "CALL" if c in calls else "PUT",
                    "strike": c.get("strike"),
                    "volume": v,
                    "oi": oi,
                })
        unusual.sort(key=lambda x: -x["volume"])

        return {
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "pc_vol_ratio": round(pc_vol_ratio, 3),
            "pc_oi_ratio": round(pc_oi_ratio, 3),
            "unusual_count": len(unusual),
            "unusual_top": unusual[:3],
        }
    except Exception:
        return None


def _score_options_flow(metrics: dict) -> tuple:
    """P/C 비율 기반 점수 (0~30) + 방향성."""
    if not metrics:
        return 0, "NEUTRAL", "Options data unavailable"

    pc_vol = metrics["pc_vol_ratio"]
    pc_oi  = metrics["pc_oi_ratio"]

    # P/C ratio 기준 (역방향 신호: P/C 낮음 = 콜 매수 우세 = LONG)
    # Volume P/C: 0.7 이하 = 강한 콜 우세, 1.2 이상 = 강한 풋 우세
    score = 0
    direction = "NEUTRAL"
    parts = []

    if pc_vol <= 0.6:
        score += 15
        direction = "LONG"
        parts.append(f"P/C vol {pc_vol:.2f} — strong call demand")
    elif pc_vol <= 0.8:
        score += 8
        direction = "LONG"
        parts.append(f"P/C vol {pc_vol:.2f} — mild call lean")
    elif pc_vol >= 1.4:
        score += 15
        direction = "SHORT"
        parts.append(f"P/C vol {pc_vol:.2f} — strong put demand")
    elif pc_vol >= 1.2:
        score += 8
        direction = "SHORT"
        parts.append(f"P/C vol {pc_vol:.2f} — mild put lean")
    else:
        parts.append(f"P/C vol {pc_vol:.2f} — neutral")

    # OI ratio confirmation (smaller weight)
    if direction == "LONG" and pc_oi <= 0.9:
        score += 5
        parts.append(f"OI {pc_oi:.2f} confirms")
    elif direction == "SHORT" and pc_oi >= 1.1:
        score += 5
        parts.append(f"OI {pc_oi:.2f} confirms")
    elif (direction == "LONG" and pc_oi >= 1.2) or (direction == "SHORT" and pc_oi <= 0.8):
        score -= 5  # OI divergence weakens signal
        parts.append(f"OI {pc_oi:.2f} divergent")

    # Unusual activity boost
    if metrics["unusual_count"] >= 3:
        score += 10
        parts.append(f"{metrics['unusual_count']} unusual strikes")
    elif metrics["unusual_count"] >= 1:
        score += 5
        parts.append(f"{metrics['unusual_count']} unusual strikes")

    score = max(0, min(30, score))
    detail = " · ".join(parts) if parts else "No clear flow"
    return score, direction, detail


def calculate_options_flow_score(symbol: str = "SPY") -> dict:
    """
    Layer 3 결과.

    Returns
    -------
    dict with keys:
        score          : int (0~30)
        max            : int (30)
        direction      : str (LONG / SHORT / NEUTRAL)
        status         : str (LIVE / CACHED / NO_DATA)
        pc_vol_ratio   : float or None
        pc_oi_ratio    : float or None
        unusual_count  : int
        unusual_top    : list
        detail         : str
    """
    now = time.time()
    if _CACHE["data"] and now - _CACHE["at"] < CACHE_TTL_SEC:
        cached = dict(_CACHE["data"])
        cached["status"] = "CACHED"
        return cached

    chain, source = _fetch_chain(symbol)
    metrics = _parse_flow_metrics(chain)

    if not metrics:
        result = {
            "score": 0,
            "max": 30,
            "direction": "NEUTRAL",
            "status": "NO_DATA",
            "source": source,
            "pc_vol_ratio": None,
            "pc_oi_ratio": None,
            "unusual_count": 0,
            "unusual_top": [],
            "detail": "Options chain unavailable (all sources failed). Set POLYGON_API_KEY or TRADIER_API_KEY.",
        }
        return result

    score, direction, detail = _score_options_flow(metrics)
    result = {
        "score": score,
        "max": 30,
        "direction": direction,
        "status": "LIVE",
        "source": source,
        "pc_vol_ratio": metrics["pc_vol_ratio"],
        "pc_oi_ratio": metrics["pc_oi_ratio"],
        "unusual_count": metrics["unusual_count"],
        "unusual_top": metrics["unusual_top"],
        "detail": detail,
    }
    _CACHE["at"] = now
    _CACHE["data"] = result
    return result
