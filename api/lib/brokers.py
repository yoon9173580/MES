"""
브로커 추상화 계층.

지원 어댑터:
  • AlpacaPaperAdapter   — SPY 주식 paper 거래 (현재 작동)
  • TradovateAdapter     — MES/ES 실 선물 (자격증명 필요, 스텁)
  • DryRunAdapter        — 시뮬레이션 (실제 주문 없음, 로컬 로그)

환경변수:
  BROKER=alpaca | tradovate | dryrun
  ALPACA_API_KEY / ALPACA_SECRET_KEY
  TRADOVATE_USERNAME / TRADOVATE_PASSWORD / TRADOVATE_CID / TRADOVATE_SEC
  TRADOVATE_DEMO=true|false  (sim 환경 vs live)

추후 확장:
  • InteractiveBrokers (IBKR) — TWS gateway 필요
  • NinjaTrader/AMP          — REST API 부재, 별도 통합
"""
import os
import json
import time
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple


class BrokerAdapter(ABC):
    """모든 브로커 어댑터의 인터페이스."""

    name: str = "base"
    supports_futures: bool = False
    supports_equity: bool = False

    @abstractmethod
    def get_account_equity(self) -> Optional[float]:
        """현재 계좌 잔고 (USD)."""

    @abstractmethod
    def get_open_positions(self) -> list:
        """오픈 포지션 목록 (브로커 raw)."""

    @abstractmethod
    def place_bracket_order(self, symbol: str, qty: int, side: str,
                             take_profit: float, stop_loss: float) -> Optional[dict]:
        """SL+TP 동반 진입 주문."""

    @abstractmethod
    def is_ready(self) -> Tuple[bool, str]:
        """자격증명/연결 상태 확인. (ok, reason) 반환."""


# ── 1. Alpaca Paper (현재 trading_bot 호환) ─────────────────────────
class AlpacaPaperAdapter(BrokerAdapter):
    name = "alpaca_paper"
    supports_equity = True
    supports_futures = False  # Alpaca는 선물 미지원

    BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(self):
        self.key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID", "")
        self.sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY", "")
        self.headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.sec,
        }

    def is_ready(self):
        if not self.key or not self.sec:
            return False, "ALPACA_API_KEY / ALPACA_SECRET_KEY 미설정"
        return True, "OK"

    def get_account_equity(self):
        try:
            r = requests.get(f"{self.BASE_URL}/v2/account", headers=self.headers, timeout=8)
            r.raise_for_status()
            return float(r.json().get("equity", 0))
        except Exception:
            return None

    def get_open_positions(self):
        try:
            r = requests.get(f"{self.BASE_URL}/v2/positions", headers=self.headers, timeout=8)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def place_bracket_order(self, symbol, qty, side, take_profit, stop_loss):
        order = {
            "symbol": symbol, "qty": qty, "side": side,
            "type": "market", "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": round(take_profit, 2)},
            "stop_loss":   {"stop_price":  round(stop_loss, 2)},
        }
        try:
            r = requests.post(f"{self.BASE_URL}/v2/orders", json=order,
                              headers=self.headers, timeout=8)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        return None


# ── 2. Tradovate (MES/ES 선물 — 자격증명 필요) ──────────────────────
class TradovateAdapter(BrokerAdapter):
    """Tradovate REST API 어댑터.

    OAuth 흐름:
      1) POST /auth/accesstokenrequest → access token (24h)
      2) Subsequent calls use Bearer token
      3) Re-auth on 401

    실 거래 활성화 전 필요한 env vars:
      TRADOVATE_USERNAME, TRADOVATE_PASSWORD,
      TRADOVATE_CID (client id from Tradovate API portal),
      TRADOVATE_SEC (client secret),
      TRADOVATE_DEMO ('true' = demo, 'false' = live)
    """
    name = "tradovate"
    supports_futures = True
    supports_equity = False

    @property
    def base_url(self):
        return ("https://demo.tradovateapi.com/v1"
                if os.getenv("TRADOVATE_DEMO", "true").lower() == "true"
                else "https://live.tradovateapi.com/v1")

    def __init__(self):
        self.username = os.getenv("TRADOVATE_USERNAME", "")
        self.password = os.getenv("TRADOVATE_PASSWORD", "")
        self.cid      = os.getenv("TRADOVATE_CID", "")
        self.sec      = os.getenv("TRADOVATE_SEC", "")
        self._token   = None
        self._token_exp = 0
        self._account_id = None

    def is_ready(self):
        missing = [k for k, v in [
            ("TRADOVATE_USERNAME", self.username),
            ("TRADOVATE_PASSWORD", self.password),
            ("TRADOVATE_CID", self.cid),
            ("TRADOVATE_SEC", self.sec),
        ] if not v]
        if missing:
            return False, f"Tradovate 자격증명 누락: {', '.join(missing)}"
        ok, _ = self._ensure_token()
        return ok, "OK" if ok else "Tradovate 인증 실패"

    def _ensure_token(self):
        if self._token and time.time() < self._token_exp - 300:
            return True, self._token
        body = {
            "name": self.username,
            "password": self.password,
            "appId": "MILLI",
            "appVersion": "1.0",
            "cid": self.cid,
            "sec": self.sec,
        }
        try:
            r = requests.post(f"{self.base_url}/auth/accesstokenrequest",
                              json=body, timeout=8)
            if r.status_code != 200:
                return False, None
            d = r.json()
            self._token = d.get("accessToken")
            # expires in 80 minutes by default
            self._token_exp = time.time() + 80 * 60
            return True, self._token
        except Exception:
            return False, None

    def _headers(self):
        ok, tok = self._ensure_token()
        if not ok:
            return None
        return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

    def _account(self):
        if self._account_id:
            return self._account_id
        h = self._headers()
        if not h:
            return None
        try:
            r = requests.get(f"{self.base_url}/account/list", headers=h, timeout=8)
            if r.status_code == 200:
                accs = r.json()
                if accs:
                    self._account_id = accs[0].get("id")
            return self._account_id
        except Exception:
            return None

    def get_account_equity(self):
        h = self._headers()
        acc = self._account()
        if not h or not acc:
            return None
        try:
            r = requests.get(f"{self.base_url}/cashBalance/getcashbalancesnapshot",
                             params={"accountId": acc}, headers=h, timeout=8)
            if r.status_code == 200:
                d = r.json()
                return float(d.get("totalCashValue", 0))
        except Exception:
            pass
        return None

    def get_open_positions(self):
        h = self._headers()
        acc = self._account()
        if not h or not acc:
            return []
        try:
            r = requests.get(f"{self.base_url}/position/list", headers=h, timeout=8)
            if r.status_code == 200:
                positions = [p for p in r.json() if p.get("accountId") == acc and p.get("netPos", 0) != 0]
                return positions
        except Exception:
            pass
        return []

    def place_bracket_order(self, symbol, qty, side, take_profit, stop_loss):
        """OSO (Order Sends Order) bracket: market entry + OCO TP/SL."""
        h = self._headers()
        acc = self._account()
        if not h or not acc:
            return None
        # Tradovate OSO body
        body = {
            "accountSpec": self.username,
            "accountId": acc,
            "action": side.capitalize(),  # "Buy" / "Sell"
            "symbol": symbol,              # e.g. "MESM6"
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
            "bracket1": {
                "action": "Sell" if side == "buy" else "Buy",
                "orderType": "Limit",
                "price": round(take_profit, 2),
            },
            "bracket2": {
                "action": "Sell" if side == "buy" else "Buy",
                "orderType": "Stop",
                "stopPrice": round(stop_loss, 2),
            },
        }
        try:
            r = requests.post(f"{self.base_url}/order/placeoso", json=body, headers=h, timeout=10)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:
            pass
        return None


# ── 3. Dry-run (시뮬레이션, 실 주문 없음) ────────────────────────────
class DryRunAdapter(BrokerAdapter):
    """주문을 로컬 JSONL에만 기록. 외부 호출 0회."""
    name = "dryrun"
    supports_equity = True
    supports_futures = True

    LOG_PATH = os.path.join("/tmp" if os.getenv("VERCEL") else "data_cache",
                            "dryrun_orders.jsonl")

    def __init__(self):
        self._equity = 500000.0
        self._positions = []

    def is_ready(self):
        return True, "Dry-run mode — no broker calls"

    def get_account_equity(self):
        return self._equity

    def get_open_positions(self):
        return list(self._positions)

    def place_bracket_order(self, symbol, qty, side, take_profit, stop_loss):
        order = {
            "ts": datetime.utcnow().isoformat(),
            "symbol": symbol, "qty": qty, "side": side,
            "tp": take_profit, "sl": stop_loss,
            "mode": "DRYRUN",
        }
        try:
            os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
            with open(self.LOG_PATH, "a") as f:
                f.write(json.dumps(order) + "\n")
        except Exception:
            pass
        self._positions.append({"symbol": symbol, "qty": qty, "side": side})
        return {"id": f"dryrun-{int(time.time())}", **order}


# ── Factory ────────────────────────────────────────────────────────
_ADAPTERS = {
    "alpaca":    AlpacaPaperAdapter,
    "alpaca_paper": AlpacaPaperAdapter,
    "tradovate": TradovateAdapter,
    "dryrun":    DryRunAdapter,
}


def get_broker(name: str = None) -> BrokerAdapter:
    """환경변수 BROKER 또는 명시 인자로 어댑터 선택."""
    if not name:
        name = os.getenv("BROKER", "alpaca").lower()
    cls = _ADAPTERS.get(name, DryRunAdapter)
    return cls()
