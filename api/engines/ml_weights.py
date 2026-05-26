import os
import json
import time

# Vercel's project filesystem is read-only — only /tmp is writable at runtime.
_CACHE_DIR = "/tmp" if os.getenv("VERCEL") else "data_cache"
CACHE_FILE = os.path.join(_CACHE_DIR, "ml_weights.json")

class AdaptiveWeightEngine:
    def __init__(self):
        self.weights = {
            "technical": 1.0,
            "regime": 1.0,
            "flow": 1.0,
            "correlation": 1.0,
            "last_updated": 0,
            # ── Sample-tracking (so weights aren't trusted before warm-up) ──
            "sample_count": 0,
            "wins": 0,
            "losses": 0,
            # ── Per-layer update counts (which layer is getting most signal) ──
            "updates_per_layer": {
                "technical": 0, "regime": 0, "flow": 0, "correlation": 0,
            },
        }
        self.load_weights()

    def load_weights(self):
        """Load weights from local cache if available."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    # Validate keys
                    for k in self.weights.keys():
                        if k in data:
                            self.weights[k] = data[k]
            except Exception:
                pass

    def save_weights(self):
        """Save weights to local cache."""
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        self.weights["last_updated"] = time.time()
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.weights, f, indent=2)
        except Exception:
            pass

    def get_multipliers(self):
        """Return the current multipliers for scoring layers."""
        return {
            "technical": self.weights["technical"],
            "regime": self.weights["regime"],
            "flow": self.weights["flow"],
            "correlation": self.weights["correlation"]
        }

    def update_weights(self, trade_result: dict):
        """
        Adjust weights based on trade outcome (RL feedback).

        trade_result keys:
          - "pnl": float
          - "dominant_layer": str  (regime/correlation/technical/flow)
          - "magnitude_r": optional float — PnL in R-multiples (signed).

        Magnitude-aware update: larger wins/losses move the weight more,
        so a +2R win shifts the multiplier twice as much as a +0.5R win.
        Decay (λ=0.995) gently pulls every weight back toward 1.0 so
        no single layer can stay over-/under-weighted indefinitely.
        """
        pnl = trade_result.get("pnl", 0)
        dominant = trade_result.get("dominant_layer")
        if not dominant or dominant not in self.weights:
            return

        # Learning rate + decay (env-tunable for live A/B testing)
        lr = float(os.getenv("ML_LR", "0.05"))
        decay = float(os.getenv("ML_DECAY", "0.995"))
        max_w = float(os.getenv("ML_MAX_W", "1.5"))
        min_w = float(os.getenv("ML_MIN_W", "0.5"))

        # Magnitude scaling: 1R = 1× lr, capped at 3× to prevent
        # one outlier from dominating the weight surface.
        magnitude = trade_result.get("magnitude_r")
        if magnitude is None:
            magnitude = 1.0
        scale = max(0.25, min(3.0, abs(float(magnitude))))

        if pnl > 0:
            self.weights[dominant] = min(max_w, self.weights[dominant] + lr * scale)
            self.weights["wins"] = self.weights.get("wins", 0) + 1
        elif pnl < 0:
            self.weights[dominant] = max(min_w, self.weights[dominant] - lr * scale)
            self.weights["losses"] = self.weights.get("losses", 0) + 1

        # Track sample count + per-layer update count for transparency
        self.weights["sample_count"] = self.weights.get("sample_count", 0) + 1
        per_layer = self.weights.setdefault("updates_per_layer", {})
        per_layer[dominant] = per_layer.get(dominant, 0) + 1

        # Decay all weights toward 1.0 so stale advantages dissipate.
        for k in ("technical", "regime", "flow", "correlation"):
            cur = self.weights.get(k, 1.0)
            self.weights[k] = 1.0 + (cur - 1.0) * decay

        self.save_weights()

    def get_stats(self):
        """Public snapshot of weights + sample-size confidence.

        Used by the API to surface ML state to the frontend so users can
        see *whether* the weights are trustworthy (warm-up phase needs
        ~30+ trades before adjustments mean much).
        """
        n = self.weights.get("sample_count", 0)
        wins = self.weights.get("wins", 0)
        losses = self.weights.get("losses", 0)
        if n >= 30:
            confidence = "TRUSTED"
        elif n >= 10:
            confidence = "WARMING_UP"
        else:
            confidence = "COLD_START"
        wr = round(wins / n * 100, 1) if n else None
        return {
            "weights": {k: round(self.weights.get(k, 1.0), 3)
                        for k in ("technical", "regime", "flow", "correlation")},
            "sample_count": n,
            "wins": wins,
            "losses": losses,
            "observed_wr_pct": wr,
            "updates_per_layer": dict(self.weights.get("updates_per_layer") or {}),
            "confidence": confidence,
            "last_updated": self.weights.get("last_updated", 0),
        }

# Singleton instance
_ml_engine = AdaptiveWeightEngine()

def get_ml_multipliers():
    return _ml_engine.get_multipliers()

def get_ml_stats():
    """Snapshot of weights + sample-size confidence for the API response."""
    return _ml_engine.get_stats()

def feedback_trade_result(trade_result):
    _ml_engine.update_weights(trade_result)
