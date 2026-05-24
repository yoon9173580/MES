"""
LAYER 4 — Market Microstructure & Correlation
SPY 단독이 아닌 시장 전체 맥락에서 방향성 확인.
"""


def _score_qqq_alignment(spy_pct: float, qqq_pct: float) -> tuple:
    """QQQ and SPY direction alignment → higher confidence."""
    if spy_pct is None or qqq_pct is None:
        return 0, "QQQ data unavailable"
    if (spy_pct >= 0 and qqq_pct >= 0) or (spy_pct < 0 and qqq_pct < 0):
        return 10, f"QQQ aligned ({qqq_pct:+.2f}%) — confirmed"
    else:
        return -5, f"QQQ divergence ({qqq_pct:+.2f}%) — caution"


def _score_iwm_risk(iwm_pct: float) -> tuple:
    """IWM (small caps) as risk-on/off indicator."""
    if iwm_pct is None:
        return 0, "IWM data unavailable"
    if iwm_pct > 0.3:
        return 5, f"IWM {iwm_pct:+.2f}% — Risk-On"
    elif iwm_pct < -0.3:
        return -3, f"IWM {iwm_pct:+.2f}% — Risk-Off"
    else:
        return 0, f"IWM {iwm_pct:+.2f}% — Neutral"


def _score_sector_sync(spy_pct: float, qqq_pct: float, iwm_pct: float) -> tuple:
    """Full sector synchronization check."""
    if any(v is None for v in [spy_pct, qqq_pct, iwm_pct]):
        return 0, False, "Sector data incomplete"

    all_up = spy_pct >= 0 and qqq_pct >= 0 and iwm_pct >= 0
    all_down = spy_pct < 0 and qqq_pct < 0 and iwm_pct < 0
    synced = all_up or all_down

    if synced:
        return 5, True, "All sectors aligned"
    else:
        return 0, False, "Sector divergence"


def _score_dia_alignment(spy_pct: float, dia_pct: float) -> tuple:
    """DIA (Dow) alignment provides additional confirmation."""
    if spy_pct is None or dia_pct is None:
        return 0, "DIA data unavailable"
    if (spy_pct >= 0 and dia_pct >= 0) or (spy_pct < 0 and dia_pct < 0):
        return 3, f"DIA aligned ({dia_pct:+.2f}%)"
    else:
        return 0, f"DIA divergence ({dia_pct:+.2f}%)"


def calculate_correlation_score(pcts: dict) -> dict:
    """
    Calculate Layer 4 correlation & market microstructure score.

    Parameters
    ----------
    pcts : dict
        Percentage changes for SPY, QQQ, IWM, DIA, etc.

    Returns
    -------
    dict with keys:
        score       : int (max ~20)
        max         : int (20)
        sector_sync : bool
        details     : dict — component breakdown
    """
    spy_pct = pcts.get("SPY", 0)
    qqq_pct = pcts.get("QQQ")
    iwm_pct = pcts.get("IWM")
    dia_pct = pcts.get("DIA")
    details = {}

    # QQQ alignment
    qqq_score, qqq_detail = _score_qqq_alignment(spy_pct, qqq_pct)
    details["qqq_alignment"] = {"score": qqq_score, "detail": qqq_detail}

    # IWM risk-on/off
    iwm_score, iwm_detail = _score_iwm_risk(iwm_pct)
    details["iwm_risk"] = {"score": iwm_score, "detail": iwm_detail}

    # Full sector sync
    sync_score, synced, sync_detail = _score_sector_sync(spy_pct, qqq_pct, iwm_pct)
    details["sector_sync"] = {"score": sync_score, "detail": sync_detail}

    # DIA alignment
    dia_score, dia_detail = _score_dia_alignment(spy_pct, dia_pct)
    details["dia_alignment"] = {"score": dia_score, "detail": dia_detail}

    total = min(20, max(0, qqq_score + iwm_score + sync_score + dia_score))

    return {
        "score": total,
        "max": 20,
        "sector_sync": synced,
        "details": details,
    }
