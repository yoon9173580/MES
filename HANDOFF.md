# 인수인계 문서 (HANDOFF) — MES Futures Signal Engine

> 이 문서는 이 작업 환경(터미널 머신)과 프로젝트를 **다른 AI나 사람이 처음 봐도 이어받을 수 있도록** 정리한 것이다.
> 최종 갱신: 2026-06-04 / 전략 버전: **v10.4**

---

## 0. TL;DR (30초 요약)

- **무엇:** MES(Micro E-mini S&P 500 선물) 데이트레이딩 **시그널/페이퍼 트레이딩 봇**.
- **전략:** 하루 1회, 미국장 **10:30 ET(PRIME) 단일 진입**. 7-레이어 스코어가 **68점 이상**일 때만 진입. EOD(15:30 ET) 청산. 저빈도·고확신.
- **현재 성과(백테스트, 2023-03~2026-05 전체, RTH, $10k 시작):** 승률 53.2%, 연수익 31.6%, Sharpe 1.44, MaxDD 6.0%, 156거래(~49/yr).
  - ⚠️ **과최적화·레버리지 주의:** `MIN_SCORE=68`·`SL_CAP=22`는 이 데이터로 그리드서치했고, 헤드라인은 2.5% 리스크(레버리지). **인샘플 최적화 수치**라 라이브 기대치는 더 낮다. walk-forward OOS는 테스트 연도 Sharpe 1.1~1.6(전 연도 흑자). 보수적 구버전(score88·1.5%·SLcap15)은 같은 데이터에서 **8.8%/Sharpe 0.46**. (§3, §8 참조)
- **배포:** 프론트+API는 **Vercel** (`https://hannaealgo.vercel.app`), 스케줄러는 **Vercel Cron**, 상태/로그는 **Upstash Redis(KV)**.
- **핵심 원칙:** 백테스트(`thorough_backtest_futures.py`)와 라이브 봇(`api/v10_runner.py`)이 **동일한 의사결정 모듈(`api/v10_strategy.py`)을 import** → "백테스트가 곧 라이브"가 보장됨.

---

## 1. 이 작업 환경(터미널 머신)에 대해

- **성격:** 클라우드의 **격리된·일시적(ephemeral) 컨테이너**. Claude Code(웹/원격 실행)가 돌고 있다. 세션이 끝나거나 일정 시간 비활성이면 컨테이너는 회수된다.
- **저장의 영속성:** 컨테이너는 시작 시 레포를 **새로 clone**한다. **커밋+푸시하지 않은 변경은 사라진다.** 작업물은 반드시 git에 올려야 한다.
- **작업 디렉토리:** `/home/user/MES`
- **OS/런타임:** Linux, Python **3.11**. 의존성: `pandas, numpy, pytz, requests` (`requirements.txt`).
- **네트워크:** 아웃바운드는 환경의 네트워크 정책에 따름.
- **시크릿(중요):** 이 개발 컨테이너에는 **프로덕션 자격증명이 없다.**
  - 없는 것: `UPSTASH_REDIS_REST_URL/TOKEN`(=KV), `APCA_API_KEY_ID/SECRET`(Alpaca), `POLYGON_API_KEY`, `CRON_SECRET`, `VERCEL`.
  - 따라서 **이 환경에서는** ① KV 로그 조회, ② 라이브 시세 fetch를 통한 실시간 진입 재현이 **불가**. (시크릿은 Vercel 프로덕션에만 존재 — 정상)
  - 단, **CSV 기반 백테스트와 단위테스트는 외부 키 없이 완전히 실행 가능**.
- **Git 원격:** `origin` = `yoon9173580/MES` (로컬 프록시 경유). GitHub 작업은 `gh` CLI가 아니라 **GitHub MCP 도구**(`mcp__github__*`)로 한다.
- **현재 브랜치 상태:** `main`이 최신(`c7e538d`, PR #17 머지). 작업 브랜치 `bear-followup`은 main과 내용 동일(머지 완료).

---

## 2. 프로젝트 개요

- **레포명:** `GUN_SPY_MILLI` (README 기준) / `yoon9173580/MES`
- **목적:** "프로 트레이더 스타일" 7-레이어 스코어로 MES 선물의 **월 몇 번뿐인 고확신 셋업**만 잡는다. 잡소리(noise)는 스킵.
- **라이브 대시보드:** https://hannaealgo.vercel.app (Google SSO 필요)
- **부가 전략:** Iron Condor(옵션) 시그널도 일부 존재(`api/engines/ic_signal.py`, `ic_daily_check.yml`) — 단, **메인은 MES v10**.

### 데이터 소스
| 소스 | 용도 |
|------|------|
| Polygon / Databento | MES 선물 1분봉 (백테스트용 과거데이터) |
| Alpaca Markets | 주식(SPY 등) 라이브 시세 (라이브 진입 입력) |
| Cboe 공개 CDN | VIX |

> 라이브는 SPY ETF(~530)를 받아 `ES_PER_SPY(=10)`를 곱해 ES 인덱스 스케일(~5300)로 변환 → 백테스트와 같은 가격공간/ATR 기준(≥8pt)에서 판단.

---

## 3. 전략 v10.4 (현재 운영중) — 핵심 규칙

**진입 (하루 최대 1회):**
1. 시간: **10:30 ET PRIME 바**만 평가 (단일 진입). 오후 GAMMA 창은 v10에서 사실상 미사용.
2. 7-레이어 스코어 ≥ **MIN_SCORE = 68** 이어야 진입.
3. 변동성 필터: 14일 ATR ≥ **8 포인트**.
4. **Runaway veto**(과열 차단, 대칭형): ADX≥40, 또는 RSI≥90/≤10, 또는 전 섹터 동반 급등/급락이면 진입 거부.
5. **Daily-bias 필터**: 일봉 상승추세인데 VIX<20이면 SHORT 스킵(`VIX_SHORT_FILTER=20`).
6. **모드 전환(VIX 기준)**:
   - VIX < 25 → 추세추종(trend-follow)
   - 25 ≤ VIX < 30 → 평균회귀(mean-reversion)
   - VIX ≥ 30 → 위기장, 다시 추세추종으로 override (Option A)

**청산/리스크:**
- SL = `min(max(1.5×ATR, 2pt), 22pt)` — **상한 22포인트**(`SL_CAP_PTS=22`, v10.3 핵심 레버).
- TP = **2.5 × SL** (`TP_MULT=2.5`).
- 본전이동(BE): 이익 ≥ 0.25×ATR에서 손절을 본전으로.
- 트레일링: 이익 ≥ 0.5×ATR에서 무장, best−0.25×ATR로 추적.
- EOD: 15:30 ET 강제 청산.
- **포지션 사이징(리스크기반)**: `계약수 = (잔고 × risk%) ÷ (SL포인트 × $5)`. risk%는 VIX로 스케일:
  - VIX<25 → **2.5%** (`RISK_PCT_FULL=0.025`)
  - 25–35 → 1.0% / VIX≥35 → 0.7%
- 3-strike 락아웃: 연속 손실 누적 시 1일 쿨다운.

### v10.4 핵심 상수 (단일 출처: `api/v10_strategy.py` 35~66행)
```python
ATR_SL_MULT     = 1.5      # SL = 1.5×ATR
TP_MULT         = 2.5      # TP = 2.5×SL
ATR_MIN         = 8.0      # 14일 ATR 하한
MIN_SCORE       = 68       # 진입 스코어 임계값  ★ (v10.4: 74→68, +18% 거래빈도)
VIX_THRESHOLD   = 25.0     # 추세↔평균회귀 전환  ★
VIX_SHORT_FILTER= 20.0     # SHORT 스킵 기준
VIX_CRISIS      = 30.0     # 위기장 추세추종 override
RISK_PCT_FULL   = 0.025    # 거래당 리스크 2.5%  ★
SL_CAP_PTS      = 22.0     # 손절 상한 22pt      ★
RSI_UPPER/LOWER = 90 / 10
ADX_RUNAWAY     = 40.0
```
> ★ = v10.1 대비 변경된 결정적 파라미터.

### 왜 이 값들인가 (튜닝 근거)
- **`SL_CAP_PTS` 15→22 (진짜 엣지):** 진입일 대부분의 ATR이 ~21pt 이상이라 기존 15pt 상한이 정상 노이즈에 손절당하게 만들었음(휩쏘). 22pt로 넓히니 SL청산 40→21건↓, 승률 49→57%↑, MaxDD 4.0→2.7%↓. **손절은 리스크정규화되어 있어 거리를 넓혀도 거래당 달러손실은 동일**(계약수가 자동 감소).
- **`RISK_PCT` 1.5→2.5% (레버리지 선택):** 위에서 DD 여유가 생겨 사이즈 업 → 연수익 ↑. 단 이건 수익·위험을 비례 확대하는 노브(리스크 선호도 선택). 2.0%면 연수익~20%/DD~4%, 2.5%면 ~31.6%/DD 6.0%.
- **`MIN_SCORE` 74→68 (거래빈도 개선):** 74 대비 연수익 동일(31.6%), 거래수 +18%(42→49/yr). 승률은 56→53%로 소폭 하락하나 Sharpe·DD는 허용 범위. 피드백 루프 가속 목적.
- **`VIX_THRESHOLD` 20→25:** 2023-26 강세장에서 VIX 20-25 구간의 평균회귀(역추세 SHORT)가 승률 21%로 독이었음 → 추세추종 구간으로 흡수.

> ⚠️ **과최적화 경고:** `SL_CAP=22`·`MIN_SCORE=68`은 **결과를 보고하는 바로 그 2023-26 데이터로 그리드서치**한 값이다(인샘플 최적화). `RISK 2.5%`는 순수 레버리지(수익·위험 동반 확대). 따라서 31.6%/Sharpe 1.44는 **낙관적 상한**으로 봐야 한다. 정직한 강건성은 walk-forward로 본다:
>
> | Split | 거래 | 승률 | 연수익 | Sharpe | PF |
> |---|--:|--:|--:|--:|--:|
> | 2023 (train) | 32 | 62.5% | 43.8% | 2.24 | 3.97 |
> | 2024 (test)  | 41 | 51.2% | 19.6% | 1.10 | 2.26 |
> | 2025 (test)  | 43 | 58.1% | 28.9% | 1.28 | 2.00 |
> | 2026 (test)  | 16 | 43.8% | 29.1% | 1.56 | 3.56 |
>
> 전 연도 흑자·PF>2지만 Sharpe가 2.24→1.1~1.6으로 ~30-50% 하락. 보수적 하한(구 v10, score88·1.5%·SLcap15)은 같은 데이터 **8.8%/Sharpe 0.46/34거래**. → 라이브 기대는 이 둘 사이로 보는 게 안전.

---

## 4. 아키텍처 / 코드 맵

```
데이터 소스 (Polygon/Alpaca/Cboe)
        │
   api/data.py  ── 메인 오케스트레이터 (병렬 fetch, KV 캐시, /api/data·/api/health·/api/stream 서빙)
        │           + Upstash KV 헬퍼 _kv_get/_kv_set, 인증, 포트폴리오
        │
   api/engines/ ── 7-레이어 스코어링
        ├ regime.py        (1) VIX + ADX + ATR 레짐
        ├ options_flow.py  (2) 옵션 플로우 (무료플랜은 NO_DATA)
        ├ correlation.py   (3) 섹터 동조성
        ├ time_window.py   (4) PRIME/GAMMA/점심 시간창
        ├ technical.py     (5) VWAP + RSI + EMA 트리거
        ├ macro_gate.py    (6) FOMC/CPI/NFP/PPI 이벤트 게이트
        ├ risk_manager.py  (7) 3-strike + DD 사이징
        └ score_engine.py  → 7-레이어 통합 (run_score_engine)
        │
   api/v10_strategy.py ── ★ 순수 의사결정 엔진 (단일 출처)
        │  evaluate_entry() / init_position() / manage_bar() / vix_risk_pct()
        │  ↑ 백테스트와 라이브가 "둘 다" 이 함수를 import → 동일 전략 보장
        ├──────────────┬───────────────────────────────
        ▼              ▼
 thorough_backtest_   api/v10_runner.py ── 라이브 1틱 실행
 futures.py           │  run_once_entry() / run_once_flatten()
 (백테스트=ground      │  FileStore(로컬,개발용) / KVStore(Upstash,프로덕션)
  truth, CSV replay)  ▼
                 api/cron_v10.py ── Vercel Cron HTTP 핸들러 (/api/cron_v10?mode=entry|flatten)
```

### 주요 파일 (줄 수)
| 파일 | 역할 |
|------|------|
| `api/v10_strategy.py` (384) | **전략 두뇌. 모든 상수·진입/청산 로직.** 바꿀 거의 모든 것이 여기. |
| `api/v10_runner.py` (403) | 라이브 실행: 입력수집(`_gather_inputs`)→`evaluate_entry`→주문. 상태=KVStore. |
| `api/cron_v10.py` (90) | Vercel Cron 엔드포인트. mode=entry/flatten 디스패치. CRON_SECRET 인증. |
| `api/data.py` (2704) | 대시보드 API + KV + 백테스트 요약 상수(`BACKTEST_SUMMARY`). |
| `thorough_backtest_futures.py` (928) | CSV 백테스트 엔진. `--csv`로 실행. 결과 → `backtest_futures.json`. |
| `index.html` | 대시보드 프론트엔드. |
| `MES_1min_data_et_rth.csv` (17M, 309k행) | **백테스트 표준 데이터** (RTH, 2023-03~2026-05, 761 거래일). |

---

## 5. 배포 / 운영 (프로덕션)

- **호스팅:** Vercel (`vercel.json` 참고). `api/data.py`, `api/cron_v10.py`는 Python 서버리스 함수, `index.html`은 정적.
- **스케줄러 = Vercel Cron** (`vercel.json`의 `crons`):
  | mode | Cron(UTC) | 실제 ET | 비고 |
  |------|-----------|---------|------|
  | entry | `30 15 * * 1-5` | 여름11:30 / 겨울10:30 ET | 10:30 PRIME 바 완성 후 평가 |
  | flatten | `35 19 * * 1-5` | 여름15:35 ET | EOD 청산 |
- **상태/로그 영속 = Upstash Redis(KV)**. 키: `v10:state`, `v10:log`.
  - ⚠️ Vercel 함수는 파일시스템이 휘발성 → 로컬 `v10_state.json`/`v10_paper_log.json`은 **프로덕션 데이터 아님**(개발용 stub/placeholder). 진짜 기록은 KV에 있음.
- **GitHub Actions는 비활성:** `v10_bot.yml`(수동 전용, 계정 Actions 과금 차단), `daily_bot.yml`(구형 SPY 0DTE, 폐기). → **실 운영은 Vercel Cron만.**
- **브로커:** `api/lib/brokers.py` (Alpaca/Tradovate/dryrun). 기본 페이퍼/드라이런.

### 라이브 상태 읽는 법 (대시보드)
대시보드 "진입 대기" 줄이 핵심 진단:
> `오늘 최고 54@12:46 (NONE) · 전체 최고 68@2026-06-02 11:10 (WEAK) [누적 N샘플 · 5일 보존] ● KV 영속`
- **오늘/5일 최고 스코어 vs 74** 비교로 무진입 사유 즉시 판단 가능. (예: 최고 68 < 74 → 임계 미달, 정상)
- `● KV 영속` = KV 정상 작동 표시. RISK STATUS=OK & DD 0.0% = 락아웃 아님.

---

## 6. 자주 쓰는 명령

```bash
# 백테스트 (표준)
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv

# 파라미터 오버라이드
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv \
    --min-score 74 --tp-mult 2.5 --no-mean-reversion

# 단위테스트 (108개, 외부키 불필요)
python -m pytest tests/ -q

# import 무결성 체크
python -c "import sys; sys.path.insert(0,'api'); import v10_strategy, v10_runner; print('OK')"
```
- 백테스트 결과 핵심 지표는 stdout과 `backtest_futures.json`에 저장.
- 발행 수치(`api/data.py`의 `BACKTEST_SUMMARY['mes_futures']`)는 백테스트 출력과 **일치하도록 수동 갱신**해야 함.

> 💡 **개발 팁(이번 세션에서 쓴 기법):** CLI 인자가 없는 상수(`SL_CAP`, `RISK_PCT` 등)를 그리드서치할 때, 해당 상수를 `float(os.getenv("MES_XXX", "기본값"))`으로 임시 변경해 env로 스윕한 뒤, 최적값을 하드코딩으로 되돌리고 env 훅을 제거했다. (지금은 모두 정리됨 — 잔존 `os.getenv("MES_` 없음)

---

## 7. 진행 이력 (최근)

| 버전 | 변경 | 결과(승률/연수익/Sharpe) |
|------|------|------|
| v10.1 | MIN_SCORE 88→60, TP_MULT 1.5→2.5 | 43%/7.9%/0.27 |
| v10.2 | VIX_TH 20→25, VIX_SHORT_FILTER 신설, MIN_SCORE→74 | 49%/15.6%/1.01 |
| v10.3 | SL_CAP 15→22, RISK_PCT 1.5→2.5% | 57%/31.8%/1.57 |
| **v10.4** | **MIN_SCORE 74→68 (+18% 거래빈도, 42→49/yr)** | **53%/31.6%/1.44** |

- **베어마켓 보강(2022)**: Option A(VIX≥30 추세추종)+Option C(VIX 사이징) 적용. Option B(ADX 트렌드베어)와 direction-aware veto는 **MEAN_REVERSION과 방향 충돌 버그**로 제거/롤백 → **대칭형 veto 유지**.
- `bear_market_2022` 상태: `DATA_NOT_AVAILABLE` — 2022 Databento 데이터 미보유. `MES_1min_data_2022_et_rth.csv` 다운로드 후 재백테스트 필요.

---

## 8. 알아둘 함정 / 주의사항

1. **백테스트=라이브 동기화가 생명.** 상수는 `api/v10_strategy.py`에만 두고, 백테스트/러너는 그걸 import해야 한다. 한쪽만 바꾸면 라이브가 발행 성과와 달라진다. (단, 백테스트의 `sl_cap`은 509행에서 한 번 더 명시적으로 22를 넘기므로 함께 맞춰야 함.)
2. **로컬 상태파일 ≠ 프로덕션.** `v10_state.json`/`v10_paper_log.json`의 `{"test":true}`류 내용은 무시. 진짜는 KV.
3. **"무진입"은 대개 정상.** 평균 6.2거래일에 1회(전체 날의 16%만 진입), 과거 최장 36거래일 무진입. 보통 사유는 `score < 74`. 이벤트(NFP/CPI/FOMC) 직전엔 변동성↓로 스코어가 더 낮게 나옴.
4. **시크릿은 프로덕션에만.** 이 컨테이너에서 라이브 재현/ KV조회 불가. 필요하면 사용자가 키를 주입하거나 Vercel/Upstash 콘솔에서 확인.
5. **"Sharpe 2.05" 옛 표기 오류 주의.** 과거 `data.py`에 Sharpe=profit_factor로 잘못 표기된 흔적이 있었음. 실제 Sharpe는 백테스트 코드 공식 기준(현재 v10.3=1.56).
6. **MEAN_REVERSION은 강세장에서 해롭다.** 2023-26 데이터에선 순수 추세추종이 더 나음. VIX_THRESHOLD를 25로 올려 평균회귀 구간을 축소한 이유.
7. **브랜치 작업 규칙(이 세션 한정):** 개발은 `claude/...` 류 지정 브랜치에서, 푸시는 `git push -u origin <branch>` (네트워크 실패시 지수백오프 재시도). PR은 명시 요청시에만.

---

## 9. 다음에 할 만한 일 (열린 과제)

- [ ] **2022 베어마켓 재백테스트** (Databento 2022 데이터) → `bear_market_2022` 요약을 ACTUAL로 갱신.
- [ ] 빈도/품질 재균형 검토: 저변동 국면에서 진입이 너무 드물면 `MIN_SCORE`를 70 등으로 미세조정(승률/Sharpe 트레이드오프 재평가).
- [ ] 파라미터 한계 도달 → 추가 수익은 **새 엣지(신규 피처/시간대/종목분산)** 필요. 단 과최적화 검증 필수.
- [ ] (선택) RISK_PCT를 사용자 리스크 선호도에 맞춰 확정(2.0% 보수 ~ 2.5% 현행 ~ 3%+ 공격).

---

## 10. 빠른 점검 체크리스트 (인수자가 처음 할 것)

```bash
cd /home/user/MES
git log --oneline -3                  # main이 v10.3(c7e538d)인지
python -m pytest tests/ -q            # 108 passed 확인
python thorough_backtest_futures.py --csv MES_1min_data_et_rth.csv \
  | grep -E "Win Rate|Annual|Sharpe|Max Draw"   # 57.4/31.6/1.56/5.8 확인
grep -E "^MIN_SCORE|^SL_CAP_PTS|^RISK_PCT_FULL|^VIX_THRESHOLD" api/v10_strategy.py
```
값이 위 표와 일치하면 정상. 라이브 점검은 https://hannaealgo.vercel.app 대시보드의 "진입 대기" 줄(오늘/5일 최고 스코어 vs 74)로.
