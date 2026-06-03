# v10 상시 워커 배포 가이드

`trading_bot.py --worker` 는 백테스트를 **분 단위로 그대로 재현**하는 상시
프로세스입니다. 진입(10:30 ET 단일) + 장중 트레일링/본전이동(BE) 청산까지
`api/v10_strategy.py` 의 **동일한 함수**(`evaluate_entry`, `init_position`,
`manage_bar`)로 처리하므로, 백테스트(243거래·Sharpe 2.05)와 같은 의사결정을
라이브 데이터에서 수행합니다.

Vercel Cron(하루 2회)은 트레일링/BE를 못 하지만, 이 워커는 매분 돌면서
그걸 합니다. **가장 충실한 배포 방식**입니다.

---

## 1) 무엇이 필요한가

- 24시간 켜져 있는 리눅스 호스트 1대 (아래 무료/저가 옵션)
- Python 3.11+
- 환경변수: 시장데이터(Alpaca/Polygon) + (선택)Upstash KV
  - `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`
  - `POLYGON_API_KEY` (폴백)
  - `BROKER=dryrun` (페이퍼; 실거래 시 `tradovate` + TRADOVATE_* 키)
  - `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` (KV 공유 시; 생략하면
    로컬 파일 상태)

## 2) 호스트 옵션 (저렴한 순)

| 옵션 | 비용 | 비고 |
|------|------|------|
| **Oracle Cloud Always-Free VM** | **$0 평생** | Ampere A1 / AMD micro. 추천 |
| Fly.io | $0~ | 작은 머신 무료 한도 |
| Google Cloud e2-micro | $0 (us-west1 등) | always-free 한도 |
| Railway / Render | $5~ | 가장 셋업 쉬움 |
| Hetzner / DigitalOcean VPS | $4~6/월 | 완전한 제어 |
| 집 라즈베리파이 / 미니PC | 전기값 | 전원만 유지 |

## 3) Oracle Cloud Always-Free 예시

```bash
# (Oracle Cloud 콘솔에서 Always-Free VM 생성: Ubuntu 22.04, Ampere A1)
ssh ubuntu@<vm-ip>

sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/yoon9173580/mes.git
cd mes
pip3 install -r requirements.txt   # 또는: pip3 install pandas numpy pytz requests

# 환경변수 파일 작성
cat > ~/mes/.env <<'EOF'
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
POLYGON_API_KEY=...
BROKER=dryrun
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
EOF

# 수동 테스트 (장중에 한 번 돌려보기)
set -a; source ~/mes/.env; set +a
python3 trading_bot.py --worker --store kv --poll 60
```

## 4) systemd 서비스로 상시 구동 (재부팅·크래시 자동 복구)

```bash
sudo tee /etc/systemd/system/v10-worker.service > /dev/null <<'EOF'
[Unit]
Description=MES v10 paper worker
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mes
EnvironmentFile=/home/ubuntu/mes/.env
ExecStart=/usr/bin/python3 trading_bot.py --worker --store kv --poll 60
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now v10-worker
sudo systemctl status v10-worker        # 상태 확인
journalctl -u v10-worker -f             # 실시간 로그
```

## 5) Vercel Cron 과의 관계

- 워커와 Vercel Cron은 **같은 KV 상태**(`--store kv`)를 공유합니다.
- 둘 다 켜두면: 워커가 10:30에 먼저 진입해 `last_trade_date` 를 잠그므로
  11:30 Vercel entry cron 은 `SKIP_DONE` 됩니다 (중복 없음).
- **권장:** 워커를 메인으로 쓰고, Vercel entry cron 은 비활성화하거나 백업으로만
  둡니다. 워커가 트레일링/BE 까지 처리하니 가장 정확합니다.
- 대시보드(`/api/data`)는 KV 의 `v10:log` / `v10:state` 를 그대로 읽어
  거래 내역을 표시합니다.

## 6) 동작 요약 (매분)

```
평일 09:30–16:00 ET 안에서:
  포지션 없음 + 10:30–12:00 + 오늘 미거래 → run_once_entry (진입)
  포지션 있음                           → run_once_monitor (트레일/BE 또는 청산)
  15:30 이후 + 포지션 있음              → EOD 청산
그 외 시간                              → 대기(sleep poll)
```

상태/로그는 `--store kv` 면 Upstash, 아니면 로컬 `v10_state.json` /
`v10_paper_log.json` 에 저장됩니다.
