# download_1min_data.py
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from tqdm import tqdm
import time

# Ensure UTF-8 output encoding for emojis and Korean characters in Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==================== 설정 ====================
API_KEY = os.getenv("POLYGON_API_KEY")

# Fallback: Read from .env if not found in environment variables
if not API_KEY and os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.strip().startswith("POLYGON_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()

SYMBOL = "SPY"             # MES 선물은 "MES" 또는 "MESZ4" (만기별) 사용
START_DATE = "2021-01-01"   # 시작일 (Polygon Stocks Starter = 5yr lookback; free = 2yr, older chunks fail gracefully)
END_DATE = "2026-05-31"     # 종료일 (today)
TIMEFRAME = "1"             # 1 = 1분봉
OUTPUT_CSV = f"{SYMBOL}_1min_data.csv"

if not API_KEY:
    raise ValueError("POLYGON_API_KEY가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인해주세요.")

client = RESTClient(API_KEY)

def download_1min_bars():
    all_data = []
    current_start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    print(f"📥 {SYMBOL} 1분봉 데이터 다운로드 시작...")
    
    # Calculate approximate total iterations for tqdm progress bar
    total_days = (end_dt - current_start).days
    total_chunks = (total_days + 29) // 30
    
    with tqdm(total=total_chunks, desc="Downloading") as pbar:
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt) # 30일씩 끊어서 요청 (50000개 제한 대비)
            
            try:
                aggs = client.get_aggs(
                    ticker=SYMBOL,
                    multiplier=int(TIMEFRAME),
                    timespan="minute",
                    from_=current_start.strftime("%Y-%m-%d"),
                    to=current_end.strftime("%Y-%m-%d"),
                    limit=50000
                )
                
                chunk_count = 0
                for agg in aggs:
                    all_data.append({
                        "timestamp": pd.to_datetime(agg.timestamp, unit='ms'),
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "vwap": getattr(agg, 'vwap', None)
                    })
                    chunk_count += 1
                
                print(f"\n✅ {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')} 완료 ({chunk_count:,}개 봉)")
                
            except Exception as e:
                print(f"\n❌ 오류: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')} -> {e}")
                
            current_start = current_end + timedelta(days=1)
            pbar.update(1)
            
            # rate limit (5 calls/min) 대비 안전 대기 (마지막 루프가 아니면 대기)
            if current_start < end_dt:
                time.sleep(12)
        
    # DataFrame으로 변환 후 저장
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\n🎉 완료! 총 {len(df):,}건 데이터가 {OUTPUT_CSV}에 저장되었습니다.")
        print(f"파일 크기: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    else:
        print("\n❌ 다운로드된 데이터가 없습니다.")

if __name__ == "__main__":
    download_1min_bars()
