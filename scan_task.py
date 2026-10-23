import os
import datetime
import base64
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

# ==============================
# 1. 抓取台股代號清單
# ==============================
def get_all_tw_stocks():
    stocks = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    for mode in ['2', '4']: 
        suffix = '.TW' if mode == '2' else '.TWO'
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        try:
            res = requests.get(url, headers=headers, timeout=15)
            res.encoding = 'big5'
            soup = BeautifulSoup(res.text, 'html.parser')
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) > 0 and '\u3000' in tds[0].text:
                    code, name = tds[0].text.strip().split('\u3000')
                    if code.isdigit() and len(code) == 4:
                        stocks[code + suffix] = name
        except Exception:
            pass
    return stocks

# ==============================
# 2. 技術指標計算模組
# ==============================
def add_indicators(df):
    df = df.copy()
    low_min = df['Low'].rolling(window=9, min_periods=1).min()
    high_max = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-8) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['Pct_Change'] = df['Close'].pct_change() * 100
    df['Volume_Lots'] = df['Volume'] / 1000
    df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
    return df

# ==============================
# 3. 底背離檢測邏輯
# ==============================
def check_bottom_divergence(df, price_col, ind_col, recent_w=5, older_w=20):
    if len(df) < recent_w + older_w:
        return False
    recent_start = len(df) - recent_w
    older_start = recent_start - older_w
    older_end = recent_start

    prices = df[price_col].values
    inds = df[ind_col].values

    r_p_min = np.min(prices[recent_start:])
    o_p_min = np.min(prices[older_start:older_end])
    r_i_min = np.min(inds[recent_start:])
    o_i_min = np.min(inds[older_start:older_end])

    return (r_p_min <= o_p_min * 1.02) and (r_i_min > o_i_min)

# ==============================
# 4. 主執行排程
# ==============================
def run_daily_scan():
    print("🚀 [排程啟動] 開始全市場盤後掃描...")
    stock_dict = get_all_tw_stocks()
    tickers = list(stock_dict.keys())
    
    results = []
    chunk_size = 40

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = yf.download(chunk, period="6mo", threads=False, progress=False)
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        df = data.xs(ticker, axis=1, level=0).dropna(how='all') if ticker in data.columns.get_level_values(0) else data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else:
                        df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                    if df.empty or len(df) < 35:
                        continue

                    df = add_indicators(df)
                    last_row = df.iloc[-1]
                    prev_vol = df['Volume_Lots'].iloc[-2]

                    # 1. 第一階段價量過濾
                    cond_pct = 3.4 <= last_row['Pct_Change'] <= 10.5
                    cond_vol = last_row['Volume_Lots'] >= 1000
                    vol_mult = last_row['Volume_Lots'] / (prev_vol + 1e-8)
                    cond_mult = 2.4 <= vol_mult <= 15.0

                    if cond_pct and cond_vol and cond_mult:
                        # 2. 第二階段背離驗證 (回溯25天)
                        kd_pass, macd_pass = False, False
                        for b in range(0, 25):
                            sub_df = df.iloc[:len(df)-b]
                            if not kd_pass:
                                kd_pass = check_bottom_divergence(sub_df, 'Low', 'K')
                            if not macd_pass:
                                macd_pass = check_bottom_divergence(sub_df, 'Low', 'MACD_Hist')
                            if kd_pass and macd_pass:
                                break

                        if kd_pass and macd_pass:
                            # 3. 統計前 30 天觸發次數
                            count_30d = 0
                            for b_offset in range(1, 31):
                                if len(df) <= b_offset + 1: continue
                                c_row = df.iloc[-1 - b_offset]
                                c_prev_vol = df['Volume_Lots'].iloc[-2 - b_offset]
                                if (3.4 <= c_row['Pct_Change'] <= 10.5) and (c_row['Volume_Lots'] >= 1000) and (2.4 <= (c_row['Volume_Lots'] / (c_prev_vol + 1e-8)) <= 15.0):
                                    count_30d += 1

                            results.append({
                                "股票代號": ticker.split('.')[0],
                                "股票名稱": stock_dict[ticker],
                                "觸發日期": last_row.name.strftime('%Y-%m-%d'),
                                "近30日觸發次數": count_30d,
                                "收盤價": round(float(last_row['Close']), 2),
                                "今日漲跌幅(%)": round(float(last_row['Pct_Change']), 2),
                                "當日成交量(張)": int(last_row['Volume_Lots']),
                                "量增倍數": round(vol_mult, 2),
                                "演算法建議結果": "量價雙背離"
                            })
                except Exception:
                    continue
        except Exception:
            pass

    # ==============================
    # 5. 上傳 Google Drive
    # ==============================
    if results:
        res_df = pd.DataFrame(results)
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        file_name = f"auto_scan_results_{today_str}.csv"
        csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')

        webhook_url = os.environ.get("GAS_WEBHOOK_URL")
        token = os.environ.get("GAS_SECRET_TOKEN")

        if webhook_url and token:
            payload = {
                "token": token,
                "fileName": file_name,
                "fileData": base64.b64encode(csv_bytes).decode('utf-8')
            }
            res = requests.post(webhook_url, data=payload)
            print(f"✅ 上傳結果: {res.text}")
        else:
            print("⚠️ 缺少環境變數，略過上傳")
    else:
        print("💡 今日無符合條件標的。")

if __name__ == "__main__":
    run_daily_scan()
