import os
import requests
import datetime
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.signal import argrelextrema
from bs4 import BeautifulSoup
import urllib3

# 關閉因 verify=False 產生的 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 參數與全域常數設定
# ==========================================
DEFAULT_CONFIG = {
    "algo_mode": "全部",
    "lookback_end": 0,
    "lookback_start": 0,
    "min_score": 8.0,
    "min_vcp_score": 10.0,
    "min_reso_score": 10.0,
    "min_ma240_score": 8.0,
    "min_vol_ma20": 1000,
    "use_single_div": False,
    "div_recent_w": 5,
    "div_older_w": 20,
    "pivot_left": 0,
    "pivot_right": 0,
    "recent_lows_cnt": 0,
    "older_lows_cnt": 0,
    "kou_di_5": False,
    "kou_di_10": False,
    "kou_di_20": False,
    "kou_di_60": False,
    "reso_kd_older_low": 20.0,
    "reso_kd_older_high": 90.0,
    "reso_kd_recent_low": 0.0,
    "use_macd_abs": False,
    "reso_macd_older_low": 0.0,
    "reso_macd_recent_low": 0.0,
    "reso_cross_days": 3,
    "reso_price_higher_low": False,
    "reso_macd_cross_zero": False,
    "reso_price_basis": "最低價 (Low)",
    "reso_macd_wave_logic": False,
    "require_cross_confirm": True,
    "use_cross_position_filter": True,
    "cross_position_threshold": 70.0,
    "use_weekly_filter": True,
    "enable_30d_backtest": True,
    "show_ma120_240_bear_align": True,
    "show_ma240_slope": True,
    "show_ma120_slope": True,
    "auto_upload_gdrive": True,
    "use_backtest_date": False,
    "backtest_date": str(datetime.date.today())
}

# 擴充演算法清單 (UI自動抓取此清單顯示，隨時可加)
SCAN_ALGO_CHOICES = ['全部', '底部翻轉', 'VCP', '指標共振', '量價雙背離', '長下引突破', '*回測走揚年線', '低位長下引量縮突破']

# ==========================================
# 網路與工具模組
# ==========================================
def get_yf_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'})
    return session

def fetch_all_tw_stocks():
    stocks = {}
    session = get_yf_session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/json,application/xhtml+xml'
    }
    
    def is_valid_code(c):
        # 支援 4 碼純數字 (一般股票) 與 00 開頭的 ETF (如 0050, 00631L)
        return (c.isdigit() and len(c) == 4) or (c.startswith('00') and len(c) >= 4)

    # 方法 1: 政府 OpenData API 優先 (加入 verify=False 繞過 SSL 阻擋)
    try:
        r1 = session.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=15, verify=False)
        if r1.status_code == 200:
            for item in r1.json():
                c = str(item.get('Code', '')).strip()
                if is_valid_code(c): stocks[c + '.TW'] = str(item.get('Name', '')).strip()
                
        r2 = session.get("https://www.tpex.org.tw/openapi/v1/mktdata/tpexod/BPSR", headers=headers, timeout=15, verify=False)
        if r2.status_code == 200:
            for item in r2.json():
                c = str(item.get('SecuritiesCompanyCode', '')).strip()
                if is_valid_code(c): stocks[c + '.TWO'] = str(item.get('CompanyName', '')).strip()
        
        if len(stocks) > 1000: return stocks
    except Exception: pass

    # 方法 2: FinMind 開放 API 備援 (對海外 IP 友善)
    try:
        r3 = session.get("https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo", headers=headers, timeout=15, verify=False)
        if r3.status_code == 200:
            for item in r3.json().get('data', []):
                c = str(item.get('stock_id', '')).strip()
                m = str(item.get('type', '')).strip().lower()
                if is_valid_code(c):
                    if m == 'twse': stocks[c + '.TW'] = str(item.get('stock_name', '')).strip()
                    elif m == 'tpex': stocks[c + '.TWO'] = str(item.get('stock_name', '')).strip()
        if len(stocks) > 1000: return stocks
    except Exception: pass
        
    # 方法 3: 證交所 ISIN 爬蟲 (最後防線)
    try:
        for mode in ['2', '4']: 
            suffix = '.TW' if mode == '2' else '.TWO'
            r4 = session.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}", headers=headers, timeout=15, verify=False)
            if r4.status_code == 200:
                r4.encoding = 'big5' 
                soup = BeautifulSoup(r4.text, 'html.parser')
                for tr in soup.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) > 0:
                        text = tds[0].text.strip()
                        if '\u3000' in text:
                            code, name = text.split('\u3000')
                            if is_valid_code(code): stocks[code + suffix] = name
        if len(stocks) > 1000: return stocks
    except Exception: pass

    # 方法 4: 終極保底機制 (防止系統徹底崩潰，若均失敗，給予最低限度權值股進行測試)
    if not stocks:
        fallback_list = ['2330', '2317', '2454', '2308', '2382', '2881', '2891', '2412', '2882', '2886', '1303', '2884', '1216', '2002', '2892', '2303', '3711', '2885', '2880', '1301', '2345', '3008', '3231', '2357', '1101', '5880', '2883', '2887', '2603', '2324', '2379', '2890', '2912', '1326', '2395', '3034', '2207', '2408', '2609', '4938', '2801', '6669', '1402', '5871', '1590', '2301', '2356', '2448', '3037', '2383', '2618', '2610', '9904', '9910', '1102']
        for c in fallback_list: stocks[c + '.TW'] = "權值備援標的"

    return stocks

def fetch_tw_stock_name(ticker):
    code = ticker.split('.')[0]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5, verify=False)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1')
            if h1: return h1.text.strip()
    except Exception: pass
    try:
        return yf.Ticker(ticker, session=get_yf_session()).info.get('shortName', code)
    except Exception: return code

def fetch_stock_data(symbol):
    code = str(symbol).strip().upper()
    targets = [code] if code.endswith(".TW") or code.endswith(".TWO") else [f"{code}.TW", f"{code}.TWO"]
    session = get_yf_session()
    for ticker in targets:
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period="2y")
            if not df.empty: return df, ticker
        except Exception: continue
    return pd.DataFrame(), code

def resolve_ticker(symbol, stock_dict):
    code = str(symbol).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"): return code
    if f"{code}.TW" in stock_dict: return f"{code}.TW"
    if f"{code}.TWO" in stock_dict: return f"{code}.TWO"
    return f"{code}.TW" 

class GoogleDriveUploader:
    @staticmethod
    def upload(file_name, webhook_url, token, file_path=None, csv_bytes=None):
        if not webhook_url or not webhook_url.startswith("https://script.google.com/"):
            return False, "無效的 Google Apps Script Webhook 網址"
        try:
            if csv_bytes is not None: file_data = csv_bytes
            elif file_path:
                with open(file_path, "rb") as f: file_data = f.read()
            else: return False, "無檔案資料"
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            payload = {"token": token, "fileName": file_name, "fileData": encoded_data}
            response = requests.post(webhook_url, data=payload, timeout=30, verify=False)
            result = response.json()
            if result.get("status") == "success": return True, result.get("fileUrl", "成功")
            return False, result.get("message", "未知錯誤")
        except Exception as e: return False, str(e)

# ==========================================
# 指標與策略核心 (包含所有技術分析邏輯)
# ==========================================
class TechnicalIndicators:
    @staticmethod
    def add_kd(df, n=9, m1=3, m2=3):
        df = df.copy()
        low_min = df['Low'].rolling(window=n, min_periods=1).min()
        high_max = df['High'].rolling(window=n, min_periods=1).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-8) * 100
        df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
        return df
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        df = df.copy()
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df
    @staticmethod
    def add_rsi(df, period=14):
        df = df.copy()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

class MarketRegimeFilter:
    @staticmethod
    def evaluate(session, backtest_date_obj=None):
        try:
            if backtest_date_obj:
                end_dt = backtest_date_obj + datetime.timedelta(days=1)
                start_dt = end_dt - datetime.timedelta(days=150)
                df = yf.Ticker("^TWII", session=session).history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))
            else:
                df = yf.Ticker("^TWII", session=session).history(period="3mo")
            if df.empty: return None
            close = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            basis_value = close 
            if close > ma20 and ma20 > ma60: regime = "🟢 多頭排列 (做多環境佳，可適度放大部位)"
            elif close < ma20 and ma20 < ma60: regime = "🔴 空頭弱勢 (系統性風險高，建議縮小部位或觀望)"
            else: regime = "🟡 震盪整理 (選股不選市，嚴格執行停損)"
            return {
                "加權指數收盤": f"{close:.2f}", "月線 (MA20)": f"{ma20:.2f}",
                "季線 (MA60)": f"{ma60:.2f}", "自動運算基準價值 (約當大台基礎)": f"{basis_value:.2f}",
                "大盤環境判定": regime
            }
        except Exception: return None

class ExpertTradingStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.DataFrame(index=df.index)
        if len(df) < 20:
            for c in ['RSI盤整', '多空指標數', '日KD多', '週KD多']: res[c] = 0 if c=='多空指標數' else False
            return res
        is_consolidation = df['RSI'].between(45, 55).rolling(5).sum() == 5
        kd_bull = df['K'] > df['D']
        macd_bull = df['MACD_Hist'] > 0
        rsi_bull = df['RSI'] > 50
        bull_count = kd_bull.astype(int) + macd_bull.astype(int) + rsi_bull.astype(int)
        try:
            w_df = df[['Open', 'High', 'Low', 'Close']].resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            w_df = TechnicalIndicators.add_kd(w_df)
            w_df['W_Bull'] = w_df['K'] > w_df['D']
            w_bull_series = w_df['W_Bull'].reindex(df.index, method='ffill').fillna(False)
        except Exception:
            w_bull_series = pd.Series(False, index=df.index)
        res['RSI盤整'] = is_consolidation
        res['多空指標數'] = bull_count
        res['日KD多'] = kd_bull
        res['週KD多'] = w_bull_series
        return res
    @staticmethod
    def get_position_advice(bull_count):
        if pd.isna(bull_count): return "-"
        if bull_count == 3: return "7成 (同步偏多)"
        if bull_count == 2: return "5成 (偏多震盪)"
        if bull_count == 1: return "3成 (偏空震盪)"
        return "1成以下 (同步偏空)"
    @staticmethod
    def get_multi_tf_advice(w_bull, d_bull):
        if pd.isna(w_bull) or pd.isna(d_bull): return "-"
        if w_bull and d_bull: return "🟢 同步轉多 (勝率高)"
        if w_bull and not d_bull: return "🟡 短期回檔 (勿加碼)"
        if not w_bull and d_bull: return "🟠 跌深反彈 (非回升)"
        return "🔴 同步轉空 (觀望)"

class BottomReversalStrategy:
    @staticmethod
    def evaluate(df):
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        total_range = (df['High'] - df['Low']).replace(0, 0.001)
        vol_mult = (df['Volume_Lots'] / (df.get('Vol_MA20', df['Volume_Lots'].rolling(20).mean()) + 1e-8)).clip(0.5, 3.0)
        bias20 = (df['Close'] - df.get('MA20', df['Close'].rolling(20).mean())) / (df.get('MA20', df['Close'].rolling(20).mean()) + 1e-8) * 100
        cond_low_pin = (bias20 <= 0) & (lower_shadow > body * 1.5) & (lower_shadow > total_range * 0.4)
        pct_change = (df['Close'].pct_change() * 100) if 'Pct_Change' not in df.columns else df['Pct_Change']
        cond_low_red = (bias20 <= 0) & (df['Close'] > df['Open']) & (pct_change >= 2.5)
        candle_score = pd.Series(0.0, index=df.index)
        candle_score[cond_low_pin] = 7 * (lower_shadow[cond_low_pin] / total_range[cond_low_pin]) * vol_mult[cond_low_pin]
        candle_score[cond_low_red] = 5 * vol_mult[cond_low_red]
        return candle_score

class VCPStrategy:
    @staticmethod
    def evaluate(df):
        ma20 = df.get('MA20', df['Close'].rolling(20).mean())
        ma60 = df.get('MA60', df['Close'].rolling(60).mean())
        bb_u = df.get('BB_Upper', ma20 + 2 * df['Close'].rolling(20).std())
        bb_l = df.get('BB_Lower', ma20 - 2 * df['Close'].rolling(20).std())
        bb_width = (bb_u - bb_l) / (ma20 + 1e-8) * 100
        vol_ma = df.get('Vol_MA20', df['Volume_Lots'].rolling(20).mean())
        cond_uptrend = (df['Close'] > ma20) & (ma20 > ma60)
        cond_vol_dry = df['Volume_Lots'] < vol_ma
        cond_tight_price = bb_width < 10.0
        vol_score = 10 * (1 - df['Volume_Lots'] / (vol_ma + 1e-8)).clip(0, 1)
        tight_score = 10 * (10 - bb_width) / 10
        vcp_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_uptrend & cond_vol_dry & cond_tight_price
        vcp_score[valid_mask] = vol_score[valid_mask] + tight_score[valid_mask]
        return vcp_score

class DivergenceStrategy:
    @staticmethod
    def check_bottom_divergence(df, price_col='Low', ind_col='K', cross_col1='K', cross_col2='D',
        recent_w=5, older_w=20, recent_lows_cnt=0, older_lows_cnt=0, pivot_left=0, pivot_right=0, require_cross_confirm=True, cross_days=3):
        if len(df) < recent_w + older_w: return False
        if require_cross_confirm:
            cross = (df[cross_col1] > df[cross_col2]) & (df[cross_col1].shift(1) <= df[cross_col2].shift(1))
            if not cross.iloc[-cross_days:].any(): return False
        recent_start, older_start, older_end = len(df) - recent_w, len(df) - recent_w - older_w, len(df) - recent_w
        prices, inds = df[price_col].values, df[ind_col].values
        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            return (np.min(prices[recent_start:]) <= np.min(prices[older_start:older_end]) * 1.02) and (np.min(inds[recent_start:]) > np.min(inds[older_start:older_end]))
        def get_valid_pivots_iloc(start_loc, end_loc):
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s, e = max(0, i_loc - pivot_left), min(len(prices), i_loc + pivot_right + 1)
                if prices[i_loc] == np.min(prices[s:e]): pivots.append(i_loc)
            return pivots
        r_pivots = get_valid_pivots_iloc(recent_start, len(df))
        o_pivots = get_valid_pivots_iloc(older_start, older_end)
        if recent_lows_cnt > 0: r_pivots = sorted(r_pivots, key=lambda x: prices[x])[:recent_lows_cnt]
        if older_lows_cnt > 0: o_pivots = sorted(o_pivots, key=lambda x: prices[x])[:older_lows_cnt]
        if not r_pivots: r_pivots = [recent_start + np.argmin(prices[recent_start:])]
        if not o_pivots: o_pivots = [older_start + np.argmin(prices[older_start:older_end])]
        for r_idx in r_pivots:
            for o_idx in o_pivots:
                r_i_val = np.min(inds[max(recent_start, r_idx-2) : min(len(inds), r_idx+3)])
                o_i_val = np.min(inds[max(older_start, o_idx-2) : min(older_end, o_idx+3)])
                if prices[r_idx] <= prices[o_idx] * 1.02 and r_i_val > o_i_val: return True
        return False

class IndicatorResonanceStrategy:
    @staticmethod
    def evaluate(df, recent_w=5, older_w=20, kd_older_low_th=20, kd_older_high_th=90, kd_recent_low_th=0.0, 
                 use_macd_abs=False, macd_older_low_th=0.0, macd_recent_low_th=0.0, cross_days=3,
                 require_price_higher_low=False, require_macd_cross_zero=False,
                 reso_price_basis="最低價 (Low)", use_macd_wave_logic=False, require_cross_confirm=True):
        if len(df) < recent_w + older_w: return pd.Series(0.0, index=df.index)
        price_col = 'Low' if reso_price_basis == "最低價 (Low)" else 'Close'
        recent_k_low = df['K'].rolling(window=recent_w, min_periods=1).min()
        older_k_low = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        older_k_high = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).max()
        if use_macd_wave_logic:
            sign = np.sign(df['MACD_Hist']).replace(0, np.nan).ffill().fillna(1)
            block_id = (sign != sign.shift(1)).cumsum().rename('block_id')
            neg_mask = sign < 0
            if not neg_mask.any(): recent_macd_low, older_macd_low = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
            else:
                block_mins = df['MACD_Hist'].groupby(block_id).transform('min')
                recent_val = pd.Series(np.nan, index=df.index)
                recent_val.loc[neg_mask] = block_mins[neg_mask]
                recent_macd_low = recent_val.ffill()
                unique_neg_blocks = df.loc[neg_mask, 'MACD_Hist'].groupby(block_id).min().reset_index()
                unique_neg_blocks['prev_min'] = unique_neg_blocks['MACD_Hist'].shift(1)
                prev_min_dict = dict(zip(unique_neg_blocks['block_id'], unique_neg_blocks['prev_min']))
                older_val = pd.Series(np.nan, index=df.index)
                older_val.loc[neg_mask] = block_id[neg_mask].map(prev_min_dict)
                older_macd_low = older_val.ffill()
        else:
            recent_macd_low = df['MACD_Hist'].rolling(window=recent_w, min_periods=1).min()
            older_macd_low = df['MACD_Hist'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        recent_price_low = df[price_col].rolling(window=recent_w, min_periods=1).min()
        older_price_low = df[price_col].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        kd_cross = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        if require_macd_cross_zero: macd_signal = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        else: macd_signal = ((df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))) | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))
        kd_recent_cross = kd_cross.rolling(window=cross_days, min_periods=1).max() >= 1
        macd_recent_cross = macd_signal.rolling(window=cross_days, min_periods=1).max() >= 1
        if require_cross_confirm:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th) & kd_recent_cross
            cond_macd = (recent_macd_low > older_macd_low) & macd_recent_cross
        else:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th)
            cond_macd = (recent_macd_low > older_macd_low)
        if use_macd_abs: cond_macd = cond_macd & (older_macd_low < macd_older_low_th) & (recent_macd_low > macd_recent_low_th)
        cond_price = (recent_price_low > older_price_low) if require_price_higher_low else pd.Series(True, index=df.index)
        resonance_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_kd & cond_macd & cond_price
        hist_momentum = (df['MACD_Hist'] - df['MACD_Hist'].shift(1)) * 100
        resonance_score[valid_mask] = 10.0 + hist_momentum[valid_mask].clip(0, 5) 
        return resonance_score

class PriceVolDivergenceStrategy:
    @staticmethod
    def evaluate(df, recent_w=5, older_w=20, rl_cnt=0, ol_cnt=0, p_left=0, p_right=0, req_cross=True, c_days=3, price_col='Low'):
        res = pd.Series(False, index=df.index)
        if len(df) < recent_w + older_w + 5: return res
        pct_change = (df['Close'].pct_change() * 100) if 'Pct_Change' not in df.columns else df['Pct_Change']
        cond_pct = pct_change.between(3.4, 10.5)
        cond_vol_abs = df['Volume_Lots'] >= 1000
        cond_vol_mult = (df['Volume_Lots'] / (df['Volume_Lots'].shift(1) + 1e-8)).between(2.4, 15.0) 
        valid_idx = np.where(cond_pct & cond_vol_abs & cond_vol_mult)[0]
        for idx in valid_idx:
            d_kd, d_macd = False, False
            for buffer in range(0, 25):
                check_idx = idx - buffer
                if check_idx < recent_w + older_w: continue
                sub_df = df.iloc[:check_idx+1]
                if not d_kd: d_kd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'K', 'K', 'D', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if not d_macd: d_macd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if d_kd and d_macd: break
            if d_kd and d_macd: res.iloc[idx] = True
        return res

class LongLowerShadowBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 20: return res  
        df = df.copy()
        range_hl = (df['High'] - df['Low']).replace(0, 1e-8) 
        max_oc = df[['Open', 'Close']].max(axis=1)
        min_oc = df[['Open', 'Close']].min(axis=1)
        df['Upper_Shadow_Ratio'] = (df['High'] - max_oc) / range_hl
        df['Lower_Shadow_Ratio'] = (min_oc - df['Low']) / range_hl
        df['K_Amplitude'] = range_hl / df['Low'] * 100  
        if 'Pct_Change' not in df.columns: df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        if 'MA20' not in df.columns: df['MA20'] = df['Close'].rolling(window=20).mean()
        if 'BB_Lower' not in df.columns: df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(20).std()
        cond_today = (df['Close'] > df['Open']) & (df['Pct_Change'] >= 3.0) & (df['Upper_Shadow_Ratio'] < 0.40) & (df['Volume_Lots'] > df['Volume_Lots'].shift(1)) 
        target_k_mask = (df['Lower_Shadow_Ratio'] >= 0.30) & (df['Low'] < df['Low'].shift(1)) & (df['K_Amplitude'] >= 1.5) & (df['Low'] <= df['BB_Lower'] * 1.015) & (df['Close'] < df['MA20']) 
        for i in range(1, 11):
            valid = cond_today & target_k_mask.shift(i) & (df['Close'] > df['High'].shift(i))
            res = res | valid.fillna(False)
        return res

class MA240PullbackRightSideStrategy:
    @staticmethod
    def evaluate_phase1(df):
        if len(df) < 25: return pd.Series(False, index=df.index)
        close, open_p, low = df['Close'], df['Open'], df['Low']
        ma5, ma20 = close.rolling(5).mean(), close.rolling(20).mean()
        cond_red_ma5 = (close > open_p) & (close >= ma5)
        cond_base_support = low.rolling(3).min() >= low.rolling(10).min()
        cond_bias = ((close - ma20) / (ma20 + 1e-8) * 100).between(-8.0, 6.0)
        cond_ma20_stable = ma20 >= ma20.shift(3) * 0.985
        return (cond_red_ma5 & cond_base_support & cond_bias & cond_ma20_stable).fillna(False)

    @staticmethod
    def evaluate_phase2(long_df):
        if long_df is None or len(long_df) < 240: return {"is_pass": False, "score": 0.0}
        close, open_p, low, vol = long_df['Close'], long_df['Open'], long_df['Low'], long_df['Volume']
        ma5, ma20, ma60, ma240 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean(), close.rolling(240).mean()
        cur_close, cur_open, cur_ma240, cur_ma60, cur_ma20, cur_ma5 = float(close.iloc[-1]), float(open_p.iloc[-1]), float(ma240.iloc[-1]), float(ma60.iloc[-1]), float(ma20.iloc[-1]), float(ma5.iloc[-1])
        ma240_5d_prev = float(ma240.iloc[-5])
        ma240_slope = (cur_ma240 - ma240_5d_prev) / (ma240_5d_prev + 1e-8) * 100
        score_ma240 = 3.5 if ma240_slope > 0.04 else (2.5 if ma240_slope >= 0.0 else (1.0 if ma240_slope >= -0.03 else 0.0))
        recent_15_low = float(low.iloc[-15:].min())
        tested_ma240 = (recent_15_low <= cur_ma240 * 1.035) and (recent_15_low >= cur_ma240 * 0.96)
        held_ma240 = cur_close >= cur_ma240 * 0.99
        score_pullback = 4.0 if tested_ma240 and held_ma240 else (2.5 if held_ma240 and recent_15_low <= cur_ma240 * 1.06 else (1.0 if held_ma240 else 0.0))
        score_ma = (1.5 if cur_ma20 > float(ma20.iloc[-3]) else 0) + (1.0 if cur_close >= cur_ma20 else 0) + (0.5 if cur_ma20 >= cur_ma60 else 0)
        score_right = (1.0 if cur_close > cur_open else 0) + (1.0 if cur_close >= cur_ma5 else 0) + (0.5 if cur_close >= cur_ma240 else 0)
        vol_lots = vol / 1000.0
        cur_vol, prev_vol = float(vol_lots.iloc[-1]), float(vol_lots.iloc[-2]) if len(vol_lots) > 1 else float(vol_lots.iloc[-1])
        score_vol = (1.0 if cur_vol > prev_vol else 0) + (1.0 if cur_vol >= float(vol_lots.iloc[-20:].mean()) * 0.8 else 0)
        total_score = round(score_ma240 + score_pullback + score_ma + score_right + score_vol, 2)
        is_pass = (score_ma240 >= 1.0) and held_ma240 and (score_pullback >= 2.5) and (total_score >= DEFAULT_CONFIG["min_ma240_score"])
        return {"is_pass": is_pass, "score": total_score}

class LowPositionShrinkVolumeBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 62: return res
        df = df.copy()

        if 'Volume_Lots' not in df.columns:
            df['Volume_Lots'] = df.get('Volume', pd.Series(0, index=df.index)) / 1000.0

        min_60 = df['Low'].shift(2).rolling(window=60).min()
        max_60 = df['High'].shift(2).rolling(window=60).max()
        
        pos_60 = (df['Close'].shift(1) - min_60) / (max_60 - min_60 + 1e-8)
        cond_low_pos = pos_60 <= 0.20

        body_1 = abs(df['Close'].shift(1) - df['Open'].shift(1))
        lower_shadow_1 = df[['Open', 'Close']].shift(1).min(axis=1) - df['Low'].shift(1)
        range_1 = (df['High'].shift(1) - df['Low'].shift(1)).replace(0, 1e-8)

        cond_long_shadow = (lower_shadow_1 > body_1 * 1.5) & ((lower_shadow_1 / range_1) >= 0.4)
        cond_vol_up_1 = df['Volume_Lots'].shift(1) > df['Volume_Lots'].shift(2)

        cond_today_break = df['Close'] > df['High'].shift(1)
        cond_today_vol_shrink = df['Volume_Lots'] < df['Volume_Lots'].shift(1)

        valid = cond_low_pos & cond_long_shadow & cond_vol_up_1 & cond_today_break & cond_today_vol_shrink
        return valid.fillna(False)

class WeeklyAnalysisStrategy:
    @staticmethod
    def evaluate(long_df, div_pairs, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days, price_col):
        if long_df is None or len(long_df) < 20: return {"週K共振": "無資料", "週K背離": "無資料"}
        try:
            df = long_df.copy()
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            w_df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            if len(w_df) < 20: return {"週K共振": "資料不足", "週K背離": "資料不足"}
            w_df = TechnicalIndicators.add_macd(TechnicalIndicators.add_kd(w_df))
            last_row = w_df.iloc[-1]
            kd_bull, macd_bull = last_row['K'] > last_row['D'], last_row['MACD_Hist'] > 0
            reso_str = "✅ 雙共振" if kd_bull and macd_bull else ("KD偏多" if kd_bull else ("MACD偏多" if macd_bull else "❌ 無"))
            div_res = []
            for r_w, o_w in div_pairs:
                if DivergenceStrategy.check_bottom_divergence(w_df, price_col, 'K', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days): div_res.append(f"KD({r_w},{o_w})")
                if DivergenceStrategy.check_bottom_divergence(w_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days): div_res.append(f"MACD({r_w},{o_w})")
            return {"週K共振": reso_str, "週K背離": "+".join(list(dict.fromkeys(div_res))) if div_res else "無"}
        except Exception: return {"週K共振": "錯誤", "週K背離": "錯誤"}

# ==========================================
# 高階統合引擎 (ScanEngine)
# ==========================================
class ScanEngine:
    @staticmethod
    def calculate_all_phase1_scores(df, config):
        df['Candle_Score'] = BottomReversalStrategy.evaluate(df)
        df['VCP_Score'] = VCPStrategy.evaluate(df)
        price_col = 'Low' if config.get("reso_price_basis") == "最低價 (Low)" else 'Close'
        df['Reso_Score'] = IndicatorResonanceStrategy.evaluate(
            df, recent_w=config.get("div_recent_w"), older_w=config.get("div_older_w"),
            kd_older_low_th=config.get("reso_kd_older_low"), kd_older_high_th=config.get("reso_kd_older_high"),
            kd_recent_low_th=config.get("reso_kd_recent_low"), use_macd_abs=config.get("use_macd_abs"),
            macd_older_low_th=config.get("reso_macd_older_low"), macd_recent_low_th=config.get("reso_macd_recent_low"),
            cross_days=config.get("reso_cross_days"), require_price_higher_low=config.get("reso_price_higher_low"),
            require_macd_cross_zero=config.get("reso_macd_cross_zero"), reso_price_basis=config.get("reso_price_basis"),
            use_macd_wave_logic=config.get("reso_macd_wave_logic"), require_cross_confirm=config.get("require_cross_confirm")
        )
        df['PriceVol_Div_Pass'] = PriceVolDivergenceStrategy.evaluate(
            df, recent_w=config.get("div_recent_w"), older_w=config.get("div_older_w"),
            rl_cnt=config.get("recent_lows_cnt"), ol_cnt=config.get("older_lows_cnt"),
            p_left=config.get("pivot_left"), p_right=config.get("pivot_right"),
            req_cross=config.get("require_cross_confirm"), c_days=config.get("reso_cross_days"), price_col=price_col
        )
        df['Breakout_Signal'] = LongLowerShadowBreakoutStrategy.evaluate(df)
        df['MA240_P1_Pass'] = MA240PullbackRightSideStrategy.evaluate_phase1(df)
        df['LowPos_Breakout_Pass'] = LowPositionShrinkVolumeBreakoutStrategy.evaluate(df)
        
        return df

    @staticmethod
    def check_algo_pass(row, config, algo_mode, force_pass=False):
        is_r = (algo_mode in ['全部', '底部翻轉']) and (row.get('Candle_Score', 0) >= config.get("min_score", 8.0))
        is_v = (algo_mode in ['全部', 'VCP']) and (row.get('VCP_Score', 0) >= config.get("min_vcp_score", 10.0))
        is_reso = (algo_mode in ['全部', '指標共振']) and (row.get('Reso_Score', 0) >= config.get("min_reso_score", 10.0))
        is_pv = (algo_mode in ['全部', '量價雙背離']) and row.get('PriceVol_Div_Pass', False)
        is_brk = (algo_mode in ['全部', '長下引突破']) and row.get('Breakout_Signal', False)
        is_ma240 = (algo_mode in ['全部', '*回測走揚年線']) and row.get('MA240_P1_Pass', False)
        is_lowpos_brk = (algo_mode in ['全部', '低位長下引量縮突破']) and row.get('LowPos_Breakout_Pass', False)
        
        passed = is_r or is_v or is_reso or is_pv or is_brk or is_ma240 or is_lowpos_brk or force_pass
        combo_score = (row.get('Candle_Score', 0) + row.get('VCP_Score', 0) + row.get('Reso_Score', 0) + 
                       (20 if row.get('PriceVol_Div_Pass', False) else 0) + 
                       (10 if row.get('Breakout_Signal', False) else 0) + 
                       (12 if row.get('MA240_P1_Pass', False) else 0) +
                       (15 if row.get('LowPos_Breakout_Pass', False) else 0))
        return passed, combo_score

    @staticmethod
    def generate_base_tags(item, config):
        base_tags = []
        if item.get("量價雙背離") == "✅ 是": base_tags.append("量價雙背離")
        if item.get("長下引突破") == "✅ 是": base_tags.append("長下引突破")
        if item.get("*回測走揚年線") == "✅ 是": base_tags.append("*回測走揚年線")
        if item.get("低位長下引量縮突破") == "✅ 是" or item.get("LowPos_Breakout_Pass", False): base_tags.append("低位量縮突破")
        if item.get("反轉分數", 0) >= config.get("min_score", 8.0): base_tags.append("底部翻轉")
        if item.get("VCP分數", 0) >= config.get("min_vcp_score", 10.0): base_tags.append("VCP收斂")
        if item.get("共振分數", 0) >= config.get("min_reso_score", 10.0): base_tags.append("指標共振")
        return base_tags

# ==========================================
# 財報三表地雷掃描 (FinancialAuditStrategy)
# ==========================================
class FinancialAuditStrategy:
    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value) or value is None: return np.nan
            return float(value)
        except Exception: return np.nan
    @staticmethod
    def _find_row(df, candidates):
        if df is None or df.empty: return None
        index_map = {str(x).strip().lower(): x for x in df.index}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in index_map: return index_map[key]
        for candidate in candidates:
            key = str(candidate).strip().lower()
            for idx in df.index:
                if key in str(idx).strip().lower(): return idx
        return None
    @staticmethod
    def _get_series(df, candidates):
        row = FinancialAuditStrategy._find_row(df, candidates)
        if row is None: return pd.Series(dtype=float)
        s = df.loc[row].copy()
        return pd.Series([FinancialAuditStrategy._safe_float(x) for x in s.values], index=s.index, dtype=float)
    @staticmethod
    def _normalize_statement(df):
        if df is None or df.empty: return pd.DataFrame()
        result, new_columns = df.copy(), []
        for col in result.columns:
            try:
                dt = pd.to_datetime(col)
                if hasattr(dt, "tz_localize") and dt.tz is not None: dt = dt.tz_localize(None)
                new_columns.append(dt)
            except Exception: new_columns.append(col)
        result.columns = new_columns
        try: result = result.loc[:, sorted(result.columns)]
        except Exception: pass
        return result.loc[:, ~pd.Index(result.columns).duplicated()]
    @staticmethod
    def _growth(current, previous):
        if pd.isna(current) or pd.isna(previous) or abs(previous) < 1e-9: return np.nan
        return (current - previous) / abs(previous) * 100
    @staticmethod
    def _ratio(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) or abs(b) < 1e-9 else a / b
    @staticmethod
    def evaluate(ticker_symbol, session):
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            ticker_base = str(ticker_symbol).split('.')[0]
            is_financial = ticker_base.startswith('28') or ticker_base.startswith('58') or ticker_base.startswith('60')
            try:
                if not is_financial and stock.info.get('sector') == 'Financial Services': is_financial = True
            except Exception: pass
            inc_stmt = FinancialAuditStrategy._normalize_statement(stock.quarterly_income_stmt)
            bal_sheet = FinancialAuditStrategy._normalize_statement(stock.quarterly_balance_sheet)
            cash_flow = FinancialAuditStrategy._normalize_statement(stock.quarterly_cashflow)
            if inc_stmt.empty and bal_sheet.empty and cash_flow.empty: return {"status": "error", "msg": f"無法取得 {ticker_symbol} 財報資料。"}
            
            revenue = FinancialAuditStrategy._get_series(inc_stmt, ["Total Revenue", "Operating Revenue", "Revenue"])
            net_income = FinancialAuditStrategy._get_series(inc_stmt, ["Net Income", "Net Income Common Stockholders"])
            operating_income = FinancialAuditStrategy._get_series(inc_stmt, ["Operating Income", "Operating Profit"])
            gross_profit = FinancialAuditStrategy._get_series(inc_stmt, ["Gross Profit"])
            cogs = FinancialAuditStrategy._get_series(inc_stmt, ["Cost Of Revenue", "Cost of Goods Sold", "Operating Cost"])
            cfo = FinancialAuditStrategy._get_series(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            ar = FinancialAuditStrategy._get_series(bal_sheet, ["Accounts Receivable", "Net Receivables"])
            inventory = FinancialAuditStrategy._get_series(bal_sheet, ["Inventory", "Inventories"])
            ap = FinancialAuditStrategy._get_series(bal_sheet, ["Accounts Payable", "Payables"])
            contract_liab = FinancialAuditStrategy._get_series(bal_sheet, ["Contract Liabilities", "Deferred Revenue", "Advances From Customers"])
            cash = FinancialAuditStrategy._get_series(bal_sheet, ["Cash And Cash Equivalents", "Cash Financial"])
            debt = FinancialAuditStrategy._get_series(bal_sheet, ["Total Debt", "Long Term Debt", "Current Debt"])
            t_assets = FinancialAuditStrategy._get_series(bal_sheet, ["Total Assets"])
            equity = FinancialAuditStrategy._get_series(bal_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

            all_periods = set()
            for df in [inc_stmt, bal_sheet, cash_flow]:
                if not df.empty: all_periods.update(df.columns)
            if not all_periods: return {"status": "error", "msg": "無財報期間資料"}
