import os
import json
import time
import requests
import datetime
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.signal import argrelextrema
from bs4 import BeautifulSoup
import urllib3

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
        return (c.isdigit() and len(c) == 4) or (c.startswith('00') and len(c) >= 4)

    try:
        r1 = session.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=15, verify=False)
        if r1.status_code == 200:
            for item in r1.json():
                c = str(item.get('Code', '')).strip()
                if is_valid_code(c):
                    stocks[c + '.TW'] = str(item.get('Name', '')).strip()
                
        r2 = session.get("https://www.tpex.org.tw/openapi/v1/mktdata/tpexod/BPSR", headers=headers, timeout=15, verify=False)
        if r2.status_code == 200:
            for item in r2.json():
                c = str(item.get('SecuritiesCompanyCode', '')).strip()
                if is_valid_code(c):
                    stocks[c + '.TWO'] = str(item.get('CompanyName', '')).strip()
        
        if len(stocks) > 1000:
            return stocks
    except Exception:
        pass

    try:
        r3 = session.get("https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo", headers=headers, timeout=15, verify=False)
        if r3.status_code == 200:
            for item in r3.json().get('data', []):
                c = str(item.get('stock_id', '')).strip()
                m = str(item.get('type', '')).strip().lower()
                if is_valid_code(c):
                    if m == 'twse':
                        stocks[c + '.TW'] = str(item.get('stock_name', '')).strip()
                    elif m == 'tpex':
                        stocks[c + '.TWO'] = str(item.get('stock_name', '')).strip()
        if len(stocks) > 1000:
            return stocks
    except Exception:
        pass
        
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
                            if is_valid_code(code):
                                stocks[code + suffix] = name
        if len(stocks) > 1000:
            return stocks
    except Exception:
        pass

    if not stocks:
        fallback_list = ['2330', '2317', '2454', '2308', '2382', '2881', '2891', '2412', '2882', '2886', '1303', '2884', '1216', '2002', '2892', '2303', '3711', '2885', '2880', '1301', '2345', '3008', '3231', '2357', '1101', '5880', '2883', '2887', '2603', '2324', '2379', '2890', '2912', '1326', '2395', '3034', '2207', '2408', '2609', '4938', '2801', '6669', '1402', '5871', '1590', '2301', '2356', '2448', '3037', '2383', '2618', '2610', '9904', '9910', '1102']
        for c in fallback_list:
            stocks[c + '.TW'] = "權值備援標的"

    return stocks

def fetch_tw_stock_name(ticker):
    code = ticker.split('.')[0]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5, verify=False)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1')
            if h1:
                return h1.text.strip()
    except Exception:
        pass
    try:
        return yf.Ticker(ticker, session=get_yf_session()).info.get('shortName', code)
    except Exception:
        return code

def fetch_stock_data(symbol):
    code = str(symbol).strip().upper()
    targets = [code] if code.endswith(".TW") or code.endswith(".TWO") else [f"{code}.TW", f"{code}.TWO"]
    session = get_yf_session()
    for ticker in targets:
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period="2y")
            if not df.empty:
                return df, ticker
        except Exception:
            continue
    return pd.DataFrame(), code

def resolve_ticker(symbol, stock_dict):
    code = str(symbol).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    if f"{code}.TW" in stock_dict:
        return f"{code}.TW"
    if f"{code}.TWO" in stock_dict:
        return f"{code}.TWO"
    return f"{code}.TW" 

class GoogleDriveUploader:
    @staticmethod
    def upload(file_name, webhook_url, token, file_path=None, csv_bytes=None, file_bytes=None):
        if not webhook_url or not webhook_url.startswith("https://script.google.com/"):
            return False, "無效的 Google Apps Script Webhook 網址"
        try:
            actual_bytes = file_bytes if file_bytes is not None else csv_bytes
            
            if actual_bytes is not None:
                file_data = actual_bytes
            elif file_path:
                with open(file_path, "rb") as f:
                    file_data = f.read()
            else:
                return False, "無檔案資料"
                
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            
            mime_type = "text/csv"
            if file_name.endswith('.xlsx'):
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
            payload = {
                "token": token, 
                "fileName": file_name, 
                "fileData": encoded_data,
                "mimeType": mime_type
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=30, verify=False)
            
            result = response.json()
            if result.get("status") == "success":
                return True, result.get("fileUrl", "成功")
            return False, result.get("message", "未知錯誤")
        except Exception as e:
            return False, str(e)

# ==========================================
# 指標與策略核心 
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
            if df.empty:
                return None
            close = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            basis_value = close 
            
            if close > ma20 and ma20 > ma60:
                regime = "🟢 多頭排列 (做多環境佳，可適度放大部位)"
            elif close < ma20 and ma20 < ma60:
                regime = "🔴 空頭弱勢 (系統性風險高，建議縮小部位或觀望)"
            else:
                regime = "🟡 震盪整理 (選股不選市，嚴格執行停損)"
                
            return {
                "加權指數收盤": f"{close:.2f}",
                "月線 (MA20)": f"{ma20:.2f}",
                "季線 (MA60)": f"{ma60:.2f}",
                "自動運算基準價值 (約當大台基礎)": f"{basis_value:.2f}",
                "大盤環境判定": regime
            }
        except Exception:
            return None

class ExpertTradingStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.DataFrame(index=df.index)
        if len(df) < 20:
            for c in ['RSI盤整', '多空指標數', '日KD多', '週KD多']:
                res[c] = 0 if c == '多空指標數' else False
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
        if pd.isna(bull_count):
            return "-"
        if bull_count == 3:
            return "7成 (同步偏多)"
        if bull_count == 2:
            return "5成 (偏多震盪)"
        if bull_count == 1:
            return "3成 (偏空震盪)"
        return "1成以下 (同步偏空)"
        
    @staticmethod
    def get_multi_tf_advice(w_bull, d_bull):
        if pd.isna(w_bull) or pd.isna(d_bull):
            return "-"
        if w_bull and d_bull:
            return "🟢 同步轉多 (勝率高)"
        if w_bull and not d_bull:
            return "🟡 短期回檔 (勿加碼)"
        if not w_bull and d_bull:
            return "🟠 跌深反彈 (非回升)"
        return "🔴 同步轉空 (觀望)"

class BottomReversalStrategy:
    @staticmethod
    def evaluate(df):
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        total_range = (df['High'] - df['Low']).replace(0, 0.001)
        
        vol_ma = df.get('Vol_MA20', df['Volume_Lots'].rolling(20).mean())
        vol_mult = (df['Volume_Lots'] / (vol_ma + 1e-8)).clip(0.5, 3.0)
        
        ma20 = df.get('MA20', df['Close'].rolling(20).mean())
        bias20 = (df['Close'] - ma20) / (ma20 + 1e-8) * 100
        
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
        std20 = df['Close'].rolling(20).std()
        
        bb_u = df.get('BB_Upper', ma20 + 2 * std20)
        bb_l = df.get('BB_Lower', ma20 - 2 * std20)
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
        if len(df) < recent_w + older_w:
            return False
            
        if require_cross_confirm:
            cross = (df[cross_col1] > df[cross_col2]) & (df[cross_col1].shift(1) <= df[cross_col2].shift(1))
            if not cross.iloc[-cross_days:].any():
                return False
                
        recent_start = len(df) - recent_w
        older_start = len(df) - recent_w - older_w
        older_end = len(df) - recent_w
        prices, inds = df[price_col].values, df[ind_col].values
        
        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            return (np.min(prices[recent_start:]) <= np.min(prices[older_start:older_end]) * 1.02) and (np.min(inds[recent_start:]) > np.min(inds[older_start:older_end]))
            
        def get_valid_pivots_iloc(start_loc, end_loc):
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s = max(0, i_loc - pivot_left)
                e = min(len(prices), i_loc + pivot_right + 1)
                if prices[i_loc] == np.min(prices[s:e]):
                    pivots.append(i_loc)
            return pivots
            
        r_pivots = get_valid_pivots_iloc(recent_start, len(df))
        o_pivots = get_valid_pivots_iloc(older_start, older_end)
        
        if recent_lows_cnt > 0:
            r_pivots = sorted(r_pivots, key=lambda x: prices[x])[:recent_lows_cnt]
        if older_lows_cnt > 0:
            o_pivots = sorted(o_pivots, key=lambda x: prices[x])[:older_lows_cnt]
            
        if not r_pivots:
            r_pivots = [recent_start + np.argmin(prices[recent_start:])]
        if not o_pivots:
            o_pivots = [older_start + np.argmin(prices[older_start:older_end])]
            
        for r_idx in r_pivots:
            for o_idx in o_pivots:
                r_i_val = np.min(inds[max(recent_start, r_idx-2) : min(len(inds), r_idx+3)])
                o_i_val = np.min(inds[max(older_start, o_idx-2) : min(older_end, o_idx+3)])
                if prices[r_idx] <= prices[o_idx] * 1.02 and r_i_val > o_i_val:
                    return True
                    
        return False

class IndicatorResonanceStrategy:
    @staticmethod
    def evaluate(df, recent_w=5, older_w=20, kd_older_low_th=20, kd_older_high_th=90, kd_recent_low_th=0.0, 
                 use_macd_abs=False, macd_older_low_th=0.0, macd_recent_low_th=0.0, cross_days=3,
                 require_price_higher_low=False, require_macd_cross_zero=False,
                 reso_price_basis="最低價 (Low)", use_macd_wave_logic=False, require_cross_confirm=True):
        if len(df) < recent_w + older_w:
            return pd.Series(0.0, index=df.index)
            
        price_col = 'Low' if reso_price_basis == "最低價 (Low)" else 'Close'
        recent_k_low = df['K'].rolling(window=recent_w, min_periods=1).min()
        older_k_low = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        older_k_high = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).max()
        
        if use_macd_wave_logic:
            sign = np.sign(df['MACD_Hist']).replace(0, np.nan).ffill().fillna(1)
            block_id = (sign != sign.shift(1)).cumsum().rename('block_id')
            neg_mask = sign < 0
            if not neg_mask.any():
                recent_macd_low = pd.Series(np.nan, index=df.index)
                older_macd_low = pd.Series(np.nan, index=df.index)
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
        if require_macd_cross_zero:
            macd_signal = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        else:
            macd_signal = ((df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))) | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))
            
        kd_recent_cross = kd_cross.rolling(window=cross_days, min_periods=1).max() >= 1
        macd_recent_cross = macd_signal.rolling(window=cross_days, min_periods=1).max() >= 1
        
        if require_cross_confirm:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th) & kd_recent_cross
            cond_macd = (recent_macd_low > older_macd_low) & macd_recent_cross
        else:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th)
            cond_macd = (recent_macd_low > older_macd_low)
            
        if use_macd_abs:
            cond_macd = cond_macd & (older_macd_low < macd_older_low_th) & (recent_macd_low > macd_recent_low_th)
            
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
        if len(df) < recent_w + older_w + 5:
            return res
            
        pct_change = (df['Close'].pct_change() * 100) if 'Pct_Change' not in df.columns else df['Pct_Change']
        cond_pct = pct_change.between(3.4, 10.5)
        cond_vol_abs = df['Volume_Lots'] >= 1000
        cond_vol_mult = (df['Volume_Lots'] / (df['Volume_Lots'].shift(1) + 1e-8)).between(2.4, 15.0) 
        
        valid_idx = np.where(cond_pct & cond_vol_abs & cond_vol_mult)[0]
        
        for idx in valid_idx:
            d_kd = False
            d_macd = False
            for buffer in range(0, 25):
                check_idx = idx - buffer
                if check_idx < recent_w + older_w:
                    continue
                sub_df = df.iloc[:check_idx+1]
                if not d_kd:
                    d_kd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'K', 'K', 'D', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if not d_macd:
                    d_macd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if d_kd and d_macd:
                    break
            if d_kd and d_macd:
                res.iloc[idx] = True
        return res

class LongLowerShadowBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 20:
            return res  
            
        df = df.copy()
        range_hl = (df['High'] - df['Low']).replace(0, 1e-8) 
        max_oc = df[['Open', 'Close']].max(axis=1)
        min_oc = df[['Open', 'Close']].min(axis=1)
        
        df['Upper_Shadow_Ratio'] = (df['High'] - max_oc) / range_hl
        df['Lower_Shadow_Ratio'] = (min_oc - df['Low']) / range_hl
        df['K_Amplitude'] = range_hl / df['Low'] * 100  
        
        if 'Pct_Change' not in df.columns:
            df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if 'BB_Lower' not in df.columns:
            df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(20).std()
            
        cond_today = (df['Close'] > df['Open']) & (df['Pct_Change'] >= 3.0) & (df['Upper_Shadow_Ratio'] < 0.40) & (df['Volume_Lots'] > df['Volume_Lots'].shift(1)) 
        target_k_mask = (df['Lower_Shadow_Ratio'] >= 0.30) & (df['Low'] < df['Low'].shift(1)) & (df['K_Amplitude'] >= 1.5) & (df['Low'] <= df['BB_Lower'] * 1.015) & (df['Close'] < df['MA20']) 
        
        for i in range(1, 11):
            valid = cond_today & target_k_mask.shift(i) & (df['Close'] > df['High'].shift(i))
            res = res | valid.fillna(False)
        return res

class MA240PullbackRightSideStrategy:
    @staticmethod
    def evaluate_phase1(df):
        if len(df) < 25:
            return pd.Series(False, index=df.index)
        close, open_p, low = df['Close'], df['Open'], df['Low']
        ma5, ma20 = close.rolling(5).mean(), close.rolling(20).mean()
        
        cond_red_ma5 = (close > open_p) & (close >= ma5)
        cond_base_support = low.rolling(3).min() >= low.rolling(10).min()
        cond_bias = ((close - ma20) / (ma20 + 1e-8) * 100).between(-8.0, 6.0)
        cond_ma20_stable = ma20 >= ma20.shift(3) * 0.985
        
        return (cond_red_ma5 & cond_base_support & cond_bias & cond_ma20_stable).fillna(False)

    @staticmethod
    def evaluate_phase2(long_df):
        if long_df is None or len(long_df) < 240:
            return {"is_pass": False, "score": 0.0}
            
        close = long_df['Close']
        open_p = long_df['Open']
        low = long_df['Low']
        vol = long_df['Volume']
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma240 = close.rolling(240).mean()
        
        cur_close = float(close.iloc[-1])
        cur_open = float(open_p.iloc[-1])
        cur_ma240 = float(ma240.iloc[-1])
        cur_ma60 = float(ma60.iloc[-1])
        cur_ma20 = float(ma20.iloc[-1])
        cur_ma5 = float(ma5.iloc[-1])
        
        ma240_5d_prev = float(ma240.iloc[-5])
        ma240_slope = (cur_ma240 - ma240_5d_prev) / (ma240_5d_prev + 1e-8) * 100
        
        score_ma240 = 0.0
        if ma240_slope > 0.04:
            score_ma240 = 3.5
        elif ma240_slope >= 0.0:
            score_ma240 = 2.5
        elif ma240_slope >= -0.03:
            score_ma240 = 1.0
            
        recent_15_low = float(low.iloc[-15:].min())
        tested_ma240 = (recent_15_low <= cur_ma240 * 1.035) and (recent_15_low >= cur_ma240 * 0.96)
        held_ma240 = cur_close >= cur_ma240 * 0.99
        
        score_pullback = 0.0
        if tested_ma240 and held_ma240:
            score_pullback = 4.0
        elif held_ma240 and recent_15_low <= cur_ma240 * 1.06:
            score_pullback = 2.5
        elif held_ma240:
            score_pullback = 1.0
            
        score_ma = 0.0
        if cur_ma20 > float(ma20.iloc[-3]):
            score_ma += 1.5
        if cur_close >= cur_ma20:
            score_ma += 1.0
        if cur_ma20 >= cur_ma60:
            score_ma += 0.5
            
        score_right = 0.0
        if cur_close > cur_open:
            score_right += 1.0
        if cur_close >= cur_ma5:
            score_right += 1.0
        if cur_close >= cur_ma240:
            score_right += 0.5
            
        vol_lots = vol / 1000.0
        cur_vol = float(vol_lots.iloc[-1])
        prev_vol = float(vol_lots.iloc[-2]) if len(vol_lots) > 1 else float(vol_lots.iloc[-1])
        
        score_vol = 0.0
        if cur_vol > prev_vol:
            score_vol += 1.0
        if cur_vol >= float(vol_lots.iloc[-20:].mean()) * 0.8:
            score_vol += 1.0
            
        total_score = round(score_ma240 + score_pullback + score_ma + score_right + score_vol, 2)
        is_pass = (score_ma240 >= 1.0) and held_ma240 and (score_pullback >= 2.5) and (total_score >= DEFAULT_CONFIG["min_ma240_score"])
        
        return {"is_pass": is_pass, "score": total_score}

    @staticmethod
    def evaluate_ma_entanglement(long_df):
        if long_df is None or len(long_df) < 240:
            return {
                "ma_long_diff_pct": np.nan, "ma_long_tangled": "無資料",
                "ma_short_diff_pct": np.nan, "ma_short_tangled": "無資料"
            }
        
        close = long_df['Close']
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        ma120 = close.rolling(120).mean().iloc[-1]
        ma240 = close.rolling(240).mean().iloc[-1]
        
        ma120_prev = close.rolling(120).mean().iloc[-5]
        ma240_prev = close.rolling(240).mean().iloc[-5]
        
        ma120_slope = (ma120 - ma120_prev) / (ma120_prev + 1e-8) * 100
        ma240_slope = (ma240 - ma240_prev) / (ma240_prev + 1e-8) * 100
        
        # 第一組：長天期 (60, 120, 240)
        long_mas = [ma60, ma120, ma240]
        if any(pd.isna(long_mas)):
            long_diff_pct = np.nan
            long_tangled = "無資料"
        else:
            long_max = max(long_mas)
            long_min = min(long_mas)
            long_diff_pct = (long_max - long_min) / (long_min + 1e-8) * 100
            
            is_120_flat = abs(ma120_slope) < 0.05
            is_240_flat = abs(ma240_slope) < 0.05
            
            # 原始糾結判定邏輯移轉至此，參考長天期均線
            long_tangled = "均線糾結" if (long_diff_pct < 6.0 and is_120_flat and is_240_flat) else "否"

        # 第二組：短天期 (5, 10, 20)
        short_mas = [ma5, ma10, ma20]
        if any(pd.isna(short_mas)):
            short_diff_pct = np.nan
            short_tangled = "無資料"
        else:
            short_max = max(short_mas)
            short_min = min(short_mas)
            short_diff_pct = (short_max - short_min) / (short_min + 1e-8) * 100
            # 短天期糾結判定 (條件放寬為單純差異小於 4%)
            short_tangled = "均線糾結" if short_diff_pct < 4.0 else "否"
            
        return {
            "ma_long_diff_pct": round(long_diff_pct, 2) if pd.notna(long_diff_pct) else np.nan,
            "ma_long_tangled": long_tangled,
            "ma_short_diff_pct": round(short_diff_pct, 2) if pd.notna(short_diff_pct) else np.nan,
            "ma_short_tangled": short_tangled
        }

class LowPositionShrinkVolumeBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 62:
            return res
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
        if long_df is None or len(long_df) < 20:
            return {"週K共振": "無資料", "週K背離": "無資料"}
        try:
            df = long_df.copy()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            w_df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            
            if len(w_df) < 20:
                return {"週K共振": "資料不足", "週K背離": "資料不足"}
                
            w_df = TechnicalIndicators.add_macd(TechnicalIndicators.add_kd(w_df))
            last_row = w_df.iloc[-1]
            kd_bull = last_row['K'] > last_row['D']
            macd_bull = last_row['MACD_Hist'] > 0
            
            if kd_bull and macd_bull:
                reso_str = "✅ 雙共振"
            elif kd_bull:
                reso_str = "KD偏多"
            elif macd_bull:
                reso_str = "MACD偏多"
            else:
                reso_str = "❌ 無"
                
            div_res = []
            for r_w, o_w in div_pairs:
                if DivergenceStrategy.check_bottom_divergence(w_df, price_col, 'K', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days):
                    div_res.append(f"KD({r_w},{o_w})")
                if DivergenceStrategy.check_bottom_divergence(w_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days):
                    div_res.append(f"MACD({r_w},{o_w})")
                    
            div_str = "+".join(list(dict.fromkeys(div_res))) if div_res else "無"
            return {"週K共振": reso_str, "週K背離": div_str}
        except Exception:
            return {"週K共振": "錯誤", "週K背離": "錯誤"}

class AdvancedBottomStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.DataFrame(index=df.index, columns=[
            'Adv_Signal_A', 'Adv_Signal_B', 'Adv_Low_1', 'Adv_Low_2',
            'Adv_Neckline_1', 'Adv_Support_RS', 'Adv_Target_1x', 'Adv_Target_3x'
        ])
        for c in ['Adv_Signal_A', 'Adv_Signal_B']: res[c] = False
        for c in ['Adv_Low_1', 'Adv_Low_2', 'Adv_Neckline_1', 'Adv_Support_RS', 'Adv_Target_1x', 'Adv_Target_3x']: res[c] = np.nan

        if len(df) < 30:
            return res

        vol_ma5 = df.get('Volume_Lots', df['Volume'] / 1000.0).rolling(5).mean().values
        ma20 = df.get('MA20', df['Close'].rolling(20).mean())
        bb_lower = df.get('BB_Lower', ma20 - 2 * df['Close'].rolling(20).std()).values

        lows = df['Low'].values
        highs = df['High'].values
        closes = df['Close'].values
        opens = df['Open'].values
        k_vals = df.get('K', pd.Series(50, index=df.index)).values
        d_vals = df.get('D', pd.Series(50, index=df.index)).values
        macd_h = df.get('MACD_Hist', pd.Series(0, index=df.index)).values
        macd = df.get('MACD', pd.Series(0, index=df.index)).values
        vol = df.get('Volume_Lots', df['Volume'] / 1000.0).values
        
        pivot_lows_idx = argrelextrema(lows, np.less, order=5)[0]
        pivot_highs_idx = argrelextrema(highs, np.greater, order=5)[0]

        for i in range(15, len(df)):
            valid_lows = [idx for idx in pivot_lows_idx if idx < i - 1]
            if len(valid_lows) < 2: continue
            idx_L1, idx_L2 = valid_lows[-2], valid_lows[-1]
            
            if lows[idx_L2] >= lows[idx_L1]: continue

            kd_div = (k_vals[idx_L2] > k_vals[idx_L1]) or (d_vals[idx_L2] > d_vals[idx_L1])
            macd_div = (macd_h[idx_L2] > macd_h[idx_L1]) or (macd[idx_L2] > macd[idx_L1])
            if not (kd_div and macd_div): continue

            valid_highs = [idx for idx in pivot_highs_idx if idx < idx_L1]
            if valid_highs:
                neckline_1 = highs[valid_highs[-1]]
            else:
                neckline_1 = np.max(highs[max(0, idx_L1-10):idx_L1+1])

            if idx_L2 > idx_L1 + 1:
                diffs = np.diff(closes[idx_L1:idx_L2+1])
                if len(diffs) > 0:
                    drop_idx = idx_L1 + np.argmin(diffs)
                    sup_rs = min(closes[drop_idx], opens[drop_idx])
                else:
                    sup_rs = min(closes[idx_L1], opens[idx_L1])
            else:
                sup_rs = min(closes[idx_L1], opens[idx_L1])

            red_candles = sum(1 for j in range(idx_L2, i+1) if closes[j] > opens[j])

            sig_A = (closes[i] > neckline_1) and (vol[i] > vol_ma5[i]) and \
                    (k_vals[i] >= 20) and (closes[i] > opens[i]) and \
                    (red_candles >= 2)

            sig_A_history = any((closes[j] > neckline_1 and vol[j] > vol_ma5[j] and closes[j] > opens[j] and k_vals[j] >= 20) for j in range(idx_L2, i))
            
            sig_B = False
            if sig_A_history:
                in_zone = (lows[i] <= sup_rs) and (lows[i] >= lows[idx_L1] * 0.95)
                near_bb = (lows[i] <= bb_lower[i] * 1.02)
                if (in_zone or near_bb) and (closes[i] >= lows[idx_L2]) and (vol[i] < vol_ma5[i]):
                    sig_B = True

            if sig_A or sig_B:
                res.iloc[i, res.columns.get_loc('Adv_Signal_A')] = sig_A
                res.iloc[i, res.columns.get_loc('Adv_Signal_B')] = sig_B
                res.iloc[i, res.columns.get_loc('Adv_Low_1')] = lows[idx_L1]
                res.iloc[i, res.columns.get_loc('Adv_Low_2')] = lows[idx_L2]
                res.iloc[i, res.columns.get_loc('Adv_Neckline_1')] = neckline_1
                res.iloc[i, res.columns.get_loc('Adv_Support_RS')] = sup_rs

                neckline_2 = np.max(highs[idx_L2:i+1])
                res.iloc[i, res.columns.get_loc('Adv_Target_1x')] = neckline_2 + (neckline_2 - lows[idx_L2])
                res.iloc[i, res.columns.get_loc('Adv_Target_3x')] = neckline_2 + (neckline_2 - lows[idx_L2]) * 3

        return res

# ==========================================
# 鬆散耦合架構：Scan 演算法註冊中心 (Registry)
# ==========================================
class IScanAlgorithm:
    @property
    def name(self) -> str:
        raise NotImplementedError
    def calculate(self, df, config) -> pd.DataFrame:
        return df
    def check_pass(self, row, config) -> tuple[bool, float]:
        return False, 0.0
    def get_tags(self, row, config) -> list[str]:
        return []
    def get_metrics(self, row, adv_src=None) -> dict:
        return {}

class AlgoBottomReversal(IScanAlgorithm):
    @property
    def name(self): return '底部翻轉'
    def calculate(self, df, config):
        df['Candle_Score'] = BottomReversalStrategy.evaluate(df)
        return df
    def check_pass(self, row, config):
        score = row.get('Candle_Score', 0)
        return score >= config.get("min_score", 8.0), score
    def get_tags(self, row, config):
        val = row.get('Candle_Score', 0) if isinstance(row, pd.Series) else row.get('反轉分數', 0)
        return ["底部翻轉"] if val >= config.get("min_score", 8.0) else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('Candle_Score', 0) if isinstance(row, pd.Series) else row.get('反轉分數', 0)
        return {"反轉分數": round(float(val), 2) if pd.notna(val) else 0.0}

class AlgoVCP(IScanAlgorithm):
    @property
    def name(self): return 'VCP'
    def calculate(self, df, config):
        df['VCP_Score'] = VCPStrategy.evaluate(df)
        return df
    def check_pass(self, row, config):
        score = row.get('VCP_Score', 0)
        return score >= config.get("min_vcp_score", 10.0), score
    def get_tags(self, row, config):
        val = row.get('VCP_Score', 0) if isinstance(row, pd.Series) else row.get('VCP分數', 0)
        return ["VCP收斂"] if val >= config.get("min_vcp_score", 10.0) else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('VCP_Score', 0) if isinstance(row, pd.Series) else row.get('VCP分數', 0)
        return {"VCP分數": round(float(val), 2) if pd.notna(val) else 0.0}

class AlgoIndicatorResonance(IScanAlgorithm):
    @property
    def name(self): return '指標共振'
    def calculate(self, df, config):
        df['Reso_Score'] = IndicatorResonanceStrategy.evaluate(
            df, recent_w=config.get("div_recent_w"), older_w=config.get("div_older_w"),
            kd_older_low_th=config.get("reso_kd_older_low"), kd_older_high_th=config.get("reso_kd_older_high"),
            kd_recent_low_th=config.get("reso_kd_recent_low"), use_macd_abs=config.get("use_macd_abs"),
            macd_older_low_th=config.get("reso_macd_older_low"), macd_recent_low_th=config.get("reso_macd_recent_low"),
            cross_days=config.get("reso_cross_days"), require_price_higher_low=config.get("reso_price_higher_low"),
            require_macd_cross_zero=config.get("reso_macd_cross_zero"), reso_price_basis=config.get("reso_price_basis"),
            use_macd_wave_logic=config.get("reso_macd_wave_logic"), require_cross_confirm=config.get("require_cross_confirm")
        )
        return df
    def check_pass(self, row, config):
        score = row.get('Reso_Score', 0)
        return score >= config.get("min_reso_score", 10.0), score
    def get_tags(self, row, config):
        val = row.get('Reso_Score', 0) if isinstance(row, pd.Series) else row.get('共振分數', 0)
        return ["指標共振"] if val >= config.get("min_reso_score", 10.0) else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('Reso_Score', 0) if isinstance(row, pd.Series) else row.get('共振分數', 0)
        return {"共振分數": round(float(val), 2) if pd.notna(val) else 0.0}

class AlgoPriceVolDiv(IScanAlgorithm):
    @property
    def name(self): return '量價雙背離'
    def calculate(self, df, config):
        price_col = 'Low' if config.get("reso_price_basis") == "最低價 (Low)" else 'Close'
        df['PriceVol_Div_Pass'] = PriceVolDivergenceStrategy.evaluate(
            df, recent_w=config.get("div_recent_w"), older_w=config.get("div_older_w"),
            rl_cnt=config.get("recent_lows_cnt"), ol_cnt=config.get("older_lows_cnt"),
            p_left=config.get("pivot_left"), p_right=config.get("pivot_right"),
            req_cross=config.get("require_cross_confirm"), c_days=config.get("reso_cross_days"), price_col=price_col
        )
        return df
    def check_pass(self, row, config):
        passed = bool(row.get('PriceVol_Div_Pass', False))
        return passed, 20.0 if passed else 0.0
    def get_tags(self, row, config):
        val = row.get('PriceVol_Div_Pass', False) if isinstance(row, pd.Series) else (row.get('量價雙背離') == '✅ 是')
        return ["量價雙背離"] if val else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('PriceVol_Div_Pass', False) if isinstance(row, pd.Series) else (row.get('量價雙背離') == '✅ 是')
        return {"量價雙背離": "✅ 是" if val else "❌ 否"}

class AlgoLongLowerShadowBreakout(IScanAlgorithm):
    @property
    def name(self): return '長下引突破'
    def calculate(self, df, config):
        df['Breakout_Signal'] = LongLowerShadowBreakoutStrategy.evaluate(df)
        return df
    def check_pass(self, row, config):
        passed = bool(row.get('Breakout_Signal', False))
        return passed, 10.0 if passed else 0.0
    def get_tags(self, row, config):
        val = row.get('Breakout_Signal', False) if isinstance(row, pd.Series) else (row.get('長下引突破') == '✅ 是')
        return ["長下引突破"] if val else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('Breakout_Signal', False) if isinstance(row, pd.Series) else (row.get('長下引突破') == '✅ 是')
        return {"長下引突破": "✅ 是" if val else "❌ 否"}

class AlgoMA240Pullback(IScanAlgorithm):
    @property
    def name(self): return '*回測走揚年線'
    def calculate(self, df, config):
        df['MA240_P1_Pass'] = MA240PullbackRightSideStrategy.evaluate_phase1(df)
        return df
    def check_pass(self, row, config):
        passed = bool(row.get('MA240_P1_Pass', False))
        return passed, 12.0 if passed else 0.0
    def get_tags(self, row, config):
        return ["*回測走揚年線"] if row.get("*回測走揚年線") == "✅ 是" else []
    def get_metrics(self, row, adv_src=None):
        return {"*回測走揚年線": "❌ 否", "*年線回測分數": 0.0}

class AlgoLowPosBreakout(IScanAlgorithm):
    @property
    def name(self): return '低位長下引量縮突破'
    def calculate(self, df, config):
        df['LowPos_Breakout_Pass'] = LowPositionShrinkVolumeBreakoutStrategy.evaluate(df)
        return df
    def check_pass(self, row, config):
        passed = bool(row.get('LowPos_Breakout_Pass', False))
        return passed, 15.0 if passed else 0.0
    def get_tags(self, row, config):
        val = row.get('LowPos_Breakout_Pass', False) if isinstance(row, pd.Series) else (row.get('低位量縮突破') == '✅ 是')
        return ["低位量縮突破"] if val else []
    def get_metrics(self, row, adv_src=None):
        val = row.get('LowPos_Breakout_Pass', False) if isinstance(row, pd.Series) else (row.get('低位量縮突破') == '✅ 是')
        return {"低位量縮突破": "✅ 是" if val else "❌ 否"}

class AlgoAdvancedBottom(IScanAlgorithm):
    @property
    def name(self): return '雙底正背離與右肩回測'
    def calculate(self, df, config):
        adv_res = AdvancedBottomStrategy.evaluate(df)
        for col in adv_res.columns:
            df[col] = adv_res[col]
        return df
    def check_pass(self, row, config):
        sig_a = bool(row.get('Adv_Signal_A', False))
        sig_b = bool(row.get('Adv_Signal_B', False))
        combo = 0.0
        if sig_a: combo += 25.0
        if sig_b: combo += 30.0
        return (sig_a or sig_b), combo
    def get_tags(self, row, config):
        tags = []
        a_val = row.get("Adv_Signal_A")
        b_val = row.get("Adv_Signal_B")
        if a_val == "✅ 是" or (isinstance(a_val, bool) and a_val):
            tags.append("雙底突破(底倉)")
        if b_val == "✅ 是" or (isinstance(b_val, bool) and b_val):
            tags.append("右肩回測(加碼)")
        return tags
    def get_metrics(self, row, adv_src=None):
        src = adv_src if adv_src is not None else row
        def safe_round(val):
            return round(float(val), 2) if pd.notna(val) else "-"
        return {
            "左腳(Low1)": safe_round(src.get('Adv_Low_1', np.nan)),
            "破底(Low2/停損)": safe_round(src.get('Adv_Low_2', np.nan)),
            "上影線頸線": safe_round(src.get('Adv_Neckline_1', np.nan)),
            "右肩回測價": safe_round(src.get('Adv_Support_RS', np.nan)),
            "預估1段目標": safe_round(src.get('Adv_Target_1x', np.nan)),
            "預估3段目標": safe_round(src.get('Adv_Target_3x', np.nan))
        }

ALGO_REGISTRY = [
    AlgoBottomReversal(),
    AlgoVCP(),
    AlgoIndicatorResonance(),
    AlgoPriceVolDiv(),
    AlgoLongLowerShadowBreakout(),
    AlgoMA240Pullback(),
    AlgoLowPosBreakout(),
    AlgoAdvancedBottom()
]

SCAN_ALGO_CHOICES = ['全部'] + [algo.name for algo in ALGO_REGISTRY]

class ScanEngine:
    @staticmethod
    def calculate_all_phase1_scores(df, config):
        for algo in ALGO_REGISTRY:
            df = algo.calculate(df, config)
        return df

    @staticmethod
    def check_algo_pass(row, config, algo_mode, force_pass=False):
        passed = force_pass
        combo_score = 0
        for algo in ALGO_REGISTRY:
            is_pass, score = algo.check_pass(row, config)
            combo_score += score
            if (algo_mode == '全部' or algo_mode == algo.name) and is_pass:
                passed = True
        return passed, combo_score

    @staticmethod
    def generate_base_tags(item, config):
        base_tags = []
        for algo in ALGO_REGISTRY:
            base_tags.extend(algo.get_tags(item, config))
        return base_tags
        
    @staticmethod
    def get_all_metrics(row, adv_src=None):
        metrics = {}
        for algo in ALGO_REGISTRY:
            metrics.update(algo.get_metrics(row, adv_src))
        return metrics
        
    @staticmethod
    def get_all_metric_keys():
        keys = []
        dummy_row = pd.Series(dtype=float)
        for algo in ALGO_REGISTRY:
            keys.extend(algo.get_metrics(dummy_row, dummy_row).keys())
        return list(dict.fromkeys(keys))


class FinancialAuditStrategy:
    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value) or value is None:
                return np.nan
            return float(value)
        except Exception:
            return np.nan
            
    @staticmethod
    def _find_row(df, candidates):
        if df is None or df.empty:
            return None
        index_map = {str(x).strip().lower(): x for x in df.index}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in index_map:
                return index_map[key]
        for candidate in candidates:
            key = str(candidate).strip().lower()
            for idx in df.index:
                if key in str(idx).strip().lower():
                    return idx
        return None
        
    @staticmethod
    def _get_series(df, candidates):
        row = FinancialAuditStrategy._find_row(df, candidates)
        if row is None:
            return pd.Series(dtype=float)
        s = df.loc[row].copy()
        return pd.Series([FinancialAuditStrategy._safe_float(x) for x in s.values], index=s.index, dtype=float)
        
    @staticmethod
    def _normalize_statement(df):
        if df is None or df.empty:
            return pd.DataFrame()
        result = df.copy()
        new_columns = []
        for col in result.columns:
            try:
                dt = pd.to_datetime(col)
                if hasattr(dt, "tz_localize") and dt.tz is not None:
                    dt = dt.tz_localize(None)
                new_columns.append(dt)
            except Exception:
                new_columns.append(col)
        result.columns = new_columns
        try:
            result = result.loc[:, sorted(result.columns)]
        except Exception:
            pass
        return result.loc[:, ~pd.Index(result.columns).duplicated()]
        
    @staticmethod
    def _growth(current, previous):
        if pd.isna(current) or pd.isna(previous) or abs(previous) < 1e-9:
            return np.nan
        return (current - previous) / abs(previous) * 100
        
    @staticmethod
    def _ratio(a, b):
        if pd.isna(a) or pd.isna(b) or abs(b) < 1e-9:
            return np.nan
        return a / b
        
    @staticmethod
    def evaluate(ticker_symbol, session):
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            ticker_base = str(ticker_symbol).split('.')[0]
            is_financial = False
            
            if ticker_base.startswith(('28', '58', '60')):
                is_financial = True
            else:
                try:
                    if stock.info.get('sector') == 'Financial Services':
                        is_financial = True
                except Exception:
                    pass
                    
            inc_stmt = FinancialAuditStrategy._normalize_statement(stock.quarterly_income_stmt)
            bal_sheet = FinancialAuditStrategy._normalize_statement(stock.quarterly_balance_sheet)
            cash_flow = FinancialAuditStrategy._normalize_statement(stock.quarterly_cashflow)
            
            if inc_stmt.empty and bal_sheet.empty and cash_flow.empty:
                return {"status": "error", "msg": f"無法取得 {ticker_symbol} 財報資料。"}
            
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
            for df_stmt in [inc_stmt, bal_sheet, cash_flow]:
                if not df_stmt.empty:
                    all_periods.update(df_stmt.columns)
                    
            if not all_periods:
                return {"status": "error", "msg": "無財報期間資料"}
            
            sorted_periods = sorted(list(all_periods))
            periods_8q = sorted_periods[-8:] if len(sorted_periods) >= 8 else sorted_periods
            periods_4q = sorted_periods[-4:] if len(sorted_periods) >= 4 else sorted_periods
            latest = sorted_periods[-1]
            q1_ago = sorted_periods[-2] if len(sorted_periods) >= 2 else None
            q4_ago = sorted_periods[-5] if len(sorted_periods) >= 5 else None

            def val_at(s, p):
                if not s.empty and p in s.index:
                    return FinancialAuditStrategy._safe_float(s.loc[p])
                return np.nan
                
            def get_trend(s, periods):
                return [val_at(s, p) for p in periods]
            
            v_rev = val_at(revenue, latest)
            v_ni = val_at(net_income, latest)
            v_cfo = val_at(cfo, latest)
            v_gp = val_at(gross_profit, latest)
            v_op = val_at(operating_income, latest)
            v_cogs = val_at(cogs, latest)
            v_ar = val_at(ar, latest)
            v_inv = val_at(inventory, latest)
            v_cl = val_at(contract_liab, latest)
            v_cash = val_at(cash, latest)
            v_debt = val_at(debt, latest) if pd.notna(val_at(debt, latest)) else 0.0
            v_assets = val_at(t_assets, latest)
            v_equity = val_at(equity, latest)

            t4_rev = get_trend(revenue, periods_4q)
            t4_ni = get_trend(net_income, periods_4q)
            t4_cfo = get_trend(cfo, periods_4q)
            t4_ar = get_trend(ar, periods_4q)
            t4_inv = get_trend(inventory, periods_4q)
            t4_cl = get_trend(contract_liab, periods_4q)
            t8_cl = get_trend(contract_liab, periods_8q)

            rev_qoq = FinancialAuditStrategy._growth(v_rev, val_at(revenue, q1_ago))
            rev_yoy = FinancialAuditStrategy._growth(v_rev, val_at(revenue, q4_ago))
            ni_qoq = FinancialAuditStrategy._growth(v_ni, val_at(net_income, q1_ago))
            ni_yoy = FinancialAuditStrategy._growth(v_ni, val_at(net_income, q4_ago))
            cfo_qoq = FinancialAuditStrategy._growth(v_cfo, val_at(cfo, q1_ago))
            cfo_yoy = FinancialAuditStrategy._growth(v_cfo, val_at(cfo, q4_ago))
            cl_qoq = FinancialAuditStrategy._growth(v_cl, val_at(contract_liab, q1_ago))
            cl_yoy = FinancialAuditStrategy._growth(v_cl, val_at(contract_liab, q4_ago))
            ar_qoq = FinancialAuditStrategy._growth(v_ar, val_at(ar, q1_ago))
            ar_yoy = FinancialAuditStrategy._growth(v_ar, val_at(ar, q4_ago))
            inv_qoq = FinancialAuditStrategy._growth(v_inv, val_at(inventory, q1_ago))
            inv_yoy = FinancialAuditStrategy._growth(v_inv, val_at(inventory, q4_ago))

            gm_latest = FinancialAuditStrategy._ratio(v_gp, v_rev) * 100
            om_latest = FinancialAuditStrategy._ratio(v_op, v_rev) * 100
            cfo_ni_ratio = FinancialAuditStrategy._ratio(v_cfo, v_ni)
            cl_rev_ratio = FinancialAuditStrategy._ratio(v_cl, v_rev) * 100
            roe_annual = (v_ni / v_equity * 4 * 100) if pd.notna(v_ni) and pd.notna(v_equity) and v_equity > 0 else np.nan
            roa_annual = (v_ni / v_assets * 4 * 100) if pd.notna(v_ni) and pd.notna(v_assets) and v_assets > 0 else np.nan
            equity_multiplier = (v_assets / v_equity) if pd.notna(v_assets) and pd.notna(v_equity) and v_equity > 0 else np.nan
            dso = (v_ar / v_rev) * 90 if (pd.notna(v_rev) and v_rev > 0 and pd.notna(v_ar)) else np.nan
            
            if pd.notna(v_cogs) and v_cogs > 0 and pd.notna(v_inv):
                inv_days = v_inv / v_cogs * 90
            elif pd.notna(v_rev) and v_rev > 0 and pd.notna(v_inv):
                inv_days = v_inv / v_rev * 90
            else:
                inv_days = np.nan

            cross_analysis = []
            risk_resonance = []
            positives = []
            watch_list = []
            quality_score = 50
            risk_score = 20
            
            has_cl = not contract_liab.empty and not pd.isna(v_cl)

            if pd.notna(v_ni) and v_ni < 0:
                quality_score -= 15
                risk_score += 15
            if pd.notna(v_op) and v_op < 0:
                quality_score -= 10
                risk_score += 10
                
            if ((pd.notna(v_op) and v_op < 0) or (pd.notna(v_ni) and v_ni < 0)) and (pd.notna(v_cfo) and v_cfo < 0):
                risk_score += 25
                quality_score -= 15
                risk_resonance.append("🚨【本業造血枯竭】本業營運虧損且營運現金流持續淨流出。")

            loss_count = sum([1 for val in t4_ni if pd.notna(val) and val < 0])
            if loss_count >= 2:
                risk_score += 15
                quality_score -= 10
                risk_resonance.append("🚨【長期營運虧損】近四季多次出現虧損。")

            cl_status = "無資料"
            cl_state = "None"
            
            if has_cl:
                cl_qoq_val = cl_qoq if pd.notna(cl_qoq) else 0
                cl_yoy_val = cl_yoy if pd.notna(cl_yoy) else 0
                rev_qoq_val = rev_qoq if pd.notna(rev_qoq) else 0
                rev_yoy_val = rev_yoy if pd.notna(rev_yoy) else 0
                inv_inc = (pd.notna(inv_qoq) and inv_qoq > 0) or (pd.notna(inv_yoy) and inv_yoy > 0)
                
                if (cl_yoy_val > 10 or cl_qoq_val > 10) and inv_inc:
                    cl_state = "A"
                    cl_status = "🚀 正向擴張 (接單強勁，營運資金積極備料)"
                    cross_analysis.append("【合約負債】客戶預付款與遞延收入高增長伴隨存貨增加，具備未來營收潛力。")
                elif (cl_qoq_val < 0 or cl_yoy_val < -5) and (rev_qoq_val > 0 or rev_yoy_val > 0):
                    cl_state = "B"
                    cl_status = "🟢 履約轉化 (前期訂單陸續驗收並轉列營收)"
                elif cl_yoy_val < -15 and rev_yoy_val <= 0:
                    cl_state = "C"
                    cl_status = "🔴 可能異常 (訂單動能萎縮)"
                    risk_resonance.append("🚨【訂單動能衰退】合約負債大幅減少且營收熄火。")
                elif cl_yoy_val > 5:
                    cl_status = "🟢 正向 (訂單/預付款穩健增加)"
                else:
                    cl_status = "⚪ 中性波動"
            else:
                cl_status = "⚪ 合約負債資料不足"

            inv_surge = (pd.notna(inv_qoq) and inv_qoq > 15) or (pd.notna(inv_yoy) and inv_yoy > 20)
            has_rev_support = (pd.notna(rev_qoq) and rev_qoq > 10) or (pd.notna(rev_yoy) and rev_yoy > 10)
            rev_declining = (pd.notna(rev_qoq) and rev_qoq < 0) or (pd.notna(rev_yoy) and rev_yoy < 0)

            if inv_surge:
                if cl_state == "A" or has_rev_support:
                    quality_score += 10
                    risk_score -= 10
                    positives.append("存貨大幅增加具備「合約負債」或「營收」成長支撐，屬健康備料。")
                elif cl_state == "C" or (rev_declining and pd.notna(v_cfo) and v_cfo < 0):
                    risk_score += 25
                    risk_resonance.append("🚨【存貨去化危機】營收轉弱且缺乏訂單支撐，存貨卻大幅上升。")
                else:
                    watch_list.append("存貨急升需留意後續營收是否跟上。")

            is_growth_expansion = False
            is_black_bankruptcy = False
            
            if pd.notna(v_ni) and v_ni > 0 and pd.notna(v_cfo) and v_cfo < 0:
                rev_stalled = (pd.notna(rev_qoq) and rev_qoq <= 5) and (pd.notna(rev_yoy) and rev_yoy <= 0)
                if rev_stalled and cl_state != "A" and not inv_surge:
                    if sum([1 for val in t4_cfo if pd.notna(val) and val < 0]) >= 2:
                        is_black_bankruptcy = True
                        risk_score += 30
                        risk_resonance.append("🚨【黑字倒閉/流動性風險】帳面獲利但營業現金流持續為負，無擴張支撐。")
                else:
                    if cl_state == "A" or (has_rev_support and ((pd.notna(inv_qoq) and inv_qoq > 0) or (pd.notna(ar_qoq) and ar_qoq > 0))):
                        is_growth_expansion = True
                        quality_score += 15
                        risk_score -= 10
                        positives.append("🟢 成長型營運資金擴張 (CFO為負純屬接單備料墊高營運資金)。")

            quality_score = min(100, max(0, int(quality_score)))
            risk_score = min(100, max(0, int(risk_score)))
            if is_growth_expansion:
                risk_score = min(risk_score, 35)

            if is_growth_expansion and quality_score >= 70 and risk_score <= 35:
                overall_grade = "🚀 擴張成長 (大單備料期)"
            elif risk_score >= 60:
                overall_grade = "🔴 高度風險 (地雷特徵明顯)"
            elif risk_score >= 40:
                overall_grade = "🟠 觀察警戒 (財務結構存在弱點)"
            elif quality_score >= 75 and risk_score < 30:
                overall_grade = "💎 卓越績優 (護城河深厚)"
            else:
                overall_grade = "🟡 穩健平庸"

            def fmt(v, suffix=""):
                return f"{v:,.2f}{suffix}" if pd.notna(v) else "-"
            
            if not cross_analysis:
                cross_analysis.append("未偵測到明顯的三表交叉異常或正向特徵。")
            if not risk_resonance:
                risk_resonance.append("目前三表結構中，無重大風險共振鏈。")
            if not positives:
                positives.append("無顯著突出之護城河優勢。")
            if not watch_list:
                watch_list.append("持續觀察基本面變化與毛利率維持能力。")

            return {
                "status": "success",
                "代號": ticker_symbol, 
                "當期季報": latest.strftime("%Y-%m-%d") if hasattr(latest, "strftime") else str(latest),
                "比較基準": "最近 4~8 季綜合評估",
                "綜合評級": overall_grade, 
                "體質分數": quality_score,
                "風險分數": risk_score,
                "core_metrics": [
                    {"指標": "營業收入", "最新": fmt(v_rev), "QoQ": fmt(rev_qoq, "%"), "YoY": fmt(rev_yoy, "%"), "4Q趨勢": t4_rev},
                    {"指標": "稅後淨利", "最新": fmt(v_ni), "QoQ": fmt(ni_qoq, "%"), "YoY": fmt(ni_yoy, "%"), "4Q趨勢": t4_ni},
                    {"指標": "營業現金流", "最新": fmt(v_cfo), "QoQ": fmt(cfo_qoq, "%"), "YoY": fmt(cfo_yoy, "%"), "4Q趨勢": t4_cfo},
                    {"指標": "合約負債", "最新": fmt(v_cl), "QoQ": fmt(cl_qoq, "%"), "YoY": fmt(cl_yoy, "%"), "4Q趨勢": t4_cl},
                ],
                "cl_latest": fmt(v_cl), "cl_qoq": fmt(cl_qoq, "%"), "cl_yoy": fmt(cl_yoy, "%"), 
                "cl_rev_ratio": fmt(cl_rev_ratio, "%"), "cl_status": cl_status, "cl_8q": t8_cl,
                "cross_analysis": cross_analysis, "risk_resonance": risk_resonance, 
                "positives": positives, "watch_list": watch_list,
                "ROE年化(%)": roe_annual, "ROA年化(%)": roa_annual, "營收成長(%)": rev_qoq,
                "毛利率": gm_latest, "營益率": om_latest, "CFO/淨利": cfo_ni_ratio,
                "CCC_now": inv_days + dso if pd.notna(inv_days) and pd.notna(dso) else np.nan,
                "DSO_now": dso, "DIO_now": inv_days,
                "警訊數": len(risk_resonance) if risk_resonance[0] != "目前三表結構中，無重大風險共振鏈。" else 0,
                "正向訊號": " | ".join(positives),
                "診斷明細": " | ".join(risk_resonance)
            }
        except Exception as e:
            return {"status": "error", "msg": f"發生錯誤：{str(e)}"}

class GeometricPatternStrategy:
    @staticmethod
    def calculate_geometric_patterns(df, high_col, low_col, window_order=5):
        high_idx = argrelextrema(df[high_col].values, np.greater, order=window_order)[0]
        low_idx = argrelextrema(df[low_col].values, np.less, order=window_order)[0]
        result = {"bull": None, "bear": None}
        
        if len(low_idx) >= 2 and len(high_idx) >= 1:
            v1_idx, v2_idx = low_idx[-2], low_idx[-1]
            trough1, trough2 = df[low_col].iloc[v1_idx], df[low_col].iloc[v2_idx]
            between_highs = df[high_col].iloc[v1_idx:v2_idx]
            neckline = between_highs.max() if not between_highs.empty else df[high_col].iloc[high_idx[-1]]
            w_height = neckline - min(trough1, trough2)
            result["bull"] = {
                "trough1": trough1, "date1": df.index[v1_idx].strftime('%m/%d'),
                "trough2": trough2, "date2": df.index[v2_idx].strftime('%m/%d'),
                "neckline": neckline, "height": w_height,
                "target_1to1": neckline + w_height, "target_1618": neckline + (w_height * 1.618)
            }
            
        if len(high_idx) >= 2 and len(low_idx) >= 1:
            p1_idx, p2_idx = high_idx[-2], high_idx[-1]
            peak1, peak2 = df[high_col].iloc[p1_idx], df[high_col].iloc[p2_idx]
            between_lows = df[low_col].iloc[p1_idx:p2_idx]
            neckline = between_lows.min() if not between_lows.empty else df[low_col].iloc[low_idx[-1]]
            m_height = max(peak1, peak2) - neckline
            result["bear"] = {
                "peak1": peak1, "date1": df.index[p1_idx].strftime('%m/%d'),
                "peak2": peak2, "date2": df.index[p2_idx].strftime('%m/%d'),
                "neckline": neckline, "height": m_height,
                "target_1to1": neckline - m_height, "target_1618": neckline - (m_height * 1.618)
            }
        return result

@dataclass
class QuantConfig:
    window_order: int = 5             
    price_sim_threshold: float = 0.03 
    min_time_gap: int = 8             
    max_time_gap: int = 60            
    breakout_margin: float = 0.01     
    breakout_days: int = 2            
    vol_multiplier: float = 1.2       
    risk_margin: float = 0.02         
    enable_regime_filter: bool = True 
    w_max_percentile: float = 0.60    
    m_min_percentile: float = 0.40    

class PatternState:
    FORMING = "FORMING (潛在型態形成中)"
    NECKLINE_TEST = "NECKLINE_TEST (價格接近頸線測試)"
    DAY1_BREAK_WARNING = "DAY1_BREAK_WARNING (首日突破/跌破預警，待次日確認)"
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED (有效突破/跌破頸線)"
    RETEST = "RETEST (突破後回測頸線)"
    BREAKOUT_FAILED = "BREAKOUT_FAILED (假突破/跌破失敗)"
    INVALIDATED = "INVALIDATED (價格破壞原始極值/失效)"
    NONE = "NONE (尚未形成有效型態)"

@dataclass
class SwingPoint:
    idx: int
    date: pd.Timestamp
    price: float
    conf_date: pd.Timestamp        

@dataclass
class GeometricPattern:
    p_type: str                     
    swing1: SwingPoint
    swing2: SwingPoint
    neckline: SwingPoint
    height: float
    target_1to1: float
    target_1618: float
    invalidation_level: float
    regime_note: str = "符合波段位階"
    state: str = PatternState.FORMING
    breakout_date: Optional[pd.Timestamp] = None
    vol_confirmed: bool = False

@dataclass
class DiagonalChannelPattern:
    channel_type: str               
    high1: SwingPoint
    high2: SwingPoint
    low1: SwingPoint
    low2: SwingPoint
    upper_slope: float
    lower_slope: float
    proj_upper: float
    proj_lower: float
    channel_height: float
    target_up: float
    target_down: float
    state: str                     
    break_date: Optional[pd.Timestamp] = None
    vol_confirmed: bool = False

class QuantPatternEngine:
    def __init__(self, config: QuantConfig = QuantConfig()):
        self.cfg = config
        
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            if 'Price' in df.columns.names:
                df.columns = df.columns.get_level_values('Price')
            else:
                df.columns = df.columns.get_level_values(0)
        df = df.copy()
        valid_mask = (df['High'] >= df['Low']) & (df['High'] >= df['Open']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Open']) & (df['Low'] <= df['Close']) & (df['Volume'] >= 0) & (df['Close'] > 0)
        df = df[valid_mask]
        return df[~df.index.duplicated(keep='first')].sort_index()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['20MA'] = df['Close'].rolling(20).mean()
        df['60MA'] = df['Close'].rolling(60).mean()
        df['120MA'] = df['Close'].rolling(120).mean()
        
        rolling_std = df['Close'].rolling(20).std()
        df['STD20'] = rolling_std
        df['BB_UCL'] = df['20MA'] + 2 * rolling_std
        df['BB_LCL'] = df['20MA'] - 2 * rolling_std
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Z_Score'] = np.nan_to_num(np.where(rolling_std > 0, (df['Close'] - df['20MA']) / rolling_std, 0.0), nan=0.0)
            
        df['Vol_20MA'] = df['Volume'].rolling(20).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Vol_Ratio'] = np.nan_to_num(np.where(df['Vol_20MA'] > 0, df['Volume'] / df['Vol_20MA'], 1.0), nan=1.0)
            
        roll_min60 = df['Low'].rolling(60, min_periods=20).min()
        roll_max60 = df['High'].rolling(60, min_periods=20).max()
        df['Pos_60D'] = (df['Close'] - roll_min60) / (roll_max60 - roll_min60 + 1e-8)
        
        return df

    def find_swings(self, df: pd.DataFrame, series: pd.Series, is_high: bool) -> List[SwingPoint]:
        swings = []
        vals = series.values
        dates = df.index
        w = self.cfg.window_order
        
        for i in range(w, len(vals) - w):
            window = vals[i-w:i+w+1]
            center = vals[i]
            if is_high and center == np.max(window) and center > np.max(vals[i-w:i]):
                swings.append(SwingPoint(i, dates[i], float(center), dates[i+w]))
            elif not is_high and center == np.min(window) and center < np.min(vals[i-w:i]):
                swings.append(SwingPoint(i, dates[i], float(center), dates[i+w]))
        return swings

    def identify_patterns(self, df: pd.DataFrame, highs: List[SwingPoint], lows: List[SwingPoint], p_type: str, high_series: pd.Series, low_series: pd.Series) -> List[GeometricPattern]:
        patterns = []
        cfg = self.cfg
        swings_list = lows if p_type == 'W' else highs
        
        if len(swings_list) < 2:
            return []
        
        for i in range(len(swings_list) - 1):
            s1 = swings_list[i]
            s2 = swings_list[i+1]
            
            if not (cfg.min_time_gap <= s2.idx - s1.idx <= cfg.max_time_gap):
                continue
            if abs(s1.price - s2.price) / min(s1.price, s2.price) > cfg.price_sim_threshold:
                continue
            
            mid_slice = high_series.iloc[s1.idx+1:s2.idx] if p_type == 'W' else low_series.iloc[s1.idx+1:s2.idx]
            if mid_slice.empty:
                continue
            
            neck_price = float(mid_slice.max()) if p_type == 'W' else float(mid_slice.min())
            if p_type == 'W' and neck_price <= max(s1.price, s2.price):
                continue
            if p_type == 'M' and neck_price >= min(s1.price, s2.price):
                continue
            
            neck_idx = s1.idx + 1 + (mid_slice.values.argmax() if p_type == 'W' else mid_slice.values.argmin())
            neckline = SwingPoint(neck_idx, df.index[neck_idx], neck_price, df.index[neck_idx])
            
            lookback = max(0, s1.idx - 60)
            hist_low = float(low_series.iloc[lookback:s1.idx].min()) if s1.idx > lookback else float(s1.price)
            hist_high = float(high_series.iloc[lookback:s1.idx].max()) if s1.idx > lookback else float(s1.price)
            
            pos = (min(s1.price, s2.price) if p_type == 'W' else max(s1.price, s2.price) - hist_low) / (hist_high - hist_low + 1e-8)
            
            if cfg.enable_regime_filter:
                if p_type == 'W' and pos > cfg.w_max_percentile:
                    continue
                if p_type == 'M' and pos < cfg.m_min_percentile:
                    continue
                
            if p_type == 'W':
                regime_str = f"波段低位階 ({pos*100:.0f}%)" if pos <= 0.4 else f"中繼整理 ({pos*100:.0f}%)"
                h = abs(neck_price - min(s1.price, s2.price))
                target1 = neck_price + h
                target1618 = neck_price + (h * 1.618)
                inv_lvl = min(s1.price, s2.price) * (1 - cfg.risk_margin)
            else:
                regime_str = f"波段高位階 ({pos*100:.0f}%)" if pos >= 0.6 else f"中繼震盪 ({pos*100:.0f}%)"
                h = abs(max(s1.price, s2.price) - neck_price)
                target1 = neck_price - h
                target1618 = neck_price - (h * 1.618)
                inv_lvl = max(s1.price, s2.price) * (1 + cfg.risk_margin)
            
            patterns.append(GeometricPattern(
                p_type=p_type, swing1=s1, swing2=s2, neckline=neckline, height=h,
                target_1to1=target1, target_1618=target1618, invalidation_level=inv_lvl,
                regime_note=regime_str
            ))
            
        return patterns

    def evaluate_state_machine(self, df: pd.DataFrame, pat: GeometricPattern) -> GeometricPattern:
        eval_start = pat.swing2.idx + 1
        if eval_start >= len(df):
            return pat
            
        consec_break = 0
        cur_state = PatternState.FORMING
        t_date = None
        t_vol = False
        
        for i in range(eval_start, len(df)):
            close_p = df['Close'].iloc[i]
            vol_r = df['Vol_Ratio'].iloc[i]
            cur_dt = df.index[i]
            
            if (pat.p_type == 'W' and close_p < pat.invalidation_level) or (pat.p_type == 'M' and close_p > pat.invalidation_level):
                cur_state = PatternState.INVALIDATED
                pat.breakout_date = None
                pat.vol_confirmed = False
                break
                
            is_break = (close_p >= pat.neckline.price * (1 + self.cfg.breakout_margin)) if pat.p_type == 'W' else (close_p <= pat.neckline.price * (1 - self.cfg.breakout_margin))
            is_fail = (close_p < pat.neckline.price * (1 - self.cfg.breakout_margin)) if pat.p_type == 'W' else (close_p > pat.neckline.price * (1 + self.cfg.breakout_margin))
            
            if cur_state in [PatternState.FORMING, PatternState.NECKLINE_TEST, PatternState.DAY1_BREAK_WARNING, PatternState.BREAKOUT_FAILED]:
                if is_break:
                    consec_break += 1
                    if consec_break == 1:
                        t_date = cur_dt
                        t_vol = (vol_r >= self.cfg.vol_multiplier)
                    else:
                        t_vol = t_vol or (vol_r >= self.cfg.vol_multiplier)
                        
                    cur_state = PatternState.BREAKOUT_CONFIRMED if consec_break >= self.cfg.breakout_days else PatternState.DAY1_BREAK_WARNING
                    pat.breakout_date = t_date
                    pat.vol_confirmed = t_vol
                else:
                    consec_break = 0
                    t_date = None
                    t_vol = False
                    if cur_state != PatternState.BREAKOUT_FAILED:
                        pat.breakout_date = None
                        pat.vol_confirmed = False
                        cur_state = PatternState.NECKLINE_TEST if (close_p >= pat.neckline.price * 0.98 if pat.p_type == 'W' else close_p <= pat.neckline.price * 1.02) else PatternState.FORMING
            elif cur_state == PatternState.BREAKOUT_CONFIRMED:
                if is_fail:
                    cur_state = PatternState.BREAKOUT_FAILED
                    consec_break = 0
                elif not is_break:
                    cur_state = PatternState.RETEST
            elif cur_state == PatternState.RETEST:
                if is_fail:
                    cur_state = PatternState.BREAKOUT_FAILED
                    consec_break = 0
                elif is_break:
                    cur_state = PatternState.BREAKOUT_CONFIRMED
                    
        pat.state = cur_state
        return pat

    def identify_diagonal_channel(self, df: pd.DataFrame, high_series: pd.Series, low_series: pd.Series) -> Optional[DiagonalChannelPattern]:
        highs = self.find_swings(df, high_series, True)
        lows = self.find_swings(df, low_series, False)
        
        if len(highs) < 2 or len(lows) < 2:
            return None
            
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]
        
        if (h2.idx - h1.idx) < 4 or (l2.idx - l1.idx) < 4:
            return None
            
        u_slope = (h2.price - h1.price) / (h2.idx - h1.idx)
        l_slope = (l2.price - l1.price) / (l2.idx - l1.idx)
        up_pct = (u_slope / h1.price) * 100
        lp_pct = (l_slope / l1.price) * 100
        last_idx = len(df) - 1
        
        p_u = h2.price + u_slope * (last_idx - h2.idx)
        p_l = l2.price + l_slope * (last_idx - l2.idx)
        
        if up_pct > 0.03 and lp_pct > 0.03:
            c_type = "上升收斂楔形" if lp_pct > up_pct * 1.15 else "上升趨勢通道"
        elif up_pct < -0.03 and lp_pct < -0.03:
            c_type = "下降收斂楔形" if up_pct < lp_pct * 1.15 else "下降趨勢通道"
        elif up_pct < -0.02 and lp_pct > 0.02:
            c_type = "對稱三角收斂"
        elif abs(up_pct) <= 0.03 and lp_pct > 0.03:
            c_type = "上升直角三角形"
        elif up_pct < -0.03 and abs(lp_pct) <= 0.03:
            c_type = "下降直角三角形"
        else:
            c_type = "橫向區間箱型通道"
            
        if p_u < p_l:
            c_type += " (⚠️ 趨勢線失效)"
        
        h_diff = max(h1.price, h2.price) - min(l1.price, l2.price)
        state = "WITHIN_CHANNEL (於趨勢軌道內運行)"
        b_dt = None
        v_conf = False
        
        start_idx = min(max(h2.idx, l2.idx) + self.cfg.window_order, last_idx)
        for i in range(start_idx, len(df)):
            c = df['Close'].iloc[i]
            vr = df['Vol_Ratio'].iloc[i]
            uv = h2.price + u_slope * (i - h2.idx)
            lv = l2.price + l_slope * (i - l2.idx)
            
            if uv < lv:
                state = "CHANNEL_INVALID (⚠️ 軌道已交叉失效)"
                b_dt = None
                v_conf = False
                continue
            
            if c < lv * (1 - self.cfg.breakout_margin * 0.5):
                state = "CHANNEL_BREAKDOWN (🚨 跌破下軌上升支撐線)"
                if not b_dt:
                    b_dt = df.index[i]
                    v_conf = (vr >= self.cfg.vol_multiplier)
                else:
                    v_conf = v_conf or (vr >= self.cfg.vol_multiplier)
            elif c > uv * (1 + self.cfg.breakout_margin * 0.5):
                state = "CHANNEL_BREAKOUT (🚀 突破上軌壓力趨勢線)"
                if not b_dt:
                    b_dt = df.index[i]
                    v_conf = (vr >= self.cfg.vol_multiplier)
                else:
                    v_conf = v_conf or (vr >= self.cfg.vol_multiplier)
            else:
                state = "WITHIN_CHANNEL (於趨勢軌道內運行)"
                b_dt = None
                v_conf = False
                
        return DiagonalChannelPattern(
            channel_type=c_type, high1=h1, high2=h2, low1=l1, low2=l2, upper_slope=u_slope, lower_slope=l_slope,
            proj_upper=p_u, proj_lower=p_l, channel_height=h_diff, target_up=p_u + h_diff, target_down=p_l - h_diff,
            state=state, break_date=b_dt, vol_confirmed=v_conf
        )

    def analyze_pipeline(self, df: pd.DataFrame, high_series: pd.Series, low_series: pd.Series) -> Dict[str, Optional[GeometricPattern]]:
        highs = self.find_swings(df, high_series, True)
        lows = self.find_swings(df, low_series, False)
        
        eval_w = [self.evaluate_state_machine(df, p) for p in self.identify_patterns(df, highs, lows, 'W', high_series, low_series)]
        eval_m = [self.evaluate_state_machine(df, p) for p in self.identify_patterns(df, highs, lows, 'M', high_series, low_series)]
        
        active_w = [p for p in eval_w if p.state != PatternState.INVALIDATED and (len(df) - p.swing2.idx) < 120]
        active_m = [p for p in eval_m if p.state != PatternState.INVALIDATED and (len(df) - p.swing2.idx) < 120]
        
        return {'W': active_w[-1] if active_w else None, 'M': active_m[-1] if active_m else None}


# ==========================================
# 跨市場波動率體制與避險/反彈監控模組 (純 yfinance)
# ==========================================
@dataclass
class VolatilityMonitorConfig:
    vix_wartime_th: float = 25.0
    vxn_wartime_th: float = 30.0
    vvix_alert_th: float = 95.0
    local_vol_low_pct: float = 20.0
    local_vol_lookback: int = 60
    compression_window: int = 10
    download_period: str = "6mo"
    display_months: int = 3


class CrossMarketVolatilityMonitor:
    TICKERS = {
        "^VIX": "美股大盤恐慌指數 (VIX)",
        "^VVIX": "美股二階波動率 (VVIX)",
        "^VXN": "那斯達克科技恐慌 (VXN)",
        "^TWII": "台灣加權指數 (TWII)"
    }

    @staticmethod
    def _to_taipei_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        idx = pd.to_datetime(df.index)
        try:
            if idx.tz is None:
                idx = idx.tz_localize("Asia/Taipei")
            else:
                idx = idx.tz_convert("Asia/Taipei")
        except Exception:
            tw_tz = datetime.timezone(datetime.timedelta(hours=8))
            if idx.tz is None:
                idx = idx.tz_localize(tw_tz)
            else:
                idx = idx.tz_convert(tw_tz)
        df.index = idx
        return df[~df.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _extract_single_ohlcv(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            lv0 = df.columns.get_level_values(0)
            lv1 = df.columns.get_level_values(1)
            if ticker in lv0:
                df = df.xs(ticker, axis=1, level=0)
            elif ticker in lv1:
                df = df.xs(ticker, axis=1, level=1)
            elif "Close" in lv0:
                df.columns = lv0
            elif "Close" in lv1:
                df.columns = lv1
            else:
                return pd.DataFrame()
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if "Close" not in needed:
            return pd.DataFrame()
        df = df[needed].dropna(subset=["Close"])
        return CrossMarketVolatilityMonitor._to_taipei_index(df)

    @staticmethod
    def _fetch_market_data(session, period: str = "6mo") -> Dict[str, pd.DataFrame]:
        tickers = list(CrossMarketVolatilityMonitor.TICKERS.keys())
        data_map = {}
        try:
            raw_all = yf.download(
                tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                session=session
            )
            for t in tickers:
                df_t = CrossMarketVolatilityMonitor._extract_single_ohlcv(raw_all, t)
                if df_t.empty:
                    single_raw = yf.download(
                        t,
                        period=period,
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                        session=session
                    )
                    df_t = CrossMarketVolatilityMonitor._extract_single_ohlcv(single_raw, t)
                data_map[t] = df_t
        except Exception:
            for t in tickers:
                try:
                    single_raw = yf.download(
                        t,
                        period=period,
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                        session=session
                    )
                    data_map[t] = CrossMarketVolatilityMonitor._extract_single_ohlcv(single_raw, t)
                except Exception:
                    data_map[t] = pd.DataFrame()
        return data_map

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int = 60) -> pd.Series:
        def _pct(arr):
            valid = arr[~np.isnan(arr)]
            if len(valid) < 2:
                return np.nan
            return float(np.sum(valid <= valid[-1]) / len(valid) * 100.0)
        return series.rolling(window=window, min_periods=20).apply(_pct, raw=True)

    @staticmethod
    def _compute_indicators(data_map: Dict[str, pd.DataFrame], cfg: VolatilityMonitorConfig) -> Dict[str, pd.DataFrame]:
        enriched = {}
        for ticker, raw_df in data_map.items():
            if raw_df is None or raw_df.empty:
                enriched[ticker] = pd.DataFrame()
                continue
            df = raw_df.copy()
            close = df["Close"].astype(float)
            df["Pct_Change"] = close.pct_change() * 100.0
            df["MA5"] = close.rolling(5).mean()
            df["MA10"] = close.rolling(10).mean()
            df["MA20"] = close.rolling(20).mean()
            df["Above_MA5"] = close > df["MA5"]
            df["Above_MA20"] = close > df["MA20"]

            if ticker in ("^VIX", "^VXN"):
                std20 = close.rolling(20).std()
                df["BB_Upper"] = df["MA20"] + 2.0 * std20
                df["BIAS20"] = (close - df["MA20"]) / (df["MA20"] + 1e-8) * 100.0
                df["Break_BB_Upper"] = close > df["BB_Upper"]

            if ticker == "^TWII":
                log_ret = np.log(close / close.shift(1))
                df["HV20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100.0
                df["HV20_MA5"] = df["HV20"].rolling(5).mean()
                df["HV20_MA20"] = df["HV20"].rolling(20).mean()
                df["HV20_Pct60"] = CrossMarketVolatilityMonitor._rolling_percentile(df["HV20"], cfg.local_vol_lookback)

                high = df["High"].astype(float) if "High" in df.columns else close
                low = df["Low"].astype(float) if "Low" in df.columns else close
                prev_c = close.shift(1)
                tr = pd.concat([
                    (high - low),
                    (high - prev_c).abs(),
                    (low - prev_c).abs()
                ], axis=1).max(axis=1)
                df["ATR14"] = tr.ewm(alpha=1 / 14.0, min_periods=14, adjust=False).mean()
                df["ATR14_Pct"] = df["ATR14"] / (close + 1e-8) * 100.0
                df["ATR14_Pct_MA5"] = df["ATR14_Pct"].rolling(5).mean()
                df["ATR14_Pct_MA20"] = df["ATR14_Pct"].rolling(20).mean()
                df["ATR14_Pct_Pct60"] = CrossMarketVolatilityMonitor._rolling_percentile(df["ATR14_Pct"], cfg.local_vol_lookback)

            enriched[ticker] = df
        return enriched

    @staticmethod
    def _slice_display_window(df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        cutoff = df.index.max() - pd.DateOffset(months=months)
        return df[df.index >= cutoff].copy()

    @staticmethod
    def _check_compression_breakout(val_s: pd.Series, ma20_s: pd.Series, pct60_s: pd.Series, cfg: VolatilityMonitorConfig) -> bool:
        if len(val_s) < 3:
            return False
        cur_v, prev_v = val_s.iloc[-1], val_s.iloc[-2]
        cur_ma, prev_ma = ma20_s.iloc[-1], ma20_s.iloc[-2]
        if any(pd.isna([cur_v, prev_v, cur_ma, prev_ma])):
            return False
        cross_up = (cur_v > cur_ma) and (prev_v <= prev_ma)
        if not cross_up:
            return False
        start_idx = max(0, len(pct60_s) - 1 - cfg.compression_window)
        prior_window = pct60_s.iloc[start_idx:-1].dropna()
        if prior_window.empty:
            return False
        return bool(float(prior_window.min()) < cfg.local_vol_low_pct)

    @staticmethod
    def evaluate(session=None, config: Optional[VolatilityMonitorConfig] = None) -> Dict:
        cfg = config if config is not None else VolatilityMonitorConfig()
        yf_sess = session if session is not None else get_yf_session()
        tw_tz = datetime.timezone(datetime.timedelta(hours=8))
        eval_time_str = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")

        raw_map = CrossMarketVolatilityMonitor._fetch_market_data(yf_sess, period=cfg.download_period)
        missing_core = [t for t in ("^VIX", "^VXN", "^TWII") if raw_map.get(t) is None or raw_map[t].empty or len(raw_map[t]) < 25]
        if missing_core:
            return {
                "status": "error",
                "msg": f"無法取得完整波動率監控資料，缺少或筆數不足標的：{', '.join(missing_core)}",
                "eval_time": eval_time_str
            }

        enriched = CrossMarketVolatilityMonitor._compute_indicators(raw_map, cfg)
        vix_df = enriched["^VIX"]
        vxn_df = enriched["^VXN"]
        vvix_df = enriched.get("^VVIX", pd.DataFrame())
        twii_df = enriched["^TWII"]

        # 計算科技股風險溢價 (VXN - VIX)
        vix_close_norm = vix_df["Close"].copy()
        vix_close_norm.index = vix_close_norm.index.normalize()
        vxn_close_norm = vxn_df["Close"].copy()
        vxn_close_norm.index = vxn_close_norm.index.normalize()
        spread_df = pd.concat([
            vxn_close_norm.rename("VXN"),
            vix_close_norm.rename("VIX")
        ], axis=1).dropna()
        spread_df["Spread"] = spread_df["VXN"] - spread_df["VIX"]
        spread_df["Spread_MA5"] = spread_df["Spread"].rolling(5).mean()
        spread_df["Spread_MA20"] = spread_df["Spread"].rolling(20).mean()
        spread_df["Spread_Diff"] = spread_df["Spread"].diff()

        vix_last = vix_df.iloc[-1]
        vxn_last = vxn_df.iloc[-1]
        twii_last = twii_df.iloc[-1]
        vvix_available = not vvix_df.empty and len(vvix_df) >= 20
        vvix_last = vvix_df.iloc[-1] if vvix_available else None

        vix_c, vix_ma5, vix_ma20 = float(vix_last["Close"]), float(vix_last["MA5"]), float(vix_last["MA20"])
        vix_bb_u, vix_bias20 = float(vix_last["BB_Upper"]), float(vix_last["BIAS20"])
        vix_break_bb = bool(vix_last["Break_BB_Upper"])

        vxn_c, vxn_ma5, vxn_ma20 = float(vxn_last["Close"]), float(vxn_last["MA5"]), float(vxn_last["MA20"])
        vxn_bb_u, vxn_bias20 = float(vxn_last["BB_Upper"]), float(vxn_last["BIAS20"])
        vxn_break_bb = bool(vxn_last["Break_BB_Upper"])

        vvix_c = float(vvix_last["Close"]) if vvix_available else np.nan
        vvix_ma5 = float(vvix_last["MA5"]) if vvix_available else np.nan
        vvix_ma20 = float(vvix_last["MA20"]) if vvix_available else np.nan

        twii_c = float(twii_last["Close"])
        twii_ma5 = float(twii_last["MA5"])
        twii_ma10 = float(twii_last["MA10"])
        twii_ma20 = float(twii_last["MA20"])

        spread_c = float(spread_df["Spread"].iloc[-1]) if not spread_df.empty else (vxn_c - vix_c)
        spread_diff = float(spread_df["Spread_Diff"].iloc[-1]) if len(spread_df) >= 2 else np.nan
        spread_ma5 = float(spread_df["Spread_MA5"].iloc[-1]) if not spread_df.empty else np.nan
        spread_ma20 = float(spread_df["Spread_MA20"].iloc[-1]) if not spread_df.empty else np.nan
        spread_expanding = bool(pd.notna(spread_diff) and spread_diff > 0)

        hv20_c = float(twii_last["HV20"]) if pd.notna(twii_last["HV20"]) else np.nan
        hv20_ma5 = float(twii_last["HV20_MA5"]) if pd.notna(twii_last["HV20_MA5"]) else np.nan
        hv20_ma20 = float(twii_last["HV20_MA20"]) if pd.notna(twii_last["HV20_MA20"]) else np.nan
        hv20_pct60 = float(twii_last["HV20_Pct60"]) if pd.notna(twii_last["HV20_Pct60"]) else np.nan

        atr_c = float(twii_last["ATR14_Pct"]) if pd.notna(twii_last["ATR14_Pct"]) else np.nan
        atr_ma5 = float(twii_last["ATR14_Pct_MA5"]) if pd.notna(twii_last["ATR14_Pct_MA5"]) else np.nan
        atr_ma20 = float(twii_last["ATR14_Pct_MA20"]) if pd.notna(twii_last["ATR14_Pct_MA20"]) else np.nan
        atr_pct60 = float(twii_last["ATR14_Pct_Pct60"]) if pd.notna(twii_last["ATR14_Pct_Pct60"]) else np.nan

        # ==========================================
        # 核心體制切換與四狀態判斷
        # ==========================================
        wartime_reasons = []
        if vix_c > cfg.vix_wartime_th:
            wartime_reasons.append(f"VIX={vix_c:.2f} > {cfg.vix_wartime_th:.0f}")
        if vxn_c > cfg.vxn_wartime_th:
            wartime_reasons.append(f"VXN={vxn_c:.2f} > {cfg.vxn_wartime_th:.0f}")
        if vix_break_bb:
            wartime_reasons.append(f"VIX({vix_c:.2f})突破20日布林上軌({vix_bb_u:.2f})")
        if vxn_break_bb:
            wartime_reasons.append(f"VXN({vxn_c:.2f})突破20日布林上軌({vxn_bb_u:.2f})")

        is_wartime = len(wartime_reasons) > 0
        panic_divergence = False
        alert_reminders = []
        trigger_reasons = []

        # 檢查恐慌高點下移背離 (TWII 破底但 VIX/VXN 未過前高)
        if len(twii_df) >= 25 and len(vix_df) >= 25 and len(vxn_df) >= 25:
            twii_low_col = "Low" if "Low" in twii_df.columns else "Close"
            vix_high_col = "High" if "High" in vix_df.columns else "Close"
            vxn_high_col = "High" if "High" in vxn_df.columns else "Close"
            twii_new_low = float(twii_df[twii_low_col].iloc[-5:].min()) < float(twii_df[twii_low_col].iloc[-25:-5].min())
            vix_lower_high = float(vix_df[vix_high_col].iloc[-5:].max()) < float(vix_df[vix_high_col].iloc[-25:-5].max())
            vxn_lower_high = float(vxn_df[vxn_high_col].iloc[-5:].max()) < float(vxn_df[vxn_high_col].iloc[-25:-5].max())
            panic_divergence = bool(twii_new_low and (vix_lower_high or vxn_lower_high))

        # 檢查平時本地波動與 VVIX 條件 (預先計算，供狀態判別與儀表板顯示)
        c1_vvix = bool(vvix_available and pd.notna(vvix_c) and pd.notna(vvix_ma20) and (vvix_c > cfg.vvix_alert_th) and (vvix_c > vvix_ma20))
        twii_below_ma = bool((twii_c < twii_ma10) or (twii_c < twii_ma20))
        hv20_breakout = CrossMarketVolatilityMonitor._check_compression_breakout(
            twii_df["HV20"], twii_df["HV20_MA20"], twii_df["HV20_Pct60"], cfg
        )
        atr_breakout = CrossMarketVolatilityMonitor._check_compression_breakout(
            twii_df["ATR14_Pct"], twii_df["ATR14_Pct_MA20"], twii_df["ATR14_Pct_Pct60"], cfg
        )
        c2_hv20 = bool(hv20_breakout and twii_below_ma)
        c2_atr = bool(atr_breakout and twii_below_ma)

        if is_wartime:
            regime = "【戰時恐慌警戒區】（系統性流動性殺盤）"
            trigger_reasons.extend(wartime_reasons)
            if (vix_c < vix_ma5) and (vxn_c < vxn_ma5):
                state_code = "A"
                state_label = "狀態 A（止跌反彈確認）"
                severity = "success"
                trigger_reasons.append(f"VIX({vix_c:.2f}) < 5MA({vix_ma5:.2f}) 且 VXN({vxn_c:.2f}) < 5MA({vxn_ma5:.2f})")
                if panic_divergence:
                    alert_reminders.append("🔥【恐慌高點下移背離】^TWII 破底但美股波動率未過前高，止跌反彈勝率顯著提升。")
                advice = "恐慌退潮，可關注台股止跌反彈（現貨/期貨右側多單或選擇權 Bull Put Spread 賺 IV 收斂）。"
            else:
                state_code = "B"
                state_label = "狀態 B（尚未止血警戒）"
                severity = "error"
                if vix_c >= vix_ma5:
                    trigger_reasons.append(f"VIX({vix_c:.2f}) 仍在 5MA({vix_ma5:.2f}) 之上")
                if vxn_c >= vxn_ma5:
                    trigger_reasons.append(f"VXN({vxn_c:.2f}) 仍在 5MA({vxn_ma5:.2f}) 之上")
                if (vxn_c > vxn_ma5) and spread_expanding:
                    alert_reminders.append(f"🚨【科技股高危警示】VXN 壓在 5MA 之上且 VXN-VIX 風險溢價擴大至 {spread_c:.2f}（日變動 {spread_diff:+.2f}）。")
                advice = "科技股系統性拋售未結束，切勿重倉接刀，忽略個股左側底背離訊號。"
        else:
            regime = "【平時常態震盪區】（非系統性環境）"
            if c1_vvix:
                trigger_reasons.append(f"VVIX({vvix_c:.2f}) > 95 且站上 20MA({vvix_ma20:.2f})")
                alert_reminders.append(f"⚠️【VVIX 尾部避險提醒】美股聰明錢買進尾部保險（VVIX={vvix_c:.2f} > 20MA {vvix_ma20:.2f}）。")

            if c2_hv20 and c2_atr:
                trigger_reasons.append("HV20 與 ATR14% 同步自 60 日低分位(<20%)向上突破 20MA 且 ^TWII 跌破 10MA/20MA")
                alert_reminders.append(
                    f"🚨【台股本地波動雙指標觸發提醒】HV20({hv20_c:.2f}% > 20MA {hv20_ma20:.2f}%) 與 "
                    f"ATR14%({atr_c:.2f}% > 20MA {atr_ma20:.2f}%) 同步由 60 日低分位(<20%)向上突破，且加權指數跌破均線！"
                )
            elif c2_hv20:
                trigger_reasons.append(f"HV20({hv20_c:.2f}%) 自 60 日低分位(<20%)向上突破 20MA({hv20_ma20:.2f}%) 且 ^TWII 跌破 10MA/20MA")
                alert_reminders.append(
                    f"🚨【台股本地波動觸發提醒：HV20】20 日年化已實現波動率 HV20({hv20_c:.2f}%) "
                    f"自 60 日低分位(<20%)由下往上突破 20MA({hv20_ma20:.2f}%)，且加權指數跌破 10MA/20MA！"
                )
            elif c2_atr:
                trigger_reasons.append(f"ATR14%({atr_c:.2f}%) 自 60 日低分位(<20%)向上突破 20MA({atr_ma20:.2f}%) 且 ^TWII 跌破 10MA/20MA")
                alert_reminders.append(
                    f"🚨【台股本地波動觸發提醒：ATR14%】ATR14 百分比({atr_c:.2f}%) "
                    f"自 60 日低分位(<20%)由下往上突破 20MA({atr_ma20:.2f}%)，且加權指數跌破 10MA/20MA！"
                )

            if c1_vvix or c2_hv20 or c2_atr:
                state_code = "C"
                state_label = "狀態 C（大跌前避險預警）"
                severity = "warning"
                advice = "大盤高檔波動率壓縮突破，建議提前買進低成本價外 Put（次週選/近月）或破月線啟動微台指（TMF）動態對沖。"
            else:
                state_code = "D"
                state_label = "狀態 D（結構正常）"
                severity = "info"
                trigger_reasons.append("VIX <= 25、VXN <= 30 且無 VVIX 或台股低波動突破預警")
                advice = "無系統性風險，關閉波動率雜訊干擾，回歸個股均線與籌碼操作。"

        def _fmt_bool(flag):
            if pd.isna(flag):
                return "-"
            return "✅ 是" if bool(flag) else "❌ 否"

        def _fmt_num(val, digits=2, suffix=""):
            return f"{val:.{digits}f}{suffix}" if pd.notna(val) else "-"

        metrics_rows = [
            {
                "標的 / 指標": "^VIX (美股大盤恐慌)",
                "資料日期 (UTC+8)": vix_df.index[-1].strftime("%Y-%m-%d"),
                "最新數值": round(vix_c, 2),
                "日漲跌幅(%)": round(float(vix_last["Pct_Change"]), 2) if pd.notna(vix_last["Pct_Change"]) else np.nan,
                "5MA": round(vix_ma5, 2),
                "20MA": round(vix_ma20, 2),
                "站上5MA": _fmt_bool(vix_last["Above_MA5"]),
                "站上20MA": _fmt_bool(vix_last["Above_MA20"]),
                "進階位階 / 備註": f"20日布林上軌: {vix_bb_u:.2f} ({'⚠️突破' if vix_break_bb else '軌內'}) | 20MA乖離: {vix_bias20:+.2f}%"
            },
            {
                "標的 / 指標": "^VXN (科技股恐慌)",
                "資料日期 (UTC+8)": vxn_df.index[-1].strftime("%Y-%m-%d"),
                "最新數值": round(vxn_c, 2),
                "日漲跌幅(%)": round(float(vxn_last["Pct_Change"]), 2) if pd.notna(vxn_last["Pct_Change"]) else np.nan,
                "5MA": round(vxn_ma5, 2),
                "20MA": round(vxn_ma20, 2),
                "站上5MA": _fmt_bool(vxn_last["Above_MA5"]),
                "站上20MA": _fmt_bool(vxn_last["Above_MA20"]),
                "進階位階 / 備註": f"20日布林上軌: {vxn_bb_u:.2f} ({'⚠️突破' if vxn_break_bb else '軌內'}) | 20MA乖離: {vxn_bias20:+.2f}%"
            },
            {
                "標的 / 指標": "^VVIX (美股二階波動)",
                "資料日期 (UTC+8)": vvix_df.index[-1].strftime("%Y-%m-%d") if vvix_available else "-",
                "最新數值": round(vvix_c, 2) if pd.notna(vvix_c) else np.nan,
                "日漲跌幅(%)": round(float(vvix_last["Pct_Change"]), 2) if (vvix_available and pd.notna(vvix_last["Pct_Change"])) else np.nan,
                "5MA": round(vvix_ma5, 2) if pd.notna(vvix_ma5) else np.nan,
                "20MA": round(vvix_ma20, 2) if pd.notna(vvix_ma20) else np.nan,
                "站上5MA": _fmt_bool(vvix_last["Above_MA5"]) if vvix_available else "-",
                "站上20MA": _fmt_bool(vvix_last["Above_MA20"]) if vvix_available else "-",
                "進階位階 / 備註": f"警戒門檻: >{cfg.vvix_alert_th:.0f} 且 >20MA ({'🚨觸發' if c1_vvix else '正常'})" if vvix_available else "資料暫缺"
            },
            {
                "標的 / 指標": "^TWII (台灣加權指數)",
                "資料日期 (UTC+8)": twii_df.index[-1].strftime("%Y-%m-%d"),
                "最新數值": round(twii_c, 2),
                "日漲跌幅(%)": round(float(twii_last["Pct_Change"]), 2) if pd.notna(twii_last["Pct_Change"]) else np.nan,
                "5MA": round(twii_ma5, 2),
                "20MA": round(twii_ma20, 2),
                "站上5MA": _fmt_bool(twii_last["Above_MA5"]),
                "站上20MA": _fmt_bool(twii_last["Above_MA20"]),
                "進階位階 / 備註": f"10MA: {twii_ma10:.2f} | 跌破10MA或20MA: {'⚠️是' if twii_below_ma else '否'}"
            },
            {
                "標的 / 指標": "科技股風險溢價 (VXN-VIX)",
                "資料日期 (UTC+8)": spread_df.index[-1].strftime("%Y-%m-%d") if not spread_df.empty else "-",
                "最新數值": round(spread_c, 2),
                "日漲跌幅(%)": np.nan,
                "5MA": round(spread_ma5, 2) if pd.notna(spread_ma5) else np.nan,
                "20MA": round(spread_ma20, 2) if pd.notna(spread_ma20) else np.nan,
                "站上5MA": _fmt_bool(spread_c > spread_ma5) if pd.notna(spread_ma5) else "-",
                "站上20MA": _fmt_bool(spread_c > spread_ma20) if pd.notna(spread_ma20) else "-",
                "進階位階 / 備註": f"單日變動: {_fmt_num(spread_diff, 2)} ({'⚠️利差擴大' if spread_expanding else '利差收斂/持平'})"
            },
            {
                "標的 / 指標": "台股本地波動 HV20(%)",
                "資料日期 (UTC+8)": twii_df.index[-1].strftime("%Y-%m-%d"),
                "最新數值": round(hv20_c, 2) if pd.notna(hv20_c) else np.nan,
                "日漲跌幅(%)": np.nan,
                "5MA": round(hv20_ma5, 2) if pd.notna(hv20_ma5) else np.nan,
                "20MA": round(hv20_ma20, 2) if pd.notna(hv20_ma20) else np.nan,
                "站上5MA": _fmt_bool(hv20_c > hv20_ma5) if pd.notna(hv20_c) and pd.notna(hv20_ma5) else "-",
                "站上20MA": _fmt_bool(hv20_c > hv20_ma20) if pd.notna(hv20_c) and pd.notna(hv20_ma20) else "-",
                "進階位階 / 備註": f"60日百分位: {_fmt_num(hv20_pct60, 1, '%')} | 低分位突破狀態C: {'🚨觸發(HV20)' if c2_hv20 else '無'}"
            },
            {
                "標的 / 指標": "台股本地波動 ATR14(%)",
                "資料日期 (UTC+8)": twii_df.index[-1].strftime("%Y-%m-%d"),
                "最新數值": round(atr_c, 2) if pd.notna(atr_c) else np.nan,
                "日漲跌幅(%)": np.nan,
                "5MA": round(atr_ma5, 2) if pd.notna(atr_ma5) else np.nan,
                "20MA": round(atr_ma20, 2) if pd.notna(atr_ma20) else np.nan,
                "站上5MA": _fmt_bool(atr_c > atr_ma5) if pd.notna(atr_c) and pd.notna(atr_ma5) else "-",
                "站上20MA": _fmt_bool(atr_c > atr_ma20) if pd.notna(atr_c) and pd.notna(atr_ma20) else "-",
                "進階位階 / 備註": f"60日百分位: {_fmt_num(atr_pct60, 1, '%')} | 低分位突破狀態C: {'🚨觸發(ATR14%)' if c2_atr else '無'}"
            }
        ]

        history_3m = {
            "^VIX": CrossMarketVolatilityMonitor._slice_display_window(vix_df, cfg.display_months),
            "^VXN": CrossMarketVolatilityMonitor._slice_display_window(vxn_df, cfg.display_months),
            "^VVIX": CrossMarketVolatilityMonitor._slice_display_window(vvix_df, cfg.display_months),
            "^TWII": CrossMarketVolatilityMonitor._slice_display_window(twii_df, cfg.display_months),
            "SPREAD": CrossMarketVolatilityMonitor._slice_display_window(spread_df, cfg.display_months)
        }

        return {
            "status": "success",
            "eval_time": eval_time_str,
            "regime": regime,
            "is_wartime": is_wartime,
            "state_code": state_code,
            "state_label": state_label,
            "severity": severity,
            "panic_divergence": panic_divergence,
            "c_triggers": {
                "vvix": c1_vvix,
                "hv20": c2_hv20,
                "atr14": c2_atr
            },
            "trigger_reasons": trigger_reasons,
            "alert_reminders": alert_reminders,
            "advice": advice,
            "snapshot": {
                "vix": vix_c, "vix_ma5": vix_ma5, "vix_ma20": vix_ma20, "vix_bb_u": vix_bb_u, "vix_bias20": vix_bias20,
                "vxn": vxn_c, "vxn_ma5": vxn_ma5, "vxn_ma20": vxn_ma20, "vxn_bb_u": vxn_bb_u, "vxn_bias20": vxn_bias20,
                "vvix": vvix_c, "vvix_ma5": vvix_ma5, "vvix_ma20": vvix_ma20,
                "twii": twii_c, "twii_ma5": twii_ma5, "twii_ma10": twii_ma10, "twii_ma20": twii_ma20,
                "spread": spread_c, "spread_diff": spread_diff, "spread_expanding": spread_expanding,
                "hv20": hv20_c, "hv20_ma20": hv20_ma20, "hv20_pct60": hv20_pct60,
                "atr14_pct": atr_c, "atr14_pct_ma20": atr_ma20, "atr14_pct_pct60": atr_pct60
            },
            "metrics_table": metrics_rows,
            "history": history_3m
        }

    @staticmethod
    def to_report_dataframe(result: Dict) -> pd.DataFrame:
        if not result or result.get("status") != "success":
            err_msg = result.get("msg", "監控執行失敗") if isinstance(result, dict) else "無資料"
            return pd.DataFrame([{"分類": "錯誤", "標的 / 指標": "跨市場波動率監控", "最新數值 / 狀態": err_msg}])

        summary_rows = [
            {
                "分類": "體制總覽",
                "標的 / 指標": "更新時間 (UTC+8)",
                "資料日期 (UTC+8)": "-",
                "最新數值 / 狀態": result["eval_time"],
                "日漲跌幅(%)": "-",
                "5MA": "-",
                "20MA": "-",
                "站上5MA": "-",
                "站上20MA": "-",
                "進階位階 / 說明": "資料源：純 yfinance (無 VIXTWN 爬蟲)"
            },
            {
                "分類": "體制總覽",
                "標的 / 指標": "當前市場體制",
                "資料日期 (UTC+8)": "-",
                "最新數值 / 狀態": result["regime"],
                "日漲跌幅(%)": "-",
                "5MA": "-",
                "20MA": "-",
                "站上5MA": "-",
                "站上20MA": "-",
                "進階位階 / 說明": " ； ".join(result["trigger_reasons"]) if result["trigger_reasons"] else "-"
            },
            {
                "分類": "體制總覽",
                "標的 / 指標": "交易訊號狀態",
                "資料日期 (UTC+8)": "-",
                "最新數值 / 狀態": result["state_label"],
                "日漲跌幅(%)": "-",
                "5MA": "-",
                "20MA": "-",
                "站上5MA": "-",
                "站上20MA": "-",
                "進階位階 / 說明": "恐慌高點下移背離" if result.get("panic_divergence") else "-"
            },
            {
                "分類": "體制總覽",
                "標的 / 指標": "關鍵提醒事項",
                "資料日期 (UTC+8)": "-",
                "最新數值 / 狀態": " | ".join(result["alert_reminders"]) if result["alert_reminders"] else "無額外特別提醒",
                "日漲跌幅(%)": "-",
                "5MA": "-",
                "20MA": "-",
                "站上5MA": "-",
                "站上20MA": "-",
                "進階位階 / 說明": "-"
            },
            {
                "分類": "體制總覽",
                "標的 / 指標": "操作與避險建議",
                "資料日期 (UTC+8)": "-",
                "最新數值 / 狀態": result["advice"],
                "日漲跌幅(%)": "-",
                "5MA": "-",
                "20MA": "-",
                "站上5MA": "-",
                "站上20MA": "-",
                "進階位階 / 說明": "-"
            }
        ]

        detail_rows = []
        for row in result.get("metrics_table", []):
            detail_rows.append({
                "分類": "指標明細",
                "標的 / 指標": row["標的 / 指標"],
                "資料日期 (UTC+8)": row["資料日期 (UTC+8)"],
                "最新數值 / 狀態": row["最新數值"] if pd.notna(row["最新數值"]) else "-",
                "日漲跌幅(%)": row["日漲跌幅(%)"] if pd.notna(row["日漲跌幅(%)"]) else "-",
                "5MA": row["5MA"] if pd.notna(row["5MA"]) else "-",
                "20MA": row["20MA"] if pd.notna(row["20MA"]) else "-",
                "站上5MA": row["站上5MA"],
                "站上20MA": row["站上20MA"],
                "進階位階 / 說明": row["進階位階 / 備註"]
            })

        return pd.DataFrame(summary_rows + detail_rows)


# ==========================================
# 外資實質資金流向（匯率＋現貨買賣超連動）監控模組
# ==========================================
FOREIGN_FUND_FLOW_COLUMNS = [
    "日期",
    "外資單日買進(億)",
    "外資單日賣出(億)",
    "外資單日買賣超(億)",
    "當月累計買賣超(億)",
    "USD/TWD匯率",
    "匯率單日變動(%)",
    "外資資金流向判定"
]


def _parse_twse_bfi82u_foreign(payload: dict) -> Optional[Dict]:
    """從證交所 BFI82U JSON 解析「外資及陸資(不含外資自營商)」買進、賣出與買賣差額（億元）。"""
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        return None
    rows = payload.get("data") or []
    if not rows:
        return None

    target_row = None
    for row in rows:
        if not row or len(row) < 4:
            continue
        name = str(row[0]).replace(" ", "").replace("（", "(").replace("）", ")")
        if "外資及陸資" in name and "不含外資自營商" in name:
            target_row = row
            break

    if target_row is None:
        return None

    try:
        buy_val = float(str(target_row[1]).replace(",", "").strip())
        sell_val = float(str(target_row[2]).replace(",", "").strip())
        net_val = float(str(target_row[3]).replace(",", "").strip())
    except Exception:
        return None

    date_compact = str(payload.get("date", "")).strip()
    return {
        "date_compact": date_compact,
        "buy_yi": round(buy_val / 1e8, 2),
        "sell_yi": round(sell_val / 1e8, 2),
        "net_yi": round(net_val / 1e8, 2)
    }


def _request_twse_bfi82u(session: requests.Session, params: dict) -> Optional[dict]:
    url = "https://www.twse.com.tw/rwd/zh/fund/BFI82U"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.twse.com.tw/zh/trading/foreign/bfi82u.html"
    }
    try:
        resp = session.get(url, params=params, headers=headers, timeout=15, verify=False)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _normalize_target_date(target_date) -> Optional[datetime.date]:
    if target_date is None or str(target_date).strip() == "":
        return None
    if isinstance(target_date, datetime.datetime):
        return target_date.date()
    if isinstance(target_date, datetime.date):
        return target_date
    raw = str(target_date).strip().replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        return datetime.datetime.strptime(raw, "%Y%m%d").date()
    return pd.to_datetime(raw).date()


def fetch_foreign_fund_flow(target_date=None) -> Dict:
    """
    取得「外資及陸資(不含外資自營商)」現貨買賣超（含當月累計）與 USD/TWD 匯率變動，
    並依連動規則判定外資實質資金流向狀態。
    """
    tw_tz = datetime.timezone(datetime.timedelta(hours=8))
    now_tw = datetime.datetime.now(tw_tz)
    eval_time_str = now_tw.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")

    try:
        session = get_yf_session()
        parsed_target = _normalize_target_date(target_date)

        # ------------------------------------------
        # 1. 抓取證交所 BFI82U 單日外資買賣超（週末/休市自動回退）
        # ------------------------------------------
        daily_parsed = None
        resolved_date = None
        latest_twse_compact = None

        if parsed_target is None:
            latest_json = _request_twse_bfi82u(session, {"response": "json"})
            daily_parsed = _parse_twse_bfi82u_foreign(latest_json)
            if daily_parsed and len(daily_parsed["date_compact"]) == 8:
                latest_twse_compact = daily_parsed["date_compact"]
                resolved_date = datetime.datetime.strptime(latest_twse_compact, "%Y%m%d").date()

        if daily_parsed is None:
            base_dt = parsed_target if parsed_target is not None else now_tw.date()
            for back_days in range(0, 11):
                cand_dt = base_dt - datetime.timedelta(days=back_days)
                if cand_dt.weekday() >= 5:
                    continue
                cand_str = cand_dt.strftime("%Y%m%d")
                day_json = _request_twse_bfi82u(
                    session,
                    {"response": "json", "type": "day", "dayDate": cand_str, "date": cand_str}
                )
                parsed = _parse_twse_bfi82u_foreign(day_json)
                if parsed:
                    daily_parsed = parsed
                    if len(parsed["date_compact"]) == 8:
                        resolved_date = datetime.datetime.strptime(parsed["date_compact"], "%Y%m%d").date()
                    else:
                        resolved_date = cand_dt
                        daily_parsed["date_compact"] = cand_str
                    break
                time.sleep(0.25)

        if daily_parsed is None or resolved_date is None:
            return {
                "status": "error",
                "msg": "無法從台灣證交所 BFI82U 取得近期有效交易日的外資買賣超資料。",
                "eval_time": eval_time_str
            }

        resolved_compact = resolved_date.strftime("%Y%m%d")
        resolved_date_str = resolved_date.strftime("%Y-%m-%d")

        # ------------------------------------------
        # 2. 計算「當月截至該日之累計外資買賣超（億元）」
        # ------------------------------------------
        month_cum_net_yi = np.nan
        use_month_endpoint = False

        if parsed_target is None:
            use_month_endpoint = True
        else:
            if latest_twse_compact is None:
                latest_probe = _request_twse_bfi82u(session, {"response": "json"})
                probe_parsed = _parse_twse_bfi82u_foreign(latest_probe)
                if probe_parsed and len(probe_parsed["date_compact"]) == 8:
                    latest_twse_compact = probe_parsed["date_compact"]
            if latest_twse_compact and resolved_compact == latest_twse_compact:
                use_month_endpoint = True
            else:
                # 若查詢歷史月份的最後一個營業日，同樣可直接用月統計端點
                next_day = resolved_date + datetime.timedelta(days=1)
                while next_day.weekday() >= 5:
                    next_day += datetime.timedelta(days=1)
                if next_day.month != resolved_date.month:
                    use_month_endpoint = True

        if use_month_endpoint:
            month_json = _request_twse_bfi82u(
                session,
                {"response": "json", "type": "month", "monthDate": resolved_compact}
            )
            month_parsed = _parse_twse_bfi82u_foreign(month_json)
            if month_parsed:
                month_cum_net_yi = month_parsed["net_yi"]
        else:
            # 指定歷史月中日期：自該月 1 日逐日加總至 resolved_date（遇限流則回退月端點）
            cum_sum = 0.0
            valid_days = 0
            cur_d = resolved_date.replace(day=1)
            while cur_d <= resolved_date:
                if cur_d.weekday() < 5:
                    if cur_d == resolved_date:
                        cum_sum += daily_parsed["net_yi"]
                        valid_days += 1
                    else:
                        d_str = cur_d.strftime("%Y%m%d")
                        d_json = _request_twse_bfi82u(
                            session,
                            {"response": "json", "type": "day", "dayDate": d_str, "date": d_str}
                        )
                        d_parsed = _parse_twse_bfi82u_foreign(d_json)
                        if d_parsed:
                            cum_sum += d_parsed["net_yi"]
                            valid_days += 1
                        time.sleep(0.25)
                cur_d += datetime.timedelta(days=1)

            if valid_days > 0:
                month_cum_net_yi = round(cum_sum, 2)
            else:
                month_json = _request_twse_bfi82u(
                    session,
                    {"response": "json", "type": "month", "monthDate": resolved_compact}
                )
                month_parsed = _parse_twse_bfi82u_foreign(month_json)
                if month_parsed:
                    month_cum_net_yi = month_parsed["net_yi"]

        # ------------------------------------------
        # 3. 使用 yfinance 抓取 TWD=X 近 5 個交易日匯率並計算變動
        # ------------------------------------------
        fx_df = pd.DataFrame()
        fx_dl_kwargs = {"interval": "1d", "auto_adjust": False, "progress": False, "threads": False}
        if parsed_target is None:
            fx_dl_kwargs["period"] = "1mo"
        else:
            fx_dl_kwargs["start"] = (resolved_date - datetime.timedelta(days=25)).strftime("%Y-%m-%d")
            fx_dl_kwargs["end"] = (resolved_date + datetime.timedelta(days=2)).strftime("%Y-%m-%d")

        for use_custom_sess in (False, True):
            try:
                kw = fx_dl_kwargs.copy()
                if use_custom_sess:
                    kw["session"] = session
                raw_fx = yf.download("TWD=X", **kw)
                fx_df = CrossMarketVolatilityMonitor._extract_single_ohlcv(raw_fx, "TWD=X")
                if not fx_df.empty and len(fx_df) >= 2:
                    break
            except Exception:
                continue

        if fx_df.empty or len(fx_df) < 2:
            try:
                hist_fx = yf.Ticker("TWD=X").history(period="1mo", auto_adjust=False)
                fx_df = CrossMarketVolatilityMonitor._extract_single_ohlcv(hist_fx, "TWD=X")
            except Exception:
                fx_df = pd.DataFrame()

        if fx_df.empty or len(fx_df) < 2:
            return {
                "status": "error",
                "msg": "無法透過 yfinance 取得足夠的 USD/TWD (TWD=X) 匯率資料。",
                "eval_time": eval_time_str
            }

        # 對齊至 resolved_date（若 TWD=X 延遲則自動回退至最近有效交易日）
        fx_aligned = fx_df[fx_df.index.date <= resolved_date]
        if len(fx_aligned) < 2:
            fx_aligned = fx_df

        fx_5d = fx_aligned.tail(5).copy()
        fx_5d["Pct_Change"] = fx_5d["Close"].pct_change() * 100.0
        fx_5d["Diff_Yuan"] = fx_5d["Close"].diff()

        fx_rate = round(float(fx_5d["Close"].iloc[-1]), 4)
        fx_prev_rate = round(float(fx_5d["Close"].iloc[-2]), 4)
        fx_date_str = fx_5d.index[-1].strftime("%Y-%m-%d")

        fx_change_yuan = round(fx_rate - fx_prev_rate, 4)
        fx_change_jiao = round(fx_change_yuan * 10.0, 2)
        fx_change_ratio = (fx_rate - fx_prev_rate) / fx_prev_rate if fx_prev_rate > 0 else 0.0
        fx_change_pct = round(fx_change_ratio * 100.0, 2)

        fx_recent_5d_list = []
        for idx_dt, r in fx_5d.iterrows():
            fx_recent_5d_list.append({
                "日期": idx_dt.strftime("%Y-%m-%d"),
                "USD/TWD收盤": round(float(r["Close"]), 4),
                "單日變動(元)": round(float(r["Diff_Yuan"]), 4) if pd.notna(r["Diff_Yuan"]) else np.nan,
                "單日變動(角)": round(float(r["Diff_Yuan"]) * 10.0, 2) if pd.notna(r["Diff_Yuan"]) else np.nan,
                "單日變動(%)": round(float(r["Pct_Change"]), 2) if pd.notna(r["Pct_Change"]) else np.nan
            })

        # ------------------------------------------
        # 4. 外資實質匯出入狀態判定標籤 (flow_signal)
        # ------------------------------------------
        net_yi = daily_parsed["net_yi"]
        if (net_yi <= -200.0) and (fx_change_pct >= 0.35):
            flow_signal = "實質撤資匯出警報"
            severity = "error"
            signal_desc = f"外資單日大賣 {abs(net_yi):.2f} 億元且台幣單日貶值 {fx_change_pct:+.2f}%（{fx_change_jiao:+.2f} 角），呈現股匯雙殺實質匯出。"
        elif (net_yi <= -100.0) and (fx_change_pct <= 0.0):
            flow_signal = "帳上換股停泊（未匯出）"
            severity = "warning"
            signal_desc = f"外資單日賣超 {abs(net_yi):.2f} 億元但台幣持平或升值（{fx_change_pct:+.2f}%／{fx_change_jiao:+.2f} 角），資金偏向留滯帳上換股。"
        elif (net_yi >= 150.0) and (fx_change_pct <= -0.30):
            flow_signal = "實質資金匯入偏多"
            severity = "success"
            signal_desc = f"外資單日大買 {net_yi:.2f} 億元且台幣升值 {abs(fx_change_pct):.2f}%（{fx_change_jiao:+.2f} 角），具實質熱錢匯入支撐。"
        else:
            flow_signal = "中性常態波動"
            severity = "info"
            signal_desc = f"外資單日買賣超 {net_yi:+.2f} 億元、匯率變動 {fx_change_pct:+.2f}%（{fx_change_jiao:+.2f} 角），落於常態波動區間。"

        record_excel = {
            "日期": resolved_date_str,
            "外資單日買進(億)": daily_parsed["buy_yi"],
            "外資單日賣出(億)": daily_parsed["sell_yi"],
            "外資單日買賣超(億)": net_yi,
            "當月累計買賣超(億)": month_cum_net_yi if pd.notna(month_cum_net_yi) else np.nan,
            "USD/TWD匯率": fx_rate,
            "匯率單日變動(%)": round(fx_change_ratio, 6),
            "外資資金流向判定": flow_signal
        }

        return {
            "status": "success",
            "msg": "",
            "eval_time": eval_time_str,
            "date": resolved_date_str,
            "date_compact": resolved_compact,
            "fx_date": fx_date_str,
            "foreign_buy_yi": daily_parsed["buy_yi"],
            "foreign_sell_yi": daily_parsed["sell_yi"],
            "foreign_net_yi": net_yi,
            "month_cum_net_yi": month_cum_net_yi if pd.notna(month_cum_net_yi) else np.nan,
            "fx_rate": fx_rate,
            "fx_prev_rate": fx_prev_rate,
            "fx_change_pct": fx_change_pct,
            "fx_change_ratio": round(fx_change_ratio, 6),
            "fx_change_yuan": fx_change_yuan,
            "fx_change_jiao": fx_change_jiao,
            "flow_signal": flow_signal,
            "severity": severity,
            "signal_desc": signal_desc,
            "fx_recent_5d": fx_recent_5d_list,
            "record": record_excel
        }

    except Exception as e:
        return {
            "status": "error",
            "msg": f"外資實質資金流向監控執行異常：{str(e)}",
            "eval_time": eval_time_str
        }


def build_foreign_fund_flow_df(flow_res: Dict, for_excel: bool = False) -> pd.DataFrame:
    """將 fetch_foreign_fund_flow() 結果轉為標準 8 欄 DataFrame。"""
    if not isinstance(flow_res, dict) or flow_res.get("status") != "success":
        err_msg = flow_res.get("msg", "資料取得失敗") if isinstance(flow_res, dict) else "無資料"
        return pd.DataFrame([{
            "日期": "-",
            "外資單日買進(億)": np.nan,
            "外資單日賣出(億)": np.nan,
            "外資單日買賣超(億)": np.nan,
            "當月累計買賣超(億)": np.nan,
            "USD/TWD匯率": np.nan,
            "匯率單日變動(%)": np.nan if for_excel else "-",
            "外資資金流向判定": f"資料異常：{err_msg}"
        }], columns=FOREIGN_FUND_FLOW_COLUMNS)

    rec = flow_res["record"].copy()
    if not for_excel:
        rec["匯率單日變動(%)"] = f"{flow_res['fx_change_pct']:+.2f}%"
    return pd.DataFrame([rec], columns=FOREIGN_FUND_FLOW_COLUMNS)


# ==========================================
# 空頭避風港強勢股（方案 B）盤後選股掃描模組
# ==========================================
SAFE_HAVEN_COLUMNS = [
    "股票代號",
    "股票名稱",
    "狀態判定",
    "最新收盤價",
    "60MA季線價",
    "10MA移動停利參考價",
    "距60MA乖離率(%)",
    "15日超額報酬RS(%)",
    "5日均量(張)"
]


class BearMarketSafeHavenStrategy:
    """空頭避風港強勢股（方案 B）核心量化策略引擎。"""

    GATE_BLOCKED_MSG = "目前大盤尚處於第一波恐慌期或未進入第二波回測壓力測試，暫不啟動方案 B 掃描"
    STATUS_BREAKOUT = "🚀 突破點火（右側試單訊號）"
    STATUS_HOLDING = "🛡️ 通過雙波測試（抗跌蓄勢觀察中）"
    MIN_REQUIRED_BARS = 85

    @staticmethod
    def filter_common_stocks(stock_dict: Dict[str, str]) -> Dict[str, str]:
        """
        從 fetch_all_tw_stocks() 回傳字典中過濾上市(.TW)與上櫃(.TWO)所有普通股：
        僅保留 4 碼純數字且非 00 開頭的代號，並排除 ETF、權證、ETN 與 TDR（91 開頭或 -DR 結尾）。
        """
        filtered = {}
        if not isinstance(stock_dict, dict):
            return filtered

        for ticker, name in stock_dict.items():
            t_str = str(ticker).strip().upper()
            if not (t_str.endswith(".TW") or t_str.endswith(".TWO")):
                continue
            code = t_str.split(".")[0]
            name_str = str(name).strip()
            if not (code.isdigit() and len(code) == 4):
                continue
            if code.startswith("00"):
                continue
            if code.startswith("91") or name_str.upper().endswith("-DR"):
                continue
            filtered[t_str] = name_str
        return filtered

    @staticmethod
    def _normalize_ohlcv_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """統一去除時區、正規化日線日期索引、去重排序並清理無效列。"""
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            lv0 = out.columns.get_level_values(0)
            lv1 = out.columns.get_level_values(1)
            if "Close" in lv0:
                out.columns = lv0
            elif "Close" in lv1:
                out.columns = lv1
            else:
                return pd.DataFrame()

        out = out.loc[:, ~pd.Index(out.columns).duplicated()]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in out.columns for c in required_cols):
            return pd.DataFrame()

        out = out[required_cols].copy()
        for col in required_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        out = out[(out["Close"] > 0) & (out["Low"] > 0) & (out["High"] >= out["Low"]) & (out["Volume"] >= 0)]
        if out.empty:
            return pd.DataFrame()

        idx = pd.to_datetime(out.index)
        try:
            if idx.tz is not None:
                idx = idx.tz_convert("Asia/Taipei").tz_localize(None)
        except Exception:
            if idx.tz is not None:
                idx = idx.tz_localize(None)
        out.index = idx.normalize()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _extract_ticker_df(raw_df: Optional[pd.DataFrame], ticker: str, single_ticker: bool = False) -> pd.DataFrame:
        """從 yfinance 批次或單檔下載結果抽取單一標的之標準化 OHLCV DataFrame。"""
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                lv0 = raw_df.columns.get_level_values(0)
                lv1 = raw_df.columns.get_level_values(1)
                if ticker in lv0:
                    sub = raw_df.xs(ticker, axis=1, level=0)
                elif ticker in lv1:
                    sub = raw_df.xs(ticker, axis=1, level=1)
                elif single_ticker:
                    sub = raw_df.copy()
                else:
                    return pd.DataFrame()
            else:
                sub = raw_df.copy() if single_ticker else pd.DataFrame()
            return BearMarketSafeHavenStrategy._normalize_ohlcv_df(sub)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def fetch_twii_data(session=None, period: str = "6mo") -> pd.DataFrame:
        """下載加權指數 (^TWII) 近 6 個月日線資料並標準化索引。"""
        yf_sess = session if session is not None else get_yf_session()
        for use_sess in (True, False):
            try:
                kw = {
                    "period": period,
                    "interval": "1d",
                    "auto_adjust": False,
                    "progress": False,
                    "threads": False
                }
                if use_sess:
                    kw["session"] = yf_sess
                raw = yf.download("^TWII", **kw)
                df = BearMarketSafeHavenStrategy._extract_ticker_df(raw, "^TWII", single_ticker=True)
                if not df.empty and len(df) >= 75:
                    return df
            except Exception:
                continue

        try:
            hist = yf.Ticker("^TWII", session=yf_sess).history(period=period, auto_adjust=False)
            return BearMarketSafeHavenStrategy._normalize_ohlcv_df(hist)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def evaluate_market_gate(twii_df: Optional[pd.DataFrame] = None, session=None) -> Dict:
        """
        檢核大盤 (^TWII) 條件 1 與條件 2 閘門狀態：
        - 條件 1：最新收盤低於 60MA，且過去 10 個交易日至少 7 日收盤位於 60MA 之下。
        - 條件 2：近 7 個交易日最低點 <= 前段（第 -25 至 -8 日）波段低點 Low_1 * 1.01。
        """
        tw_tz = datetime.timezone(datetime.timedelta(hours=8))
        eval_time_str = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")

        df = BearMarketSafeHavenStrategy._normalize_ohlcv_df(twii_df) if twii_df is not None else BearMarketSafeHavenStrategy.fetch_twii_data(session=session)
        if df.empty or len(df) < 75:
            return {
                "status": "error",
                "eval_time": eval_time_str,
                "gate_passed": False,
                "cond1_passed": False,
                "cond2_passed": False,
                "message": "無法取得足夠的加權指數 (^TWII) 近 6 個月日線資料，無法執行方案 B 大盤閘門檢核。",
                "twii_df": pd.DataFrame()
            }

        df = df.copy()
        df["MA60"] = df["Close"].rolling(60).mean()
        if df["MA60"].iloc[-10:].isna().any():
            return {
                "status": "error",
                "eval_time": eval_time_str,
                "gate_passed": False,
                "cond1_passed": False,
                "cond2_passed": False,
                "message": "加權指數 (^TWII) 季線 (60MA) 樣本不足，無法執行方案 B 大盤閘門檢核。",
                "twii_df": df
            }

        latest_date = df.index[-1].strftime("%Y-%m-%d")
        latest_close = float(df["Close"].iloc[-1])
        latest_ma60 = float(df["MA60"].iloc[-1])

        recent_10 = df.iloc[-10:]
        below_ma60_days_10 = int((recent_10["Close"] < recent_10["MA60"]).sum())
        cond1_passed = bool((latest_close < latest_ma60) and (below_ma60_days_10 >= 7))

        prev_seg = df.iloc[-25:-7]
        recent_seg = df.iloc[-7:]

        low_1_idx = prev_seg["Low"].idxmin()
        prev_low_1 = float(prev_seg.loc[low_1_idx, "Low"])
        prev_low_1_date = low_1_idx.strftime("%Y-%m-%d")

        recent_low_idx = recent_seg["Low"].idxmin()
        recent_low_7d = float(recent_seg.loc[recent_low_idx, "Low"])
        recent_low_7d_date = recent_low_idx.strftime("%Y-%m-%d")

        low_threshold = prev_low_1 * 1.01
        cond2_passed = bool(prev_low_1 > 0 and recent_low_7d <= low_threshold)
        is_new_low = bool(recent_low_7d < prev_low_1)
        low_ratio_pct = round((recent_low_7d / prev_low_1 - 1.0) * 100.0, 2) if prev_low_1 > 0 else np.nan

        gate_passed = bool(cond1_passed and cond2_passed)
        if gate_passed:
            msg = "大盤同時滿足條件 1（破季線時間閘門）與條件 2（第二波回測／破底壓力測試），啟動方案 B 掃描。"
        else:
            msg = BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG

        return {
            "status": "success",
            "eval_time": eval_time_str,
            "gate_passed": gate_passed,
            "cond1_passed": cond1_passed,
            "cond2_passed": cond2_passed,
            "message": msg,
            "latest_date": latest_date,
            "latest_close": round(latest_close, 2),
            "latest_ma60": round(latest_ma60, 2),
            "below_ma60_days_10": below_ma60_days_10,
            "prev_low_1": round(prev_low_1, 2),
            "prev_low_1_date": prev_low_1_date,
            "recent_low_7d": round(recent_low_7d, 2),
            "recent_low_7d_date": recent_low_7d_date,
            "low_threshold": round(low_threshold, 2),
            "low_ratio_pct": low_ratio_pct,
            "is_new_low": is_new_low,
            "twii_df": df
        }

    @staticmethod
    def evaluate_single_stock(
        ticker: str,
        stock_name: str,
        stock_df: Optional[pd.DataFrame],
        twii_df: Optional[pd.DataFrame]
    ) -> Optional[Dict]:
        """
        對單一普通股執行方案 B 條件 3、4、5 量化篩選與訊號分級。
        通過全部條件則回傳標準欄位 dict，否則回傳 None。
        """
        try:
            s_df = BearMarketSafeHavenStrategy._normalize_ohlcv_df(stock_df)
            t_df = BearMarketSafeHavenStrategy._normalize_ohlcv_df(twii_df)
            if s_df.empty or t_df.empty:
                return None
            if len(s_df) < BearMarketSafeHavenStrategy.MIN_REQUIRED_BARS or len(t_df) < 25:
                return None

            s_df = s_df.copy()
            s_df["MA10"] = s_df["Close"].rolling(10).mean()
            s_df["MA60"] = s_df["Close"].rolling(60).mean()
            s_df["Vol_MA5"] = s_df["Volume"].rolling(5).mean()

            common_idx = s_df.index.intersection(t_df.index)
            if len(common_idx) < 26:
                return None

            # 確保個股最新交易日與大盤同步（排除已停牌或過舊標的）
            if common_idx[-1] < t_df.index[-2]:
                return None

            s_aligned = s_df.loc[common_idx]
            t_aligned = t_df.loc[common_idx]

            # 檢查必要指標無 NaN 斷層
            if (
                s_aligned["MA60"].iloc[-15:].isna().any()
                or pd.isna(s_aligned["MA60"].iloc[-11])
                or pd.isna(s_aligned["MA10"].iloc[-1])
                or pd.isna(s_aligned["Vol_MA5"].iloc[-1])
                or pd.isna(s_aligned["Vol_MA5"].iloc[-2])
            ):
                return None

            # 條件 3：個股全程守穩 60MA 季線多方防線
            cond3_all_above_ma60 = bool((s_aligned["Close"].iloc[-15:] > s_aligned["MA60"].iloc[-15:]).all())
            latest_ma60 = float(s_aligned["MA60"].iloc[-1])
            ma60_10d_ago = float(s_aligned["MA60"].iloc[-11])
            cond3_ma60_trend = bool(latest_ma60 > 0 and ma60_10d_ago > 0 and latest_ma60 >= ma60_10d_ago * 0.99)
            if not (cond3_all_above_ma60 and cond3_ma60_trend):
                return None

            # 條件 4：個股結構底背離（底底高 + 近 15 日對大盤超額報酬 RS > +8%）
            stock_prev_low = float(s_aligned["Low"].iloc[-25:-7].min())
            stock_recent_low = float(s_aligned["Low"].iloc[-7:].min())
            if stock_prev_low <= 0 or not (stock_recent_low > stock_prev_low * 1.02):
                return None

            stock_c_now = float(s_aligned["Close"].iloc[-1])
            stock_c_15d = float(s_aligned["Close"].iloc[-16])
            twii_c_now = float(t_aligned["Close"].iloc[-1])
            twii_c_15d = float(t_aligned["Close"].iloc[-16])
            if stock_c_15d <= 0 or twii_c_15d <= 0:
                return None

            stock_ret_15d = (stock_c_now / stock_c_15d - 1.0) * 100.0
            twii_ret_15d = (twii_c_now / twii_c_15d - 1.0) * 100.0
            rs_15d = stock_ret_15d - twii_ret_15d
            if rs_15d <= 8.0:
                return None

            # 條件 5：流動性濾網（5 日均量 > 1,500 張，即 1,500,000 股）
            vol_ma5_shares = float(s_aligned["Vol_MA5"].iloc[-1])
            if vol_ma5_shares <= 1_500_000.0:
                return None

            # 訊號分級：突破近 10 日（不含當日）最高價且當日量 > 前一日 5 日均量 1.3 倍
            prev_10d_high = float(s_aligned["High"].iloc[-11:-1].max())
            latest_vol = float(s_aligned["Volume"].iloc[-1])
            prev_vol_ma5 = float(s_aligned["Vol_MA5"].iloc[-2])
            is_breakout = bool(
                (stock_c_now > prev_10d_high)
                and (prev_vol_ma5 > 0)
                and (latest_vol > prev_vol_ma5 * 1.3)
            )
            status_label = (
                BearMarketSafeHavenStrategy.STATUS_BREAKOUT
                if is_breakout
                else BearMarketSafeHavenStrategy.STATUS_HOLDING
            )

            latest_ma10 = float(s_aligned["MA10"].iloc[-1])
            bias_60ma = (stock_c_now - latest_ma60) / latest_ma60 * 100.0
            vol_ma5_lots = vol_ma5_shares / 1000.0

            return {
                "股票代號": str(ticker).split(".")[0],
                "股票名稱": str(stock_name).strip(),
                "狀態判定": status_label,
                "最新收盤價": round(stock_c_now, 2),
                "60MA季線價": round(latest_ma60, 2),
                "10MA移動停利參考價": round(latest_ma10, 2),
                "距60MA乖離率(%)": round(bias_60ma, 2),
                "15日超額報酬RS(%)": round(rs_15d, 2),
                "5日均量(張)": round(vol_ma5_lots, 1)
            }
        except Exception:
            return None

    @staticmethod
    def _download_chunk_worker(chunk: List[str], period: str = "6mo", request_delay: float = 0.35) -> Dict[str, pd.DataFrame]:
        """單一執行緒批次下載工作函式，內建獨立 Session、Rate Limit 防護與例外隔離。"""
        result_map = {}
        worker_session = get_yf_session()
        single_mode = len(chunk) == 1
        raw = None

        for use_sess in (True, False):
            try:
                kw = {
                    "period": period,
                    "interval": "1d",
                    "auto_adjust": False,
                    "progress": False,
                    "threads": False
                }
                if use_sess:
                    kw["session"] = worker_session
                raw = yf.download(chunk, **kw)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None

        if raw is not None and not raw.empty:
            for ticker in chunk:
                try:
                    df_t = BearMarketSafeHavenStrategy._extract_ticker_df(raw, ticker, single_ticker=single_mode)
                    if not df_t.empty:
                        result_map[ticker] = df_t
                except Exception:
                    continue

        if request_delay > 0:
            time.sleep(request_delay)
        return result_map

    @staticmethod
    def scan_market(
        stock_dict: Optional[Dict[str, str]] = None,
        twii_df: Optional[pd.DataFrame] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
        max_workers: int = 5,
        chunk_size: int = 30,
        request_delay: float = 0.35,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        session=None
    ) -> Dict:
        """
        執行方案 B 完整盤後掃描流程：
        1. 先執行大盤 (^TWII) 條件 1 與條件 2 閘門檢核；未通過則立即停止，不進行個股下載。
        2. 過濾上市／上櫃普通股，優先複用 preloaded_data（若有效筆數 >= 85 日），剩餘標的以 ThreadPoolExecutor 批次下載。
        3. 執行條件 3、4、5 與訊號分級，依 15日超額報酬RS(%) 由高到低排序回傳標準 DataFrame。
        """
        yf_sess = session if session is not None else get_yf_session()
        gate_info = BearMarketSafeHavenStrategy.evaluate_market_gate(twii_df=twii_df, session=yf_sess)

        empty_df = pd.DataFrame(columns=SAFE_HAVEN_COLUMNS)
        if gate_info.get("status") != "success":
            return {
                "status": "error",
                "gate_passed": False,
                "message": gate_info.get("message", "大盤閘門檢核失敗"),
                "gate_info": gate_info,
                "scanned_count": 0,
                "reused_count": 0,
                "results_df": empty_df
            }

        if not gate_info.get("gate_passed", False):
            return {
                "status": "blocked",
                "gate_passed": False,
                "message": BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG,
                "gate_info": gate_info,
                "scanned_count": 0,
                "reused_count": 0,
                "results_df": empty_df
            }

        raw_stocks = stock_dict if stock_dict is not None else fetch_all_tw_stocks()
        common_stocks = BearMarketSafeHavenStrategy.filter_common_stocks(raw_stocks)
        tickers = list(common_stocks.keys())
        total_count = len(tickers)

        if total_count == 0:
            return {
                "status": "error",
                "gate_passed": True,
                "message": "無法取得上市／上櫃普通股清單，無法執行方案 B 掃描。",
                "gate_info": gate_info,
                "scanned_count": 0,
                "reused_count": 0,
                "results_df": empty_df
            }

        twii_clean = gate_info["twii_df"]
        matched_rows = []
        to_download = []
        reused_count = 0
        processed_count = 0

        # 優先複用呼叫端已下載之日線資料（降低重複請求）
        if isinstance(preloaded_data, dict) and preloaded_data:
            for ticker in tickers:
                cached_df = preloaded_data.get(ticker)
                if cached_df is not None and not cached_df.empty:
                    norm_df = BearMarketSafeHavenStrategy._normalize_ohlcv_df(cached_df)
                    if len(norm_df) >= BearMarketSafeHavenStrategy.MIN_REQUIRED_BARS:
                        reused_count += 1
                        processed_count += 1
                        row = BearMarketSafeHavenStrategy.evaluate_single_stock(
                            ticker=ticker,
                            stock_name=common_stocks[ticker],
                            stock_df=norm_df,
                            twii_df=twii_clean
                        )
                        if row is not None:
                            matched_rows.append(row)
                        continue
                to_download.append(ticker)
        else:
            to_download = tickers

        if progress_callback and processed_count > 0:
            try:
                progress_callback(processed_count, total_count, f"已複用快取日線 {processed_count} / {total_count} 檔...")
            except Exception:
                pass

        # 剩餘標的以 ThreadPoolExecutor 分批下載與評估
        if to_download:
            safe_chunk_size = max(1, int(chunk_size))
            safe_workers = max(1, min(int(max_workers), 8))
            chunks = [to_download[i:i + safe_chunk_size] for i in range(0, len(to_download), safe_chunk_size)]

            with ThreadPoolExecutor(max_workers=safe_workers) as executor:
                future_map = {
                    executor.submit(
                        BearMarketSafeHavenStrategy._download_chunk_worker,
                        chunk,
                        "6mo",
                        request_delay
                    ): chunk
                    for chunk in chunks
                }
                for future in as_completed(future_map):
                    chunk = future_map[future]
                    try:
                        chunk_data = future.result()
                    except Exception:
                        chunk_data = {}

                    for ticker in chunk:
                        try:
                            df_t = chunk_data.get(ticker)
                            if df_t is not None and not df_t.empty:
                                row = BearMarketSafeHavenStrategy.evaluate_single_stock(
                                    ticker=ticker,
                                    stock_name=common_stocks[ticker],
                                    stock_df=df_t,
                                    twii_df=twii_clean
                                )
                                if row is not None:
                                    matched_rows.append(row)
                        except Exception:
                            continue

                    processed_count += len(chunk)
                    if progress_callback:
                        try:
                            progress_callback(
                                min(processed_count, total_count),
                                total_count,
                                f"正在掃描方案 B 普通股：{min(processed_count, total_count)} / {total_count} 檔（已入選 {len(matched_rows)} 檔）..."
                            )
                        except Exception:
                            pass

        if matched_rows:
            results_df = pd.DataFrame(matched_rows, columns=SAFE_HAVEN_COLUMNS)
            results_df = results_df.sort_values(by="15日超額報酬RS(%)", ascending=False).reset_index(drop=True)
            msg = f"方案 B 掃描完成，共檢測 {total_count} 檔普通股，精選出 {len(results_df)} 檔空頭避風港強勢股。"
        else:
            results_df = empty_df
            msg = f"方案 B 掃描完成，共檢測 {total_count} 檔普通股，今日無同時符合 5 項嚴格條件之標的。"

        return {
            "status": "success",
            "gate_passed": True,
            "message": msg,
            "gate_info": gate_info,
            "scanned_count": total_count,
            "reused_count": reused_count,
            "results_df": results_df
        }

    @staticmethod
    def build_excel_dataframe(scan_res: Optional[Dict]) -> pd.DataFrame:
        """
        產生供 Excel 第 4 工作表（空頭避風港_方案B）寫入的標準化 DataFrame；
        若當日大盤未通過閘門或無達標標的，則輸出單列狀態說明以維持報表結構完整。
        """
        if isinstance(scan_res, dict):
            res_df = scan_res.get("results_df")
            if isinstance(res_df, pd.DataFrame) and not res_df.empty:
                return res_df[SAFE_HAVEN_COLUMNS].copy()
            status_msg = scan_res.get("message") or BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG
        else:
            status_msg = BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG

        return pd.DataFrame([{
            "股票代號": "-",
            "股票名稱": "方案 B 狀態說明",
            "狀態判定": status_msg,
            "最新收盤價": np.nan,
            "60MA季線價": np.nan,
            "10MA移動停利參考價": np.nan,
            "距60MA乖離率(%)": np.nan,
            "15日超額報酬RS(%)": np.nan,
            "5日均量(張)": np.nan
        }], columns=SAFE_HAVEN_COLUMNS)


# ==========================================
# 純 Python 威科夫週期階段與量價結構判定模組
# ==========================================
WYCKOFF_REPORT_COLUMNS = [
    "威科夫階段",
    "威科夫階段代碼",
    "威科夫事件",
    "威科夫信心分",
    "威科夫操作訊號",
    "冰線支撐",
    "阻力小溪",
    "騙線警示",
    "努力與結果",
    "威科夫判讀摘要"
]


class WyckoffPhaseAnalyzer:
    """
    100% 純 Python 確定性威科夫（Wyckoff）量價週期與結構判定引擎。
    直接接收已下載之日線 DataFrame 運算，零 AI 呼叫、零額外行情請求。
    """

    MIN_BARS = 60
    VALID_PHASES = (
        "Phase_A_B_Accumulation",
        "Phase_C_D_E_Markup",
        "Phase_A_B_Distribution",
        "Phase_C_D_E_Markdown"
    )
    VALID_EVENTS = ("SC", "AR", "ST", "Spring", "JAC", "LPS", "BC", "UT", "SOW", "LPSY")
    VALID_ACTIONS = ("WAIT_OBSERVE", "PULLBACK_BUY", "REBOUND_SELL_LPSY", "STOP_LOSS_EXIT")

    @staticmethod
    def _safe_num(val, default: float = 0.0) -> float:
        try:
            f = float(val)
            return f if np.isfinite(f) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _default_result(stock_id: str, stock_name: str, ref_price: float = 0.0, reason: str = "歷史K線不足60根，暫不判定威科夫週期。") -> Dict:
        price_val = round(WyckoffPhaseAnalyzer._safe_num(ref_price, 0.0), 2)
        summary = str(reason)[:50]
        return {
            "stock_id": str(stock_id or "").strip(),
            "stock_name": str(stock_name or "").strip(),
            "wyckoff_phase": "Phase_A_B_Accumulation",
            "phase_label_zh": "資料不足（未滿60根K線）",
            "key_event": "ST",
            "confidence_score": 0,
            "ice_line_support": float(price_val),
            "creek_resistance": float(price_val),
            "trap_warning": "無明顯騙線",
            "effort_vs_result_analysis": "有效K線不足60根，無法執行威科夫量價分析",
            "action_signal": "WAIT_OBSERVE",
            "summary_reason": summary
        }

    @staticmethod
    def _clean_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            lv0 = out.columns.get_level_values(0)
            lv1 = out.columns.get_level_values(1)
            if "Close" in lv0:
                out.columns = lv0
            elif "Close" in lv1:
                out.columns = lv1
            else:
                return pd.DataFrame()

        out = out.loc[:, ~pd.Index(out.columns).duplicated()]
        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in out.columns for c in ["Open", "High", "Low", "Close"]):
            return pd.DataFrame()
        if "Volume" not in out.columns:
            out["Volume"] = 0.0

        out = out[req_cols].copy()
        for c in req_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0.0)
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        out = out[
            (out["Open"] > 0)
            & (out["High"] > 0)
            & (out["Low"] > 0)
            & (out["Close"] > 0)
            & (out["High"] >= out["Low"])
        ]
        if out.empty:
            return pd.DataFrame()

        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _calc_5d_slope_pct(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) < 2:
            return 0.0
        cur = float(valid.iloc[-1])
        prev = float(valid.iloc[-6]) if len(valid) >= 6 else float(valid.iloc[0])
        if prev <= 0 or not np.isfinite(cur) or not np.isfinite(prev):
            return 0.0
        return (cur - prev) / prev * 100.0

    @staticmethod
    def to_report_fields(res: Optional[Dict]) -> Dict:
        """將 evaluate() 輸出字典轉換為 scan_task3.py 與 app.py 統一報表欄位。"""
        if not isinstance(res, dict):
            res = WyckoffPhaseAnalyzer._default_result("", "")
        return {
            "威科夫階段": str(res.get("phase_label_zh", "資料不足（未滿60根K線）")),
            "威科夫階段代碼": str(res.get("wyckoff_phase", "Phase_A_B_Accumulation")),
            "威科夫事件": str(res.get("key_event", "ST")),
            "威科夫信心分": int(WyckoffPhaseAnalyzer._safe_num(res.get("confidence_score", 0), 0)),
            "威科夫操作訊號": str(res.get("action_signal", "WAIT_OBSERVE")),
            "冰線支撐": round(WyckoffPhaseAnalyzer._safe_num(res.get("ice_line_support", 0.0), 0.0), 2),
            "阻力小溪": round(WyckoffPhaseAnalyzer._safe_num(res.get("creek_resistance", 0.0), 0.0), 2),
            "騙線警示": str(res.get("trap_warning", "無明顯騙線")),
            "努力與結果": str(res.get("effort_vs_result_analysis", "無資料")),
            "威科夫判讀摘要": str(res.get("summary_reason", ""))[:50]
        }

    @staticmethod
    def evaluate(df: Optional[pd.DataFrame], stock_id: str = "", stock_name: str = "") -> Dict:
        s_id = str(stock_id or "").strip()
        s_name = str(stock_name or "").strip()

        try:
            clean_df = WyckoffPhaseAnalyzer._clean_ohlcv(df)
            if clean_df.empty:
                return WyckoffPhaseAnalyzer._default_result(s_id, s_name, 0.0)

            latest_close_raw = float(clean_df["Close"].iloc[-1])
            if len(clean_df) < WyckoffPhaseAnalyzer.MIN_BARS:
                return WyckoffPhaseAnalyzer._default_result(s_id, s_name, latest_close_raw)

            work = clean_df.copy()
            close = work["Close"]
            high = work["High"]
            low = work["Low"]
            open_p = work["Open"]
            vol = work["Volume"]

            # 1. 基礎量價與結構特徵計算
            work["MA20"] = close.rolling(20, min_periods=20).mean()
            work["MA60"] = close.rolling(60, min_periods=60).mean()
            work["MA120"] = close.rolling(120, min_periods=60).mean()
            work["Vol_MA5"] = vol.rolling(5, min_periods=1).mean()
            work["Vol_MA10"] = vol.rolling(10, min_periods=1).mean()
            work["Vol_MA20"] = vol.rolling(20, min_periods=1).mean()

            with np.errstate(divide="ignore", invalid="ignore"):
                work["Vol_Ratio"] = np.where(work["Vol_MA20"] > 0, vol / work["Vol_MA20"], 0.0)
            work["Vol_Ratio"] = pd.Series(work["Vol_Ratio"], index=work.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)

            work["Pct_Change"] = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
            prev_c = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_c).abs(),
                (low - prev_c).abs()
            ], axis=1).max(axis=1)
            work["ATR14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

            cur_close = float(close.iloc[-1])
            cur_open = float(open_p.iloc[-1])
            cur_ma20 = WyckoffPhaseAnalyzer._safe_num(work["MA20"].iloc[-1], cur_close)
            prev_ma20 = WyckoffPhaseAnalyzer._safe_num(work["MA20"].iloc[-2], cur_ma20) if len(work) >= 2 else cur_ma20
            cur_ma60 = WyckoffPhaseAnalyzer._safe_num(work["MA60"].iloc[-1], cur_close)
            cur_ma120 = WyckoffPhaseAnalyzer._safe_num(work["MA120"].iloc[-1], cur_ma60)
            cur_atr = WyckoffPhaseAnalyzer._safe_num(work["ATR14"].iloc[-1], cur_close * 0.02)
            cur_vol_ma5 = WyckoffPhaseAnalyzer._safe_num(work["Vol_MA5"].iloc[-1], 0.0)
            cur_vol_ma10 = WyckoffPhaseAnalyzer._safe_num(work["Vol_MA10"].iloc[-1], 0.0)
            cur_vol_ma20 = WyckoffPhaseAnalyzer._safe_num(work["Vol_MA20"].iloc[-1], 0.0)
            cur_vol_ratio = WyckoffPhaseAnalyzer._safe_num(work["Vol_Ratio"].iloc[-1], 0.0)
            cur_pct = WyckoffPhaseAnalyzer._safe_num(work["Pct_Change"].iloc[-1], 0.0)

            ma20_slope = WyckoffPhaseAnalyzer._calc_5d_slope_pct(work["MA20"])
            ma60_slope = WyckoffPhaseAnalyzer._calc_5d_slope_pct(work["MA60"])
            ma120_slope = WyckoffPhaseAnalyzer._calc_5d_slope_pct(work["MA120"])
            ma20_turn_up = bool(cur_ma20 >= prev_ma20)
            bias20 = ((cur_close - cur_ma20) / cur_ma20 * 100.0) if cur_ma20 > 0 else 0.0

            # 關鍵價位界定：以前段 55 日 [-60:-5] 避免近 5 日測試/突破自我參照
            ref_df = work.iloc[-60:-5]
            ref_lows = ref_df["Low"].values
            ref_highs = ref_df["High"].values

            piv_low_idx = argrelextrema(ref_lows, np.less, order=3)[0]
            piv_high_idx = argrelextrema(ref_highs, np.greater, order=3)[0]

            if len(piv_low_idx) >= 2:
                ice_line = float(np.percentile(ref_lows[piv_low_idx], 20))
            else:
                ice_line = float(np.percentile(ref_lows, 10))

            if len(piv_high_idx) >= 2:
                creek_res = float(np.percentile(ref_highs[piv_high_idx], 80))
            else:
                creek_res = float(np.percentile(ref_highs, 90))

            if creek_res <= ice_line * 1.02:
                ice_line = float(np.min(ref_lows))
                creek_res = float(np.max(ref_highs))
            if creek_res <= ice_line * 1.01:
                pad = max(cur_atr * 1.5, cur_close * 0.03)
                ice_line = max(0.01, cur_close - pad)
                creek_res = cur_close + pad

            # 120日位階百分位與回撤/漲幅基準
            win_120 = work.iloc[-min(len(work), 120):]
            low_120 = float(win_120["Low"].min())
            high_120 = float(win_120["High"].max())
            pos_120d = ((cur_close - low_120) / (high_120 - low_120)) if high_120 > low_120 else 0.5
            pos_120d = float(np.clip(pos_120d, 0.0, 1.0))

            win_60 = work.iloc[-60:]
            high_60 = float(win_60["High"].max())
            dd_60 = ((high_60 - cur_close) / high_60 * 100.0) if high_60 > 0 else 0.0
            dd_120 = ((high_120 - cur_close) / high_120 * 100.0) if high_120 > 0 else 0.0
            max_drawdown_pct = max(dd_60, dd_120)

            close_60_ago = float(win_60["Close"].iloc[0])
            ret_60d = ((cur_close / close_60_ago - 1.0) * 100.0) if close_60_ago > 0 else 0.0
            close_10_ago = float(work["Close"].iloc[-10]) if len(work) >= 10 else close_60_ago
            ret_10d = ((cur_close / close_10_ago - 1.0) * 100.0) if close_10_ago > 0 else 0.0

            # 2. 三大威科夫物理印記
            recent5 = work.iloc[-5:]
            low5_min = float(recent5["Low"].min())
            low5_pos_120d = ((low5_min - low_120) / (high_120 - low_120)) if high_120 > low_120 else 0.5
            low5_pos_120d = float(np.clip(low5_pos_120d, 0.0, 1.0))

            vol_zero_issue = (cur_vol_ma20 <= 0.0) or (float(recent5["Volume"].sum()) <= 0.0)

            # (A) 努力與結果 (Effort vs. Result)
            evr_type = "normal"
            evr_short = "量價配合正常"
            evr_vol_ratio = cur_vol_ratio
            if vol_zero_issue:
                effort_text = "近期成交量為零或資料缺漏，無法驗證量能努力"
                evr_short = "量能資料不足"
            else:
                effort_text = ""
                for idx_pos in range(len(work) - 1, len(work) - 6, -1):
                    row = work.iloc[idx_pos]
                    vr = WyckoffPhaseAnalyzer._safe_num(row["Vol_Ratio"], 0.0)
                    if vr <= 1.5:
                        continue
                    o_v = float(row["Open"])
                    h_v = float(row["High"])
                    l_v = float(row["Low"])
                    c_v = float(row["Close"])
                    pct_v = WyckoffPhaseAnalyzer._safe_num(row["Pct_Change"], 0.0)

                    body = abs(c_v - o_v)
                    body_pct = (body / o_v * 100.0) if o_v > 0 else 0.0
                    total_rng = max(h_v - l_v, 1e-8)
                    upper_sh = h_v - max(o_v, c_v)
                    lower_sh = min(o_v, c_v) - l_v

                    is_bear_engulf = False
                    if idx_pos > 0:
                        prev_row = work.iloc[idx_pos - 1]
                        p_o, p_c = float(prev_row["Open"]), float(prev_row["Close"])
                        if (c_v < o_v) and (p_c > p_o) and (o_v >= p_c) and (c_v <= p_o) and (pct_v <= -1.5):
                            is_bear_engulf = True

                    if (pct_v <= -2.5 and c_v < o_v) or is_bear_engulf:
                        evr_type = "bear_selloff"
                        evr_vol_ratio = vr
                        evr_short = "帶量長黑賣壓湧現"
                        effort_text = f"帶量長黑賣壓湧現（近5日量比{vr:.2f}倍、單日{pct_v:+.2f}%），空方力道主導"
                        break
                    if (upper_sh > 1.5 * body and upper_sh / total_rng >= 0.40) or (pos_120d >= 0.65 and body_pct < 1.2):
                        evr_type = "high_stall"
                        evr_vol_ratio = vr
                        evr_short = "高檔爆量滯漲"
                        effort_text = f"高檔爆量滯漲（量比{vr:.2f}倍、實體僅{body_pct:.2f}%且遇上檔反壓），努力與結果背離"
                        break
                    if (lower_sh > 1.5 * body and lower_sh / total_rng >= 0.40) and (pos_120d <= 0.45 or cur_close <= cur_ma20):
                        evr_type = "low_absorb"
                        evr_vol_ratio = vr
                        evr_short = "低檔爆量長下影承接"
                        effort_text = f"低檔爆量長下影承接（量比{vr:.2f}倍），買盤吸收恐慌賣壓"
                        break
                    if body_pct < 1.2:
                        evr_type = "narrow_stall"
                        evr_vol_ratio = vr
                        evr_short = "爆量實體狹窄僵持"
                        effort_text = f"爆量但實體收幅狹窄（量比{vr:.2f}倍、實體{body_pct:.2f}%），努力與結果背離"
                        break

                if not effort_text:
                    if cur_vol_ratio >= 1.2 and cur_pct > 0:
                        evr_short = "溫和放量推升"
                        effort_text = f"量價配合正常，溫和放量推動價格上行（量比{cur_vol_ratio:.2f}倍）"
                    elif cur_vol_ratio < 0.85:
                        evr_short = "量縮整理沉澱"
                        effort_text = f"量價配合正常，量縮整理籌碼沉澱（量比{cur_vol_ratio:.2f}倍）"
                    else:
                        evr_short = "量價配合正常"
                        effort_text = f"近期量價配合正常（量比{cur_vol_ratio:.2f}倍），未見顯著努力與結果背離"

            # (B) 收盤價意志 (Closing Intent)
            has_ut = bool(((recent5["High"] > creek_res) & (recent5["Close"] < creek_res)).any())
            has_spring = bool(((recent5["Low"] < ice_line) & (recent5["Close"] > ice_line)).any())
            has_sow_bar = bool(
                (
                    (recent5["Close"] < ice_line)
                    & (recent5["Close"] < recent5["Open"])
                    & ((recent5["Vol_Ratio"] >= 1.3) | (recent5["Pct_Change"] <= -2.5))
                ).any()
            )
            has_sow_break = bool(has_sow_bar and (cur_close < ice_line))
            has_jac_bar = bool(
                ((recent5["Close"] > creek_res) & (recent5["Vol_Ratio"] >= 1.3)).any()
            )
            has_jac_break = bool(has_jac_bar and (cur_close > creek_res))

            # (C) 因果堆積 (Cause & Effect)
            in_tr_mask = (win_60["Close"] >= ice_line * 0.98) & (win_60["Close"] <= creek_res * 1.02)
            tr_days = int(in_tr_mask.sum())
            tr_width_pct = ((creek_res - ice_line) / ice_line * 100.0) if ice_line > 0 else 999.0
            sufficient_cause = bool((tr_days >= 25) and (tr_width_pct <= 35.0))
            is_narrow_base = bool((tr_days >= 25) and (tr_width_pct <= 25.0))

            # 輔助結構條件
            bull_ma_align = bool((cur_close > cur_ma20) and (cur_ma20 > cur_ma60) and (ma20_slope > 0))
            bear_ma_align = bool((cur_close < cur_ma20) and (cur_ma20 < cur_ma60) and (ma20_slope < 0))
            is_high_zone = bool((pos_120d >= 0.70) or (ret_60d >= 20.0))
            is_low_zone = bool(
                (pos_120d <= 0.40)
                or (low5_pos_120d <= 0.35)
                or (max_drawdown_pct >= 15.0)
                or (is_narrow_base and pos_120d < 0.70)
            )
            is_early_breakout = bool(
                sufficient_cause
                and (pos_120d < 0.70)
                and (cur_pct >= 3.0)
                and (cur_close > cur_open)
                and (cur_vol_ratio >= 1.2)
                and (cur_close > cur_ma20)
                and (cur_close > cur_ma60)
                and ma20_turn_up
            )

            high_sideways_churn = bool(
                (cur_close >= ice_line + 0.65 * (creek_res - ice_line))
                and (cur_vol_ma20 > 0)
                and (cur_vol_ma10 > 1.15 * cur_vol_ma20)
                and (ret_10d < 2.0)
            )
            is_lps_pullback = bool(
                bull_ma_align
                and (cur_close >= ice_line * 0.99)
                and ((cur_vol_ratio <= 0.90) or (cur_vol_ma20 > 0 and cur_vol_ma5 <= cur_vol_ma20 * 0.95))
            )
            is_st_confirm = bool(
                (cur_vol_ma20 > 0)
                and (cur_vol_ma5 < 0.85 * cur_vol_ma20)
                and (cur_close >= ice_line * 0.99)
                and sufficient_cause
            )

            # 3. 四大生命週期階段互斥優先級判定（強制四擇一）
            matched_priority = True
            if has_sow_break:
                wyckoff_phase = "Phase_C_D_E_Markdown"
                key_event = "SOW"
                phase_label_zh = "派發後下跌階段 - SOW有效破冰線"
                action_signal = "STOP_LOSS_EXIT"
            elif is_high_zone and (has_ut or evr_type in ("high_stall", "narrow_stall") or high_sideways_churn):
                wyckoff_phase = "Phase_A_B_Distribution"
                if has_ut:
                    key_event = "UT"
                    phase_label_zh = "高檔派發階段 - UT誘多滯漲"
                    action_signal = "REBOUND_SELL_LPSY"
                else:
                    key_event = "BC"
                    phase_label_zh = "高檔派發階段 - BC爆量滯漲"
                    action_signal = "REBOUND_SELL_LPSY" if (evr_type == "bear_selloff" or cur_close < cur_ma20) else "WAIT_OBSERVE"
            elif (bull_ma_align and (has_jac_break or is_lps_pullback or cur_close > creek_res)) or is_early_breakout:
                wyckoff_phase = "Phase_C_D_E_Markup"
                if has_jac_break or (cur_close > creek_res and cur_vol_ratio >= 1.1):
                    key_event = "JAC"
                    phase_label_zh = "拉升主升階段 - JAC突破小溪"
                    action_signal = "WAIT_OBSERVE" if bias20 > 8.0 else "PULLBACK_BUY"
                elif is_early_breakout:
                    key_event = "JAC"
                    phase_label_zh = "拉升主升階段 - JAC起漲突破"
                    action_signal = "WAIT_OBSERVE" if bias20 > 8.0 else "PULLBACK_BUY"
                else:
                    key_event = "LPS"
                    phase_label_zh = "拉升主升階段 - LPS縮量回測支撐"
                    action_signal = "WAIT_OBSERVE" if bias20 > 8.0 else "PULLBACK_BUY"
            elif (has_spring and pos_120d < 0.70) or (
                is_low_zone and (
                    evr_type == "low_absorb"
                    or is_st_confirm
                    or (not bear_ma_align and cur_close >= ice_line * 0.98)
                )
            ):
                wyckoff_phase = "Phase_A_B_Accumulation"
                if has_spring:
                    key_event = "Spring"
                    phase_label_zh = "吸籌築底階段 - Spring破底翻"
                    action_signal = "PULLBACK_BUY"
                elif evr_type == "low_absorb":
                    key_event = "SC"
                    phase_label_zh = "吸籌築底階段 - SC拋售高潮承接"
                    action_signal = "WAIT_OBSERVE"
                elif is_st_confirm:
                    key_event = "ST"
                    phase_label_zh = "吸籌築底階段 - ST縮量二次測試"
                    action_signal = "PULLBACK_BUY"
                else:
                    key_event = "AR"
                    phase_label_zh = "吸籌築底階段 - AR自動反彈整理" if sufficient_cause else "吸籌築底階段 - AR跌深反彈(堆積不足)"
                    action_signal = "WAIT_OBSERVE"
            elif bear_ma_align:
                wyckoff_phase = "Phase_C_D_E_Markdown"
                key_event = "LPSY"
                phase_label_zh = "派發後下跌階段 - 回測LPSY反壓"
                action_signal = "REBOUND_SELL_LPSY"
            else:
                matched_priority = False
                action_signal = "WAIT_OBSERVE"
                if cur_close >= cur_ma20 and ma20_slope >= 0:
                    wyckoff_phase = "Phase_C_D_E_Markup"
                    key_event = "LPS"
                    phase_label_zh = "拉升主升階段 - 區間偏多整理"
                elif cur_close < cur_ma20 and ma20_slope < 0:
                    wyckoff_phase = "Phase_C_D_E_Markdown"
                    key_event = "LPSY"
                    phase_label_zh = "派發後下跌階段 - 均線反壓整理"
                elif cur_close >= cur_ma20 and cur_close >= cur_ma60:
                    wyckoff_phase = "Phase_A_B_Accumulation"
                    key_event = "ST"
                    phase_label_zh = "吸籌築底階段 - 築底轉強測試"
                elif pos_120d >= 0.65 and not (cur_pct >= 3.0 and cur_close > cur_open):
                    wyckoff_phase = "Phase_A_B_Distribution"
                    key_event = "BC"
                    phase_label_zh = "高檔派發階段 - 中高檔橫盤震盪"
                else:
                    wyckoff_phase = "Phase_A_B_Accumulation"
                    key_event = "ST"
                    phase_label_zh = "吸籌築底階段 - 低中檔橫盤測試"

            # 4. 信心分數 (0~100 整數)
            score = 50 if matched_priority else 30

            # 均線與位階一致性 (+0~20)
            if wyckoff_phase == "Phase_C_D_E_Markup":
                if bull_ma_align or is_early_breakout:
                    score += 10
                if ma60_slope >= 0 and ma120_slope >= -0.05:
                    score += 10
            elif wyckoff_phase == "Phase_C_D_E_Markdown":
                if bear_ma_align or cur_close < ice_line:
                    score += 10
                if ma20_slope < 0 and ma60_slope <= 0:
                    score += 10
            elif wyckoff_phase == "Phase_A_B_Distribution":
                if is_high_zone:
                    score += 10
                if pos_120d >= 0.80 or high_sideways_churn:
                    score += 10
            else:
                if is_low_zone:
                    score += 10
                if pos_120d <= 0.30 or low5_pos_120d <= 0.25 or max_drawdown_pct >= 20.0:
                    score += 10

            # 量能驗證強度 (+0~20)
            if not vol_zero_issue:
                if key_event in ("SOW", "JAC", "SC", "BC", "UT"):
                    check_vr = max(cur_vol_ratio, evr_vol_ratio)
                    if check_vr >= 1.5:
                        score += 20
                    elif check_vr >= 1.2:
                        score += 10
                elif key_event in ("LPS", "ST", "LPSY"):
                    if cur_vol_ratio <= 0.85 or (cur_vol_ma20 > 0 and cur_vol_ma5 <= cur_vol_ma20 * 0.85):
                        score += 20
                    elif cur_vol_ratio <= 1.0:
                        score += 10
                else:
                    if cur_vol_ratio >= 1.2 or cur_vol_ratio <= 0.85:
                        score += 10

            # 因果堆積與收盤意志加成 (+0~10)
            if sufficient_cause:
                score += 5
            if has_spring or has_ut or has_sow_break or has_jac_break or is_early_breakout:
                score += 5

            # 懲罰扣分
            if wyckoff_phase == "Phase_A_B_Accumulation" and key_event == "AR" and not sufficient_cause:
                score -= 10
            if wyckoff_phase == "Phase_C_D_E_Markup" and bias20 > 8.0:
                score -= 10
            if vol_zero_issue:
                score -= 10

            confidence_score = int(np.clip(round(score), 0, 100))

            # 5. 騙線警示與 50 字內結構化摘要
            if has_ut:
                trap_warning = "⚠️ UT上衝回落誘多陷阱，慎防假突破套牢"
            elif has_spring:
                trap_warning = "⚠️ Spring破底翻掃損誘空，留意主力洗盤後轉強"
            elif wyckoff_phase == "Phase_C_D_E_Markdown" and key_event == "SOW":
                trap_warning = "⚠️ 帶量有效跌破冰線，慎防破線後假反彈陷阱"
            elif wyckoff_phase == "Phase_A_B_Accumulation" and key_event == "AR" and not sufficient_cause:
                trap_warning = "⚠️ 底部橫盤堆積不足，慎防跌深逃命反彈(AR/LPSY)"
            elif wyckoff_phase == "Phase_A_B_Distribution":
                trap_warning = "⚠️ 高檔爆量滯漲，慎防主力拉高出貨派發"
            else:
                trap_warning = "無明顯騙線"

            ice_r = round(WyckoffPhaseAnalyzer._safe_num(ice_line, cur_close), 2)
            creek_r = round(WyckoffPhaseAnalyzer._safe_num(creek_res, cur_close), 2)

            short_phase_map = {
                "Phase_A_B_Accumulation": "吸籌築底",
                "Phase_C_D_E_Markup": "拉升主升",
                "Phase_A_B_Distribution": "高檔派發",
                "Phase_C_D_E_Markdown": "派發下跌"
            }
            summary_reason = f"{short_phase_map[wyckoff_phase]}{key_event}，冰線{ice_r:.2f}/小溪{creek_r:.2f}，{evr_short}。"
            if len(summary_reason) > 50:
                summary_reason = summary_reason[:49] + "。"

            return {
                "stock_id": s_id,
                "stock_name": s_name,
                "wyckoff_phase": wyckoff_phase,
                "phase_label_zh": phase_label_zh,
                "key_event": key_event,
                "confidence_score": confidence_score,
                "ice_line_support": float(ice_r),
                "creek_resistance": float(creek_r),
                "trap_warning": trap_warning,
                "effort_vs_result_analysis": effort_text,
                "action_signal": action_signal,
                "summary_reason": summary_reason
            }
        except Exception:
            return WyckoffPhaseAnalyzer._default_result(s_id, s_name, 0.0, "威科夫運算發生異常，已降級回傳預設狀態。")
