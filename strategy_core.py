import os
import json
import time
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
# 實體波段留倉量（扣除當日沖銷）後置檢核模組
# ==========================================
_TWTB4U_DATE_CACHE: Dict[str, Dict[str, float]] = {}


def _fetch_twse_twtb4u_by_date(session: requests.Session, date_compact: str) -> Dict[str, float]:
    """抓取證交所 TWTB4U 單日上市當沖成交股數並轉換為張數，寫入執行期間快取。"""
    if date_compact in _TWTB4U_DATE_CACHE:
        return _TWTB4U_DATE_CACHE[date_compact]

    day_map: Dict[str, float] = {}
    url = "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.twse.com.tw/zh/trading/dayTrading/twtb4u.html"
    }
    try:
        resp = session.get(
            url,
            params={"response": "json", "date": date_compact},
            headers=headers,
            timeout=15,
            verify=False
        )
        time.sleep(0.5)
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("stat") == "OK":
                tables = payload.get("tables") or []
                target_fields = []
                target_rows = []
                for tbl in tables:
                    fields = tbl.get("fields") or []
                    if any("證券代號" in str(f) for f in fields):
                        target_fields = [str(f).strip() for f in fields]
                        target_rows = tbl.get("data") or []
                        break
                if not target_rows and payload.get("data"):
                    target_fields = [str(f).strip() for f in (payload.get("fields") or [])]
                    target_rows = payload.get("data") or []

                vol_col_idx = 3
                for idx, fname in enumerate(target_fields):
                    if "成交股數" in fname:
                        vol_col_idx = idx
                        break

                for row in target_rows:
                    if not row or len(row) <= vol_col_idx:
                        continue
                    code = str(row[0]).strip()
                    raw_vol = str(row[vol_col_idx]).replace(",", "").strip()
                    try:
                        day_map[code] = float(raw_vol) / 1000.0
                    except Exception:
                        continue
    except Exception:
        pass

    _TWTB4U_DATE_CACHE[date_compact] = day_map
    return day_map


def fetch_day_trading_volume(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    串接 FinMind (TaiwanStockDayTrading) 或證交所 OpenAPI (TWTB4U) 取得每日當沖張數。
    回傳以無時區正規化日期為索引之 DataFrame，包含欄位 `day_trade_vol` (張)。
    """
    empty_df = pd.DataFrame(columns=["day_trade_vol"])
    empty_df.index = pd.DatetimeIndex([], name="date")

    code = str(stock_id).strip().upper().split(".")[0]
    if not code:
        return empty_df

    try:
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()
    except Exception:
        return empty_df

    session = get_yf_session()

    # 1. 第一順位：FinMind TaiwanStockDayTrading（支援上市 .TW 與上櫃 .TWO 區間查詢）
    try:
        params = {
            "dataset": "TaiwanStockDayTrading",
            "data_id": code,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d")
        }
        fm_token = os.environ.get("FINMIND_TOKEN", "").strip()
        if fm_token:
            params["token"] = fm_token

        resp = session.get(
            "https://api.finmindtrade.com/api/v4/data",
            params=params,
            timeout=15,
            verify=False
        )
        time.sleep(0.3)

        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, dict) and str(payload.get("msg", "")).lower() == "success":
                rows = payload.get("data") or []
                if rows:
                    records = []
                    for r in rows:
                        d_str = r.get("date")
                        vol_raw = r.get("Volume", 0)
                        if d_str is None:
                            continue
                        try:
                            vol_lots = float(vol_raw) / 1000.0
                        except Exception:
                            vol_lots = 0.0
                        records.append({"date": pd.to_datetime(d_str).normalize(), "day_trade_vol": max(0.0, vol_lots)})
                    if records:
                        dt_df = pd.DataFrame(records).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
                        return dt_df[["day_trade_vol"]]
                else:
                    # FinMind 成功回傳但該股區間無當沖紀錄（如暫停先賣後買或無當沖）
                    return empty_df
    except Exception:
        pass

    # 2. 第二順位備援：台灣證交所 OpenAPI (TWTB4U)
    try:
        records = []
        b_days = pd.bdate_range(start=start_dt, end=end_dt)
        for dt_ts in b_days:
            d_compact = dt_ts.strftime("%Y%m%d")
            day_map = _fetch_twse_twtb4u_by_date(session, d_compact)
            if code in day_map:
                records.append({"date": dt_ts.normalize(), "day_trade_vol": max(0.0, float(day_map[code]))})
        if records:
            dt_df = pd.DataFrame(records).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
            return dt_df[["day_trade_vol"]]
    except Exception:
        pass

    return empty_df


def calc_real_volume_metrics(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """
    計算扣除當日沖銷後的實體波段留倉量指標與威科夫高當沖假爆量陷阱判定。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    required_cols = {"Volume", "High", "Low", "Close"}
    if not required_cols.issubset(set(out.columns)):
        return out

    idx_norm = pd.to_datetime(out.index)
    if idx_norm.tz is not None:
        idx_norm = idx_norm.tz_localize(None)
    idx_norm = idx_norm.normalize()

    end_dt = idx_norm.max()
    start_dt = min(idx_norm.min(), end_dt - pd.Timedelta(days=45))

    dt_df = fetch_day_trading_volume(
        stock_id=stock_id,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d")
    )

    if not dt_df.empty:
        aligned_dt = dt_df["day_trade_vol"].reindex(idx_norm).fillna(0.0).values
    else:
        aligned_dt = np.zeros(len(out), dtype=float)

    total_vol = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0) / 1000.0
    day_trade_vol = pd.Series(aligned_dt, index=out.index, dtype=float).fillna(0.0).clip(lower=0.0)

    real_vol = (total_vol - day_trade_vol).clip(lower=0.0)
    day_trade_ratio = pd.Series(
        np.where(total_vol > 0, (day_trade_vol / total_vol).clip(0.0, 1.0), 0.0),
        index=out.index,
        dtype=float
    )

    real_vol_ma5 = real_vol.rolling(5).mean()
    real_vol_ma20 = real_vol.rolling(20).mean()
    real_vol_ratio = pd.Series(
        np.where((real_vol_ma20 > 0) & pd.notna(real_vol_ma20), real_vol / real_vol_ma20, 0.0),
        index=out.index,
        dtype=float
    )

    total_vol_ma20 = total_vol.rolling(20).mean()
    hl_range = (pd.to_numeric(out["High"], errors="coerce") - pd.to_numeric(out["Low"], errors="coerce")).replace(0, np.nan)
    upper_weak_ratio = (pd.to_numeric(out["High"], errors="coerce") - pd.to_numeric(out["Close"], errors="coerce")) / hl_range

    is_valid_vol_breakout = (
        (day_trade_ratio <= 0.40) &
        pd.notna(real_vol_ma20) &
        (real_vol_ma20 > 0) &
        (real_vol >= real_vol_ma20 * 1.5)
    ).fillna(False)

    is_fake_vol_trap = (
        pd.notna(total_vol_ma20) &
        (total_vol_ma20 > 0) &
        (total_vol >= total_vol_ma20 * 2.0) &
        (day_trade_ratio >= 0.60) &
        (upper_weak_ratio >= 0.50)
    ).fillna(False)

    vol_structure_tag = np.where(
        is_fake_vol_trap,
        "UTAD_FAKE_VOLUME_TRAP",
        np.where(is_valid_vol_breakout, "✅ 實體有效放量", "⚪ 一般量能結構")
    )

    out["day_trade_vol"] = day_trade_vol
    out["real_vol"] = real_vol
    out["day_trade_ratio"] = day_trade_ratio
    out["real_vol_ma5"] = real_vol_ma5
    out["real_vol_ma20"] = real_vol_ma20
    out["real_vol_ratio"] = real_vol_ratio
    out["is_valid_vol_breakout"] = is_valid_vol_breakout
    out["is_fake_vol_trap"] = is_fake_vol_trap
    out["vol_structure_tag"] = vol_structure_tag

    return out


def get_latest_real_volume_summary(df: pd.DataFrame, stock_id: str) -> Dict:
    """回傳最新一列實體留倉量摘要欄位，供 scan_task3.py 與 app.py 統一呼叫。"""
    default_res = {
        "當沖率(%)": 0.0,
        "實體留倉量(張)": 0,
        "實體量比": 0.0,
        "量能結構標籤": "⚪ 一般量能結構",
        "is_fake_vol_trap": False,
        "is_valid_vol_breakout": False
    }
    try:
        metrics_df = calc_real_volume_metrics(df, stock_id)
        if metrics_df is None or metrics_df.empty or "vol_structure_tag" not in metrics_df.columns:
            return default_res

        last = metrics_df.iloc[-1]
        dt_ratio_pct = round(float(last["day_trade_ratio"]) * 100.0, 2) if pd.notna(last["day_trade_ratio"]) else 0.0
        real_vol_lots = int(round(float(last["real_vol"]))) if pd.notna(last["real_vol"]) else 0
        real_vol_ratio = round(float(last["real_vol_ratio"]), 2) if pd.notna(last["real_vol_ratio"]) else 0.0

        return {
            "當沖率(%)": dt_ratio_pct,
            "實體留倉量(張)": real_vol_lots,
            "實體量比": real_vol_ratio,
            "量能結構標籤": str(last["vol_structure_tag"]),
            "is_fake_vol_trap": bool(last["is_fake_vol_trap"]),
            "is_valid_vol_breakout": bool(last["is_valid_vol_breakout"])
        }
    except Exception:
        return default_res
