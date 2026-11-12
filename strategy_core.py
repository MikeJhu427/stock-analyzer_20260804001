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

# 🚀 這裡新增了「雙底正背離與右肩回測」選項
SCAN_ALGO_CHOICES = ['全部', '底部翻轉', 'VCP', '指標共振', '量價雙背離', '長下引突破', '*回測走揚年線', '低位長下引量縮突破', '雙底正背離與右肩回測']

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
    def upload(file_name, webhook_url, token, file_path=None, csv_bytes=None):
        if not webhook_url or not webhook_url.startswith("https://script.google.com/"):
            return False, "無效的 Google Apps Script Webhook 網址"
        try:
            if csv_bytes is not None:
                file_data = csv_bytes
            elif file_path:
                with open(file_path, "rb") as f:
                    file_data = f.read()
            else:
                return False, "無檔案資料"
                
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            payload = {"token": token, "fileName": file_name, "fileData": encoded_data}
            response = requests.post(webhook_url, data=payload, timeout=30, verify=False)
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

# 🚀 這裡新增了全新的【雙底正背離與右肩回測 (AdvancedBottomStrategy)】模組
class AdvancedBottomStrategy:
    @staticmethod
    def evaluate(df):
        # 初始化輸出結構
        res = pd.DataFrame(index=df.index, columns=[
            'Adv_Signal_A', 'Adv_Signal_B', 'Adv_Low_1', 'Adv_Low_2',
            'Adv_Neckline_1', 'Adv_Support_RS', 'Adv_Target_1x', 'Adv_Target_3x'
        ])
        for c in ['Adv_Signal_A', 'Adv_Signal_B']: res[c] = False
        for c in ['Adv_Low_1', 'Adv_Low_2', 'Adv_Neckline_1', 'Adv_Support_RS', 'Adv_Target_1x', 'Adv_Target_3x']: res[c] = np.nan

        if len(df) < 60:
            return res

        # 預計算與取得相關數值
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
        
        # 尋找所有波段轉折極值 (預設 Order=5)
        pivot_lows_idx = argrelextrema(lows, np.less, order=5)[0]
        pivot_highs_idx = argrelextrema(highs, np.greater, order=5)[0]

        # 透過時間迴圈計算確保不引用未來數據
        for i in range(20, len(df)):
            # 取得當前 K 棒(i)之前的有效低點極值
            valid_lows = [idx for idx in pivot_lows_idx if idx < i - 1]
            if len(valid_lows) < 2: continue

            idx_L1, idx_L2 = valid_lows[-2], valid_lows[-1]
            
            # 條件 1：破底確認 (Low_2 < Low_1)
            if lows[idx_L2] >= lows[idx_L1]: continue

            # 條件 2：正背離驗證 (KD 與 MACD)
            kd_div = (k_vals[idx_L2] > k_vals[idx_L1]) or (d_vals[idx_L2] > d_vals[idx_L1])
            macd_div = (macd_h[idx_L2] > macd_h[idx_L1]) or (macd[idx_L2] > macd[idx_L1])
            if not (kd_div and macd_div): continue

            # 尋找 Neckline_1：Low_1 前方緊鄰的反彈波段上影線高點
            valid_highs_before_L1 = [idx for idx in pivot_highs_idx if idx < idx_L1]
            if valid_highs_before_L1:
                neckline_1 = highs[valid_highs_before_L1[-1]]
            else:
                neckline_1 = np.max(highs[max(0, idx_L1-10):idx_L1+1])

            # 尋找 Support_RightShoulder：破底長黑前的起跌實體底 (擷取 L1 到 L2 之間單日跌幅最大的前一天)
            if idx_L2 > idx_L1 + 1:
                diffs = np.diff(closes[idx_L1:idx_L2+1])
                if len(diffs) > 0:
                    drop_idx = idx_L1 + np.argmin(diffs)
                    sup_rs = min(closes[drop_idx], opens[drop_idx])
                else:
                    sup_rs = min(closes[idx_L1], opens[idx_L1])
            else:
                sup_rs = min(closes[idx_L1], opens[idx_L1])

            # 計算 Low_2 以來的止跌紅K數量
            red_candles_since_L2 = sum(1 for j in range(idx_L2, i+1) if closes[j] > opens[j])

            # 訊號 A (底倉突破)：收盤大於上影線頸線 + 帶量 + K值>=20 + 第二根紅K
            sig_A = (closes[i] > neckline_1) and (vol[i] > vol_ma5[i]) and \
                    (k_vals[i] >= 20) and (closes[i] > opens[i]) and \
                    (red_candles_since_L2 >= 2)

            # 訊號 B (右肩回測加碼)：先前出現過 Signal A，且回測到 Support 區間或布林下軌量縮不破底
            sig_A_history = any((closes[j] > neckline_1 and vol[j] > vol_ma5[j] and closes[j] > opens[j] and k_vals[j] >= 20) for j in range(idx_L2, i))
            
            sig_B = False
            if sig_A_history:
                in_zone = (lows[i] <= sup_rs) and (lows[i] >= lows[idx_L1] * 0.95)
                near_bb = (lows[i] <= bb_lower[i] * 1.02)
                if (in_zone or near_bb) and (closes[i] >= lows[idx_L2]) and (vol[i] < vol_ma5[i]):
                    sig_B = True

            # 若觸發任一訊號，寫入點位與目標價
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
        
        # 🚀 在 Phase 1 併入新的 AdvancedBottomStrategy 演算法模組
        adv_res = AdvancedBottomStrategy.evaluate(df)
        for col in adv_res.columns:
            df[col] = adv_res[col]
            
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
        
        # 🚀 這裡新增抓取雙底背離突破(Signal A)與回測加碼(Signal B)的命中判定
        is_adv = (algo_mode in ['全部', '雙底正背離與右肩回測']) and (row.get('Adv_Signal_A', False) or row.get('Adv_Signal_B', False))
        
        passed = is_r or is_v or is_reso or is_pv or is_brk or is_ma240 or is_lowpos_brk or is_adv or force_pass
        
        combo_score = 0
        combo_score += row.get('Candle_Score', 0)
        combo_score += row.get('VCP_Score', 0)
        combo_score += row.get('Reso_Score', 0)
        combo_score += 20 if row.get('PriceVol_Div_Pass', False) else 0
        combo_score += 10 if row.get('Breakout_Signal', False) else 0
        combo_score += 12 if row.get('MA240_P1_Pass', False) else 0
        combo_score += 15 if row.get('LowPos_Breakout_Pass', False) else 0
        # 🚀 給予進階背離策略極高的 combo 加權分，讓這類標的排在較上面
        combo_score += 25 if row.get('Adv_Signal_A', False) else 0
        combo_score += 30 if row.get('Adv_Signal_B', False) else 0
        
        return passed, combo_score

    @staticmethod
    def generate_base_tags(item, config):
        base_tags = []
        # 🚀 加入新的標籤顯示
        if item.get("Adv_Signal_A") == "✅ 是":
            base_tags.append("雙底突破(底倉)")
        if item.get("Adv_Signal_B") == "✅ 是":
            base_tags.append("右肩回測(加碼)")
            
        if item.get("量價雙背離") == "✅ 是":
            base_tags.append("量價雙背離")
        if item.get("長下引突破") == "✅ 是":
            base_tags.append("長下引突破")
        if item.get("*回測走揚年線") == "✅ 是":
            base_tags.append("*回測走揚年線")
        if item.get("低位長下引量縮突破") == "✅ 是" or item.get("LowPos_Breakout_Pass", False):
            base_tags.append("低位量縮突破")
        if item.get("反轉分數", 0) >= config.get("min_score", 8.0):
            base_tags.append("底部翻轉")
        if item.get("VCP分數", 0) >= config.get("min_vcp_score", 10.0):
            base_tags.append("VCP收斂")
        if item.get("共振分數", 0) >= config.get("min_reso_score", 10.0):
            base_tags.append("指標共振")
        return base_tags

# ==========================================
# 財報三表地雷掃描 (FinancialAuditStrategy)
# ==========================================
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
            return {"status": "error", "msg": f"解析發生錯誤：{str(e)}"}

# ==========================================
# 幾何形態測距與量化引擎 (QuantPatternEngine)
# ==========================================
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
