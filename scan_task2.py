import os
import json
import warnings
import logging
import requests
import datetime
import time
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# 核心參數設定 (100% 比照 app.py 預設值)
# ==========================================
CONFIG = {
    "algo_mode": "全部",               
    "lookback_end": 0,
    "lookback_start": 0,
    "min_score": 8.0,                  
    "min_vcp_score": 10.0,             
    "min_reso_score": 10.0,            
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
    "show_ma120_slope": True            
}

# ==========================================
# 模組 1：技術指標計算
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

# ==========================================
# 老手實戰交易濾網
# ==========================================
class ExpertTradingStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.DataFrame(index=df.index)
        if len(df) < 20:
            res['RSI盤整'] = False
            res['多空指標數'] = 0
            res['日KD多'] = False
            res['週KD多'] = False
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

# ==========================================
# 週K線共振與背離獨立運算模組
# ==========================================
class WeeklyAnalysisStrategy:
    @staticmethod
    def evaluate(long_df, div_pairs, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days, price_col):
        if long_df is None or long_df.empty or len(long_df) < 20:
            return {"週K共振": "無資料", "週K背離": "無資料"}
        
        try:
            df = long_df.copy()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            w_df = df.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if len(w_df) < 20:
                return {"週K共振": "資料不足", "週K背離": "資料不足"}
                
            w_df = TechnicalIndicators.add_kd(w_df)
            w_df = TechnicalIndicators.add_macd(w_df)
            
            last_row = w_df.iloc[-1]
            kd_bull = last_row['K'] > last_row['D']
            macd_bull = last_row['MACD_Hist'] > 0
            
            if kd_bull and macd_bull: reso_str = "✅ 雙共振"
            elif kd_bull: reso_str = "KD偏多"
            elif macd_bull: reso_str = "MACD偏多"
            else: reso_str = "❌ 無"
            
            div_res = []
            for r_w, o_w in div_pairs:
                d_kd = DivergenceStrategy.check_bottom_divergence(
                    w_df, price_col, 'K', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                )
                d_macd = DivergenceStrategy.check_bottom_divergence(
                    w_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                )
                if d_kd: div_res.append(f"KD({r_w},{o_w})")
                if d_macd: div_res.append(f"MACD({r_w},{o_w})")
                
            div_str = "+".join(list(dict.fromkeys(div_res))) if div_res else "無"
            return {"週K共振": reso_str, "週K背離": div_str}
        except Exception:
            return {"週K共振": "錯誤", "週K背離": "錯誤"}

# ==========================================
# 模組 2：策略演算法核心
# ==========================================
class BottomReversalStrategy:
    @staticmethod
    def evaluate(df):
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        total_range = (df['High'] - df['Low']).replace(0, 0.001)
        vol_mult = (df['Volume_Lots'] / (df['Vol_MA20'] + 1e-8)).clip(0.5, 3.0)

        cond_low_pin = (df['BIAS20'] <= 0) & (lower_shadow > body * 1.5) & (lower_shadow > total_range * 0.4)
        cond_low_red = (df['BIAS20'] <= 0) & (df['Close'] > df['Open']) & (df['Pct_Change'] >= 2.5)

        candle_score = pd.Series(0.0, index=df.index)
        candle_score[cond_low_pin] = 7 * (lower_shadow[cond_low_pin] / total_range[cond_low_pin]) * vol_mult[cond_low_pin]
        candle_score[cond_low_red] = 5 * vol_mult[cond_low_red]
        return candle_score

class VCPStrategy:
    @staticmethod
    def evaluate(df):
        bb_width = (df['BB_Upper'] - df['BB_Lower']) / (df['MA20'] + 1e-8) * 100
        cond_uptrend = (df['Close'] > df['MA20']) & (df['MA20'] > df['MA60'])
        cond_vol_dry = df['Volume_Lots'] < df['Vol_MA20']
        cond_tight_price = bb_width < 10.0

        vol_score = 10 * (1 - df['Volume_Lots'] / (df['Vol_MA20'] + 1e-8)).clip(0, 1)
        tight_score = 10 * (10 - bb_width) / 10

        vcp_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_uptrend & cond_vol_dry & cond_tight_price
        vcp_score[valid_mask] = vol_score[valid_mask] + tight_score[valid_mask]
        return vcp_score

class DivergenceStrategy:
    @staticmethod
    def check_bottom_divergence(
        df, price_col='Low', ind_col='K', cross_col1='K', cross_col2='D',
        recent_w=5, older_w=20, recent_lows_cnt=0, older_lows_cnt=0,
        pivot_left=0, pivot_right=0, require_cross_confirm=True, cross_days=3
    ):
        if len(df) < recent_w + older_w: 
            return False
            
        if require_cross_confirm:
            cross = (df[cross_col1] > df[cross_col2]) & (df[cross_col1].shift(1) <= df[cross_col2].shift(1))
            if not cross.iloc[-cross_days:].any():
                return False

        recent_start = len(df) - recent_w
        older_start = recent_start - older_w
        older_end = recent_start
        
        prices = df[price_col].values
        inds = df[ind_col].values
        
        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            r_p_min = np.min(prices[recent_start:])
            o_p_min = np.min(prices[older_start:older_end])
            r_i_min = np.min(inds[recent_start:])
            o_i_min = np.min(inds[older_start:older_end])
            return (r_p_min <= o_p_min * 1.02) and (r_i_min > o_i_min)

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
            macd_turn_up = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))
            macd_signal = macd_turn_up | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))

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
        if len(df) < recent_w + older_w + 5: return res
        
        cond_pct = df['Pct_Change'].between(3.4, 10.5)
        cond_vol_abs = df['Volume_Lots'] >= 1000
        prev_vol = df['Volume_Lots'].shift(1)
        cond_vol_mult = (df['Volume_Lots'] / (prev_vol + 1e-8)).between(2.4, 15.0) 
        
        pv_mask = cond_pct & cond_vol_abs & cond_vol_mult
        valid_idx = np.where(pv_mask)[0]
        
        for idx in valid_idx:
            d_kd, d_macd = False, False
            for buffer in range(0, 25):
                check_idx = idx - buffer
                if check_idx < recent_w + older_w: continue
                sub_df = df.iloc[:check_idx+1]
                if not d_kd:
                    d_kd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'K', 'K', 'D', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if not d_macd:
                    d_macd = DivergenceStrategy.check_bottom_divergence(sub_df, price_col, 'MACD_Hist', 'MACD', 'MACD_Signal', recent_w, older_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days)
                if d_kd and d_macd: break 
            if d_kd and d_macd:
                res.iloc[idx] = True
        return res

# --- 新增模組：長長下引突破 (進階容錯優化版：適應真實資料源微小差異) ---
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
        
        if 'Pct_Change' not in df.columns:
            df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if 'Vol_MA20' not in df.columns:
            df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
        if 'BB_Lower' not in df.columns:
            std20 = df['Close'].rolling(20).std()
            df['BB_Lower'] = df['MA20'] - 2 * std20

        # 1. 當日條件：紅K、漲幅>=3.0%(微調)、上影線放寬至 < 40%，且「今日量 > 昨日量」
        cond_today = (
            (df['Close'] > df['Open']) & 
            (df['Pct_Change'] >= 3.0) & 
            (df['Upper_Shadow_Ratio'] < 0.40) & 
            (df['Volume_Lots'] > df['Volume_Lots'].shift(1)) 
        )
        
        # 2. 關鍵K線條件：下影線佔>=30%、最低價破前低、振幅>=1.5%、跌破布林下軌(給1.5%緩衝)、收盤在月線下
        target_k_mask = (
            (df['Lower_Shadow_Ratio'] >= 0.30) & 
            (df['Low'] < df['Low'].shift(1)) &     
            (df['K_Amplitude'] >= 1.5) & 
            (df['Low'] <= df['BB_Lower'] * 1.015) & 
            (df['Close'] < df['MA20']) 
        )
        
        # 3. 尋找前 10 日內是否符合並突破
        for i in range(1, 11):
            is_target_k_past = target_k_mask.shift(i)
            target_k_high = df['High'].shift(i)
            
            cond_breakout = df['Close'] > target_k_high
            
            valid = cond_today & is_target_k_past & cond_breakout
            res = res | valid.fillna(False)
            
        return res

# ==========================================
# 網路請求與台股標的擷取
# ==========================================
def get_yf_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    return session

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
                if len(tds) > 0:
                    text = tds[0].text.strip()
                    if '\u3000' in text:
                        code, name = text.split('\u3000')
                        if code.isdigit() and len(code) == 4:
                            stocks[code + suffix] = name
        except Exception: pass
    return stocks

# ==========================================
# Google Apps Script Webhook 上傳模組
# ==========================================
def upload_to_gdrive(csv_bytes, file_name, webhook_url, token):
    try:
        encoded_data = base64.b64encode(csv_bytes).decode('utf-8')
        payload = {"token": token, "fileName": file_name, "fileData": encoded_data}
        response = requests.post(webhook_url, data=payload, timeout=30)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 主流程：全市場兩階段深度掃描引擎
# ==========================================
def run_full_scan():
    print("🚀 [Step 1] 初始化：正在獲取台股上市櫃清單...")
    yf_session = get_yf_session()
    stock_dict = get_all_tw_stocks()
    tickers = list(stock_dict.keys())
    print(f"✅ 成功獲取 {len(tickers)} 檔標的，開始執行第一階段篩選...")

    div_pairs = [(CONFIG["div_recent_w"], CONFIG["div_older_w"])] if CONFIG["use_single_div"] else [(5, 20), (5, 60), (20, 60)]
    max_older_w = max(pair[1] for pair in div_pairs)
    extra_buffer_days = 30 if CONFIG["enable_30d_backtest"] else 0
    total_needed_days = CONFIG["lookback_start"] + max_older_w + CONFIG["pivot_left"] + 70 + extra_buffer_days

    dl_period = "3mo" if total_needed_days <= 60 else "6mo" if total_needed_days <= 120 else "1y"
    reversal_candidates = {}
    chunk_size = 40

    # --- Phase 1: 快速運算指標與初篩 ---
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = yf.download(chunk, period=dl_period, threads=False, progress=False, session=yf_session)
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(0):
                            df = data.xs(ticker, axis=1, level=0).dropna(how='all')
                        elif ticker in data.columns.get_level_values(1):
                            df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                        else: df = pd.DataFrame()
                    else:
                        df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                    if df.empty or len(df) <= CONFIG["lookback_start"] + 20: continue

                    df = TechnicalIndicators.add_kd(df)
                    df = TechnicalIndicators.add_macd(df)
                    df = TechnicalIndicators.add_rsi(df)
                    expert_res_df = ExpertTradingStrategy.evaluate(df)

                    df['Pct_Change'] = df['Close'].pct_change() * 100
                    df['Volume_Lots'] = df['Volume'] / 1000
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['MA60'] = df['Close'].rolling(60).mean()
                    df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8) * 100
                    df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()

                    std20 = df['Close'].rolling(20).std()
                    df['BB_Upper'] = df['MA20'] + 2 * std20
                    df['BB_Lower'] = df['MA20'] - 2 * std20

                    df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
                    if CONFIG["reso_macd_cross_zero"]:
                        df['MACD_Cross'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
                    else:
                        macd_turn_up = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))
                        df['MACD_Cross'] = macd_turn_up | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))

                    df['KD_Cross_Recent'] = df['KD_Cross'].rolling(window=CONFIG["reso_cross_days"], min_periods=1).max() >= 1
                    df['MACD_Cross_Recent'] = df['MACD_Cross'].rolling(window=CONFIG["reso_cross_days"], min_periods=1).max() >= 1

                    # 五大演算法打分 (加入長下引突破)
                    df['Candle_Score'] = BottomReversalStrategy.evaluate(df)
                    df['VCP_Score'] = VCPStrategy.evaluate(df)
                    price_col_for_div_eval = 'Low' if CONFIG["reso_price_basis"] == "最低價 (Low)" else 'Close'

                    df['Reso_Score'] = IndicatorResonanceStrategy.evaluate(
                        df, recent_w=CONFIG["div_recent_w"], older_w=CONFIG["div_older_w"],
                        kd_older_low_th=CONFIG["reso_kd_older_low"], kd_older_high_th=CONFIG["reso_kd_older_high"],
                        kd_recent_low_th=CONFIG["reso_kd_recent_low"], use_macd_abs=CONFIG["use_macd_abs"],
                        macd_older_low_th=CONFIG["reso_macd_older_low"], macd_recent_low_th=CONFIG["reso_macd_recent_low"],
                        cross_days=CONFIG["reso_cross_days"], require_price_higher_low=CONFIG["reso_price_higher_low"],
                        require_macd_cross_zero=CONFIG["reso_macd_cross_zero"], reso_price_basis=CONFIG["reso_price_basis"],
                        use_macd_wave_logic=CONFIG["reso_macd_wave_logic"], require_cross_confirm=CONFIG["require_cross_confirm"]
                    )

                    df['PriceVol_Div_Pass'] = PriceVolDivergenceStrategy.evaluate(
                        df, recent_w=CONFIG["div_recent_w"], older_w=CONFIG["div_older_w"],
                        rl_cnt=CONFIG["recent_lows_cnt"], ol_cnt=CONFIG["older_lows_cnt"],
                        p_left=CONFIG["pivot_left"], p_right=CONFIG["pivot_right"],
                        req_cross=CONFIG["require_cross_confirm"], c_days=CONFIG["reso_cross_days"],
                        price_col=price_col_for_div_eval
                    )

                    # 新增：長長下引突破訊號運算
                    df['Breakout_Signal'] = LongLowerShadowBreakoutStrategy.evaluate(df)

                    best_combined_score = -1
                    best_row, best_offset = None, 0
                    best_vol_mult = 0.0

                    for offset in range(CONFIG["lookback_end"], CONFIG["lookback_start"] + 1):
                        if len(df) <= offset + 1: continue
                        t_row = df.iloc[-1 - offset]
                        r_score, v_score, reso_score, vol_ma = t_row['Candle_Score'], t_row['VCP_Score'], t_row['Reso_Score'], t_row['Vol_MA20']

                        prev_vol = df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset + 1 else np.nan
                        vol_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                        if vol_ma >= CONFIG["min_vol_ma20"]:
                            is_r_pass = r_score >= CONFIG["min_score"]
                            is_v_pass = v_score >= CONFIG["min_vcp_score"]
                            is_reso_pass = reso_score >= CONFIG["min_reso_score"]
                            is_pv_div_pass = t_row.get('PriceVol_Div_Pass', False)
                            is_breakout_pass = t_row.get('Breakout_Signal', False)  # 取得當下突破訊號

                            # 只要符合任何一項邏輯(包含新邏輯)即列為候選
                            if is_r_pass or is_v_pass or is_reso_pass or is_pv_div_pass or is_breakout_pass:
                                combo_score = r_score + v_score + reso_score + (20 if is_pv_div_pass else 0) + (10 if is_breakout_pass else 0)
                                if combo_score > best_combined_score or best_row is None:
                                    best_combined_score = combo_score
                                    best_row, best_offset = t_row, offset
                                    best_vol_mult = vol_mult

                    # 30天回測統計
                    count_30d = 0
                    if CONFIG["enable_30d_backtest"] and best_row is not None:
                        for b_offset in range(best_offset + 1, best_offset + 31):
                            if len(df) <= b_offset + 1: continue
                            c_row = df.iloc[-1 - b_offset]
                            c_vol_ma = c_row['Vol_MA20']
                            if pd.isna(c_vol_ma): continue

                            if c_vol_ma >= CONFIG["min_vol_ma20"]:
                                c_r = c_row['Candle_Score'] >= CONFIG["min_score"]
                                c_v = c_row['VCP_Score'] >= CONFIG["min_vcp_score"]
                                c_reso = c_row['Reso_Score'] >= CONFIG["min_reso_score"]
                                c_pv = c_row.get('PriceVol_Div_Pass', False)
                                c_brk = c_row.get('Breakout_Signal', False) # 統計也加入新邏輯
                                if c_r or c_v or c_reso or c_pv or c_brk: count_30d += 1

                    if best_row is not None:
                        clean_ticker = ticker.split('.')[0]
                        last_expert = expert_res_df.iloc[-1 - best_offset]
                        rec_status = "-"
                        if CONFIG["use_cross_position_filter"]:
                            k_val = float(best_row['K'])
                            rec_status = "✅ 建議參考" if k_val < CONFIG["cross_position_threshold"] else "⚠️ 位置偏高"

                        # 將新邏輯結果正確寫入候選名單，傳遞給 Phase 2
                        reversal_candidates[ticker] = {
                            "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                            "股票代號": clean_ticker, "股票名稱": stock_dict[ticker],
                            "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                            "近30日觸發次數": count_30d if CONFIG["enable_30d_backtest"] else "-",
                            "當日收收盤": round(float(best_row['Close']), 2),
                            "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                            "當日成交量(張)": int(best_row['Volume_Lots']),
                            "量增倍數": round(best_vol_mult, 2),
                            "月均量(張)": int(best_row['Vol_MA20']),
                            "DIF": round(float(best_row['MACD']), 3),
                            "KD金叉判定": "✅ 是" if best_row['KD_Cross_Recent'] else "❌ 否",
                            "MACD金叉判定": "✅ 是" if best_row['MACD_Cross_Recent'] else "❌ 否",
                            "反轉分數": round(float(best_row['Candle_Score']), 2),
                            "VCP分數": round(float(best_row['VCP_Score']), 2),
                            "共振分數": round(float(best_row['Reso_Score']), 2),
                            "量價雙背離": "✅ 是" if best_row.get('PriceVol_Div_Pass', False) else "❌ 否",
                            "長下引突破": "✅ 是" if best_row.get('Breakout_Signal', False) else "❌ 否",  # 新增欄位
                            "進場位置建議": rec_status,
                            "盤整過濾": "✅ 盤整中(建議停用指標)" if last_expert['RSI盤整'] else "趨勢盤",
                            "建議水位": ExpertTradingStrategy.get_position_advice(last_expert['多空指標數']),
                            "多週期KD": ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])
                        }
                except Exception: continue
        except Exception: pass
        time.sleep(1.0)

    reversal_list = list(reversal_candidates.values())
    print(f"🎯 第一階段篩選完畢，共入選 {len(reversal_list)} 檔標的。")

    if not reversal_list:
        print("💡 今日全市場無符合條件標的，任務結束。")
        return

    # --- Phase 2: 長天期下載與第二階段深度指標運算 ---
    print("🚀 [Step 2] 正在下載入選標的 2 年資料以運算週K、年線與背離...")
    candidate_tickers = [item["_Full_Ticker"] for item in reversal_list]
    long_data = yf.download(candidate_tickers, period="2y", threads=False, progress=False, session=yf_session)

    final_results = []
    price_col_for_div = 'Low' if CONFIG["reso_price_basis"] == "最低價 (Low)" else 'Close'

    for item in reversal_list:
        try:
            ticker = item.pop("_Full_Ticker")
            specific_offset = item.pop("_Offset")
            daily_df = item.pop("_Daily_DF")
            has_daily_div, has_m60_div = False, False

            item['年線/半年線空排'] = "無資料"
            item['半年線斜率(%)'] = "無資料"
            item['年線斜率(%)'] = "無資料"

            try:
                if isinstance(long_data.columns, pd.MultiIndex):
                    if ticker in long_data.columns.get_level_values(0):
                        long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                    elif ticker in long_data.columns.get_level_values(1):
                        long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else: long_df = pd.DataFrame()
                else:
                    long_df = long_data.dropna(how='all') if len(candidate_tickers) == 1 else pd.DataFrame()

                if not long_df.empty:
                    if specific_offset > 0:
                        long_df = long_df.iloc[:-specific_offset]

                    if len(long_df) >= 120:
                        ma120 = long_df['Close'].rolling(120).mean()
                        item['半年線斜率(%)'] = round((ma120.iloc[-1] - ma120.iloc[-2]) / ma120.iloc[-2] * 100, 3)

                    if len(long_df) >= 240:
                        ma240 = long_df['Close'].rolling(240).mean()
                        item['年線斜率(%)'] = round((ma240.iloc[-1] - ma240.iloc[-2]) / ma240.iloc[-2] * 100, 3)
                        item['年線/半年線空排'] = "✅ 是" if ma120.iloc[-1] < ma240.iloc[-1] else "❌ 否"

                    if CONFIG["use_weekly_filter"]:
                        wk_res = WeeklyAnalysisStrategy.evaluate(
                            long_df, div_pairs, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"],
                            CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"],
                            CONFIG["reso_cross_days"], price_col_for_div
                        )
                        item["週K共振"] = wk_res["週K共振"]
                        item["週K背離"] = wk_res["週K背離"]
            except Exception:
                if CONFIG["use_weekly_filter"]:
                    item["週K共振"] = "錯誤"
                    item["週K背離"] = "錯誤"

            if not daily_df.empty:
                if specific_offset > 0:
                    daily_df = daily_df.iloc[:-specific_offset]

                for r_w, o_w in div_pairs:
                    d_kd, d_macd = False, False
                    for b in range(0, 25):
                        if len(daily_df) <= b + r_w + o_w: break
                        temp_df = daily_df.iloc[:len(daily_df)-b] if b > 0 else daily_df
                        if not d_kd:
                            d_kd = DivergenceStrategy.check_bottom_divergence(
                                temp_df, price_col_for_div, 'K', 'K', 'D', r_w, o_w, CONFIG["recent_lows_cnt"],
                                CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"],
                                CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"]
                            )
                        if not d_macd:
                            d_macd = DivergenceStrategy.check_bottom_divergence(
                                temp_df, price_col_for_div, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, CONFIG["recent_lows_cnt"],
                                CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"],
                                CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"]
                            )
                        if d_kd and d_macd: break
                    
                    res_tags = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                    item[f"日K背離({r_w},{o_w})"] = "+".join(res_tags) if res_tags else "無"
                    if res_tags: has_daily_div = True

            try:
                time.sleep(0.5)
                m60_df = yf.Ticker(ticker, session=yf_session).history(interval="60m", period="730d")
                if specific_offset > 0 and not daily_df.empty:
                    target_date = daily_df.index[-1].date()
                    m60_df = m60_df[[d.date() <= target_date for d in m60_df.index]]

                if not m60_df.empty:
                    m60_df = TechnicalIndicators.add_macd(TechnicalIndicators.add_kd(m60_df))
                    for r_w, o_w in div_pairs:
                        d_kd, d_macd = False, False
                        for b in range(0, 20):
                            if len(m60_df) <= b + r_w + o_w: break
                            temp_df = m60_df.iloc[:len(m60_df)-b] if b > 0 else m60_df
                            if not d_kd:
                                d_kd = DivergenceStrategy.check_bottom_divergence(
                                    temp_df, price_col_for_div, 'K', 'K', 'D', r_w, o_w, CONFIG["recent_lows_cnt"],
                                    CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"],
                                    CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"]
                                )
                            if not d_macd:
                                d_macd = DivergenceStrategy.check_bottom_divergence(
                                    temp_df, price_col_for_div, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, CONFIG["recent_lows_cnt"],
                                    CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"],
                                    CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"]
                                )
                            if d_kd and d_macd: break
                        res_tags = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                        item[f"60分K背離({r_w},{o_w})"] = "+".join(res_tags) if res_tags else "無"
                        if res_tags: has_m60_div = True
                else:
                    for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"
            except Exception:
                for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"

            base_tags = []
            if item.get("量價雙背離") == "✅ 是": base_tags.append("量價雙背離")
            if item.get("長下引突破") == "✅ 是": base_tags.append("長下引突破") # 將新邏輯加入演算法建議結果
            if item["反轉分數"] >= CONFIG["min_score"]: base_tags.append("底部翻轉")
            if item["VCP分數"] >= CONFIG["min_vcp_score"]: base_tags.append("VCP收斂")
            if item["共振分數"] >= CONFIG["min_reso_score"]: base_tags.append("指標共振")

            div_tag = " + 雙級別共振" if (has_daily_div and has_m60_div) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if has_m60_div else " (無背離)"))
            item["演算法建議結果"] = " & ".join(base_tags) + div_tag

            final_results.append(item)
        except Exception: continue

    res_df = pd.DataFrame(final_results)

    # 確保 `長下引突破` 被加入 CSV 輸出的 base_cols 中
    base_cols = ['股票代號', '股票名稱', '觸發日期']
    if CONFIG["enable_30d_backtest"]:
        base_cols.append('近30日觸發次數')

    base_cols.extend([
        '演算法建議結果', '長下引突破', '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
        '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', 'KD金叉判定', 'MACD金叉判定', 
        '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)'
    ])

    if CONFIG["use_weekly_filter"]:
        insert_idx = base_cols.index('盤整過濾') + 1
        base_cols.insert(insert_idx, '週K共振')
        base_cols.insert(insert_idx + 1, '週K背離')

    if CONFIG["show_ma120_240_bear_align"]:
        base_cols.append('年線/半年線空排')
    if CONFIG["show_ma240_slope"]:
        base_cols.append('年線斜率(%)')
    if CONFIG["show_ma120_slope"]:
        base_cols.append('半年線斜率(%)')

    div_cols = [c for c in res_df.columns if '背離' in c and c not in base_cols]
    cols = base_cols + div_cols
    res_df = res_df[cols].sort_values(by=["共振分數", "反轉分數", "VCP分數"], ascending=[False, False, False]).reset_index(drop=True)
    res_df.index = res_df.index + 1

    print("📊 掃描完成！最終報表標的預覽：")
    print(res_df[['股票代號', '股票名稱', '演算法建議結果', '當日收盤', '量增倍數']].to_string())

    # --- 上傳 Google Drive ---
    webhook_url = os.environ.get("GAS_WEBHOOK_URL", "").strip()
    token = os.environ.get("GAS_SECRET_TOKEN", "").strip()

    if webhook_url and token:
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        file_name = f"auto_scan_all_{today_str}.csv"
        csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')

        print(f"☁️ 正在將報表上傳至 Google Drive: {file_name} ...")
        res = upload_to_gdrive(csv_bytes, file_name, webhook_url, token)
        print(f"✅ 上傳結果: {res}")
    else:
        print("⚠️ 未檢測到完整的 GAS_WEBHOOK_URL 或 GAS_SECRET_TOKEN 環境變數，略過上傳。")

if __name__ == "__main__":
    run_full_scan()
