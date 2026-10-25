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
    "algo_mode": "全部",               # 啟用全部演算法 (底部翻轉 / VCP / 指標共振 / 量價雙背離)
    "lookback_end": 0,
    "lookback_start": 0,
    "min_score": 8.0,                  # 底部翻轉最低分
    "min_vcp_score": 10.0,             # VCP 最低分
    "min_reso_score": 10.0,            # 指標共振最低分
    "min_vol_ma20": 1000,              # 月均量門檻 (張)
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
    "use_weekly_filter": True,          # 週K共振與背離
    "enable_30d_backtest": True,        # 30天觸發次數統計
    "show_ma120_240_bear_align": True,  # 年線/半年線空排
    "show_ma240_slope": True,           # 年線斜率
    "show_ma120_slope": True            # 半年線斜率
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

# ==========================================
# 新增模組：長長下引突破 (獨立邏輯)
# ==========================================
class LongLowerShadowBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 10:
            return res
            
        df = df.copy()
        
        # 依照需求定義計算欄位
        # 避免最高與最低相等導致除以零
        range_hl = (df['High'] - df['Low']).replace(0, 1e-8) 
        max_oc = df[['Open', 'Close']].max(axis=1)
        min_oc = df[['Open', 'Close']].min(axis=1)
        
        df['Upper_Shadow_Ratio'] = (df['High'] - max_oc) / range_hl
        df['Lower_Shadow_Ratio'] = (min_oc - df['Low']) / range_hl
        
        if 'Pct_Change' not in df.columns:
            df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
            
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()

        # 1.1 當日條件判斷
        # 紅k(收盤>開盤)、漲幅+3.5%、上引佔比<15%、收盤價格<20均線
        cond_today = (
            (df['Close'] > df['Open']) & 
            (df['Pct_Change'] >= 3.5) & 
            (df['Upper_Shadow_Ratio'] < 0.15) & 
            (df['Close'] < df['MA20'])
        )
        
        # 1.2 前n天關鍵K線條件
        # 下引佔比>50% 且 開盤 < 前一日最低點
        target_k_mask = (df['Lower_Shadow_Ratio'] > 0.50) & (df['Open'] < df['Low'].shift(1))
        
        # 1.3 尋找前5日(不含當日)是否出現關鍵K線，且今日收盤 > 該K線最高價
        for i in range(1, 6):
            # i天前是否為關鍵K線
            is_target_k_past = target_k_mask.shift(i)
            # 該關鍵K線的最高價
            target_k_high = df['High'].shift(i)
            
            # 當日收盤突破關鍵K線最高價
            cond_breakout = df['Close'] > target_k_high
            
            # 綜合條件：當日滿足 cond_today 且 i天前為關鍵K線 且 當日發生突破
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
            # 依據需求，設定 auto_adjust=True 讓回傳的皆為還原股價
            data = yf.download(chunk, period=dl_period, threads=False, progress=False, session=yf_session, auto_adjust=True)
            for ticker in chunk:
                try:
                    # 處理 yfinance 多重索引欄位格式
                    if isinstance(data.columns, pd.MultiIndex):
                        df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else:
                        # 只有一檔股票時的回傳處理
                        df = data.dropna(how='all')

                    if len(df) < 60:
                        continue
                        
                    # 建立基礎技術欄位以利各策略模組順暢運作
                    df['Volume_Lots'] = df['Volume'] / 1000
                    df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['MA60'] = df['Close'].rolling(60).mean()
                    df['BB_Center'] = df['MA20']
                    df['BB_Std'] = df['Close'].rolling(20).std()
                    df['BB_Upper'] = df['BB_Center'] + 2 * df['BB_Std']
                    df['BB_Lower'] = df['BB_Center'] - 2 * df['BB_Std']
                    df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
                    df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
                    
                    # 加入共用技術指標
                    df = TechnicalIndicators.add_kd(df)
                    df = TechnicalIndicators.add_macd(df)
                    df = TechnicalIndicators.add_rsi(df)
                    
                    # 執行原有策略檢測
                    bottom_score = BottomReversalStrategy.evaluate(df)
                    vcp_score = VCPStrategy.evaluate(df)
                    reso_score = IndicatorResonanceStrategy.evaluate(df, 
                        recent_w=CONFIG['div_recent_w'], 
                        older_w=CONFIG['div_older_w']
                    )
                    
                    # 執行新增獨立策略：長長下引突破
                    breakout_signal = LongLowerShadowBreakoutStrategy.evaluate(df)
                    
                    # 取得當日(或指定回溯日)的運算結果
                    last_idx = -1 - CONFIG['lookback_start']
                    latest = df.iloc[last_idx]
                    
                    # 判斷是否觸發各策略
                    is_bottom = bottom_score.iloc[last_idx] >= CONFIG['min_score']
                    is_vcp = vcp_score.iloc[last_idx] >= CONFIG['min_vcp_score']
                    is_reso = reso_score.iloc[last_idx] >= CONFIG['min_reso_score']
                    is_breakout = breakout_signal.iloc[last_idx]
                    
                    # 新邏輯獨立於其他邏輯，只要符合任何一項（包含新邏輯）即列為候選
                    if is_bottom or is_vcp or is_reso or is_breakout:
                        reversal_candidates[ticker] = {
                            "代號": ticker,
                            "名稱": stock_dict.get(ticker, ""),
                            "收盤價": round(latest['Close'], 2),
                            "漲跌幅(%)": round(latest['Pct_Change'], 2),
                            "成交量(張)": round(latest['Volume_Lots'], 0),
                            "底部翻轉": "✅" if is_bottom else "-",
                            "VCP收斂": "✅" if is_vcp else "-",
                            "指標共振": "✅" if is_reso else "-",
                            "長下引突破": "🔥觸發" if is_breakout else "-"
                        }

                except Exception as e:
                    # 忽略單一個股錯誤，確保流程繼續
                    pass
        except Exception as e:
            # 忽略批次下載錯誤
            pass

    # 彙整與輸出結果
    df_result = pd.DataFrame.from_dict(reversal_candidates, orient='index')
    if not df_result.empty:
        print(f"\n🎉 掃描完成！共發現 {len(df_result)} 檔符合條件的標的：")
        print(df_result.to_string(index=False))
    else:
        print("\n🥺 本次掃描無符合條件的標的。")
        
    return df_result

if __name__ == "__main__":
    run_full_scan()
