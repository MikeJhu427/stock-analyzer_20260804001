import os
import json
import warnings
import logging
import requests
import datetime
import time  # 新增：用於連線延遲防阻擋
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True

# ==========================================
# 參數設定檔管理模組 (JSON 本機儲存)
# ==========================================
PARAMS_FILE = "params_config.json"

DEFAULT_PARAMS = {
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
    "kou_di_5": True,
    "kou_di_10": False,
    "kou_di_20": True,
    "kou_di_60": True,
    "reso_kd_older_low": 20.0,
    "reso_kd_older_high": 50.0,
    "reso_kd_recent_low": 20.0,
    "use_macd_abs": False,
    "reso_macd_older_low": 0.0,
    "reso_macd_recent_low": 0.0,
    "reso_cross_days": 3,
    "use_backtest_date": False,
    "backtest_date": str(datetime.date.today())
}

def load_config():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"讀取設定檔失敗，將套用預設值: {e}")
    return {"last_used": "預設參數 (Default)", "profiles": {"預設參數 (Default)": DEFAULT_PARAMS.copy()}}

def save_config(config):
    try:
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"儲存設定檔失敗: {e}")

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

# ==========================================
# 模組 2：大盤位階與策略演算法核心
# ==========================================
class MarketRegimeFilter:
    """大盤位階與期貨基準動態濾網"""
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
        except Exception as e:
            return None

class BottomReversalStrategy:
    """左側交易：低檔強力翻轉判定模組"""
    @staticmethod
    def evaluate(df):
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        total_range = (df['High'] - df['Low']).replace(0, 0.001)
        vol_mult = (df['Volume_Lots'] / df['Vol_MA20']).clip(0.5, 3.0)

        cond_low_pin = (df['BIAS20'] <= 0) & (lower_shadow > body * 1.5) & (lower_shadow > total_range * 0.4)
        cond_low_red = (df['BIAS20'] <= 0) & (df['Close'] > df['Open']) & (df['Pct_Change'] >= 2.5)

        candle_score = pd.Series(0.0, index=df.index)
        candle_score[cond_low_pin] = 7 * (lower_shadow[cond_low_pin] / total_range[cond_low_pin]) * vol_mult[cond_low_pin]
        candle_score[cond_low_red] = 5 * vol_mult[cond_low_red]
        return candle_score

class VCPStrategy:
    """右側交易：VCP波動收斂型態判定模組"""
    @staticmethod
    def evaluate(df):
        bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] * 100
        cond_uptrend = (df['Close'] > df['MA20']) & (df['MA20'] > df['MA60'])
        cond_vol_dry = df['Volume_Lots'] < df['Vol_MA20']
        cond_tight_price = bb_width < 10.0

        vol_score = 10 * (1 - df['Volume_Lots'] / df['Vol_MA20']).clip(0, 1)
        tight_score = 10 * (10 - bb_width) / 10

        vcp_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_uptrend & cond_vol_dry & cond_tight_price
        vcp_score[valid_mask] = vol_score[valid_mask] + tight_score[valid_mask]
        return vcp_score

class IndicatorResonanceStrategy:
    """雙指標共振起漲：精細化KD與MACD波段條件限制"""
    @staticmethod
    def evaluate(df, recent_w=5, older_w=20, kd_older_low_th=20, kd_older_high_th=50, kd_recent_low_th=20, 
                 use_macd_abs=False, macd_older_low_th=0, macd_recent_low_th=0, cross_days=3):
        if len(df) < recent_w + older_w:
            return pd.Series(0.0, index=df.index)

        recent_k_low = df['K'].rolling(window=recent_w, min_periods=1).min()
        older_k_low = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        older_k_high = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).max()
        
        recent_macd_low = df['MACD'].rolling(window=recent_w, min_periods=1).min()
        older_macd_low = df['MACD'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        
        recent_price_low = df['Low'].rolling(window=recent_w, min_periods=1).min()
        older_price_low = df['Low'].shift(recent_w).rolling(window=older_w, min_periods=1).min()

        kd_cross = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        macd_cross = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)

        kd_recent_cross = kd_cross.rolling(window=cross_days, min_periods=1).max() >= 1
        macd_recent_cross = macd_cross.rolling(window=cross_days, min_periods=1).max() >= 1

        cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th) & kd_recent_cross
        
        cond_macd = (recent_macd_low > older_macd_low) & macd_recent_cross
        if use_macd_abs:
            cond_macd = cond_macd & (older_macd_low < macd_older_low_th) & (recent_macd_low > macd_recent_low_th)
            
        cond_price = recent_price_low > older_price_low

        resonance_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_kd & cond_macd & cond_price
        
        hist_momentum = (df['MACD_Hist'] - df['MACD_Hist'].shift(1)) * 100
        resonance_score[valid_mask] = 10.0 + hist_momentum[valid_mask].clip(0, 5) 
        
        return resonance_score

class DivergenceStrategy:
    """底背離判定模組"""
    @staticmethod
    def check_bottom_divergence(
        df, price_col='Low', ind_col='K', ind_signal_col='D', 
        recent_w=20, older_w=60, recent_lows_cnt=0, older_lows_cnt=0,
        pivot_left=0, pivot_right=0
    ):
        if len(df) < older_w: return False
        recent_start, recent_end = len(df) - recent_w, len(df)
        older_start, older_end = len(df) - older_w, recent_start
        if recent_start < 0 or older_start < 0: return False
            
        prices, k_vals, d_vals = df[price_col].values, df[ind_col].values, df[ind_signal_col].values
        recent_prices = prices[recent_start:recent_end]
        if len(recent_prices) == 0: return False
        
        idx1_iloc = recent_start + np.argmin(recent_prices)
        p1, i1 = prices[idx1_iloc], k_vals[idx1_iloc]
        
        def check_divergence_condition(p_iloc):
            p2, i2 = prices[p_iloc], k_vals[p_iloc]
            if not (p2 > p1 and i2 < i1): return False 
            s_idx, e_idx = min(idx1_iloc, p_iloc), max(idx1_iloc, p_iloc)
            if e_idx - s_idx + 1 > 2:
                cross_found = False
                for j in range(s_idx + 1, e_idx + 1):
                    if k_vals[j] < d_vals[j] and k_vals[j-1] >= d_vals[j-1]:
                        cross_found = True
                        break
                if not cross_found: return False
            else: return False
            return True

        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            older_prices = prices[older_start:older_end]
            if len(older_prices) == 0: return False
            idx2_iloc = older_start + np.argmin(older_prices)
            return check_divergence_condition(idx2_iloc)

        def get_valid_pivots_iloc(start_loc, end_loc):
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s, e = max(0, i_loc - pivot_left), min(len(prices), i_loc + pivot_right + 1)
                if prices[i_loc] == np.min(prices[s:e]): pivots.append(i_loc)
            return pivots

        if recent_lows_cnt > 0:
            recent_pivots_iloc = get_valid_pivots_iloc(recent_start, recent_end)
            if idx1_iloc in recent_pivots_iloc: recent_pivots_iloc.remove(idx1_iloc)
            if not recent_pivots_iloc: return False
            for p_iloc in sorted(recent_pivots_iloc, key=lambda x: prices[x])[:recent_lows_cnt]:
                if not check_divergence_condition(p_iloc): return False
                    
        if older_lows_cnt > 0:
            older_pivots_iloc = get_valid_pivots_iloc(older_start, older_end)
            if not older_pivots_iloc: return False
            for p_iloc in sorted(older_pivots_iloc, key=lambda x: prices[x])[:older_lows_cnt]:
                if not check_divergence_condition(p_iloc): return False
                    
        return True

# ==========================================
# 資料抓取與共用函式 (含防阻擋機制)
# ==========================================
def get_yf_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'})
    return session

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tw_stocks():
    """強制重試與數量驗證，確保上市與上櫃股票皆完整抓取 (約1700檔)"""
    stocks = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for mode in ['2', '4']: # 2=上市, 4=上櫃
        suffix = '.TW' if mode == '2' else '.TWO'
        # 若失敗最多重試 3 次
        for attempt in range(3):
            try:
                url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
                res = requests.get(url, headers=headers, timeout=15)
                res.encoding = 'big5' # 強制 Big5 解碼避免亂碼
                soup = BeautifulSoup(res.text, 'html.parser')
                
                fetched_count = 0
                for tr in soup.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) > 0:
                        text = tds[0].text.strip()
                        if '\u3000' in text:
                            code, name = text.split('\u3000')
                            if code.isdigit() and len(code) == 4:
                                stocks[code + suffix] = name
                                fetched_count += 1
                
                # 確保成功抓取足夠數量的股票才算成功 (上市/上櫃一般都超過 500 檔)
                if fetched_count > 500:
                    break
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)
        # 兩次大市場請求之間延遲 2 秒，避免被 TWSE 阻擋
        time.sleep(2)
        
    return stocks

@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_stock_name(ticker):
    code = ticker.split('.')[0]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1')
            if h1: return h1.text.strip()
    except Exception: pass
    try:
        return yf.Ticker(ticker, session=get_yf_session()).info.get('shortName', code)
    except Exception: return code

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol):
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

# ==========================================
# 介面主程式與 Session State 初始化防呆
# ==========================================
st.set_page_config(page_title="台股 K線型態與位階深度解析系統", layout="wide")

if "config" not in st.session_state: st.session_state.config = load_config()
if "current_profile" not in st.session_state: st.session_state.current_profile = st.session_state.config.get("last_used", "預設參數 (Default)")

def apply_profile_to_state(profile_name):
    prof = st.session_state.config["profiles"].get(profile_name, DEFAULT_PARAMS)
    full_prof = DEFAULT_PARAMS.copy()
    full_prof.update(prof)
    for k, v in full_prof.items(): st.session_state[k] = v
    st.session_state.backtest_date_obj = pd.to_datetime(full_prof["backtest_date"]).date()
    st.session_state.current_profile = profile_name
    st.session_state.config["last_used"] = profile_name
    save_config(st.session_state.config)

missing_keys = [k for k in DEFAULT_PARAMS.keys() if k not in st.session_state]
if missing_keys or "backtest_date_obj" not in st.session_state:
    apply_profile_to_state(st.session_state.current_profile)

st.title("📈 台股 K線型態與位階深度解析系統")
st.markdown("<style>header {visibility: hidden;}</style>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 單檔深度解析", "🚀 全市場智慧掃描 (回測/翻轉/VCP/共振/背離)"])

# ----------------------------------------------------
# 頁籤 1：單檔深度解析
# ----------------------------------------------------
with tab1:
    st.write("請在下方輸入股票代號（例如：`2495`、`00631L`），系統將自動抓取近兩年資料進行診斷。")

    col1, col2 = st.columns([4, 1])
    with col1: user_input = st.text_input("輸入股票代號", value="2495", placeholder="例如：2330").strip()
    with col2: st.write(""); st.write(""); submit_btn = st.button("開始分析", type="primary")

    if submit_btn and user_input:
        with st.spinner(f"⏳ 正在抓取 [{user_input}] 資料並進行解析中..."):
            df, real_ticker = get_stock_data(user_input)
            
            if df.empty:
                st.error(f"❌ 查無 [{user_input}] 的歷史數據，請確認代號是否正確。或請稍後再試。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                
                df = TechnicalIndicators.add_kd(df)
                df = TechnicalIndicators.add_macd(df)
                
                df['Pct_Change'] = df['Close'].pct_change() * 100
                df['Volume_Lots'] = df['Volume'] / 1000
                df['Momentum_Force'] = df['Pct_Change'] * df['Volume_Lots']
                df['Prev_Close'] = df['Close'].shift(1)
                
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['Prev_MA5'] = df['MA5'].shift(1)
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
                df['Std20'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + 2 * df['Std20']
                df['BB_Lower'] = df['MA20'] - 2 * df['Std20']
                
                df['VWMA20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
                df['Prev_VWMA20'] = df['VWMA20'].shift(1)
                df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
                
                vols_arr, opens_arr, closes_arr = df['Volume'].values, df['Open'].values, df['Close'].values
                defense_arr = np.full(len(df), np.nan)
                for i in range(60, len(df)):
                    max_idx = (i - 60) + np.argmax(vols_arr[i-60:i+1])
                    defense_arr[i] = min(opens_arr[max_idx], closes_arr[max_idx])
                df['Max_Vol_Defense'] = defense_arr
                df['Prev_Defense'] = df['Max_Vol_Defense'].shift(1)
                
                df['M_Mean'] = df['Momentum_Force'].rolling(window=60).mean()
                df['M_Std'] = df['Momentum_Force'].rolling(window=60).std()
                df['Upper_Bound'] = df['M_Mean'] + 1.5 * df['M_Std']
                df['Lower_Bound'] = df['M_Mean'] - 1.5 * df['M_Std']
                
                df['Candle_Score'] = BottomReversalStrategy.evaluate(df)
                df['VCP_Score'] = VCPStrategy.evaluate(df)
                df['Reso_Score'] = IndicatorResonanceStrategy.evaluate(
                    df,
                    recent_w=st.session_state.div_recent_w,
                    older_w=st.session_state.div_older_w,
                    kd_older_low_th=st.session_state.reso_kd_older_low,
                    kd_older_high_th=st.session_state.reso_kd_older_high,
                    kd_recent_low_th=st.session_state.reso_kd_recent_low,
                    use_macd_abs=st.session_state.use_macd_abs,
                    macd_older_low_th=st.session_state.reso_macd_older_low,
                    macd_recent_low_th=st.session_state.reso_macd_recent_low,
                    cross_days=st.session_state.reso_cross_days
                )
                
                df = df.dropna(subset=['Momentum_Force', 'Max_Vol_Defense', 'VWMA20', 'Prev_MA5', 'Vol_MA20', 'MA60']).copy()
                plot_df, recent_df = df.tail(240).copy(), df.tail(60)
                
                last_row = recent_df.iloc[-1]
                last_date = recent_df.index[-1].strftime('%Y-%m-%d')
                
                st.subheader(f"📊 【{stock_name} ({real_ticker})】 深度解析與策略判定")
                
                st.markdown("### 🎯 演算法最新判定狀態 (全市場掃描標準)")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    rev_score = round(last_row['Candle_Score'], 2)
                    rev_status = "✅ 達標入選" if rev_score >= st.session_state.min_score else "❌ 未達標"
                    st.info(f"**底部翻轉分數：{rev_score}** ({rev_status})\n\n*(門檻：{st.session_state.min_score} 分)*")
                with col_b:
                    vcp_score = round(last_row['VCP_Score'], 2)
                    vcp_status = "✅ 達標入選" if vcp_score >= st.session_state.min_vcp_score else "❌ 未達標"
                    st.success(f"**VCP 收斂分數：{vcp_score}** ({vcp_status})\n\n*(門檻：{st.session_state.min_vcp_score} 分)*")
                with col_c:
                    reso_score = round(last_row['Reso_Score'], 2)
                    reso_status = "✅ 達標入選" if reso_score >= st.session_state.min_reso_score else "❌ 未達標"
                    st.warning(f"**指標共振分數：{reso_score}** ({reso_status})\n\n*(門檻：{st.session_state.min_reso_score} 分)*")
                st.markdown("---")
                
                st.markdown(f"**更新日期：{last_date} | 最新收盤價：{last_row['Close']:.2f}**")
                with st.container(height=165):
                    if last_row['Close'] > last_row['MA20']:
                        st.markdown(f"- 📈 **趨勢方向**：股價位於月線 ({last_row['MA20']:.2f}) 之上，短期波段偏**多頭**。")
                    else:
                        st.markdown(f"- 📉 **趨勢方向**：股價位於月線 ({last_row['MA20']:.2f}) 之下，短期波段偏**空頭**或弱勢整理。")
                        
                    if last_row['Close'] > last_row['VWMA20']:
                        st.markdown(f"- 💰 **籌碼狀況**：站穩加權均線 ({last_row['VWMA20']:.2f})，近期買盤有獲利，具**實質支撐**。")
                    else:
                        st.markdown(f"- ⚠️ **籌碼狀況**：低於加權均線 ({last_row['VWMA20']:.2f})，近期買盤套牢，上方有**解套賣壓**。")
                        
                    st.markdown(f"- 🛡️ **主力防線**：近兩個月最大量防守價為 **{last_row['Max_Vol_Defense']:.2f}**，此價位不破皆可偏多看待。")
                    
                    st.markdown("---")
                    st.markdown("### ⚡ 近三個月極端訊號紀錄 (由近到遠)")
                    
                    signal_logs = []
                    for date, row in recent_df.iterrows():
                        m, c, prev_c = row['Momentum_Force'], row['Close'], row['Prev_Close']
                        ma5, prev_ma5, ma20 = row['MA5'], row['Prev_MA5'], row['MA20']
                        vwma20, prev_vwma20 = row['VWMA20'], row['Prev_VWMA20']
                        defense, prev_defense = row['Max_Vol_Defense'], row['Prev_Defense']
                        vol, vol_ma = row['Volume_Lots'], row['Vol_MA20']
                        cps, res_sc = row['Candle_Score'], row['Reso_Score']
                        
                        date_str = date.strftime('%Y-%m-%d')
                        is_bull_surge = (m > row['Upper_Bound']) or (row['Pct_Change'] >= 4.0 and vol >= vol_ma * 1.5)
                        is_bear_surge = (m < row['Lower_Bound']) or (row['Pct_Change'] <= -4.0 and vol >= vol_ma * 1.5)
                        
                        if c < defense and prev_c >= prev_defense:
                            signal_logs.append(f"- ☠️ **{date_str}** | 🚨 跌破最大量防守價 **{defense:.2f}** (最後防線潰堤) | 收盤: {c:.2f}")
                        elif res_sc >= st.session_state.min_reso_score:
                            signal_logs.append(f"- 🎯 **{date_str}** | 🔥 MACD與KD底部共振發動 (買點浮現) | 共振分: {res_sc:.1f}")
                        elif cps >= st.session_state.min_score:
                            signal_logs.append(f"- ☀️ **{date_str}** | 🚀 低檔強力反轉 (爆量買盤) | 正權重: {cps:.1f}")
                        elif not is_bull_surge and not is_bear_surge:
                            if c < vwma20 and prev_c >= prev_vwma20:
                                signal_logs.append(f"- 📉 **{date_str}** | ⚠️ 跌破 VWMA 加權均線 (建議大部位減碼) | 收盤: {c:.2f}")
                            elif c < ma5 and prev_c >= prev_ma5 and c > ma20:
                                signal_logs.append(f"- 💰 **{date_str}** | ⚡ 高檔跌破 5日線 (短線獲利提早落袋) | 收盤: {c:.2f}")
                        elif is_bull_surge:
                            if c >= vwma20:
                                signal_logs.append(f"- ✅ **{date_str}** | 📈 帶量站穩加權均線 (買點浮現) | 收盤: {c:.2f}")
                        elif is_bear_surge:
                            signal_logs.append(f"- 🔴 **{date_str}** | ⚠️ 爆量長黑 (動能極弱，大戶倒貨) | 收盤: {c:.2f}")

                    signal_logs.reverse()
                    if not signal_logs: st.markdown("> 💡 近期走勢溫和，未觸發特殊訊號。")
                    else:
                        for s in signal_logs: st.markdown(s)

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.8, 1.2, 1.2]})
                
                ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='#1f77b4', linewidth=1.8)
                ax1.plot(plot_df.index, plot_df['MA5'], label='MA5', color='purple', linestyle=':', alpha=0.8, linewidth=1.5)
                ax1.plot(plot_df.index, plot_df['MA20'], label='MA20', color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.plot(plot_df.index, plot_df['VWMA20'], label='VWMA20', color='blue', linestyle='-.', linewidth=1.5, alpha=0.8)
                ax1.plot(plot_df.index, plot_df['Max_Vol_Defense'], label='Defense Line', color='teal', linestyle='-', linewidth=2.0)
                ax1.set_title(f'{stock_name} ({real_ticker}) Price & Defense System', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Price (TWD)', fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.5); ax1.legend(loc='upper left')
                
                ax2.plot(plot_df.index, plot_df['Momentum_Force'], label='Momentum (M)', color='#7f7f7f', linewidth=1.2)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
                ax2.plot(plot_df.index, plot_df['Upper_Bound'], color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='+1.5 Std')
                ax2.plot(plot_df.index, plot_df['Lower_Bound'], color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='-1.5 Std')
                ax2.set_ylabel('Momentum', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.5); ax2.legend(loc='upper left')
                
                cps_colors = ['#d62728' if val > 0 else '#2ca02c' for val in plot_df['Candle_Score']]
                ax3.bar(plot_df.index, plot_df['Candle_Score'], color=cps_colors, alpha=0.7, label='Rev Score')
                ax3.axhline(0, color='black', linestyle='-', linewidth=1.0)
                ax3.axhline(st.session_state.min_score, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Min Rev Threshold')
                ax3.set_ylabel('Algorithm Score', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.5); ax3.legend(loc='upper left')
                
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)

# ----------------------------------------------------
# 頁籤 2：全市場智慧掃描
# ----------------------------------------------------
with tab2:
    st.write("系統將自動抓取全部普通股，尋找符合「低檔翻轉」、「VCP收斂」或「雙指標共振」的標的，並針對入選標的進行多級別背離與均線扣抵判定。")
    
    with st.expander("📖 掃描參數與背離/扣抵判定定義說明", expanded=False):
        st.markdown("""
        | 參數名稱 | 模組分類 | 定義與邏輯說明 |
        | :--- | :--- | :--- |
        | **回測基準日** | 基礎過濾 | 啟用後，系統下載的歷史資料將自動截斷至該日期。讓您能精準回到過去任意一天執行策略回測。 |
        | **底部翻轉最低分數** | 基礎過濾 | 判定低檔長紅或下影線強度的核心數值。預設 8.0 分，分數越高代表買盤力道越強、型態越完美。 |
        | **VCP收斂最低分數** | 基礎過濾 | 判定右側多頭收斂的強度。預設 10.0 分，滿分 20 分。分數越高代表成交量越萎縮、布林帶越壓縮。 |
        | **指標共振最低分數** | 基礎過濾 | 判定 KD 與 MACD 的共振強度。滿分 15 分，滿足基礎條件即給 10 分，動能越強加分越多。 |
        | **月均量最低門檻** | 基礎過濾 | 剔除流動性差的殭屍股。預設 1000 張，確保標的具備足夠的市場參與度與進出空間。 |
        | **近波/前波範圍** | 背離/共振 | 定義尋找「第一低點(近波)」與「第二低點(前波)」的 K 棒區間長度。共振模組亦連動此參數。 |
        | **左X根/右Y根不破** | 背離判定 | 嚴格轉折點定義：該低點必須是往左 X 根、往右 Y 根範圍內的「絕對最低價」，避免抓到半山腰的雜訊。 |
        | **KD/MACD共振門檻** | 指標共振 | 內建 MACD 底底高核心邏輯。支援設定 KD 前波高低點、近波低點限制，以及 MACD 絕對數值濾網。並要求在指定天數內發生金叉。 |
        | **均線扣抵判斷(5/10/20/60)**| 動能濾網 | 判斷觸發日的收盤價是否大於 N 天前的收盤價。若大於(扣低)，代表均線準備上揚，具備支撐動能；若小於(扣高)，代表均線有下彎壓力。 |
        """)

    with st.expander("⚙️ 掃描與背離參數設定", expanded=True):
        profile_names = list(st.session_state.config["profiles"].keys())
        idx = profile_names.index(st.session_state.current_profile) if st.session_state.current_profile in profile_names else 0
            
        def on_profile_change(): apply_profile_to_state(st.session_state.profile_selector)

        col_p1, col_p2, col_p3, col_p4 = st.columns([3, 3, 2, 2])
        with col_p1: st.selectbox("選擇歷史設定檔", profile_names, index=idx, key="profile_selector", on_change=on_profile_change)
        with col_p2: st.text_input("儲存新名稱", placeholder="輸入自訂設定檔名稱...", key="new_profile_input")
        with col_p3:
            st.write(""); st.write("")
            if st.button("💾 儲存設定", use_container_width=True):
                new_input = st.session_state.new_profile_input.strip()
                name_to_save = new_input if new_input != "" else st.session_state.profile_selector
                if name_to_save == "預設參數 (Default)": st.error("❌ 不可覆寫系統預設參數名稱！")
                else:
                    st.session_state.backtest_date = str(st.session_state.backtest_date_obj)
                    current_vals = {k: st.session_state[k] for k in DEFAULT_PARAMS.keys()}
                    st.session_state.config["profiles"][name_to_save] = current_vals
                    st.session_state.config["last_used"] = name_to_save
                    save_config(st.session_state.config)
                    st.success(f"✅ 已成功儲存版本：'{name_to_save}'"); st.rerun()
        with col_p4:
            st.write(""); st.write("")
            if st.button("🗑️ 刪除", use_container_width=True):
                if st.session_state.profile_selector == "預設參數 (Default)": st.error("❌ 系統預設參數不可刪除！")
                else:
                    del st.session_state.config["profiles"][st.session_state.profile_selector]
                    apply_profile_to_state("預設參數 (Default)")
                    st.success("✅ 已刪除"); st.rerun()
                    
        st.markdown("---")
        
        st.markdown("**1. 演算法選擇**")
        algo_mode = st.radio("請選擇欲執行的掃描演算法", ['全部', '底部翻轉', 'VCP', '指標共振'], index=0, horizontal=True)
        if algo_mode == '全部':
            st.warning("💡 提示：您目前選擇【全部】演算法，標的只要符合「任一」條件即會列出。請務必查看表格中的『演算法建議結果』確認觸發類型！")

        st.markdown("**2. 基礎掃描參數與時光機回測設定**")
        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        with col_b1: st.checkbox("啟用指定日期回測", key="use_backtest_date")
        with col_b2: st.date_input("選擇回測基準日", key="backtest_date_obj", disabled=not st.session_state.use_backtest_date)
            
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.number_input("掃描區間(迄)：起算天數", min_value=0, max_value=1000, step=1, key="lookback_end")
            st.number_input("掃描區間(起)：回推天數", min_value=0, max_value=1000, step=1, key="lookback_start")
            if st.session_state.lookback_start < st.session_state.lookback_end: st.warning("⚠️ 「起」需大於「迄」。")
        with col_b: st.number_input("底部翻轉最低分", min_value=1.0, max_value=30.0, step=1.0, key="min_score")
        with col_c: st.number_input("VCP收斂最低分", min_value=1.0, max_value=20.0, step=1.0, key="min_vcp_score")
        with col_d: st.number_input("指標共振最低分", min_value=1.0, max_value=15.0, step=1.0, key="min_reso_score")
        with col_e: st.number_input("月均量最低門檻", min_value=0, max_value=100000, step=100, key="min_vol_ma20")
        
        st.markdown("**3. 波段檢測週期與背離條件設定 (共振演算法同步使用波段天數)**")
        st.checkbox("啟用單一組自訂背離週期 (未勾則預設比對三組：(5,20)、(5,60)、(20,60))", key="use_single_div")
        col_f, col_g, col_h, col_i, col_j, col_k = st.columns(6)
        with col_f: st.number_input("近波範圍", min_value=5, max_value=60, step=1, key="div_recent_w", disabled=not st.session_state.use_single_div)
        with col_g: st.number_input("前波範圍", min_value=10, max_value=240, step=1, key="div_older_w", disabled=not st.session_state.use_single_div)
        with col_h: st.number_input("左X根不破", min_value=0, max_value=20, step=1, key="pivot_left")
        with col_i: st.number_input("右Y根不破", min_value=0, max_value=20, step=1, key="pivot_right")
        with col_j: st.number_input("近波低點數", min_value=0, max_value=20, step=1, key="recent_lows_cnt")
        with col_k: st.number_input("前波低點數", min_value=0, max_value=20, step=1, key="older_lows_cnt")
        
        st.markdown("**4. 均線扣抵判斷設定 (扣低有利均線向上)**")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1: st.checkbox("啟用 5MA 扣抵判斷", key="kou_di_5")
        with col_k2: st.checkbox("啟用 10MA 扣抵判斷", key="kou_di_10")
        with col_k3: st.checkbox("啟用 20MA 扣抵判斷", key="kou_di_20")
        with col_k4: st.checkbox("啟用 60MA 扣抵判斷", key="kou_di_60")
            
        st.markdown("**5. 指標共振演算法進階濾網 (依據上方波段範圍天數)**")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.number_input("KD前波低點 <", value=st.session_state.reso_kd_older_low, step=1.0, key="reso_kd_older_low")
            st.number_input("KD前波高點 <", value=st.session_state.reso_kd_older_high, step=1.0, key="reso_kd_older_high")
        with col_r2:
            st.number_input("KD近波低點 >", value=st.session_state.reso_kd_recent_low, step=1.0, key="reso_kd_recent_low")
            st.number_input("近期金叉天數 <=", value=st.session_state.reso_cross_days, step=1, key="reso_cross_days")
        with col_r3:
            st.checkbox("啟用 MACD絕對數值濾網", key="use_macd_abs")
            st.number_input("MACD前波低 <", value=st.session_state.reso_macd_older_low, step=0.1, key="reso_macd_older_low", disabled=not st.session_state.use_macd_abs)
            st.number_input("MACD近波低 >", value=st.session_state.reso_macd_recent_low, step=0.1, key="reso_macd_recent_low", help="若要抓水下翻紅，請設為負數或 0.0", disabled=not st.session_state.use_macd_abs)

    st.markdown("---")
    
    if st.button("🚀 開始智慧區間掃描", type="primary"):
        status_text = st.empty(); progress_bar = st.progress(0)
        status_text.text("⏳ [初始化] 正在同步台股最新代號與名稱清單，請稍候...")
        
        yf_session = get_yf_session()
        
        div_pairs = [(st.session_state.div_recent_w, st.session_state.div_older_w)] if st.session_state.use_single_div else [(5, 20), (5, 60), (20, 60)]
        max_older_w = max(pair[1] for pair in div_pairs)
        total_needed_days = st.session_state.lookback_start + max_older_w + st.session_state.pivot_left + 70
        
        dl_kwargs = {}
        dl_60m_kwargs = {}
        
        if st.session_state.use_backtest_date:
            end_dt = st.session_state.backtest_date_obj + datetime.timedelta(days=1)
            start_dt = end_dt - datetime.timedelta(days=730)
            dl_kwargs = {"start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
            
            start_60m = datetime.date.today() - datetime.timedelta(days=729)
            if end_dt <= start_60m:
                st.warning("⚠️ 回測基準日過早 (超過 730 天)，Yahoo Finance 不支援該時期的 60 分 K 線，短線背離檢測將顯示無資料。")
                dl_60m_kwargs = None
            else:
                dl_60m_kwargs = {"start": start_60m.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
        else:
            if total_needed_days <= 60: dl_period = "3mo"
            elif total_needed_days <= 120: dl_period = "6mo"
            elif total_needed_days <= 250: dl_period = "1y"
            elif total_needed_days <= 500: dl_period = "2y"
            elif total_needed_days <= 1250: dl_period = "5y"
            else: dl_period = "10y"
            dl_period_60m = "3mo" if total_needed_days <= 60 else "6mo" if total_needed_days <= 120 else "730d"
            dl_kwargs = {"period": dl_period}
            dl_60m_kwargs = {"period": dl_period_60m}
            
        market_info = MarketRegimeFilter.evaluate(yf_session, st.session_state.backtest_date_obj if st.session_state.use_backtest_date else None)
        
        stock_dict = get_all_tw_stocks()
        
        if not stock_dict:
            status_text.empty(); progress_bar.empty(); st.error("❌ 無法取得台股清單，請檢查網路連線。")
        else:
            tickers = list(stock_dict.keys())
            reversal_candidates = {} 
            
            chunk_size = 100
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                status_text.text(f"[階段一] 正在全市場區間掃描 [{algo_mode}]：進度 {i} / {len(tickers)} 檔...")
                try:
                    data = yf.download(chunk, threads=False, progress=False, session=yf_session, **dl_kwargs)
                    for ticker in chunk:
                        try:
                            df = data.xs(ticker, axis=1, level=1).dropna(how='all') if isinstance(data.columns, pd.MultiIndex) else (data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame())
                            
                            if st.session_state.use_backtest_date:
                                target_dt = pd.to_datetime(st.session_state.backtest_date_obj)
                                df = df[df.index.tz_localize(None).normalize() <= target_dt]
                                
                            if df.empty or len(df) <= st.session_state.lookback_start + 20: continue
                            
                            df = TechnicalIndicators.add_kd(df)
                            df = TechnicalIndicators.add_macd(df)
                            
                            df['Pct_Change'] = df['Close'].pct_change() * 100
                            df['Volume_Lots'] = df['Volume'] / 1000
                            df['MA20'] = df['Close'].rolling(window=20).mean()
                            df['MA60'] = df['Close'].rolling(window=60).mean()
                            df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
                            df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
                            
                            std20 = df['Close'].rolling(window=20).std()
                            df['BB_Upper'] = df['MA20'] + 2 * std20
                            df['BB_Lower'] = df['MA20'] - 2 * std20
                            
                            df['Candle_Score'] = BottomReversalStrategy.evaluate(df) if algo_mode in ['全部', '底部翻轉'] else pd.Series(0.0, index=df.index)
                            df['VCP_Score'] = VCPStrategy.evaluate(df) if algo_mode in ['全部', 'VCP'] else pd.Series(0.0, index=df.index)
                            
                            if algo_mode in ['全部', '指標共振']:
                                df['Reso_Score'] = IndicatorResonanceStrategy.evaluate(
                                    df,
                                    recent_w=st.session_state.div_recent_w,
                                    older_w=st.session_state.div_older_w,
                                    kd_older_low_th=st.session_state.reso_kd_older_low,
                                    kd_older_high_th=st.session_state.reso_kd_older_high,
                                    kd_recent_low_th=st.session_state.reso_kd_recent_low,
                                    use_macd_abs=st.session_state.use_macd_abs,
                                    macd_older_low_th=st.session_state.reso_macd_older_low,
                                    macd_recent_low_th=st.session_state.reso_macd_recent_low,
                                    cross_days=st.session_state.reso_cross_days
                                )
                            else:
                                df['Reso_Score'] = pd.Series(0.0, index=df.index)
                            
                            best_combined_score = -1
                            best_row, best_offset = None, 0
                            
                            for offset in range(st.session_state.lookback_end, st.session_state.lookback_start + 1):
                                t_row = df.iloc[-1 - offset]
                                r_score, v_score, reso_score, vol_ma = t_row['Candle_Score'], t_row['VCP_Score'], t_row['Reso_Score'], t_row['Vol_MA20']
                                
                                if vol_ma >= st.session_state.min_vol_ma20:
                                    is_r_pass = (algo_mode in ['全部', '底部翻轉']) and (r_score >= st.session_state.min_score)
                                    is_v_pass = (algo_mode in ['全部', 'VCP']) and (v_score >= st.session_state.min_vcp_score)
                                    is_reso_pass = (algo_mode in ['全部', '指標共振']) and (reso_score >= st.session_state.min_reso_score)
                                    
                                    if is_r_pass or is_v_pass or is_reso_pass:
                                        combo_score = r_score + v_score + reso_score
                                        if combo_score > best_combined_score:
                                            best_combined_score = combo_score
                                            best_row, best_offset = t_row, offset
                            
                            if best_row is not None:
                                clean_ticker = ticker.split('.')[0]
                                reversal_candidates[ticker] = {
                                    "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                                    "股票代號": clean_ticker, "股票名稱": stock_dict[ticker],
                                    "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                                    "當日收盤": round(float(best_row['Close']), 2),
                                    "月均量(張)": int(best_row['Vol_MA20']),
                                    "反轉分數": round(float(best_row['Candle_Score']), 2),
                                    "VCP分數": round(float(best_row['VCP_Score']), 2),
                                    "共振分數": round(float(best_row['Reso_Score']), 2)
                                }
                        except Exception: continue
                except Exception: pass
                progress_bar.progress(min(1.0, (i + chunk_size) / len(tickers)))
            
            reversal_list = list(reversal_candidates.values())
            
            kou_di_periods = []
            if st.session_state.kou_di_5: kou_di_periods.append(5)
            if st.session_state.kou_di_10: kou_di_periods.append(10)
            if st.session_state.kou_di_20: kou_di_periods.append(20)
            if st.session_state.kou_di_60: kou_di_periods.append(60)
            
            if reversal_list:
                progress_bar.progress(0)
                status_text.text(f"[階段二] 正在分析 {len(reversal_list)} 檔入選標的之多級別背離特徵與扣抵判定...")
                
                final_results = []
                for idx, item in enumerate(reversal_list):
                    try:
                        ticker, specific_offset, daily_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
                        has_daily_div, has_m60_div = False, False
                        rl_cnt, ol_cnt = st.session_state.recent_lows_cnt, st.session_state.older_lows_cnt
                        p_left, p_right = st.session_state.pivot_left, st.session_state.pivot_right
                        
                        if not daily_df.empty:
                            if specific_offset > 0: daily_df = daily_df.iloc[:-specific_offset]
                            
                            for n in kou_di_periods:
                                if len(daily_df) >= n:
                                    curr_p = daily_df['Close'].iloc[-1]
                                    drop_p = daily_df['Close'].iloc[-n]
                                    item[f"扣抵狀態({n}MA)"] = "✅ 扣低" if curr_p > drop_p else "❌ 扣高"
                                else:
                                    item[f"扣抵狀態({n}MA)"] = "無資料"
                            
                            for r_w, o_w in div_pairs:
                                d_kd = DivergenceStrategy.check_bottom_divergence(daily_df, 'Low', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right)
                                d_macd = DivergenceStrategy.check_bottom_divergence(daily_df, 'Low', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right)
                                res = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                                item[f"日K背離({r_w},{o_w})"] = "+".join(res) if res else "無"
                                if res: has_daily_div = True
                        else:
                            for r_w, o_w in div_pairs: item[f"日K背離({r_w},{o_w})"] = "無資料"
                            for n in kou_di_periods: item[f"扣抵狀態({n}MA)"] = "無資料"
                        
                        if dl_60m_kwargs is not None:
                            m60_df = yf.Ticker(ticker, session=yf_session).history(interval="60m", **dl_60m_kwargs)
                            if specific_offset > 0 and not daily_df.empty:
                                target_date = daily_df.index[-1].date()
                                m60_df = m60_df[[d.date() <= target_date for d in m60_df.index]]
                                
                            if not m60_df.empty:
                                m60_df = TechnicalIndicators.add_macd(TechnicalIndicators.add_kd(m60_df))
                                for r_w, o_w in div_pairs:
                                    m_kd = DivergenceStrategy.check_bottom_divergence(m60_df, 'Low', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right)
                                    m_macd = DivergenceStrategy.check_bottom_divergence(m60_df, 'Low', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right)
                                    res = [x for x, b in zip(["KD", "MACD"], [m_kd, m_macd]) if b]
                                    item[f"60分K背離({r_w},{o_w})"] = "+".join(res) if res else "無"
                                    if res: has_m60_div = True
                            else:
                                for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"
                        else:
                            for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"

                        base_tags = []
                        if item["反轉分數"] >= st.session_state.min_score: base_tags.append("底部翻轉")
                        if item["VCP分數"] >= st.session_state.min_vcp_score: base_tags.append("VCP收斂")
                        if item["共振分數"] >= st.session_state.min_reso_score: base_tags.append("指標共振")
                        
                        div_tag = " + 雙級別共振" if (has_daily_div and has_m60_div) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if has_m60_div else " (無背離)"))
                        item["演算法建議結果"] = " & ".join(base_tags) + div_tag
                        
                        final_results.append(item)
                    except Exception: pass
                    progress_bar.progress(min(1.0, (idx + 1) / len(reversal_list)))

                status_text.empty(); progress_bar.empty()
                date_str = f"({st.session_state.backtest_date_obj})" if st.session_state.use_backtest_date else ""
                st.success(f"🎉 掃描完成！本次共精選出 **{len(final_results)}** 檔符合條件的標的 {date_str}。")
                
                if market_info:
                    st.markdown("### 🌐 大盤位階與期貨基準動態濾網評估結果")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("加權指數收盤", market_info["加權指數收盤"])
                    m_col2.metric("月線 (MA20)", market_info["月線 (MA20)"])
                    m_col3.metric("季線 (MA60)", market_info["季線 (MA60)"])
                    m_col4.metric("約當大台基礎", market_info["自動運算基準價值 (約當大台基礎)"])
                    st.info(f"**大盤環境判定：** {market_info['大盤環境判定']}")
                    st.markdown("---")
                
                res_df = pd.DataFrame(final_results)
                
                base_cols = ['股票代號', '股票名稱', '觸發日期', '演算法建議結果', '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)']
                div_cols = [c for c in res_df.columns if '背離' in c and c not in base_cols]
                kou_cols = [c for c in res_df.columns if '扣抵狀態' in c]
                cols = base_cols + div_cols + kou_cols
                
                res_df = res_df[cols].sort_values(by=["共振分數", "反轉分數", "VCP分數"], ascending=[False, False, False]).reset_index(drop=True)
                res_df.index = res_df.index + 1
                
                st.dataframe(res_df, use_container_width=True)
                
                csv = res_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下載建議清單 (CSV)",
                    data=csv,
                    file_name=f'stock_scan_{algo_mode}_results.csv',
                    mime='text/csv'
                )
            else:
                status_text.empty(); progress_bar.empty()
                st.info(f"掃描完成！在指定的區間與條件下，全市場無任何符合「{algo_mode}」的標的。")
