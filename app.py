import os
import json
import warnings
import logging
import requests
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
    "min_vol_ma20": 1000,
    "use_single_div": False,
    "div_recent_w": 5,
    "div_older_w": 20,
    "pivot_left": 0,
    "pivot_right": 0,
    "recent_lows_cnt": 0,
    "older_lows_cnt": 0
}

def load_config():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_used": "預設參數 (Default)", "profiles": {"預設參數 (Default)": DEFAULT_PARAMS.copy()}}

def save_config(config):
    with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# ==========================================
# 獨立模組 1：技術指標計算
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
# 獨立模組 2：背離策略判斷 
# ==========================================
class DivergenceStrategy:
    @staticmethod
    def check_bottom_divergence(
        df, price_col='Low', ind_col='K', ind_signal_col='D', 
        recent_w=20, older_w=60, 
        recent_lows_cnt=0, older_lows_cnt=0,
        pivot_left=0, pivot_right=0
    ):
        if len(df) < older_w:
            return False
            
        recent_start = len(df) - recent_w
        recent_end = len(df)
        older_start = len(df) - older_w
        older_end = recent_start
        
        if recent_start < 0 or older_start < 0 or older_start >= older_end:
            return False
            
        prices = df[price_col].values
        k_vals = df[ind_col].values
        d_vals = df[ind_signal_col].values
        
        recent_prices = prices[recent_start:recent_end]
        if len(recent_prices) == 0:
            return False
        idx1_iloc = recent_start + np.argmin(recent_prices)
        p1 = prices[idx1_iloc]
        i1 = k_vals[idx1_iloc]
        
        def check_divergence_condition(p_iloc):
            p2 = prices[p_iloc]
            i2 = k_vals[p_iloc]
            if not (p2 > p1 and i2 < i1):
                return False
                
            s_idx = min(idx1_iloc, p_iloc)
            e_idx = max(idx1_iloc, p_iloc)
            
            if e_idx - s_idx + 1 > 2:
                cross_found = False
                for j in range(s_idx + 1, e_idx + 1):
                    if k_vals[j] < d_vals[j] and k_vals[j-1] >= d_vals[j-1]:
                        cross_found = True
                        break
                if not cross_found:
                    return False
            else:
                return False
            return True

        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            older_prices = prices[older_start:older_end]
            if len(older_prices) == 0:
                return False
            idx2_iloc = older_start + np.argmin(older_prices)
            return check_divergence_condition(idx2_iloc)

        def get_valid_pivots_iloc(start_loc, end_loc):
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s = max(0, i_loc - pivot_left)
                e = min(len(prices), i_loc + pivot_right + 1)
                window = prices[s:e]
                if prices[i_loc] == np.min(window):
                    pivots.append(i_loc)
            return pivots

        if recent_lows_cnt > 0:
            recent_pivots_iloc = get_valid_pivots_iloc(recent_start, recent_end)
            if idx1_iloc in recent_pivots_iloc:
                recent_pivots_iloc.remove(idx1_iloc)
            if not recent_pivots_iloc:
                return False
            recent_pivots_iloc = sorted(recent_pivots_iloc, key=lambda x: prices[x])[:recent_lows_cnt]
            for p_iloc in recent_pivots_iloc:
                if not check_divergence_condition(p_iloc):
                    return False
                    
        if older_lows_cnt > 0:
            older_pivots_iloc = get_valid_pivots_iloc(older_start, older_end)
            if not older_pivots_iloc:
                return False
            older_pivots_iloc = sorted(older_pivots_iloc, key=lambda x: prices[x])[:older_lows_cnt]
            for p_iloc in older_pivots_iloc:
                if not check_divergence_condition(p_iloc):
                    return False
                    
        return True

# ==========================================
# 資料抓取與共用函式
# ==========================================
@st.cache_data(ttl=86400)
def _cached_get_all_tw_stocks():
    """內部快取函式，抓取台股代碼"""
    stocks = {}
    for mode in ['2', '4']:
        suffix = '.TW' if mode == '2' else '.TWO'
        try:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
            # 確保使用 Big5 避免台股證交所網頁亂碼問題
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            res.encoding = 'big5'
            soup = BeautifulSoup(res.text, 'html.parser')
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) > 0:
                    text = tds[0].text.strip().replace('\u3000', ' ')
                    if ' ' in text:
                        parts = text.split(' ')
                        code = parts[0].strip()
                        name = parts[1].strip()
                        if code.isdigit() and len(code) == 4:
                            stocks[code + suffix] = name
        except Exception as e:
            pass
    return stocks

def get_all_tw_stocks():
    """【修復快取為空的問題】若抓取失敗，則強迫清除快取"""
    stocks = _cached_get_all_tw_stocks()
    if not stocks:
        _cached_get_all_tw_stocks.clear()
    return stocks

def get_tw_stock_name(ticker):
    code = ticker.split('.')[0]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            h1 = soup.find('h1')
            if h1:
                return h1.text.strip()
    except:
        pass
    try:
        return yf.Ticker(ticker).info.get('shortName', code)
    except:
        return code

def get_stock_data(symbol):
    code = str(symbol).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"):
        targets = [code]
    else:
        targets = [f"{code}.TW", f"{code}.TWO"]
        
    for ticker in targets:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if not df.empty:
            return df, ticker
    return pd.DataFrame(), code

# ==========================================
# 全市場批次掃描核心 (解決執行過久或逾時問題)
# ==========================================
def run_market_scan(stocks_dict, base_older_w, p_left, p_right, r_cnt, o_cnt):
    tickers = list(stocks_dict.keys())
    matched_stocks = []
    
    batch_size = 200
    total_batches = (len(tickers) // batch_size) + 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 定義三組不同週期的參數：(近波天數, 前波天數)
    periods = [
        (10, base_older_w),
        (20, base_older_w * 2),
        (60, base_older_w * 3)
    ]
    
    for i in range(total_batches):
        batch_tickers = tickers[i*batch_size : (i+1)*batch_size]
        if not batch_tickers:
            continue
            
        status_text.text(f"🔍 掃描進度：正在下載第 {i+1}/{total_batches} 批次資料 (採用非同步抓取)...")
        
        try:
            # 使用 yf.download 進行批次下載，大幅縮短全市場掃描時間
            data = yf.download(batch_tickers, period="6mo", threads=True, progress=False)
        except Exception:
            continue
        
        for ticker in batch_tickers:
            try:
                if len(batch_tickers) == 1:
                    df = data.copy()
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(1):
                            df = data.xs(ticker, axis=1, level=1).copy()
                        else:
                            continue
                    else:
                        df = data.copy()
                        
                df = df.dropna(subset=['Close'])
                if len(df) < periods[-1][1]: 
                    continue
                    
                df = TechnicalIndicators.add_kd(df)
                
                # 自動比對三組週期
                is_match = False
                for r_w, o_w in periods:
                    if DivergenceStrategy.check_bottom_divergence(
                        df, recent_w=r_w, older_w=o_w,
                        recent_lows_cnt=r_cnt, older_lows_cnt=o_cnt,
                        pivot_left=p_left, pivot_right=p_right
                    ):
                        is_match = True
                        break
                        
                if is_match:
                    matched_stocks.append(ticker)
            except Exception:
                continue
        
        progress_bar.progress(min(1.0, (i + 1) / total_batches))
        
    status_text.empty()
    progress_bar.empty()
    return matched_stocks

# ==========================================
# 介面主程式與 Session State 初始化
# ==========================================
st.set_page_config(page_title="台股 K線型態與位階深度解析系統", layout="wide")

if "config" not in st.session_state:
    st.session_state.config = load_config()

if "current_profile" not in st.session_state:
    st.session_state.current_profile = st.session_state.config.get("last_used", "預設參數 (Default)")

def apply_profile_to_state(profile_name):
    prof = st.session_state.config["profiles"].get(profile_name, DEFAULT_PARAMS)
    for k, v in prof.items():
        st.session_state[k] = v
    st.session_state.current_profile = profile_name
    st.session_state.config["last_used"] = profile_name
    save_config(st.session_state.config)

if "lookback_end" not in st.session_state:
    apply_profile_to_state(st.session_state.current_profile)

st.title("📈 台股 K線型態與位階深度解析系統")
st.markdown("<style>header {visibility: hidden;}</style>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 單檔深度解析", "🚀 全市場掃描 (低檔反轉+背離)"])

# ----------------------------------------------------
# 頁籤 1：單檔深度解析 (修復繪圖截斷問題)
# ----------------------------------------------------
with tab1:
    st.write("請在下方輸入股票代號（例如：`2495`、`00631L`），系統將自動抓取近兩年資料進行診斷。")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("輸入股票代號", value="2495", placeholder="例如：2330").strip()
    with col2:
        st.write("") 
        st.write("")
        submit_btn = st.button("開始分析", type="primary", key="btn_analysis")

    if submit_btn and user_input:
        with st.spinner(f"⏳ 正在抓取 [{user_input}] 資料並進行解析中..."):
            df, real_ticker = get_stock_data(user_input)
            
            if df.empty:
                st.error(f"❌ 查無 [{user_input}] 的歷史數據，請確認代號是否正確。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                st.subheader(f"📊 【{stock_name} ({real_ticker})】 K線型態解析")
                
                # 計算基礎指標
                df['Pct_Change'] = df['Close'].pct_change() * 100
                df['Volume_Lots'] = df['Volume'] / 1000
                df['Momentum_Force'] = df['Pct_Change'] * df['Volume_Lots']
                df['Prev_Close'] = df['Close'].shift(1)
                
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['Prev_MA5'] = df['MA5'].shift(1)
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
                df['Std20'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + 2 * df['Std20']
                
                df['VWMA20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
                df['Prev_VWMA20'] = df['VWMA20'].shift(1)
                df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
                
                max_vol_defense = []
                for i in range(len(df)):
                    if i < 60:
                        max_vol_defense.append(np.nan)
                    else:
                        window = df.iloc[i-60:i+1]
                        max_idx = window['Volume'].idxmax()
                        max_bar = window.loc[max_idx]
                        defense_price = min(max_bar['Open'], max_bar['Close'])
                        max_vol_defense.append(defense_price)
                df['Max_Vol_Defense'] = max_vol_defense
                df['Prev_Defense'] = df['Max_Vol_Defense'].shift(1)
                
                df['Body'] = abs(df['Close'] - df['Open'])
                df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
                df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
                df['Total_Range'] = df['High'] - df['Low']
                df['Total_Range'] = df['Total_Range'].replace(0, 0.001)
                
                df['Vol_Mult'] = (df['Volume_Lots'] / df['Vol_MA20']).clip(0.5, 3.0)
                
                cond_high_pin = (df['BIAS20'] > 4) & (df['Upper_Shadow'] > df['Body'] * 1.5) & (df['Upper_Shadow'] > df['Total_Range'] * 0.4)
                cond_high_black = (df['BIAS20'] > 3) & (df['Close'] < df['Open']) & (df['Pct_Change'] <= -2.5)
                cond_low_pin = (df['BIAS20'] <= 0) & (df['Lower_Shadow'] > df['Body'] * 1.5) & (df['Lower_Shadow'] > df['Total_Range'] * 0.4)
                cond_low_red = (df['BIAS20'] <= 0) & (df['Close'] > df['Open']) & (df['Pct_Change'] >= 2.5)
                
                df['Candle_Score'] = 0.0
                df.loc[cond_high_pin, 'Candle_Score'] = -7 * (df['Upper_Shadow'] / df['Total_Range']) * df['Vol_Mult']
                df.loc[cond_high_black, 'Candle_Score'] = -5 * df['Vol_Mult']
                df.loc[cond_low_pin, 'Candle_Score'] = 7 * (df['Lower_Shadow'] / df['Total_Range']) * df['Vol_Mult']
                df.loc[cond_low_red, 'Candle_Score'] = 5 * df['Vol_Mult']
                
                df['M_Mean'] = df['Momentum_Force'].rolling(window=60).mean()
                df['M_Std'] = df['Momentum_Force'].rolling(window=60).std()
                df['Upper_Bound'] = df['M_Mean'] + 1.5 * df['M_Std']
                df['Lower_Bound'] = df['M_Mean'] - 1.5 * df['M_Std']
                
                df = df.dropna(subset=['Momentum_Force', 'Max_Vol_Defense', 'VWMA20', 'Prev_MA5', 'Vol_MA20']).copy()
                plot_df = df.tail(240).copy()
                recent_df = df.tail(60)
                
                last_row = recent_df.iloc[-1]
                last_date = recent_df.index[-1].strftime('%Y-%m-%d')
                last_c = last_row['Close']
                last_ma20 = last_row['MA20']
                last_vwma20 = last_row['VWMA20']
                last_defense = last_row['Max_Vol_Defense']
                
                st.markdown(f"**更新日期：{last_date} | 最新收盤價：{last_c:.2f}**")
                
                # 文字結論區
                with st.container(height=165):
                    st.markdown(f"- 📈 **趨勢方向**：股價位於月線 ({last_ma20:.2f}) 之上，短期偏**多頭**。" if last_c > last_ma20 else f"- 📉 **趨勢方向**：股價位於月線之下，短期偏**空頭**。")
                    st.markdown(f"- 💰 **籌碼狀況**：站穩加權均線 ({last_vwma20:.2f})，具**實質支撐**。" if last_c > last_vwma20 else f"- ⚠️ **籌碼狀況**：低於加權均線，上方有**解套賣壓**。")
                    st.markdown(f"- 🛡️ **主力防線**：近兩個月最大量防守價為 **{last_defense:.2f}**，不破皆可偏多看待。")
                
                # 繪圖區 (修復截斷部分)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.8, 1.2, 1.2]})
                
                # 主圖: 價格與均線
                ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='#1f77b4', linewidth=1.8)
                ax1.plot(plot_df.index, plot_df['MA5'], label='MA5', color='purple', linestyle=':', alpha=0.8, linewidth=1.5)
                ax1.plot(plot_df.index, plot_df['MA20'], label='MA20', color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.plot(plot_df.index, plot_df['VWMA20'], label='VWMA20', color='blue', linestyle='-.', linewidth=1.5, alpha=0.8)
                ax1.plot(plot_df.index, plot_df['Max_Vol_Defense'], label='Defense Line', color='teal', linewidth=2)
                ax1.set_title(f"{stock_name} ({real_ticker}) - Price Analysis")
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                
                # 副圖1: 成交量
                vol_colors = ['red' if row['Close'] >= row['Open'] else 'green' for _, row in plot_df.iterrows()]
                ax2.bar(plot_df.index, plot_df['Volume_Lots'], color=vol_colors, alpha=0.6)
                ax2.plot(plot_df.index, plot_df['Vol_MA20'], label='Vol MA20', color='orange', linewidth=1.5)
                ax2.set_title("Volume (Lots)")
                ax2.legend(loc='best')
                
                # 副圖2: KD 指標
                kd_df = TechnicalIndicators.add_kd(plot_df)
                ax3.plot(kd_df.index, kd_df['K'], label='K (9)', color='blue')
                ax3.plot(kd_df.index, kd_df['D'], label='D (9)', color='orange')
                ax3.axhline(20, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(80, color='green', linestyle='--', alpha=0.5)
                ax3.set_title("KD Indicator")
                ax3.legend(loc='best')
                
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)

# ----------------------------------------------------
# 頁籤 2：全市場掃描 (解決無法掃描與還原 UI 設定)
# ----------------------------------------------------
with tab2:
    st.markdown("### 參數設定")
    st.write("自訂：第二低點(前波)範圍")
    older_w_input = st.number_input("自訂：第二低點(前波)範圍", min_value=10, max_value=120, value=20, step=5, label_visibility="collapsed")
    
    st.markdown("### 3. 嚴格轉折點與比對數量設定 (四者皆設為0時將自動關閉嚴格邏輯，退回基礎比對)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        pivot_left_input = st.number_input("轉折判定：往前抓K棒數", min_value=0, max_value=10, value=3)
        recent_lows_cnt_input = st.number_input("本波(近波)比對低點數", min_value=0, max_value=10, value=3)
    with col_t2:
        pivot_right_input = st.number_input("轉折判定：往後取K棒數", min_value=0, max_value=10, value=3)
        older_lows_cnt_input = st.number_input("前波比對低點數 (X)", min_value=0, max_value=10, value=5)
    
    scan_btn = st.button("🚀 開始智慧區間掃描", type="primary", key="btn_scan")
    
    st.info("💡 系統將自動同時比對三組週期參數進行背離運算。")
    
    if scan_btn:
        stocks_dict = get_all_tw_stocks()
        if not stocks_dict:
            st.error("❌ 無法取得台股代碼列表，請檢查網路連線或稍後再試 (系統已自動重置快取，請再次點擊掃描)。")
        else:
            with st.spinner("啟動非同步全市場掃描，請稍候..."):
                matched = run_market_scan(
                    stocks_dict, 
                    older_w_input, 
                    pivot_left_input, 
                    pivot_right_input, 
                    recent_lows_cnt_input, 
                    older_lows_cnt_input
                )
                
            if matched:
                st.success(f"🎉 掃描完成！共發現 {len(matched)} 檔符合標的。")
                for m in matched:
                    st.write(f"- {stocks_dict.get(m, m)} ({m})")
            else:
                st.info("掃描完成！在您指定的區間與條件下，市場無任何符合「低檔強力反轉」訊號的標的。")
