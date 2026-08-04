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
    """讀取本機參數設定檔，若無則建立預設結構"""
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_used": "預設參數 (Default)", "profiles": {"預設參數 (Default)": DEFAULT_PARAMS}}

def save_config(config):
    """將參數設定檔寫入本機 JSON"""
    with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# ==========================================
# 獨立模組 1：技術指標計算
# ==========================================
class TechnicalIndicators:
    @staticmethod
    def add_kd(df, n=9, m1=3, m2=3):
        """計算 KD 指標 (標準設定 9, 3, 3)"""
        df = df.copy()
        low_min = df['Low'].rolling(window=n, min_periods=1).min()
        high_max = df['High'].rolling(window=n, min_periods=1).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-8) * 100
        # 台股習慣使用平滑移動平均
        df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """計算 MACD 指標"""
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
        """
        判定底背離邏輯 (多低點嚴格驗證版)：
        1. 定義近波與前波範圍，並找出近波的「絕對最低點」(p1, i1)。
        2. 若 recent_lows_cnt, older_lows_cnt, pivot_left, pivot_right 均為 0，則退回基礎絕對低點比對。
        3. 轉折低點定義：該 K 棒價格必須是 [往前左抓 left 根, 往後右抓 right 根] 區間內的最小值。(右側若不足則抓到最新一根為止)
        4. 本波比對：在近波內找出所有轉折低點，排除絕對最低點後，取最低的 `recent_lows_cnt` 個。這幾個低點全數必須與 p1 構成背離。
        5. 前波比對：在前波內找出所有轉折低點，取最低的 `older_lows_cnt` 個。這幾個低點全數必須與 p1 構成背離。
        6. 背離定義：過去轉折點價格 > p1 (價格破底) 且 過去指標 < i1 (指標不破底)，且兩點之間指標曾發生過死叉。
        """
        if len(df) < older_w:
            return False
            
        recent_start = len(df) - recent_w
        recent_end = len(df)
        older_start = len(df) - older_w
        older_end = recent_start
        
        if recent_start < 0 or older_start < 0:
            return False
            
        # 提取資料轉為 numpy 加速運算
        prices = df[price_col].values
        k_vals = df[ind_col].values
        d_vals = df[ind_signal_col].values
        
        # 1. 尋找近波絕對第一低點
        recent_prices = prices[recent_start:recent_end]
        if len(recent_prices) == 0:
            return False
        idx1_iloc = recent_start + np.argmin(recent_prices)
        p1 = prices[idx1_iloc]
        i1 = k_vals[idx1_iloc]
        
        def check_divergence_condition(p_iloc):
            """單點背離與死叉條件驗證"""
            p2 = prices[p_iloc]
            i2 = k_vals[p_iloc]
            # 條件 A & B: 價格破底，指標不破底
            if not (p2 > p1 and i2 < i1):
                return False
                
            s_idx = min(idx1_iloc, p_iloc)
            e_idx = max(idx1_iloc, p_iloc)
            
            # 條件 C: 兩點之間必須經歷過死叉
            if e_idx - s_idx + 1 > 2:
                cross_found = False
                for j in range(s_idx + 1, e_idx + 1):
                    # 死叉：快線小於慢線，且前一根快線大於等於慢線
                    if k_vals[j] < d_vals[j] and k_vals[j-1] >= d_vals[j-1]:
                        cross_found = True
                        break
                if not cross_found:
                    return False
            else:
                return False
            return True

        # ==========================
        # 判斷是否全為 0，若是，則關閉嚴格多點判定，退回基礎的絕對單低點比對
        # ==========================
        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            older_prices = prices[older_start:older_end]
            if len(older_prices) == 0:
                return False
            idx2_iloc = older_start + np.argmin(older_prices)
            return check_divergence_condition(idx2_iloc)

        # 嚴格轉折判定函式
        def get_valid_pivots_iloc(start_loc, end_loc):
            """取得區間內的所有轉折低點索引"""
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s = max(0, i_loc - pivot_left)
                # 往右取若遇到邊界，min() 自動支援切片至最後一筆
                e = min(len(prices), i_loc + pivot_right + 1)
                window = prices[s:e]
                if prices[i_loc] == np.min(window):
                    pivots.append(i_loc)
            return pivots

        # ==========================
        # 2. 本波(近波) 其他低點比對
        # ==========================
        if recent_lows_cnt > 0:
            recent_pivots_iloc = get_valid_pivots_iloc(recent_start, recent_end)
            # 排除本波絕對最低點
            if idx1_iloc in recent_pivots_iloc:
                recent_pivots_iloc.remove(idx1_iloc)
                
            # 若本波內沒有其他低點，視為無法滿足多點比對條件
            if not recent_pivots_iloc:
                return False
                
            # 取本波價格最低的前 N 個轉折低點
            recent_pivots_iloc = sorted(recent_pivots_iloc, key=lambda x: prices[x])[:recent_lows_cnt]
            
            # 本波找出的低點 "全數" 都必須滿足背離條件
            for p_iloc in recent_pivots_iloc:
                if not check_divergence_condition(p_iloc):
                    return False
                    
        # ==========================
        # 3. 前波低點比對
        # ==========================
        if older_lows_cnt > 0:
            older_pivots_iloc = get_valid_pivots_iloc(older_start, older_end)
            if not older_pivots_iloc:
                return False
                
            # 取前波價格最低的前 M 個轉折低點
            older_pivots_iloc = sorted(older_pivots_iloc, key=lambda x: prices[x])[:older_lows_cnt]
            
            # 前波找出的低點 "全數" 都必須滿足背離條件
            for p_iloc in older_pivots_iloc:
                if not check_divergence_condition(p_iloc):
                    return False
                    
        # 全部條件皆通過，判定底背離強烈成立
        return True

# ==========================================
# 資料抓取與共用函式
# ==========================================
@st.cache_data(ttl=86400)
def get_all_tw_stocks():
    """抓取台股上市/上櫃全部普通股代號與名稱"""
    stocks = {}
    for mode in ['2', '4']:
        suffix = '.TW' if mode == '2' else '.TWO'
        try:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) > 0:
                    text = tds[0].text.strip()
                    if '\u3000' in text:
                        code, name = text.split('\u3000')
                        if code.isdigit() and len(code) == 4:
                            stocks[code + suffix] = name
        except:
            pass
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
# 介面主程式與 Session State 初始化
# ==========================================
st.set_page_config(page_title="台股 K線型態與位階深度解析系統", layout="wide")

# 載入設定檔到 Session State
if "config" not in st.session_state:
    st.session_state.config = load_config()

if "current_profile" not in st.session_state:
    st.session_state.current_profile = st.session_state.config.get("last_used", "預設參數 (Default)")

def apply_profile_to_state(profile_name):
    """將指定的參數檔內容套用到 st.session_state，並記錄為最後使用版本"""
    prof = st.session_state.config["profiles"].get(profile_name, DEFAULT_PARAMS)
    for k, v in prof.items():
        st.session_state[k] = v
    # 更新當前設定與記錄
    st.session_state.current_profile = profile_name
    st.session_state.config["last_used"] = profile_name
    save_config(st.session_state.config)

# 系統首次執行時，將當前設定檔寫入 Widget 對應的 Session State
if "lookback_end" not in st.session_state:
    apply_profile_to_state(st.session_state.current_profile)

st.title("📈 台股 K線型態與位階深度解析系統")

# 隱藏右上角工具列（包含 GitHub 連結）
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📊 單檔深度解析", "🚀 全市場掃描 (低檔反轉+背離)"])

# ----------------------------------------------------
# 頁籤 1：單檔深度解析
# ----------------------------------------------------
with tab1:
    st.write("請在下方輸入股票代號（例如：`2495`、`00631L`），系統將自動抓取近兩年資料進行診斷。")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("輸入股票代號", value="2495", placeholder="例如：2330").strip()
    with col2:
        st.write("") 
        st.write("")
        submit_btn = st.button("開始分析", type="primary")

    if submit_btn and user_input:
        with st.spinner(f"⏳ 正在抓取 [{user_input}] 資料並進行解析中..."):
            df, real_ticker = get_stock_data(user_input)
            
            if df.empty:
                st.error(f"❌ 查無 [{user_input}] 的歷史數據，請確認代號是否正確。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                st.subheader(f"📊 【{stock_name} ({real_ticker})】 K線型態解析")
                
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
                df['Candle_Score_EMA'] = df['Candle_Score'].ewm(span=3).mean()
                
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
                
                with st.container(height=165):
                    if last_c > last_ma20:
                        st.markdown(f"- 📈 **趨勢方向**：股價位於月線 ({last_ma20:.2f}) 之上，短期波段偏**多頭**。")
                    else:
                        st.markdown(f"- 📉 **趨勢方向**：股價位於月線 ({last_ma20:.2f}) 之下，短期波段偏**空頭**或弱勢整理。")
                        
                    if last_c > last_vwma20:
                        st.markdown(f"- 💰 **籌碼狀況**：站穩加權均線 ({last_vwma20:.2f})，近期買盤有獲利，具**實質支撐**。")
                    else:
                        st.markdown(f"- ⚠️ **籌碼狀況**：低於加權均線 ({last_vwma20:.2f})，近期買盤套牢，上方有**解套賣壓**。")
                        
                    st.markdown(f"- 🛡️ **主力防線**：近兩個月最大量防守價為 **{last_defense:.2f}**，此價位不破皆可偏多看待。")
                    
                    st.markdown("---")
                    st.markdown("### ⚡ 近三個月極端訊號紀錄 (由近到遠)")
                    
                    signal_logs = []
                    for date, row in recent_df.iterrows():
                        m = row['Momentum_Force']
                        c = row['Close']
                        prev_c = row['Prev_Close']
                        ma5 = row['MA5']
                        prev_ma5 = row['Prev_MA5']
                        ma20 = row['MA20']
                        bb_upper = row['BB_Upper']
                        vwma20 = row['VWMA20']
                        prev_vwma20 = row['Prev_VWMA20']
                        defense = row['Max_Vol_Defense']
                        prev_defense = row['Prev_Defense']
                        bias20 = row['BIAS20']
                        ub = row['Upper_Bound']
                        lb = row['Lower_Bound']
                        vol = row['Volume_Lots']
                        vol_ma = row['Vol_MA20']
                        cps = row['Candle_Score']
                        
                        date_str = date.strftime('%Y-%m-%d')
                        is_bull_surge = (m > ub) or (row['Pct_Change'] >= 4.0 and vol >= vol_ma * 1.5)
                        is_bear_surge = (m < lb) or (row['Pct_Change'] <= -4.0 and vol >= vol_ma * 1.5)
                        
                        if c < defense and prev_c >= prev_defense:
                            signal_logs.append(f"- ☠️ **{date_str}** | 🚨 跌破最大量防守價 **{defense:.2f}** (最後防線潰堤) | 收盤: {c:.2f}")
                        elif cps <= -5.0:
                            signal_logs.append(f"- ⛈️ **{date_str}** | 🚨 高檔變盤型態 (長上影/長黑) | 負權重: {cps:.1f}")
                        elif cps >= 5.0:
                            signal_logs.append(f"- ☀️ **{date_str}** | 🚀 低檔強力反轉 (長下影/長紅) | 正權重: {cps:.1f}")
                        elif not is_bull_surge and not is_bear_surge:
                            if c < vwma20 and prev_c >= prev_vwma20:
                                signal_logs.append(f"- 📉 **{date_str}** | ⚠️ 跌破 VWMA 加權均線 (建議大部位減碼) | 收盤: {c:.2f}")
                            elif c < ma5 and prev_c >= prev_ma5 and c > ma20:
                                signal_logs.append(f"- 💰 **{date_str}** | ⚡ 高檔跌破 5日線 (短線獲利提早落袋) | 收盤: {c:.2f}")
                        elif is_bull_surge:
                            if bias20 > 12.0 or c >= bb_upper * 0.98:
                                signal_logs.append(f"- ✋ **{date_str}** | ⚠️ 高檔過熱 (正乖離 {bias20:.1f}% 或觸布林頂) | 收盤: {c:.2f}")
                            elif c >= vwma20:
                                signal_logs.append(f"- ✅ **{date_str}** | 📈 帶量站穩加權均線 (買點浮現) | 收盤: {c:.2f}")
                        elif is_bear_surge:
                            signal_logs.append(f"- 🔴 **{date_str}** | ⚠️ 爆量長黑 (動能極弱，大戶倒貨) | 收盤: {c:.2f}")

                    signal_logs.reverse()
                    if not signal_logs:
                        st.markdown("> 💡 近期走勢溫和，並未觸發極端爆量、變盤型態或跌破防線等特殊訊號。")
                    else:
                        for s in signal_logs:
                            st.markdown(s)

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.8, 1.2, 1.2]})
                
                ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='#1f77b4', linewidth=1.8)
                ax1.plot(plot_df.index, plot_df['MA5'], label='MA5', color='purple', linestyle=':', alpha=0.8, linewidth=1.5)
                ax1.plot(plot_df.index, plot_df['MA20'], label='MA20', color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.plot(plot_df.index, plot_df['VWMA20'], label='VWMA20', color='blue', linestyle='-.', linewidth=1.5, alpha=0.8)
                ax1.plot(plot_df.index, plot_df['Max_Vol_Defense'], label='Defense Line', color='teal', linestyle='-', linewidth=2.0)
                
                ax1.set_title(f'{stock_name} ({real_ticker}) Price & Defense System', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Price (TWD)', fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.5)
                ax1.legend(loc='upper left', framealpha=0.9)
                
                ax2.plot(plot_df.index, plot_df['Momentum_Force'], label='Momentum (M)', color='#7f7f7f', linewidth=1.2)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Zero Axis')
                ax2.plot(plot_df.index, plot_df['Upper_Bound'], color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='+1.5 Std')
                ax2.plot(plot_df.index, plot_df['Lower_Bound'], color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='-1.5 Std')
                
                ax2.set_ylabel('Momentum', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.5)
                ax2.legend(loc='upper left', framealpha=0.9)
                
                cps_colors = ['#d62728' if val > 0 else '#2ca02c' for val in plot_df['Candle_Score']]
                ax3.bar(plot_df.index, plot_df['Candle_Score'], color=cps_colors, alpha=0.7, label='Candle Score')
                ax3.plot(plot_df.index, plot_df['Candle_Score_EMA'], color='black', linewidth=1.5, linestyle='--', label='Score EMA')
                ax3.axhline(0, color='black', linestyle='-', linewidth=1.0)
                ax3.axhline(5, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='+5 Bullish')
                ax3.axhline(-5, color='green', linestyle=':', alpha=0.6, linewidth=1.5, label='-5 Bearish')
                
                ax3.set_ylabel('Score', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.5)
                ax3.legend(loc='upper left', framealpha=0.9)
                
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)

    with st.expander("📖 點我看【系統決策圖示與防線意義說明】"):
        st.markdown("""
        **🔹 買方極端訊號：**
        - ☀️ **低檔強力反轉** : 底部型態確認 (長下影/長紅)，為極佳抄底買點。
        - 🚀 / ✅ **帶量突破** : 動能爆發且站穩加權均線，為順勢波段起漲點。
        - 🟢 **低檔爆量強彈** : 跌深後首度出現大買盤，可嘗試小部位卡位。
        
        **🔸 賣方與避險訊號：**
        - ☠️ **跌破防守底線** : 股價摔穿「近60日最大量防守價」，主力全面棄守，無條件清倉！
        - ⛈️ **高檔變盤型態** : 出現高檔避雷針或烏雲罩頂，主力出貨警訊，嚴禁追高。
        - 📉 **跌破加權均線** : 波段趨勢由強轉弱，建議大部位減碼。
        - 💰 **跌破 5日線** : 短線動能衰竭，建議短線獲利提早落袋一半。
        - ✋ **高檔過熱** : 股價正乖離過高或觸碰布林頂，容易拉回，嚴禁追高。
        - 🩸 / 🔴 **爆量長黑** : 動能呈現極端負值，為大戶倒貨特徵，必須果斷停損/避險。
        """)

# ----------------------------------------------------
# 頁籤 2：全市場掃描 (包含嚴格轉折低點比對與參數管理)
# ----------------------------------------------------
with tab2:
    st.write("系統將自動抓取全部普通股資料，尋找指定區間內符合「低檔強力反轉」的標的，並進一步檢測多組技術底背離。")
    
    with st.expander("⚙️ 掃描與背離參數設定", expanded=True):
        st.markdown("**📂 參數設定檔管理**")
        profile_names = list(st.session_state.config["profiles"].keys())
        idx = profile_names.index(st.session_state.current_profile) if st.session_state.current_profile in profile_names else 0
        
        col_p1, col_p2, col_p3, col_p4 = st.columns([3, 3, 2, 2])
        with col_p1:
            selected_profile = st.selectbox(
                "選擇歷史設定檔", 
                profile_names, 
                index=idx
            )
            
            # 【核心修復】：利用同步狀態檢查，一旦發現下拉選單變更，立刻覆蓋 Session 並強制重繪
            if selected_profile != st.session_state.current_profile:
                apply_profile_to_state(selected_profile)
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                
        with col_p2:
            new_profile_name = st.text_input("儲存新名稱", placeholder="輸入自訂設定檔名稱...")
        with col_p3:
            st.write("")
            st.write("")
            if st.button("💾 儲存設定", use_container_width=True):
                name_to_save = new_profile_name.strip() or selected_profile
                # 收集當下最新的所有參數，準備寫入設定檔
                current_vals = {
                    "lookback_end": st.session_state.lookback_end,
                    "lookback_start": st.session_state.lookback_start,
                    "min_score": st.session_state.min_score,
                    "min_vol_ma20": st.session_state.min_vol_ma20,
                    "use_single_div": st.session_state.use_single_div,
                    "div_recent_w": st.session_state.div_recent_w,
                    "div_older_w": st.session_state.div_older_w,
                    "pivot_left": st.session_state.pivot_left,
                    "pivot_right": st.session_state.pivot_right,
                    "recent_lows_cnt": st.session_state.recent_lows_cnt,
                    "older_lows_cnt": st.session_state.older_lows_cnt
                }
                st.session_state.config["profiles"][name_to_save] = current_vals
                st.session_state.config["last_used"] = name_to_save
                st.session_state.current_profile = name_to_save
                save_config(st.session_state.config)
                st.success(f"已成功儲存版本：'{name_to_save}'")
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                
        with col_p4:
            st.write("")
            st.write("")
            if st.button("🗑️ 刪除", use_container_width=True):
                if selected_profile == "預設參數 (Default)":
                    st.error("系統預設參數不可刪除！")
                else:
                    del st.session_state.config["profiles"][selected_profile]
                    apply_profile_to_state("預設參數 (Default)")
                    st.success(f"已刪除版本：'{selected_profile}'")
                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                    
        st.markdown("---")

        st.markdown("**1. 基礎掃描參數**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lookback_end = st.number_input(
                "掃描區間：從幾天前起算？ (最新為 0)", 
                min_value=0, max_value=1000, step=1,
                key="lookback_end",
                help="例如設定 0 表示從最新交易日開始往回掃描。"
            )
            lookback_start = st.number_input(
                "掃描區間：最多回推至幾天前？", 
                min_value=0, max_value=1000, step=1,
                key="lookback_start",
                help="若此數值大於起算天數，系統會掃描該段期間(多日)內任何曾觸發反轉訊號的股票。"
            )
            # 防呆機制：因綁定 Session State，改為判斷變數
            if lookback_start < lookback_end:
                st.warning("⚠️ 「最多回推天數」不能小於「起算天數」，已自動為您修正。")
                lookback_start = lookback_end
                
        with col_b:
            min_score = st.number_input(
                "最低反轉權重分數", 
                min_value=1.0, max_value=30.0, step=1.0,
                key="min_score",
                help="【已最佳化】預設 8.0，過濾弱勢小紅K，確保底部成型力道。"
            )
        with col_c:
            min_vol_ma20 = st.number_input(
                "月均量最低門檻 (張)", 
                min_value=0, max_value=100000, step=100,
                key="min_vol_ma20",
                help="【已最佳化】預設 1000 張，避開流動性不佳的殭屍股。"
            )
        
        st.markdown("**2. 背離檢測週期設定**")
        use_single_div = st.checkbox("啟用單一組自訂背離週期 (未勾選則預設比對三組：(5,20)、(5,60)、(20,60))", key="use_single_div")
        col_d, col_e = st.columns(2)
        with col_d:
            div_recent_w = st.number_input(
                "自訂：第一低點(近波)範圍", 
                min_value=5, max_value=60, step=1,
                key="div_recent_w",
                disabled=not use_single_div
            )
        with col_e:
            div_older_w = st.number_input(
                "自訂：第二低點(前波)範圍", 
                min_value=10, max_value=240, step=1,
                key="div_older_w",
                disabled=not use_single_div
            )

        st.markdown("**3. 嚴格轉折點與比對數量設定 (四者皆設為 0 時將自動關閉嚴格邏輯，退回基礎比對)**")
        col_f, col_g, col_h, col_i = st.columns(4)
        with col_f:
            pivot_left = st.number_input(
                "轉折判定：往前抓K棒數", 
                min_value=0, max_value=20, step=1,
                key="pivot_left",
                help="低點定義：價格必須是(往前+往後)這段區間內的最低價。皆設為 0 時關閉多點嚴格邏輯。"
            )
        with col_g:
            pivot_right = st.number_input(
                "轉折判定：往後取K棒數", 
                min_value=0, max_value=20, step=1,
                key="pivot_right",
                help="遇到最新資料往後數量不足時，會自動抓到最新一筆為止。皆設為 0 時關閉多點嚴格邏輯。"
            )
        with col_h:
            recent_lows_cnt = st.number_input(
                "本波(近波)比對低點數", 
                min_value=0, max_value=20, step=1,
                key="recent_lows_cnt",
                help="排除絕對最低點後，找出近波前幾低的轉折點，全都必須滿足底背離。皆設為 0 時關閉此邏輯。"
            )
        with col_i:
            older_lows_cnt = st.number_input(
                "前波比對低點數 (X)", 
                min_value=0, max_value=20, step=1,
                key="older_lows_cnt",
                help="找出前波前幾低的轉折點，也都必須全部滿足底背離才成立。皆設為 0 時關閉此邏輯。"
            )

    st.markdown("---")
    
    if st.button("🚀 開始智慧區間掃描", type="primary"):
        # 根據勾選狀態，決定檢測一組或三組背離參數
        if use_single_div:
            div_pairs = [(div_recent_w, div_older_w)]
            st.info(f"💡 系統將使用您自訂的週期參數 `({div_recent_w}, {div_older_w})` 進行背離運算。")
        else:
            div_pairs = [(5, 20), (5, 60), (20, 60)]
            st.info(f"💡 系統將自動同時比對三組週期參數進行背離運算。")
            
        max_older_w = max(pair[1] for pair in div_pairs)
        
        # 動態計算需要的歷史資料天數 (最多回推天數 + 指標與背離所需的緩衝天數 + 轉折判定緩衝)
        buffer_days = max_older_w + pivot_left + 30 
        total_needed_days = lookback_start + buffer_days
        
        # 根據所需天數，決定 yfinance 下載的 period，避免抓取過多無用資料
        if total_needed_days <= 60:
            dl_period = "3mo"
        elif total_needed_days <= 120:
            dl_period = "6mo"
        elif total_needed_days <= 250:
            dl_period = "1y"
        elif total_needed_days <= 500:
            dl_period = "2y"
        elif total_needed_days <= 1250:
            dl_period = "5y"
        else:
            dl_period = "10y"
            
        # 針對 60分K 動態優化 (最高限制為 730d)
        if total_needed_days <= 60:
            dl_period_60m = "3mo"
        elif total_needed_days <= 120:
            dl_period_60m = "6mo"
        else:
            dl_period_60m = "730d"

        stock_dict = get_all_tw_stocks()
        if not stock_dict:
            st.error("❌ 無法取得台股清單，請檢查網路連線。")
        else:
            tickers = list(stock_dict.keys())
            reversal_candidates = {} 
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ----------------------------------------------------
            # 第一階段：區間粗篩
            # ----------------------------------------------------
            chunk_size = 100
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                status_text.text(f"[階段一] 正在全市場區間掃描：進度 {i} / {len(tickers)} 檔...")
                try:
                    data = yf.download(chunk, period=dl_period, threads=True, progress=False)
                    for ticker in chunk:
                        try:
                            if len(chunk) == 1:
                                df = data.copy()
                            else:
                                df = pd.DataFrame({
                                    'Open': data['Open'][ticker],
                                    'High': data['High'][ticker],
                                    'Low': data['Low'][ticker],
                                    'Close': data['Close'][ticker],
                                    'Volume': data['Volume'][ticker]
                                }).dropna()
                            
                            # 確保資料長度足夠涵蓋掃描區間與緩衝天數
                            if len(df) <= lookback_start + 20: 
                                continue
                                
                            df['Pct_Change'] = df['Close'].pct_change() * 100
                            df['Volume_Lots'] = df['Volume'] / 1000
                            df['MA20'] = df['Close'].rolling(window=20).mean()
                            df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
                            df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
                            
                            df['Body'] = abs(df['Close'] - df['Open'])
                            df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
                            df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
                            df['Total_Range'] = df['High'] - df['Low']
                            df['Total_Range'] = df['Total_Range'].replace(0, 0.001)
                            
                            df['Vol_Mult'] = (df['Volume_Lots'] / df['Vol_MA20']).clip(0.5, 3.0)
                            
                            cond_low_pin = (df['BIAS20'] <= 0) & (df['Lower_Shadow'] > df['Body'] * 1.5) & (df['Lower_Shadow'] > df['Total_Range'] * 0.4)
                            cond_low_red = (df['BIAS20'] <= 0) & (df['Close'] > df['Open']) & (df['Pct_Change'] >= 2.5)
                            
                            df['Candle_Score'] = 0.0
                            df.loc[cond_low_pin, 'Candle_Score'] = 7 * (df['Lower_Shadow'] / df['Total_Range']) * df['Vol_Mult']
                            df.loc[cond_low_red, 'Candle_Score'] = 5 * df['Vol_Mult']
                            
                            best_score_in_range = -1
                            best_target_row = None
                            offset_for_best_row = 0
                            
                            for offset in range(lookback_end, lookback_start + 1):
                                target_idx = -1 - offset
                                target_row = df.iloc[target_idx]
                                current_score = target_row['Candle_Score']
                                current_vol_ma20 = target_row['Vol_MA20']
                                
                                if current_score >= min_score and current_vol_ma20 >= min_vol_ma20:
                                    if current_score > best_score_in_range:
                                        best_score_in_range = current_score
                                        best_target_row = target_row
                                        offset_for_best_row = offset
                            
                            if best_target_row is not None:
                                clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
                                reversal_candidates[ticker] = {
                                    "_Full_Ticker": ticker,
                                    "_Offset": offset_for_best_row, 
                                    "股票代號": clean_ticker,
                                    "股票名稱": stock_dict[ticker],
                                    "觸發日期": best_target_row.name.strftime('%Y-%m-%d'),
                                    "當日收盤": round(float(best_target_row['Close']), 2),
                                    "月均量(張)": int(best_target_row['Vol_MA20']),
                                    "反轉權重分數": round(float(best_score_in_range), 2)
                                }
                        except Exception:
                            continue
                except Exception:
                    pass
                progress_bar.progress(min(1.0, (i + chunk_size) / len(tickers)))
            
            reversal_list = list(reversal_candidates.values())
            
            # ----------------------------------------------------
            # 第二階段：針對入選標的進行 日K/60分K 背離深度檢測
            # ----------------------------------------------------
            if reversal_list:
                progress_bar.progress(0)
                status_text.text(f"[階段二] 正在分析 {len(reversal_list)} 檔入選標的之多級別背離特徵...")
                
                final_results = []
                for idx, item in enumerate(reversal_list):
                    try:
                        ticker = item.pop("_Full_Ticker")
                        specific_offset = item.pop("_Offset") 
                        
                        has_daily_div = False
                        has_m60_div = False
                        
                        # ================= 1. 日K背離判定 =================
                        daily_df = yf.Ticker(ticker).history(period=dl_period)
                        if not daily_df.empty:
                            if specific_offset > 0 and len(daily_df) > specific_offset:
                                daily_df = daily_df.iloc[:-specific_offset]
                                
                            daily_df = TechnicalIndicators.add_kd(daily_df)
                            daily_df = TechnicalIndicators.add_macd(daily_df)
                            
                            for r_w, o_w in div_pairs:
                                d_kd = DivergenceStrategy.check_bottom_divergence(
                                    daily_df, 'Low', 'K', 'D', 
                                    r_w, o_w, recent_lows_cnt, older_lows_cnt, pivot_left, pivot_right
                                )
                                d_macd = DivergenceStrategy.check_bottom_divergence(
                                    daily_df, 'Low', 'MACD', 'MACD_Signal', 
                                    r_w, o_w, recent_lows_cnt, older_lows_cnt, pivot_left, pivot_right
                                )
                                
                                res = []
                                if d_kd: res.append("KD")
                                if d_macd: res.append("MACD")
                                
                                if res:
                                    item[f"日K背離({r_w},{o_w})"] = "+".join(res)
                                    has_daily_div = True
                                else:
                                    item[f"日K背離({r_w},{o_w})"] = "無"
                        else:
                            for r_w, o_w in div_pairs:
                                item[f"日K背離({r_w},{o_w})"] = "無資料"
                        
                        # ================= 2. 60分K背離判定 =================
                        m60_df = yf.Ticker(ticker).history(period=dl_period_60m, interval="60m")
                        if specific_offset > 0 and not daily_df.empty:
                            target_date = daily_df.index[-1].date()
                            mask = [d.date() <= target_date for d in m60_df.index]
                            m60_df = m60_df[mask]
                            
                        if not m60_df.empty:
                            m60_df = TechnicalIndicators.add_kd(m60_df)
                            m60_df = TechnicalIndicators.add_macd(m60_df)
                            
                            for r_w, o_w in div_pairs:
                                m_kd = DivergenceStrategy.check_bottom_divergence(
                                    m60_df, 'Low', 'K', 'D', 
                                    r_w, o_w, recent_lows_cnt, older_lows_cnt, pivot_left, pivot_right
                                )
                                m_macd = DivergenceStrategy.check_bottom_divergence(
                                    m60_df, 'Low', 'MACD', 'MACD_Signal', 
                                    r_w, o_w, recent_lows_cnt, older_lows_cnt, pivot_left, pivot_right
                                )
                                
                                res = []
                                if m_kd: res.append("KD")
                                if m_macd: res.append("MACD")
                                
                                if res:
                                    item[f"60分K背離({r_w},{o_w})"] = "+".join(res)
                                    has_m60_div = True
                                else:
                                    item[f"60分K背離({r_w},{o_w})"] = "無"
                        else:
                            for r_w, o_w in div_pairs:
                                item[f"60分K背離({r_w},{o_w})"] = "無資料"

                        # ================= 3. 綜合背離分類建議 =================
                        if has_daily_div and has_m60_div:
                            item["背離分類建議"] = "⭐⭐⭐ 雙級別多週期共振 (強烈建議)"
                        elif has_daily_div:
                            item["背離分類建議"] = "⭐⭐ 波段佈局 (日K級別共振)"
                        elif has_m60_div:
                            item["背離分類建議"] = "⭐ 短線進場 (60分K級別共振)"
                        else:
                            item["背離分類建議"] = "一般反轉 (無背離)"
                        
                        final_results.append(item)
                    except Exception as e:
                        for r_w, o_w in div_pairs:
                            item[f"日K背離({r_w},{o_w})"] = "資料異常"
                            item[f"60分K背離({r_w},{o_w})"] = "資料異常"
                        item["背離分類建議"] = "一般反轉 (無背離)"
                        final_results.append(item)
                        
                    progress_bar.progress(min(1.0, (idx + 1) / len(reversal_list)))

                # 顯示最終結果
                status_text.empty()
                progress_bar.empty()
                st.success(f"🎉 區間智慧掃描完成！本次共精選出 **{len(final_results)}** 檔標的。")
                
                res_df = pd.DataFrame(final_results)
                res_df = res_df.sort_values(by="反轉權重分數", ascending=False).reset_index(drop=True)
                res_df.index = res_df.index + 1
                
                st.dataframe(res_df, use_container_width=True)
            else:
                status_text.empty()
                progress_bar.empty()
                st.info("掃描完成！在您指定的區間與條件下，市場無任何符合「低檔強力反轉」訊號的標的。")
