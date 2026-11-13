import os
import io
import json
import warnings
import datetime
import time
import re
import importlib 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import strategy_core as core

importlib.reload(core)
warnings.filterwarnings("ignore")

OUTPUT_COLUMN_CONFIG = {
    "show_ma120_240_bear_align": True,
    "show_ma240_slope": True,
    "show_ma120_slope": True
}
PARAMS_FILE = "params_config.json"

def load_config():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if "profiles" not in config: config["profiles"] = {}
                config["profiles"]["預設參數 (Default)"] = core.DEFAULT_CONFIG.copy()
                return config
        except Exception: pass
    return {"last_used": "預設參數 (Default)", "profiles": {"預設參數 (Default)": core.DEFAULT_CONFIG.copy()}}

def save_config(config):
    try:
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception: pass

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tw_stocks(cache_version="v5"):
    stocks = core.fetch_all_tw_stocks()
    if not stocks:
        get_all_tw_stocks.clear()
    return stocks

@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_stock_name(ticker):
    return core.fetch_tw_stock_name(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol):
    return core.fetch_stock_data(symbol)

st.set_page_config(page_title="台股 K線型態與位階深度解析系統", layout="wide")

if "audit_mode" not in st.session_state: st.session_state.audit_mode = "單檔查詢"
if "batch_input_area" not in st.session_state: st.session_state.batch_input_area = "2330, 2454, 3231, 6147, 1909"
if "run_batch_audit" not in st.session_state: st.session_state.run_batch_audit = False
if "enable_60m_adv_scan" not in st.session_state: st.session_state.enable_60m_adv_scan = False
if "config" not in st.session_state: st.session_state.config = load_config()
if "current_profile" not in st.session_state: st.session_state.current_profile = st.session_state.config.get("last_used", "預設參數 (Default)")

def apply_profile_to_state(profile_name):
    prof = st.session_state.config["profiles"].get(profile_name, core.DEFAULT_CONFIG)
    full_prof = core.DEFAULT_CONFIG.copy()
    full_prof.update(prof)
    for k, v in full_prof.items(): st.session_state[k] = v
    st.session_state.backtest_date_obj = pd.to_datetime(full_prof["backtest_date"]).date()
    st.session_state.current_profile = profile_name
    st.session_state.config["last_used"] = profile_name
    save_config(st.session_state.config)

missing_keys = [k for k in core.DEFAULT_CONFIG.keys() if k not in st.session_state]
if missing_keys or "backtest_date_obj" not in st.session_state:
    apply_profile_to_state(st.session_state.current_profile)

st.title("📈 台股 K線型態與位階深度解析系統")
st.markdown("<style>header {visibility: hidden;}</style>", unsafe_allow_html=True)
st.markdown("""
    <style>
    div.row-widget.stRadio > div{flex-direction:row;flex-wrap:wrap;}
    div.row-widget.stRadio > div > label{background-color: #f0f2f6; padding: 10px 15px; border-radius: 8px; margin-right: 10px; margin-bottom: 10px; cursor: pointer;}
    </style>
""", unsafe_allow_html=True)

page_options = [
    "📊 單檔深度解析", 
    "🚀 全市場智慧掃描", 
    "📑 財報三表地雷掃描", 
    "📐 幾何型態滿足點測距", 
    "🤖 量化型態狀態機 (程式A)"
]
selected_page = st.radio("📌 快速切換功能分頁：", page_options, horizontal=True)

# ----------------------------------------------------
# 頁籤 1：單檔深度解析
# ----------------------------------------------------
if selected_page == "📊 單檔深度解析":
    st.write("請在下方輸入股票代號（例如：`2495`、`00631L`），系統將自動抓取近兩年資料進行診斷。")
    col1, col2 = st.columns([4, 1])
    with col1: user_input = st.text_input("輸入股票代號", value="2495", placeholder="例如：2330").strip()
    with col2: st.write(""); st.write(""); submit_btn = st.button("開始分析", type="primary")

    if submit_btn and user_input:
        with st.spinner(f"⏳ 正在抓取 [{user_input}] 資料並進行解析中..."):
            df, real_ticker = get_stock_data(user_input)
            if df.empty:
                st.error(f"❌ 查無 [{user_input}] 的歷史數據，請確認代號是否正確。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                df = core.TechnicalIndicators.add_kd(df)
                df = core.TechnicalIndicators.add_macd(df)
                df = core.TechnicalIndicators.add_rsi(df)

                df['Pct_Change'] = df['Close'].pct_change() * 100
                df['Volume_Lots'] = df['Volume'] / 1000
                df['Momentum_Force'] = df['Pct_Change'] * df['Volume_Lots']
                df['Prev_Close'] = df['Close'].shift(1)

                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['Prev_MA5'] = df['MA5'].shift(1)
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8) * 100
                df['Std20'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + 2 * df['Std20']
                df['BB_Lower'] = df['MA20'] - 2 * df['Std20']

                df['VWMA20'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
                df['Prev_VWMA20'] = df['VWMA20'].shift(1)
                df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()

                vols_arr = df['Volume'].values
                opens_arr = df['Open'].values
                closes_arr = df['Close'].values
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

                config_dict = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys() if k in st.session_state}
                df = core.ScanEngine.calculate_all_phase1_scores(df, config_dict)

                df = df.dropna(subset=['Momentum_Force', 'Max_Vol_Defense', 'VWMA20', 'Prev_MA5', 'Vol_MA20', 'MA60']).copy()
                plot_df, recent_df = df.tail(240).copy(), df.tail(60)
                last_row = recent_df.iloc[-1]
                last_date = recent_df.index[-1].strftime('%Y-%m-%d')

                st.subheader(f"📊 【{stock_name} ({real_ticker})】 深度解析與策略判定")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    rev_score = round(last_row.get('Candle_Score', 0), 2)
                    rev_status = "✅ 達標入選" if rev_score >= st.session_state.min_score else "❌ 未達標"
                    st.info(f"**底部翻轉分數：{rev_score}** ({rev_status})\n\n*(門檻：{st.session_state.min_score} 分)*")
                with col_b:
                    vcp_score = round(last_row.get('VCP_Score', 0), 2)
                    vcp_status = "✅ 達標入選" if vcp_score >= st.session_state.min_vcp_score else "❌ 未達標"
                    st.success(f"**VCP 收斂分數：{vcp_score}** ({vcp_status})\n\n*(門檻：{st.session_state.min_vcp_score} 分)*")
                with col_c:
                    reso_score = round(last_row.get('Reso_Score', 0), 2)
                    reso_status = "✅ 達標入選" if reso_score >= st.session_state.min_reso_score else "❌ 未達標"
                    st.warning(f"**指標共振分數：{reso_score}** ({reso_status})\n\n*(門檻：{st.session_state.min_reso_score} 分)*")
                st.markdown("---")

                expert_res_df = core.ExpertTradingStrategy.evaluate(df)
                last_expert = expert_res_df.iloc[-1]
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    if last_expert['RSI盤整']: st.error("⚠️ **震盪盤整中**\n\n(RSI 45~55糾結，建議停用指標改看區間)")
                    else: st.success("✅ **具備趨勢動能**\n\n(脫離盤整區，指標參考價值高)")
                with col_e2:
                    st.info(f"**多週期 KD 狀態：**\n\n{core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])}")
                with col_e3:
                    st.warning(f"**三指標共振持股建議：**\n\n{core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數'])} (同步偏多數: {last_expert['多空指標數']})")
                st.markdown("---")
                
                if 'Adv_Signal_A' in last_row and (last_row['Adv_Signal_A'] or last_row['Adv_Signal_B']):
                    st.success(f"🚨 **量化雙底正背離模組觸發**\n\n"
                               f"- 觸發狀態：{'✅ 底倉突破 (Signal A)' if last_row['Adv_Signal_A'] else '✅ 右肩回測加碼 (Signal B)'}\n"
                               f"- 左腳低點：{last_row['Adv_Low_1']:.2f} | 破底停損：{last_row['Adv_Low_2']:.2f}\n"
                               f"- 上影線頸線：{last_row['Adv_Neckline_1']:.2f} | 右肩回測基準：{last_row.get('Adv_Support_RS', 0):.2f}\n"
                               f"- 預估 1 段目標：{last_row['Adv_Target_1x']:.2f} | 預估 3 段目標：{last_row['Adv_Target_3x']:.2f}")
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

                cps_colors = ['#d62728' if val > 0 else '#2ca02c' for val in plot_df.get('Candle_Score', [0]*len(plot_df))]
                if 'Candle_Score' in plot_df:
                    ax3.bar(plot_df.index, plot_df['Candle_Score'], color=cps_colors, alpha=0.7, label='Rev Score')
                ax3.axhline(0, color='black', linestyle='-', linewidth=1.0)
                ax3.axhline(st.session_state.min_score, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Min Rev Threshold')
                ax3.set_ylabel('Algorithm Score', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.5); ax3.legend(loc='upper left')

                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

# ----------------------------------------------------
# 頁籤 2：全市場智慧掃描
# ----------------------------------------------------
elif selected_page == "🚀 全市場智慧掃描":
    st.write("系統將自動抓取全部普通股，尋找符合「低檔翻轉」、「VCP收斂」、「雙指標共振」或「回測走揚年線」等標的，並進行多級別背離與均線扣抵判定。")

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
                    current_vals = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys() if k in st.session_state}
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

        st.markdown("**1. 掃描模式與演算法選擇**")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
        with col_m1: scan_target_mode = st.radio("掃描對象", ['全市場掃描', '指定個股測試'], horizontal=True)
        with col_m2: test_ticker_input = st.text_input("輸入測試股票代號", value="6147") if scan_target_mode == '指定個股測試' else ""
        with col_m3: algo_mode = st.radio("請選擇欲執行的掃描演算法", core.SCAN_ALGO_CHOICES, horizontal=True)

        st.markdown("**2. 基礎掃描參數、時光機回測與雲端備份設定**")
        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns([1, 1.2, 1, 1.2, 1.4])
        with col_b1: st.checkbox("啟用指定日期回測", key="use_backtest_date")
        with col_b2: st.date_input("選擇回測基準日", key="backtest_date_obj", disabled=not st.session_state.use_backtest_date)
        with col_b3: st.checkbox("近 30日觸發統計", key="enable_30d_backtest")
        with col_b4: st.checkbox("自動上傳 Google Drive", key="auto_upload_gdrive")
        with col_b5: st.checkbox("啟用 60分K 雙底/右肩掃描 (耗時較長)", key="enable_60m_adv_scan")

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.number_input("掃描區間(迄)：起算天數", min_value=0, max_value=1000, step=1, key="lookback_end")
            st.number_input("掃描區間(起)：回推天數", min_value=0, max_value=1000, step=1, key="lookback_start")
        with col_b: st.number_input("底部翻轉最低分", min_value=1.0, max_value=30.0, step=1.0, key="min_score")
        with col_c: st.number_input("VCP收斂最低分", min_value=1.0, max_value=20.0, step=1.0, key="min_vcp_score")
        with col_d: st.number_input("指標共振最低分", min_value=1.0, max_value=15.0, step=1.0, key="min_reso_score")
        with col_e: st.number_input("月均量最低門檻", min_value=0, max_value=100000, step=100, key="min_vol_ma20")

        st.markdown("**3. 波段檢測週期與背離條件設定**")
        st.checkbox("啟用單一組自訂背離週期 (未勾則預設比對三組：(5,20)、(5,60)、(20,60))", key="use_single_div")
        col_f, col_g, col_h, col_i, col_j, col_k = st.columns(6)
        with col_f: st.number_input("近波範圍", min_value=5, max_value=60, step=1, key="div_recent_w", disabled=not st.session_state.use_single_div)
        with col_g: st.number_input("前波範圍", min_value=10, max_value=240, step=1, key="div_older_w", disabled=not st.session_state.use_single_div)
        with col_h: st.number_input("左X根不破", min_value=0, max_value=20, step=1, key="pivot_left")
        with col_i: st.number_input("右Y根不破", min_value=0, max_value=20, step=1, key="pivot_right")
        with col_j: st.number_input("近波低點數", min_value=0, max_value=20, step=1, key="recent_lows_cnt")
        with col_k: st.number_input("前波低點數", min_value=0, max_value=20, step=1, key="older_lows_cnt")

        st.markdown("**4. 扣抵與進階濾網**")
        st.checkbox("嚴格要求金叉確認背離與共振", key="require_cross_confirm")
        st.checkbox("啟用進場位置建議 (K值過高提示)", key="use_cross_position_filter")
        st.checkbox("啟用週K線共振與底背離判定", key="use_weekly_filter")

    if st.button("🚀 開始智慧區間掃描", type="primary"):
        status_text, progress_bar = st.empty(), st.progress(0)
        stock_dict = get_all_tw_stocks()
        
        if not stock_dict:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ 無法取得台股清單。 (請確認伺服器網路狀態或稍後再試)")
            st.stop()

        tickers = [core.resolve_ticker(test_ticker_input, stock_dict)] if scan_target_mode == "指定個股測試" else list(stock_dict.keys())
        config_dict = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys() if k in st.session_state}
        config_dict["backtest_date_obj"] = st.session_state.backtest_date_obj
        config_dict["enable_60m_adv_scan"] = st.session_state.enable_60m_adv_scan

        div_pairs = [(config_dict["div_recent_w"], config_dict["div_older_w"])] if config_dict["use_single_div"] else [(5, 20), (5, 60), (20, 60)]

        dl_kwargs = {"period": "2y"}
        if config_dict["use_backtest_date"]:
            end_dt = config_dict["backtest_date_obj"] + datetime.timedelta(days=1)
            dl_kwargs = {"start": (end_dt - datetime.timedelta(days=730)).strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}

        yf_session = core.get_yf_session()
        market_info = core.MarketRegimeFilter.evaluate(yf_session, config_dict["backtest_date_obj"] if config_dict["use_backtest_date"] else None)
        reversal_candidates, chunk_size = {}, 40

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            status_text.text(f"正在全市場掃描：進度 {min(i+chunk_size, len(tickers))} / {len(tickers)} 檔...")
            try:
                data = yf.download(chunk, threads=False, progress=False, session=yf_session, **dl_kwargs)
                
                data_60m = None
                if config_dict.get("enable_60m_adv_scan", False):
                    data_60m = yf.download(chunk, period="1mo", interval="60m", threads=False, progress=False, session=yf_session)

                for ticker in chunk:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            if ticker in data.columns.get_level_values(0): df = data.xs(ticker, axis=1, level=0).dropna(how='all')
                            elif ticker in data.columns.get_level_values(1): df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else: df = pd.DataFrame()
                        else: df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                        if config_dict["use_backtest_date"]:
                            df = df[df.index.tz_localize(None).normalize() <= pd.to_datetime(config_dict["backtest_date_obj"])]
                        if df.empty or len(df) <= config_dict["lookback_start"] + 20: continue

                        df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
                        df = core.TechnicalIndicators.add_rsi(df)
                        expert_res_df = core.ExpertTradingStrategy.evaluate(df)

                        df['Pct_Change'] = df['Close'].pct_change() * 100
                        df['Volume_Lots'] = df['Volume'] / 1000
                        df['MA20'], df['MA60'] = df['Close'].rolling(20).mean(), df['Close'].rolling(60).mean()
                        df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()
                        std20 = df['Close'].rolling(20).std()
                        df['BB_Upper'], df['BB_Lower'] = df['MA20'] + 2 * std20, df['MA20'] - 2 * std20

                        df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
                        df['MACD_Cross'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
                        df['KD_Cross_Recent'] = df['KD_Cross'].rolling(window=config_dict["reso_cross_days"], min_periods=1).max() >= 1
                        df['MACD_Cross_Recent'] = df['MACD_Cross'].rolling(window=config_dict["reso_cross_days"], min_periods=1).max() >= 1

                        df = core.ScanEngine.calculate_all_phase1_scores(df, config_dict)
                        if scan_target_mode == "指定個股測試":
                            st.session_state.test_stock_df = pd.concat([df, expert_res_df], axis=1)
                            st.session_state.test_ticker_name = stock_dict.get(ticker, ticker)

                        best_combined_score, best_row, best_offset, best_vol_mult = -1, None, 0, 0.0
                        force_pass = (scan_target_mode == "指定個股測試")

                        for offset in range(config_dict["lookback_end"], config_dict["lookback_start"] + 1):
                            if len(df) <= offset + 1: continue
                            t_row = df.iloc[-1 - offset]
                            prev_vol = df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset+1 else np.nan
                            vol_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                            if t_row.get('Vol_MA20', 0) >= config_dict["min_vol_ma20"] or force_pass:
                                is_pass, combo = core.ScanEngine.check_algo_pass(t_row, config_dict, algo_mode, force_pass)
                                if is_pass and (combo > best_combined_score or best_row is None):
                                    best_combined_score, best_row, best_offset, best_vol_mult = combo, t_row, offset, vol_mult

                        pass_60m_adv = False
                        adv_60m_data = None
                        if config_dict.get("enable_60m_adv_scan", False) and data_60m is not None:
                            if isinstance(data_60m.columns, pd.MultiIndex):
                                if ticker in data_60m.columns.get_level_values(0): df_60m = data_60m.xs(ticker, axis=1, level=0).dropna(how='all')
                                elif ticker in data_60m.columns.get_level_values(1): df_60m = data_60m.xs(ticker, axis=1, level=1).dropna(how='all')
                                else: df_60m = pd.DataFrame()
                            else: df_60m = data_60m.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()
                            
                            if config_dict["use_backtest_date"] and not df_60m.empty:
                                target_date_pd = pd.to_datetime(config_dict["backtest_date_obj"])
                                df_60m = df_60m[df_60m.index.tz_localize(None).normalize() <= target_date_pd]

                            if not df_60m.empty and len(df_60m) >= 60:
                                df_60m = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df_60m))
                                df_60m['Volume_Lots'] = df_60m.get('Volume', pd.Series(0)) / 1000.0
                                df_60m['MA20'] = df_60m['Close'].rolling(20).mean()
                                df_60m['BB_Lower'] = df_60m['MA20'] - 2 * df_60m['Close'].rolling(20).std()
                                
                                adv_res_60m = core.AdvancedBottomStrategy.evaluate(df_60m)
                                if not adv_res_60m.empty:
                                    target_eval_date = df.index[-1 - best_offset].date() if (best_row is not None and best_offset >= 0) else df.index[-1].date()
                                    adv_res_60m_target_day = adv_res_60m[adv_res_60m.index.tz_localize(None).date == target_eval_date]
                                    
                                    if not adv_res_60m_target_day.empty:
                                        triggered_rows = adv_res_60m_target_day[(adv_res_60m_target_day['Adv_Signal_A'] == True) | (adv_res_60m_target_day['Adv_Signal_B'] == True)]
                                        if not triggered_rows.empty:
                                            pass_60m_adv = True
                                            adv_60m_data = triggered_rows.iloc[-1]

                        if (best_row is not None and best_combined_score >= 0) or pass_60m_adv or force_pass:
                            if best_row is None:
                                best_row = df.iloc[-1]
                                best_offset = 0
                                prev_v = df['Volume_Lots'].iloc[-2] if len(df) > 1 else np.nan
                                best_vol_mult = (best_row['Volume_Lots'] / prev_v) if pd.notna(prev_v) and prev_v > 0 else 0.0

                            count_30d = 0
                            if config_dict["enable_30d_backtest"]:
                                for b_offset in range(best_offset + 1, min(len(df)-1, best_offset + 31)):
                                    c_row = df.iloc[-1 - b_offset]
                                    if c_row.get('Vol_MA20', 0) >= config_dict["min_vol_ma20"] or force_pass:
                                        c_pass, _ = core.ScanEngine.check_algo_pass(c_row, config_dict, algo_mode, force_pass)
                                        if c_pass:
                                            count_30d += 1

                            last_expert = expert_res_df.iloc[-1 - best_offset]
                            rec_status = "✅ 建議參考" if float(best_row['K']) < config_dict.get("cross_position_threshold", 70) else "⚠️ 位置偏高"
                            adv_src = adv_60m_data if pass_60m_adv and adv_60m_data is not None else best_row

                            base_info = {
                                "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                                "_Pass_60m_Adv": pass_60m_adv, "_Adv_60m_Data": adv_60m_data,
                                "股票代號": ticker.split('.')[0], "股票名稱": stock_dict[ticker],
                                "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                                "近30日觸發次數": count_30d if config_dict["enable_30d_backtest"] else "-",
                                "當日收盤": round(float(best_row['Close']), 2), "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                                "當日成交量(張)": int(best_row['Volume_Lots']), "量增倍數": round(best_vol_mult, 2), "月均量(張)": int(best_row['Vol_MA20']),
                                "DIF": round(float(best_row['MACD']), 3), "當日KD(K值)": round(float(best_row['K']), 2),
                                "KD金叉判定": "✅ 是" if best_row['KD_Cross_Recent'] else "❌ 否",
                                "MACD金叉判定": "✅ 是" if best_row['MACD_Cross_Recent'] else "❌ 否",
                                "進場位置建議": rec_status, "盤整過濾": "✅ 盤整中" if last_expert['RSI盤整'] else "趨勢盤",
                                "建議水位": core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數']),
                                "多週期KD": core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])
                            }
                            
                            dynamic_metrics = core.ScanEngine.get_all_metrics(best_row, adv_src)
                            base_info.update(dynamic_metrics)
                            
                            reversal_candidates[ticker] = base_info
                    except Exception: pass
            except Exception: pass
            progress_bar.progress(min(1.0, (i + chunk_size) / len(tickers)))

        reversal_list, final_results = list(reversal_candidates.values()), []
        if reversal_list:
            progress_bar.progress(0); status_text.text("正在下載初選標的長天期資料以計算深層背離...")
            long_data = yf.download([item["_Full_Ticker"] for item in reversal_list], threads=False, progress=False, session=yf_session, **dl_kwargs)

            for idx, item in enumerate(reversal_list):
                try:
                    ticker, specific_offset, daily_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
                    pass_60m_adv, adv_60m_data = item.pop("_Pass_60m_Adv", False), item.pop("_Adv_60m_Data", None)
                    has_daily_div, has_m60_div = False, False
                    item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"
                    item['均線最大最小差異百分比(%)'] = "-"
                    item['均線糾結狀態'] = "-"

                    try:
                        if isinstance(long_data.columns, pd.MultiIndex):
                            if ticker in long_data.columns.get_level_values(0): long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                            elif ticker in long_data.columns.get_level_values(1): long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else: long_df = pd.DataFrame()
                        else: long_df = long_data.dropna(how='all') if len(reversal_list) == 1 else pd.DataFrame()

                        if not long_df.empty:
                            if config_dict["use_backtest_date"]: long_df = long_df[long_df.index.tz_localize(None).normalize() <= pd.to_datetime(config_dict["backtest_date_obj"])]
                            if specific_offset > 0: long_df = long_df.iloc[:-specific_offset]

                            if len(long_df) >= 240:
                                ent_res = core.MA240PullbackRightSideStrategy.evaluate_ma_entanglement(long_df)
                                item['均線最大最小差異百分比(%)'] = ent_res['ma_diff_pct']
                                item['均線糾結狀態'] = ent_res['ma_tangled']

                            if len(long_df) >= 120: item['半年線斜率(%)'] = round((long_df['Close'].rolling(120).mean().iloc[-1] - long_df['Close'].rolling(120).mean().iloc[-2]) / long_df['Close'].rolling(120).mean().iloc[-2] * 100, 3)
                            if len(long_df) >= 240:
                                item['年線斜率(%)'] = round((long_df['Close'].rolling(240).mean().iloc[-1] - long_df['Close'].rolling(240).mean().iloc[-2]) / long_df['Close'].rolling(240).mean().iloc[-2] * 100, 3)
                                item['年線/半年線空排'] = "✅ 是" if long_df['Close'].rolling(120).mean().iloc[-1] < long_df['Close'].rolling(240).mean().iloc[-1] else "❌ 否"

                                ma240_res = core.MA240PullbackRightSideStrategy.evaluate_phase2(long_df)
                                item['*回測走揚年線'] = "✅ 是" if ma240_res["is_pass"] else "❌ 否"
                                item['*年線回測分數'] = ma240_res["score"]

                            if config_dict["use_weekly_filter"]:
                                wk_res = core.WeeklyAnalysisStrategy.evaluate(long_df, div_pairs, config_dict["recent_lows_cnt"], config_dict["older_lows_cnt"], config_dict["pivot_left"], config_dict["pivot_right"], config_dict["require_cross_confirm"], config_dict["reso_cross_days"], 'Low' if config_dict["reso_price_basis"] == "最低價 (Low)" else 'Close')
                                item["週K共振"], item["週K背離"] = wk_res["週K共振"], wk_res["週K背離"]
                    except Exception:
                        if config_dict["use_weekly_filter"]: item["週K共振"] = item["週K背離"] = "錯誤"

                    if not daily_df.empty:
                        if specific_offset > 0: daily_df = daily_df.iloc[:-specific_offset]
                        for r_w, o_w in div_pairs:
                            d_kd, d_macd = False, False
                            for b in range(0, 25):
                                if len(daily_df) <= b + r_w + o_w: break
                                temp_df = daily_df.iloc[:len(daily_df)-b] if b > 0 else daily_df
                                if not d_kd: d_kd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'K', 'K', 'D', r_w, o_w, config_dict["recent_lows_cnt"], config_dict["older_lows_cnt"], config_dict["pivot_left"], config_dict["pivot_right"], config_dict["require_cross_confirm"], config_dict["reso_cross_days"])
                                if not d_macd: d_macd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, config_dict["recent_lows_cnt"], config_dict["older_lows_cnt"], config_dict["pivot_left"], config_dict["pivot_right"], config_dict["require_cross_confirm"], config_dict["reso_cross_days"])
                                if d_kd and d_macd: break
                            res_tags = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                            item[f"日K背離({r_w},{o_w})"] = "+".join(res_tags) if res_tags else "無"
                            if res_tags: has_daily_div = True

                    base_tags = core.ScanEngine.generate_base_tags(item, config_dict)
                    div_tag = " + 雙級別共振" if (has_daily_div and pass_60m_adv) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if pass_60m_adv else " (無背離)"))
                    
                    if pass_60m_adv and adv_60m_data is not None:
                        sig_name = "【🔥 60分K底倉突破】" if adv_60m_data.get('Adv_Signal_A') else "【🔥 60分K右肩回測】"
                        base_tags.insert(0, sig_name)

                    item["演算法建議結果"] = " & ".join(base_tags) + div_tag
                    final_results.append(item)
                except Exception: pass
                progress_bar.progress(min(1.0, (idx + 1) / len(reversal_list)))

        status_text.empty(); progress_bar.empty()
        res_df = pd.DataFrame(final_results)

        base_cols = ['股票代號', '股票名稱', '觸發日期']
        if config_dict["enable_30d_backtest"]: base_cols.append('近30日觸發次數')
        base_cols.extend([
            '演算法建議結果', '均線最大最小差異百分比(%)', '均線糾結狀態', '當日KD(K值)'
        ])
        
        base_cols.extend(core.ScanEngine.get_all_metric_keys())
        
        base_cols.extend([
            '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
            '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', 'KD金叉判定', 'MACD金叉判定', '當日收盤', '月均量(張)'
        ])
        if config_dict["use_weekly_filter"]: base_cols.extend(['週K共振', '週K背離'])
        base_cols.extend(['年線/半年線空排', '年線斜率(%)', '半年線斜率(%)'])

        cols = base_cols + [c for c in res_df.columns if '背離' in c and c not in base_cols]
        if not res_df.empty:
            res_df = res_df[[c for c in cols if c in res_df.columns]]
            
            sort_keys = [k for k in ["*年線回測分數", "共振分數", "反轉分數", "VCP分數"] if k in res_df.columns]
            if sort_keys:
                res_df = res_df.sort_values(by=sort_keys, ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1

        st.session_state.scan_res_df, st.session_state.scan_algo_mode, st.session_state.market_info = res_df, algo_mode, market_info
        st.session_state.scan_msg = f"🎉 掃描完成！本次共精選出 **{len(res_df)}** 檔標的。"

    if "scan_res_df" in st.session_state:
        st.success(st.session_state.scan_msg)
        if st.session_state.market_info:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("加權指數收盤", st.session_state.market_info["加權指數收盤"])
            c2.metric("月線 (MA20)", st.session_state.market_info["月線 (MA20)"])
            c3.metric("季線 (MA60)", st.session_state.market_info["季線 (MA60)"])
            c4.metric("約當大台基礎", st.session_state.market_info["自動運算基準價值 (約當大台基礎)"])
            st.info(f"**大盤環境判定：** {st.session_state.market_info['大盤環境判定']}")

        if not st.session_state.scan_res_df.empty:
            st.dataframe(st.session_state.scan_res_df, use_container_width=True)
            try:
                buffer = io.BytesIO()
                st.session_state.scan_res_df.to_excel(buffer, index=False, engine='openpyxl')
                file_name = f"stock_scan_results_{datetime.datetime.now().strftime('%Y-%m-%d %H%M')}.xlsx"
                st.download_button(label="📥 下載建議清單 (Excel)", data=buffer.getvalue(), file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            except ImportError:
                st.error("⚠️ 系統缺少 Excel 輸出套件 (openpyxl)。已為您自動降級為 CSV 格式下載。請於環境配置或 requirements.txt 中加入 `openpyxl` 以啟用 Excel 匯出功能。")
                csv = st.session_state.scan_res_df.to_csv(index=False).encode('utf-8-sig')
                file_name_csv = f"stock_scan_results_{datetime.datetime.now().strftime('%Y-%m-%d %H%M')}.csv"
                st.download_button(label="📥 下載建議清單 (CSV 備用)", data=csv, file_name=file_name_csv, mime='text/csv')

            def transfer_to_audit():
                tickers_str = ", ".join(st.session_state.scan_res_df['股票代號'].astype(str).tolist())
                st.session_state.batch_input_area = tickers_str
                st.session_state.run_batch_audit = True
                st.session_state.audit_mode = "自選股批次掃描"

            st.button("🚀 將上述標的傳送至【財報地雷與體質分析】進行基本面分析", type="primary", on_click=transfer_to_audit)
            if st.session_state.get('run_batch_audit', False):
                st.success("✅ **傳送成功！已自動載入標的。** 請切換至 **【📑 財報三表地雷掃描】** 分頁。")

# ----------------------------------------------------
# 頁籤 3：財報三表地雷掃描 
# ----------------------------------------------------
elif selected_page == "📑 財報三表地雷掃描":
    def render_trend_sparkline(trend_list):
        if not trend_list or len(trend_list) < 2: return "資料不足"
        valid_vals = [v for v in trend_list if pd.notna(v)]
        if len(valid_vals) < 2: return "資料不足"
        res = []
        for i in range(1, len(valid_vals)):
            if valid_vals[i] > valid_vals[i-1]: res.append("↗")
            elif valid_vals[i] < valid_vals[i-1]: res.append("↘")
            else: res.append("→")
        return " ".join(res[-3:])

    st.write("透過財報三表（損益表、資產負債表、現金流量表）交叉勾稽，提供**「三表交叉異常診斷」**與**「動態合約負債與擴張防誤判引擎」**，挖掘具備真實造血力與安全結構的卓越企業。")
    audit_mode = st.radio("請選擇操作模式", ["單檔查詢", "自選股批次掃描"], horizontal=True, key="audit_mode")

    if audit_mode == "單檔查詢":
        c1, c2 = st.columns([4, 1])
        with c1: audit_input = st.text_input("輸入單一股票代號", value="2495", key="audit_single").strip()
        with c2: st.write(""); st.write(""); audit_btn = st.button("健檢財報", type="primary", use_container_width=True)

        if audit_btn and audit_input:
            with st.spinner(f"⏳ 正在抓取 [{audit_input}] 季報資料並進行三表交叉診斷..."):
                stock_dict = get_all_tw_stocks()
                yf_ticker = core.resolve_ticker(audit_input, stock_dict)
                name = stock_dict.get(yf_ticker, audit_input)

                res = core.FinancialAuditStrategy.evaluate(yf_ticker, core.get_yf_session())
                if res["status"] == "error":
                    st.error(res["msg"])
                else:
                    st.subheader(f"📑 {name} ({audit_input}) 財報三表體質與風險診斷報告")
                    st.markdown(f"**當期季報：{res['當期季報']} | 分析期間：{res['比較基準']}**")
                    st.markdown("---")
                    st.markdown("### 【一、總體診斷】")
                    st.markdown(f"#### 綜合評級：{res['綜合評級']}")
                    c_sc1, c_sc2 = st.columns(2)
                    with c_sc1: st.metric("🏆 體質綜合評分 (滿分100)", f"{res['體質分數']} 分")
                    with c_sc2: st.metric("🛡️ 財務風險評分 (越低越好)", f"{res['風險分數']} 分")

                    st.markdown("---")
                    st.markdown("### 【二、三表核心指標】")
                    core_df_data = []
                    for row in res["core_metrics"]:
                        r_dict = row.copy()
                        r_dict["4Q趨勢"] = render_trend_sparkline(row["4Q趨勢"]) if isinstance(row["4Q趨勢"], list) else row["4Q趨勢"]
                        core_df_data.append(r_dict)
                    st.dataframe(pd.DataFrame(core_df_data), use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 【三、合約負債專區 (Contract Liabilities)】")
                    cl_c1, cl_c2, cl_c3, cl_c4 = st.columns(4)
                    cl_c1.metric("最新合約負債 (千)", res['cl_latest'])
                    cl_c2.metric("QoQ 季增率", res['cl_qoq'])
                    cl_c3.metric("YoY 年增率", res['cl_yoy'])
                    cl_c4.metric("佔營收比重", res['cl_rev_ratio'])
                    st.info(f"**合約負債判定狀態：** {res['cl_status']}")
                    if len([v for v in res['cl_8q'] if pd.notna(v)]) >= 5:
                        st.write(f"**歷史 8 季趨勢軌跡：** `{render_trend_sparkline(res['cl_8q'])}`")

                    st.markdown("---")
                    st.markdown("### 【四、三表交叉診斷】")
                    for text in res["cross_analysis"]: st.markdown(f"- {text}")
                    st.markdown("---")
                    st.markdown("### 【五、風險共振】")
                    for text in res["risk_resonance"]:
                        if "無重大風險" in text: st.success(f"- {text}")
                        else: st.error(f"- {text}")
                    st.markdown("---")
                    st.markdown("### 【六、正向特徵】")
                    for text in res["positives"]:
                        if "無顯著突出" in text: st.warning(f"- {text}")
                        else: st.success(f"- 🌟 {text}")
                    st.markdown("---")
                    st.markdown("### 【七、下一季追蹤事項】")
                    for idx, text in enumerate(res["watch_list"]): st.markdown(f"{idx+1}. {text}")
    else:
        st.write("請貼上你想健檢的股票清單（可使用逗號、空白、或換行分隔），系統將產出包含**「體質評分」**與**「財務風險」**的綜合比較清單。")
        batch_input = st.text_area("輸入自選股清單", height=100, key="batch_input_area")
        batch_btn = st.button("執行批次交叉健檢", type="primary")

        if batch_btn or st.session_state.get('run_batch_audit', False):
            if st.session_state.get('run_batch_audit', False): st.session_state.run_batch_audit = False
            raw_tickers = re.split(r'[,\s\n]+', batch_input.strip())
            valid_tickers = [t.strip() for t in raw_tickers if t.strip()]

            if not valid_tickers: st.warning("⚠️ 找不到有效的股票代號。")
            else:
                stock_dict = get_all_tw_stocks()
                session = core.get_yf_session()
                batch_results = []
                status_text, p_bar = st.empty(), st.progress(0)

                for idx, t_input in enumerate(valid_tickers):
                    status_text.text(f"⏳ 正在分析 {t_input} ... ({idx+1}/{len(valid_tickers)})")
                    yf_t = core.resolve_ticker(t_input, stock_dict)
                    name = stock_dict.get(yf_t, t_input)
                    res = core.FinancialAuditStrategy.evaluate(yf_t, session)
                    if res["status"] == "success":
                        batch_results.append({
                            "代號": t_input, "名稱": name, "當期季報": res["當期季報"],
                            "綜合評級": res["綜合評級"], "體質分數": res["體質分數"], "風險分數": res["風險分數"],
                            "合約負債判定": res["cl_status"].split(" ")[0] if " " in res["cl_status"] else res["cl_status"],
                            "ROE年化(%)": f"{res['ROE年化(%)']:.2f}%" if pd.notna(res.get('ROE年化(%)')) else "-",
                            "ROA年化(%)": f"{res['ROA年化(%)']:.2f}%" if pd.notna(res.get('ROA年化(%)')) else "-",
                            "營收YoY(%)": f"{res['營收成長(%)']:.2f}%" if pd.notna(res['營收成長(%)']) else "-",
                            "毛利率(%)": f"{res['毛利率']:.2f}%" if pd.notna(res['毛利率']) else "-",
                            "營益率(%)": f"{res['營益率']:.2f}%" if pd.notna(res['營益率']) else "-",
                            "CFO/淨利(倍)": f"{res['CFO/淨利']:.2f}" if pd.notna(res['CFO/淨利']) else "-",
                            "CCC週期(天)": f"{res['CCC_now']:.0f}" if pd.notna(res['CCC_now']) else "-",
                            "收現天數(DSO)": f"{res['DSO_now']:.0f}" if pd.notna(res['DSO_now']) else "-",
                            "週轉天數(DIO)": f"{res['DIO_now']:.0f}" if pd.notna(res['DIO_now']) else "-",
                            "警訊數": res["警訊數"], "護城河特徵": res["正向訊號"], "警示明細": res["診斷明細"]
                        })
                    time.sleep(0.5)
                    p_bar.progress(min(1.0, (idx + 1) / len(valid_tickers)))

                status_text.empty(); p_bar.empty()
                if batch_results:
                    st.success(f"🎉 批次健檢完成！共成功分析 {len(batch_results)} 檔標的。")
                    batch_df = pd.DataFrame(batch_results)
                    batch_df = batch_df.sort_values(by=["體質分數", "風險分數"], ascending=[False, True]).reset_index(drop=True)
                    batch_df.index = batch_df.index + 1
                    st.dataframe(batch_df, use_container_width=True)
                    try:
                        buffer = io.BytesIO()
                        batch_df.to_excel(buffer, index=False, engine='openpyxl')
                        file_name = f"financial_cross_audit_{datetime.datetime.now().strftime('%Y-%m-%d %H%M')}.xlsx"
                        st.download_button(label="📥 下載三表交叉財報健檢報告 (Excel)", data=buffer.getvalue(), file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    except ImportError:
                        st.error("⚠️ 系統缺少 Excel 輸出套件 (openpyxl)。已為您自動降級為 CSV 格式下載。請於環境配置或 requirements.txt 中加入 `openpyxl` 以啟用 Excel 匯出功能。")
                        csv = batch_df.to_csv(index=False).encode('utf-8-sig')
                        file_name_csv = f"financial_cross_audit_{datetime.datetime.now().strftime('%Y-%m-%d %H%M')}.csv"
                        st.download_button(label="📥 下載三表交叉財報健檢報告 (CSV 備用)", data=csv, file_name=file_name_csv, mime='text/csv')

# ----------------------------------------------------
# 頁籤 4：幾何型態滿足點測距 
# ----------------------------------------------------
elif selected_page == "📐 幾何型態滿足點測距":
    st.write("利用量化演算法自動尋找波段轉折極值，並提供**「模式一：實體收盤價（過濾雜訊，對齊主力籌碼）」**與**「模式二：影線極值（包含極端恐慌斷頭與流動性插針）」**的雙重 1:1 及 1.618 幾何延伸測距。")

    col_g1, col_g2 = col_columns = st.columns([4, 1])
    with col_g1: geo_input = st.text_input("輸入股票代號", value="2609", key="geo_single", placeholder="例如：2330").strip()
    with col_g2: st.write(""); st.write(""); geo_btn = st.button("計算幾何滿足點", type="primary", use_container_width=True)

    if geo_btn and geo_input:
        with st.spinner(f"⏳ 正在計算 [{geo_input}] 幾何滿足點與 SPC 管制界限..."):
            df, real_ticker = get_stock_data(geo_input)
            if df.empty:
                st.error("❌ 查無歷史數據，請確認代號是否正確。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                df['20MA'] = df['Close'].rolling(window=20).mean()
                df['60MA'] = df['Close'].rolling(window=60).mean()
                df['120MA'] = df['Close'].rolling(window=120).mean()
                rolling_std = df['Close'].rolling(window=20).std()
                df['SPC_UCL'] = df['20MA'] + (2 * rolling_std)
                df['SPC_LCL'] = df['20MA'] - (2 * rolling_std)
                df['Z_Score'] = (df['Close'] - df['20MA']) / rolling_std
                latest = df.iloc[-1]

                st.subheader(f"📐 【{stock_name} ({real_ticker})】 量化幾何與 SPC 統計分析表")
                st.markdown(f"**資料日期：{df.index[-1].strftime('%Y-%m-%d')}**")

                st.markdown("#### 1. 均線位階與 SPC 管制極限 (動態標準差)")
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                c_s1.metric("最新收盤價", f"{latest['Close']:.2f}")
                c_s2.metric("月線 / 季線", f"{latest['20MA']:.2f} / {latest['60MA']:.2f}")
                c_s3.metric("半年線 (120MA)", f"{latest['120MA']:.2f}" if not pd.isna(latest['120MA']) else "-")
                c_s4.metric("當前 Z-Score", f"{latest['Z_Score']:.2f}")

                z_state = "常態波動區間段"
                if latest['Z_Score'] < -2.0: z_state = "🚨 【極端負乖離超跌】 (空方拋售竭盡 / 統計異常)"
                elif latest['Z_Score'] > 2.0: z_state = "🔥 【極端正乖離過熱】 (多方推升超買 / 統計異常)"
                st.info(f"**布林上軌 (+2σ)：** {latest['SPC_UCL']:.2f} ｜ **布林下軌 (-2σ)：** {latest['SPC_LCL']:.2f} ｜ **統計狀態：** {z_state}")
                st.markdown("---")

                res_mode1 = core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'Close', 'Close')
                res_mode2 = core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'High', 'Low')

                def render_pattern_result(res_dict, title):
                    st.markdown(f"#### {title}")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if res_dict["bull"]:
                            b = res_dict["bull"]
                            st.success(f"**▲ 多方進攻：W 底突破測距**\n\n"
                                       f"- **波段雙底**：{b['trough1']:.2f} ({b['date1']}) 與 {b['trough2']:.2f} ({b['date2']})\n"
                                       f"- **進攻頸線**：{b['neckline']:.2f} 元\n"
                                       f"- **型態高度**：{b['height']:.2f} 元\n"
                                       f"- **1:1 漲幅滿足位**：**{b['target_1to1']:.2f} 元**\n"
                                       f"- **1.618 延伸滿足位**：**{b['target_1618']:.2f} 元**")
                        else:
                            st.warning("**▲ 多方 W 底**：樣本不足，無法辨識完整雙底結構。")
                    with col_m2:
                        if res_dict["bear"]:
                            b = res_dict["bear"]
                            st.error(f"**▼ 空方摜壓：M 頭跌破測距**\n\n"
                                     f"- **波段雙峰**：{b['peak1']:.2f} ({b['date1']}) 與 {b['peak2']:.2f} ({b['date2']})\n"
                                     f"- **防守頸線**：{b['neckline']:.2f} 元\n"
                                     f"- **型態高度**：{b['height']:.2f} 元\n"
                                     f"- **1:1 跌幅滿足位**：**{b['target_1to1']:.2f} 元**\n"
                                     f"- **1.618 延伸滿足位**：**{b['target_1618']:.2f} 元**")
                        else:
                            st.warning("**▼ 空方 M 頭**：樣本不足，無法辨識完整雙峰結構。")

                render_pattern_result(res_mode1, "2. 模式一：實體收盤價模型（過濾盤中突刺，對齊主力籌碼實體成本）")
                st.markdown("---")
                render_pattern_result(res_mode2, "3. 模式二：影線極值模型（包含盤中插針、斷頭踩踏與極端低點）")

# ----------------------------------------------------
# 頁籤 5：量化型態狀態機 
# ----------------------------------------------------
elif selected_page == "🤖 量化型態狀態機 (程式A)":
    st.write("結合**「60日波段位階濾網」**、**「斜向收斂楔形/通道趨勢線引擎」**與**「首日破線即時預警狀態」**，徹底解決高檔橫盤誤判 W 底、頭頭高誤判 M 頭以及盤中長黑延遲反應等結構盲點。")

    with st.expander("⚙️ 狀態機與幾何參數微調 (選填)", expanded=False):
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1: cfg_win = st.number_input("極值確認左右 K 棒數", min_value=2, max_value=15, value=5, step=1)
        with qc2: cfg_sim = st.number_input("雙底/雙峰對稱誤差(%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5) / 100.0
        with qc3: cfg_b_days = st.number_input("突破連續確認天數", min_value=1, max_value=5, value=2, step=1)
        with qc4: cfg_regime = st.checkbox("啟用波段位階濾網 (防高檔W底/頭頭高M頭)", value=True)

    col_q1, col_q2, col_q3 = st.columns([2.5, 1.5, 1])
    with col_q1: q_input = st.text_input("輸入股票代號", value="2495", key="quant_engine_input", placeholder="請填寫欲檢測之股票代號 (如: 2495)").strip()
    with col_q2: intraday_p = st.number_input("盤中即時價模擬 (0為不啟用)", min_value=0.0, value=0.0, step=0.1)
    with col_q3:
        st.write(""); st.write("")
        q_btn = st.button("啟量化狀態機", type="primary", use_container_width=True)

    if q_btn and q_input:
        with st.spinner(f"⏳ 正在載入 [{q_input}] 資料並通過量化引擎審查..."):
            df, real_ticker = get_stock_data(q_input)
            if df.empty:
                st.error("❌ 查無歷史數據，請確認代號是否正確。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                custom_cfg = core.QuantConfig(
                    window_order=int(cfg_win),
                    price_sim_threshold=float(cfg_sim),
                    breakout_days=int(cfg_b_days),
                    enable_regime_filter=bool(cfg_regime)
                )
                engine = core.QuantPatternEngine(custom_cfg)
                df = engine.clean_and_validate_data(df)

                if intraday_p > 0:
                    last_dt = df.index[-1]
                    today_date = pd.Timestamp.now().date()
                    if last_dt.date() == today_date:
                        df.iloc[-1, df.columns.get_loc('High')] = max(float(df['High'].iloc[-1]), intraday_p)
                        df.iloc[-1, df.columns.get_loc('Low')] = min(float(df['Low'].iloc[-1]), intraday_p)
                        df.iloc[-1, df.columns.get_loc('Close')] = float(intraday_p)
                    else:
                        sim_dt = last_dt + pd.Timedelta(days=1)
                        prev_c = float(df['Close'].iloc[-1])
                        sim_row = pd.DataFrame({
                            'Open': [prev_c], 'High': [max(prev_c, intraday_p)],
                            'Low': [min(prev_c, intraday_p)], 'Close': [float(intraday_p)],
                            'Volume': [float(df['Volume'].iloc[-20:].mean() * 1.0)]
                        }, index=[sim_dt])
                        df = pd.concat([df, sim_row])

                df = engine.compute_indicators(df)
                close_patterns = engine.analyze_pipeline(df=df, high_series=df['Close'], low_series=df['Close'])
                shadow_patterns = engine.analyze_pipeline(df=df, high_series=df['High'], low_series=df['Low'])
                diag_close = engine.identify_diagonal_channel(df=df, high_series=df['Close'], low_series=df['Close'])
                diag_shadow = engine.identify_diagonal_channel(df=df, high_series=df['High'], low_series=df['Low'])

                latest = df.iloc[-1]
                latest_date = df.index[-1].strftime('%Y-%m-%d') + (" (含盤中即時模擬)" if intraday_p > 0 else "")
                latest_close = latest['Close']

                st.subheader(f"🤖 【{stock_name} ({real_ticker})】 台股量化狀態機分析報告")
                st.markdown(f"**最後有效交易日：** {latest_date}")
                st.markdown("---")

                st.markdown("### 一、核心量化數據與位階狀態")
                cq1, cq2, cq3, cq4, cq5 = st.columns(5)
                cq1.metric("最新基準價", f"{latest_close:.2f}")
                cq2.metric("20MA / 60MA", f"{latest['20MA']:.2f} / {latest['60MA']:.2f}")
                cq3.metric("60日位階百分位", f"{latest['Pos_60D']*100:.1f}%")
                cq4.metric("Z-Score", f"{latest['Z_Score']:.2f} σ")
                cq5.metric("量比 (相對 20MA)", f"{latest['Vol_Ratio']:.2f}x")

                z_desc = "極端負乖離 (超跌)" if latest['Z_Score'] < -2.0 else "極端正乖離 (過熱)" if latest['Z_Score'] > 2.0 else "常態隨機波動"
                ma120_str = f"{latest['120MA']:.2f}" if not pd.isna(latest['120MA']) else "樣本不足"
                st.info(f"**均線系統**：20MA: {latest['20MA']:.2f} | 60MA: {latest['60MA']:.2f} | 120MA: {ma120_str}\n\n"
                        f"**布林軌道**：上軌 {latest['BB_UCL']:.2f} | 下軌 {latest['BB_LCL']:.2f} ──► 波動狀態【{z_desc}】")

                st.markdown("---")
                st.markdown("### 二、斜向趨勢軌道與收斂楔形/通道診斷")

                def render_diagonal_block(d_pat, title, cur_p):
                    st.markdown(f"#### {title}")
                    if not d_pat:
                        st.warning("└─► 樣本極值不足，無法建構斜向趨勢軌道。")
                        return
                    st.markdown(f"- **辨識幾何結構**：**【{d_pat.channel_type}】**")
                    st.write(f"- **上軌波段高點**：{d_pat.high1.price:.2f} ({d_pat.high1.date.strftime('%m/%d')}) ──► {d_pat.high2.price:.2f} ({d_pat.high2.date.strftime('%m/%d')}) | 斜率: `{d_pat.upper_slope:+.2f}` 元/日")
                    st.write(f"- **下軌波段低點**：{d_pat.low1.price:.2f} ({d_pat.low1.date.strftime('%m/%d')}) ──► {d_pat.low2.price:.2f} ({d_pat.low2.date.strftime('%m/%d')}) | 斜率: `{d_pat.lower_slope:+.2f}` 元/日")
                    st.write(f"- **今日動態軌道界限**：上軌壓力 `{d_pat.proj_upper:.2f}` 元 ｜ 下軌支撐 `{d_pat.proj_lower:.2f}` 元")

                    if "BREAKDOWN" in d_pat.state:
                        v_str = "放量摜破" if d_pat.vol_confirmed else "跌破"
                        b_dt = d_pat.break_date.strftime('%Y-%m-%d') if d_pat.break_date else "-"
                        st.error(f"**當前軌道狀態**：{d_pat.state}\n\n- **觸發日期**：{b_dt} ({v_str}) ｜ **現價 ({cur_p:.2f}) 已跌破動態下軌 ({d_pat.proj_lower:.2f})**\n- **等幅下修測距參考**：`{d_pat.target_down:.2f}` 元")
                    elif "BREAKOUT" in d_pat.state:
                        v_str = "帶量突破" if d_pat.vol_confirmed else "突破"
                        b_dt = d_pat.break_date.strftime('%Y-%m-%d') if d_pat.break_date else "-"
                        st.success(f"**當前軌道狀態**：{d_pat.state}\n\n- **觸發日期**：{b_dt} ({v_str}) ｜ **現價 ({cur_p:.2f}) 已突破動態上軌 ({d_pat.proj_upper:.2f})**\n- **等幅上攻測距參考**：`{d_pat.target_up:.2f}` 元")
                    else:
                        st.info(f"**當前軌道狀態**：{d_pat.state}\n\n- 現價 ({cur_p:.2f}) 介於下軌支撐 `{d_pat.proj_lower:.2f}` 與上軌壓力 `{d_pat.proj_upper:.2f}` 之間。")

                dc1, dc2 = st.columns(2)
                with dc1:
                    with st.container(border=True): render_diagonal_block(diag_close, "📐 實體收盤價趨勢軌道", latest_close)
                with dc2:
                    with st.container(border=True): render_diagonal_block(diag_shadow, "📐 高低影線極值趨勢軌道", latest_close)

                st.markdown("---")
                def render_streamlit_pattern_block(p, title, l_price):
                    st.markdown(f"#### {title}")
                    if not p:
                        st.warning("└─► 結構狀態：未辨識出符合「對稱誤差」與「波段位階濾網」之水平型態。")
                        return

                    s1_str = f"{p.swing1.price:.2f} ({p.swing1.date.strftime('%m/%d')})"
                    s2_str = f"{p.swing2.price:.2f} ({p.swing2.date.strftime('%m/%d')})"
                    st.write(f"- **幾何極值**：點1 [{s1_str}] | 點2 [{s2_str}] ──► *{p.regime_note}*")
                    st.write(f"- **中間頸線價位**：{p.neckline.price:.2f} 元 ({p.neckline.date.strftime('%m/%d')})")

                    state_color = "🟢" if "BREAKOUT_CONFIRMED" in p.state else ("🟠" if "DAY1_BREAK" in p.state else ("🔴" if "FAILED" in p.state or "INVALID" in p.state else "🟡"))
                    st.markdown(f"- **當前型態狀態**：{state_color} **【{p.state}】**")

                    if p.breakout_date:
                        vol_tag = "✅ 有效放量" if p.vol_confirmed else "⚠️ 量能不足/平量"
                        st.write(f"- **突破/跌破觸發日**：{p.breakout_date.strftime('%Y-%m-%d')} ({vol_tag})")

                    st.write(f"- **測距滿足位**：1:1 等幅 `{p.target_1to1:.2f}` 元 | 1.618 延伸 `{p.target_1618:.2f}` 元")
                    st.write(f"- **結構失效防線**：`{p.invalidation_level:.2f}` 元")

                    if p.state in [core.PatternState.BREAKOUT_CONFIRMED, core.PatternState.RETEST, core.PatternState.DAY1_BREAK_WARNING]:
                        practical_sl = p.neckline.price * (1 - 0.01) if p.p_type == 'W' else p.neckline.price * (1 + 0.01)
                        risk = abs(l_price - practical_sl)
                        reward_1to1 = abs(p.target_1to1 - l_price)
                        if risk > 0 and ((p.p_type == 'W' and l_price < p.target_1to1) or (p.p_type == 'M' and l_price > p.target_1to1)):
                            rr_ratio = reward_1to1 / risk
                            st.success(f"└─► **實質風險報酬比 (至 1:1 目標)**：1 : {rr_ratio:.2f} (以頸線破線為防守)")
                        else:
                            st.info("└─► 實質風報比：當前價已達標或防守距離過近。")
                    else:
                        st.info("└─► 實質風報比：型態尚未完成確認突破，不提供先行交易風報比。")

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("### 三、水平頸線：實體收盤價模型\n*(過濾盤中突刺，錨定實體持股籌碼)*")
                    with st.container(border=True):
                        render_streamlit_pattern_block(close_patterns['W'], "▲ 多方 W 底模型", latest_close)
                        st.divider()
                        render_streamlit_pattern_block(close_patterns['M'], "▼ 空方 M 頭模型", latest_close)
                with cc2:
                    st.markdown("### 四、水平頸線：影線極值模型\n*(涵蓋盤中高低點與極端流動性)*")
                    with st.container(border=True):
                        render_streamlit_pattern_block(shadow_patterns['W'], "▲ 多方 W 底模型", latest_close)
                        st.divider()
                        render_streamlit_pattern_block(shadow_patterns['M'], "▼ 空方 M 頭模型", latest_close)

                st.markdown("---")
                st.markdown("### 五、客觀架構綜合結論")
                final_states = []
                for d_label, d_res in [("實體軌道", diag_close), ("影線軌道", diag_shadow)]:
                    if d_res and "BREAK" in d_res.state:
                        final_states.append(f"{d_label} [{d_res.channel_type.split(' ')[0]}]: {d_res.state}")
                for model_res in [close_patterns, shadow_patterns]:
                    for p in [model_res['W'], model_res['M']]:
                        if p and p.state not in [core.PatternState.NONE, core.PatternState.INVALIDATED]:
                            final_states.append(f"水平{p.p_type}型態: {p.state}")

                if final_states:
                    st.success(f"**當前主導狀態**：{' ； '.join(dict.fromkeys(final_states))}")
                else:
                    st.warning("**當前主導狀態**：目前無已確認之水平反轉型態或趨勢線突破，市場處於軌道內整理或單向波段中。")
