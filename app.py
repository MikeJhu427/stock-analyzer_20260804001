import os
import json
import warnings
import datetime
import time
import re
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import strategy_core as core

warnings.filterwarnings("ignore")

OUTPUT_COLUMN_CONFIG = {"show_ma120_240_bear_align": True, "show_ma240_slope": True, "show_ma120_slope": True}
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
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception: pass

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tw_stocks():
    return core.fetch_all_tw_stocks()

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
st.markdown("""<style>div.row-widget.stRadio > div{flex-direction:row;flex-wrap:wrap;}
div.row-widget.stRadio > div > label{background-color: #f0f2f6; padding: 10px 15px; border-radius: 8px; margin-right: 10px; margin-bottom: 10px; cursor: pointer;}</style>""", unsafe_allow_html=True)

selected_page = st.radio("📌 快速切換功能分頁：", ["📊 單檔深度解析", "🚀 全市場智慧掃描", "📑 財報三表地雷掃描", "📐 幾何型態滿足點測距", "🤖 量化型態狀態機 (程式A)"], horizontal=True)

# ----------------------------------------------------
# 頁籤 1：單檔深度解析
# ----------------------------------------------------
if selected_page == "📊 單檔深度解析":
    col1, col2 = st.columns([4, 1])
    with col1: user_input = st.text_input("輸入股票代號", value="2495", placeholder="例如：2330").strip()
    with col2: st.write(""); st.write(""); submit_btn = st.button("開始分析", type="primary")

    if submit_btn and user_input:
        with st.spinner(f"⏳ 正在抓取 [{user_input}] 資料並進行解析中..."):
            df, real_ticker = get_stock_data(user_input)
            if df.empty: st.error("❌ 查無歷史數據。")
            else:
                stock_name = get_tw_stock_name(real_ticker)
                df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
                df = core.TechnicalIndicators.add_rsi(df)
                
                df['Pct_Change'] = df['Close'].pct_change() * 100
                df['Volume_Lots'] = df['Volume'] / 1000
                df['Momentum_Force'] = df['Pct_Change'] * df['Volume_Lots']
                df['Prev_Close'] = df['Close'].shift(1)
                df['MA5'], df['Prev_MA5'] = df['Close'].rolling(5).mean(), df['Close'].rolling(5).mean().shift(1)
                df['MA20'], df['MA60'] = df['Close'].rolling(20).mean(), df['Close'].rolling(60).mean()
                df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8) * 100
                df['Std20'] = df['Close'].rolling(20).std()
                df['BB_Upper'], df['BB_Lower'] = df['MA20'] + 2 * df['Std20'], df['MA20'] - 2 * df['Std20']
                df['VWMA20'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
                df['Prev_VWMA20'], df['Vol_MA20'] = df['VWMA20'].shift(1), df['Volume_Lots'].rolling(20).mean()
                
                vols, opens, closes, defense = df['Volume'].values, df['Open'].values, df['Close'].values, np.full(len(df), np.nan)
                for i in range(60, len(df)):
                    max_idx = (i - 60) + np.argmax(vols[i-60:i+1])
                    defense[i] = min(opens[max_idx], closes[max_idx])
                df['Max_Vol_Defense'], df['Prev_Defense'] = defense, pd.Series(defense).shift(1)
                df['M_Mean'], df['M_Std'] = df['Momentum_Force'].rolling(60).mean(), df['Momentum_Force'].rolling(60).std()
                df['Upper_Bound'], df['Lower_Bound'] = df['M_Mean'] + 1.5 * df['M_Std'], df['M_Mean'] - 1.5 * df['M_Std']
                
                # 自動利用 core 套用所有策略計分
                config_dict = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys() if k in st.session_state}
                df = core.ScanEngine.calculate_all_phase1_scores(df, config_dict)
                
                df = df.dropna(subset=['Momentum_Force', 'Max_Vol_Defense', 'VWMA20', 'Prev_MA5']).copy()
                plot_df, recent_df = df.tail(240).copy(), df.tail(60)
                last_row, last_date = recent_df.iloc[-1], recent_df.index[-1].strftime('%Y-%m-%d')
                
                st.subheader(f"📊 【{stock_name} ({real_ticker})】 深度解析與策略判定")
                col_a, col_b, col_c = st.columns(3)
                with col_a: st.info(f"**底部翻轉分數：{round(last_row['Candle_Score'], 2)}**")
                with col_b: st.success(f"**VCP 收斂分數：{round(last_row['VCP_Score'], 2)}**")
                with col_c: st.warning(f"**指標共振分數：{round(last_row['Reso_Score'], 2)}**")
                st.markdown("---")
                
                expert_res_df = core.ExpertTradingStrategy.evaluate(df)
                last_expert = expert_res_df.iloc[-1]
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    if last_expert['RSI盤整']: st.error("⚠️ **震盪盤整中**")
                    else: st.success("✅ **具備趨勢動能**")
                with col_e2: st.info(f"**多週期 KD 狀態：**\n\n{core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])}")
                with col_e3: st.warning(f"**三指標共振持股建議：**\n\n{core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數'])}")
                
                st.markdown("---")
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.8, 1.2, 1.2]})
                ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='#1f77b4')
                ax1.plot(plot_df.index, plot_df['MA20'], label='MA20', color='orange', linestyle='--')
                ax1.plot(plot_df.index, plot_df['VWMA20'], label='VWMA20', color='blue', linestyle='-.')
                ax1.plot(plot_df.index, plot_df['Max_Vol_Defense'], label='Defense Line', color='teal')
                ax1.legend(); ax1.grid(True, alpha=0.5)
                ax2.plot(plot_df.index, plot_df['Momentum_Force'], color='#7f7f7f'); ax2.axhline(0, color='black', linestyle='--')
                ax2.plot(plot_df.index, plot_df['Upper_Bound'], color='green', linestyle=':'); ax2.plot(plot_df.index, plot_df['Lower_Bound'], color='red', linestyle=':')
                ax3.bar(plot_df.index, plot_df['Candle_Score'], color=['#d62728' if v > 0 else '#2ca02c' for v in plot_df['Candle_Score']])
                ax3.axhline(st.session_state.min_score, color='red', linestyle=':')
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

# ----------------------------------------------------
# 頁籤 2：全市場智慧掃描
# ----------------------------------------------------
elif selected_page == "🚀 全市場智慧掃描":
    with st.expander("⚙️ 掃描與背離參數設定", expanded=True):
        profile_names = list(st.session_state.config["profiles"].keys())
        idx = profile_names.index(st.session_state.current_profile) if st.session_state.current_profile in profile_names else 0
        def on_profile_change(): apply_profile_to_state(st.session_state.profile_selector)

        col_p1, col_p2, col_p3, col_p4 = st.columns([3, 3, 2, 2])
        with col_p1: st.selectbox("選擇歷史設定檔", profile_names, index=idx, key="profile_selector", on_change=on_profile_change)
        with col_p2: st.text_input("儲存新名稱", key="new_profile_input")
        with col_p3:
            st.write(""); st.write("")
            if st.button("💾 儲存設定", use_container_width=True):
                name_to_save = st.session_state.new_profile_input.strip() or st.session_state.profile_selector
                if name_to_save == "預設參數 (Default)": st.error("❌ 不可覆寫系統預設！")
                else:
                    st.session_state.backtest_date = str(st.session_state.backtest_date_obj)
                    st.session_state.config["profiles"][name_to_save] = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys()}
                    st.session_state.config["last_used"] = name_to_save
                    save_config(st.session_state.config); st.success(f"✅ 已儲存：'{name_to_save}'"); st.rerun()
        with col_p4:
            st.write(""); st.write("")
            if st.button("🗑️ 刪除", use_container_width=True):
                if st.session_state.profile_selector != "預設參數 (Default)":
                    del st.session_state.config["profiles"][st.session_state.profile_selector]
                    apply_profile_to_state("預設參數 (Default)")
                    st.success("✅ 已刪除"); st.rerun()

        st.markdown("**1. 掃描模式與演算法選擇**")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
        with col_m1: scan_target_mode = st.radio("掃描對象", ['全市場掃描', '指定個股測試'], horizontal=True)
        with col_m2: test_ticker_input = st.text_input("測試代號", value="6147") if scan_target_mode == '指定個股測試' else ""
        with col_m3: algo_mode = st.radio("掃描演算法", core.SCAN_ALGO_CHOICES, horizontal=True)

        st.markdown("**2. 基礎掃描參數、時光機回測與雲端備份設定**")
        col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1.2, 1, 1.2])
        with col_b1: st.checkbox("啟用指定日期回測", key="use_backtest_date")
        with col_b2: st.date_input("選擇回測基準日", key="backtest_date_obj", disabled=not st.session_state.use_backtest_date)
        with col_b3: st.checkbox("近 30日觸發統計", key="enable_30d_backtest")
        with col_b4: st.checkbox("自動上傳 Google Drive", key="auto_upload_gdrive")

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.number_input("掃描區間迄", min_value=0, key="lookback_end")
            st.number_input("掃描區間起", min_value=0, key="lookback_start")
        with col_b: st.number_input("翻轉最低分", min_value=1.0, step=1.0, key="min_score")
        with col_c: st.number_input("VCP最低分", min_value=1.0, step=1.0, key="min_vcp_score")
        with col_d: st.number_input("共振最低分", min_value=1.0, step=1.0, key="min_reso_score")
        with col_e: st.number_input("月均量下限", min_value=0, step=100, key="min_vol_ma20")

        st.markdown("**3. 波段檢測週期與背離條件設定**")
        st.checkbox("單一組自訂背離週期", key="use_single_div")
        col_f, col_g, col_h, col_i, col_j, col_k = st.columns(6)
        with col_f: st.number_input("近波範圍", min_value=5, step=1, key="div_recent_w")
        with col_g: st.number_input("前波範圍", min_value=10, step=1, key="div_older_w")
        with col_h: st.number_input("左X根不破", key="pivot_left")
        with col_i: st.number_input("右Y根不破", key="pivot_right")
        with col_j: st.number_input("近波低點數", key="recent_lows_cnt")
        with col_k: st.number_input("前波低點數", key="older_lows_cnt")

        st.markdown("**4. 進階濾網 (扣抵/絕對值/金叉確認等)**")
        st.checkbox("嚴格要求金叉確認背離與共振", key="require_cross_confirm")
        st.checkbox("啟用週K線共振與底背離判定", key="use_weekly_filter")
        st.checkbox("啟用進場位置建議", key="use_cross_position_filter")

    if st.button("🚀 開始智慧區間掃描", type="primary"):
        status_text, progress_bar = st.empty(), st.progress(0)
        stock_dict = get_all_tw_stocks()
        if not stock_dict: st.stop()
        
        tickers = [core.resolve_ticker(test_ticker_input, stock_dict)] if scan_target_mode == "指定個股測試" else list(stock_dict.keys())
        config_dict = {k: st.session_state[k] for k in core.DEFAULT_CONFIG.keys() if k in st.session_state}
        
        div_pairs = [(config_dict["div_recent_w"], config_dict["div_older_w"])] if config_dict["use_single_div"] else [(5, 20), (5, 60), (20, 60)]
        max_older_w = max(pair[1] for pair in div_pairs)
        dl_kwargs = {"period": "2y"}
        if config_dict["use_backtest_date"]:
            end_dt = config_dict["backtest_date_obj"] + datetime.timedelta(days=1)
            dl_kwargs = {"start": (end_dt - datetime.timedelta(days=730)).strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
            
        yf_session = core.get_yf_session()
        market_info = core.MarketRegimeFilter.evaluate(yf_session, config_dict["backtest_date_obj"] if config_dict["use_backtest_date"] else None)
        reversal_candidates, chunk_size = {}, 40

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            status_text.text(f"正在全市場掃描：進度 {min(i+chunk_size, len(tickers))} / {len(tickers)}...")
            try:
                data = yf.download(chunk, threads=False, progress=False, session=yf_session, **dl_kwargs)
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

                        df['Pct_Change'], df['Volume_Lots'] = df['Close'].pct_change() * 100, df['Volume'] / 1000
                        df['MA20'], df['MA60'] = df['Close'].rolling(20).mean(), df['Close'].rolling(60).mean()
                        df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()
                        std20 = df['Close'].rolling(20).std()
                        df['BB_Upper'], df['BB_Lower'] = df['MA20'] + 2 * std20, df['MA20'] - 2 * std20
                        
                        df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
                        df['MACD_Cross'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
                        df['KD_Cross_Recent'] = df['KD_Cross'].rolling(window=config_dict["reso_cross_days"], min_periods=1).max() >= 1
                        df['MACD_Cross_Recent'] = df['MACD_Cross'].rolling(window=config_dict["reso_cross_days"], min_periods=1).max() >= 1

                        # 一鍵呼叫核心 API 算完所有演算法
                        df = core.ScanEngine.calculate_all_phase1_scores(df, config_dict)
                        if scan_target_mode == "指定個股測試":
                            st.session_state.test_stock_df = pd.concat([df, expert_res_df], axis=1)
                            st.session_state.test_ticker_name = stock_dict.get(ticker, ticker)

                        best_combined_score, best_row, best_offset, best_vol_mult = -1, None, 0, 0.0
                        force_pass = (scan_target_mode == "指定個股測試")

                        for offset in range(config_dict["lookback_end"], config_dict["lookback_start"] + 1):
                            if len(df) <= offset + 1: continue
                            t_row, prev_vol = df.iloc[-1 - offset], df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset+1 else np.nan
                            vol_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                            if t_row.get('Vol_MA20', 0) >= config_dict["min_vol_ma20"] or force_pass:
                                is_pass, combo = core.ScanEngine.check_algo_pass(t_row, config_dict, algo_mode, force_pass)
                                if is_pass and (combo > best_combined_score or best_row is None):
                                    best_combined_score, best_row, best_offset, best_vol_mult = combo, t_row, offset, vol_mult

                        count_30d = 0
                        if config_dict["enable_30d_backtest"] and best_row is not None:
                            for b_offset in range(best_offset + 1, best_offset + 31):
                                if len(df) <= b_offset + 1: continue
                                c_row = df.iloc[-1 - b_offset]
                                if c_row.get('Vol_MA20', 0) >= config_dict["min_vol_ma20"] or force_pass:
                                    c_pass, _ = core.ScanEngine.check_algo_pass(c_row, config_dict, algo_mode, force_pass)
                                    if c_pass: count_30d += 1

                        if best_row is not None:
                            last_expert = expert_res_df.iloc[-1 - best_offset]
                            rec_status = "✅ 建議參考" if float(best_row['K']) < config_dict.get("cross_position_threshold", 70) else "⚠️ 位置偏高"

                            reversal_candidates[ticker] = {
                                "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                                "股票代號": ticker.split('.')[0], "股票名稱": stock_dict[ticker],
                                "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                                "近30日觸發次數": count_30d if config_dict["enable_30d_backtest"] else "-",
                                "當日收盤": round(float(best_row['Close']), 2), "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                                "當日成交量(張)": int(best_row['Volume_Lots']), "量增倍數": round(best_vol_mult, 2), "月均量(張)": int(best_row['Vol_MA20']),
                                "DIF": round(float(best_row['MACD']), 3), "KD金叉判定": "✅ 是" if best_row['KD_Cross_Recent'] else "❌ 否",
                                "MACD金叉判定": "✅ 是" if best_row['MACD_Cross_Recent'] else "❌ 否",
                                "反轉分數": round(float(best_row['Candle_Score']), 2), "VCP分數": round(float(best_row['VCP_Score']), 2), "共振分數": round(float(best_row['Reso_Score']), 2),
                                "量價雙背離": "✅ 是" if best_row.get('PriceVol_Div_Pass') else "❌ 否", "長下引突破": "✅ 是" if best_row.get('Breakout_Signal') else "❌ 否",
                                "*回測走揚年線": "❌ 否", "*年線回測分數": 0.0,
                                "進場位置建議": rec_status, "盤整過濾": "✅ 盤整中" if last_expert['RSI盤整'] else "趨勢盤",
                                "建議水位": core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數']),
                                "多週期KD": core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])
                            }
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
                    has_daily_div, has_m60_div = False, False
                    item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"

                    try:
                        if isinstance(long_data.columns, pd.MultiIndex):
                            if ticker in long_data.columns.get_level_values(0): long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                            elif ticker in long_data.columns.get_level_values(1): long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else: long_df = pd.DataFrame()
                        else: long_df = long_data.dropna(how='all') if len(reversal_list) == 1 else pd.DataFrame()

                        if not long_df.empty:
                            if config_dict["use_backtest_date"]: long_df = long_df[long_df.index.tz_localize(None).normalize() <= pd.to_datetime(config_dict["backtest_date_obj"])]
                            if specific_offset > 0: long_df = long_df.iloc[:-specific_offset]

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
                    if not base_tags and scan_target_mode == "指定個股測試": base_tags.append("未達標(測試)")
                    div_tag = " + 雙級別共振" if (has_daily_div and has_m60_div) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if has_m60_div else " (無背離)"))
                    item["演算法建議結果"] = " & ".join(base_tags) + div_tag
                    final_results.append(item)
                except Exception: pass
                progress_bar.progress(min(1.0, (idx + 1) / len(reversal_list)))

        status_text.empty(); progress_bar.empty()
        res_df = pd.DataFrame(final_results)
        
        base_cols = ['股票代號', '股票名稱', '觸發日期']
        if config_dict["enable_30d_backtest"]: base_cols.append('近30日觸發次數')
        base_cols.extend(['演算法建議結果', '*回測走揚年線', '*年線回測分數', '長下引突破', '今日漲跌幅(%)', '當日成交量(張)', '量增倍數', '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', 'KD金叉判定', 'MACD金叉判定', '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)'])
        if config_dict["use_weekly_filter"]: base_cols.extend(['週K共振', '週K背離'])
        base_cols.extend(['年線/半年線空排', '年線斜率(%)', '半年線斜率(%)'])
        
        cols = base_cols + [c for c in res_df.columns if '背離' in c and c not in base_cols]
        if not res_df.empty:
            res_df = res_df[[c for c in cols if c in res_df.columns]].sort_values(by=["*年線回測分數", "共振分數", "反轉分數", "VCP分數"], ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1
        
        st.session_state.scan_res_df, st.session_state.scan_algo_mode, st.session_state.market_info = res_df, algo_mode, market_info
        st.session_state.scan_msg = f"🎉 掃描完成！本次共精選出 **{len(res_df)}** 檔標的。"

        if config_dict.get('auto_upload_gdrive', False) and not res_df.empty:
            webhook, token = st.secrets.get("GAS_WEBHOOK_URL"), st.secrets.get("GAS_SECRET_TOKEN")
            if webhook and token:
                name = f"stock_scan_{algo_mode}_{datetime.date.today()}.csv"
                st.session_state.upload_success, st.session_state.upload_msg = core.GoogleDriveUploader.upload(name, webhook, token, csv_bytes=res_df.to_csv(index=False).encode('utf-8-sig'))
            else: st.session_state.upload_success, st.session_state.upload_msg = False, "未設定 Secrets"

    if "scan_res_df" in st.session_state:
        st.success(st.session_state.scan_msg)
        if st.session_state.market_info:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("加權指數收盤", st.session_state.market_info["加權指數收盤"]); c2.metric("月線 (MA20)", st.session_state.market_info["月線 (MA20)"])
            c3.metric("季線 (MA60)", st.session_state.market_info["季線 (MA60)"]); c4.metric("約當大台基礎", st.session_state.market_info["自動運算基準價值 (約當大台基礎)"])
        
        if not st.session_state.scan_res_df.empty:
            st.dataframe(st.session_state.scan_res_df, use_container_width=True)

# ----------------------------------------------------
# 頁籤 3：財報三表地雷掃描
# ----------------------------------------------------
elif selected_page == "📑 財報三表地雷掃描":
    audit_mode = st.radio("請選擇操作模式", ["單檔查詢", "自選股批次掃描"], horizontal=True)
    if audit_mode == "單檔查詢":
        col1, col2 = st.columns([4, 1])
        with col1: audit_input = st.text_input("輸入單一股票代號", value="2495")
        with col2: st.write(""); st.write(""); audit_btn = st.button("健檢財報", type="primary", use_container_width=True)
        if audit_btn and audit_input:
            res = core.FinancialAuditStrategy.evaluate(core.resolve_ticker(audit_input, get_all_tw_stocks()), core.get_yf_session())
            if res["status"] == "error": st.error(res["msg"])
            else: st.json(res)  # 此處為簡化版顯示，您可以依照原版渲染表格
    else:
        batch_input = st.text_area("輸入自選股清單", height=100)
        if st.button("執行批次交叉健檢", type="primary"):
            stock_dict, session, batch_results = get_all_tw_stocks(), core.get_yf_session(), []
            valid_tickers = [t.strip() for t in re.split(r'[,\s\n]+', batch_input) if t.strip()]
            for t in valid_tickers:
                res = core.FinancialAuditStrategy.evaluate(core.resolve_ticker(t, stock_dict), session)
                if res["status"] == "success": batch_results.append(res)
            if batch_results: st.dataframe(pd.DataFrame(batch_results))

# ----------------------------------------------------
# 頁籤 4：幾何型態滿足點測距
# ----------------------------------------------------
elif selected_page == "📐 幾何型態滿足點測距":
    col1, col2 = st.columns([4, 1])
    with col1: geo_input = st.text_input("輸入股票代號", value="2609")
    with col2: st.write(""); st.write(""); geo_btn = st.button("計算幾何滿足點", type="primary")
    if geo_btn and geo_input:
        df, _ = get_stock_data(geo_input)
        if not df.empty:
            st.json(core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'Close', 'Close'))
            st.json(core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'High', 'Low'))

# ----------------------------------------------------
# 頁籤 5：量化型態狀態機 (程式A)
# ----------------------------------------------------
elif selected_page == "🤖 量化型態狀態機 (程式A)":
    q_input = st.text_input("輸入股票代號", value="2495")
    if st.button("啟動量化狀態機", type="primary"):
        df, _ = get_stock_data(q_input)
        if not df.empty:
            engine = core.QuantPatternEngine(core.QuantConfig())
            df = engine.compute_indicators(engine.clean_and_validate_data(df))
            st.json(engine.analyze_pipeline(df, df['Close'], df['Close']))
