import os
import io
import time
import datetime
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import strategy_core as core

# ==========================================
# 系統與頁面初始化
# ==========================================
st.set_page_config(page_title="台股量化與基本面分析平台 v6.0", layout="wide")

if "config" not in st.session_state:
    st.session_state.config = core.DEFAULT_CONFIG.copy()
if "scan_res_df" not in st.session_state:
    st.session_state.scan_res_df = pd.DataFrame()
if "scan_msg" not in st.session_state:
    st.session_state.scan_msg = ""
if "yf_session" not in st.session_state:
    st.session_state.yf_session = core.get_yf_session()

@st.cache_data(ttl=86400)
def get_all_stocks():
    return core.fetch_all_tw_stocks()

@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    return core.fetch_stock_data(symbol)

stock_dict = get_all_stocks()

# ==========================================
# 側邊欄：參數設定
# ==========================================
st.sidebar.title("⚙️ 策略參數設定")
algo_mode = st.sidebar.selectbox("執行掃描策略", core.SCAN_ALGO_CHOICES, index=0)
st.session_state.config["algo_mode"] = algo_mode

with st.sidebar.expander("1. 基礎與回測區間設定"):
    st.session_state.config["lookback_start"] = st.number_input("近期區間起點 (0=今天)", min_value=0, max_value=30, value=st.session_state.config["lookback_start"])
    st.session_state.config["lookback_end"] = st.number_input("近期區間終點 (0=今天)", min_value=0, max_value=30, value=st.session_state.config["lookback_end"])
    st.session_state.config["min_vol_ma20"] = st.number_input("最低月均量要求 (張)", min_value=100, max_value=10000, value=st.session_state.config["min_vol_ma20"], step=100)
    st.session_state.config["enable_30d_backtest"] = st.checkbox("啟用近30日觸發次數統計", value=st.session_state.config["enable_30d_backtest"])
    st.session_state.config["auto_upload_gdrive"] = st.checkbox("掃描後自動上傳 Google Drive (.xlsx)", value=st.session_state.config["auto_upload_gdrive"])

with st.sidebar.expander("2. 各策略及格分數門檻"):
    st.session_state.config["min_score"] = st.number_input("底部翻轉 最低及格分數", min_value=1.0, max_value=20.0, value=float(st.session_state.config["min_score"]), step=0.5)
    st.session_state.config["min_vcp_score"] = st.number_input("VCP收斂 最低及格分數", min_value=1.0, max_value=20.0, value=float(st.session_state.config["min_vcp_score"]), step=0.5)
    st.session_state.config["min_reso_score"] = st.number_input("指標共振 最低及格分數", min_value=1.0, max_value=20.0, value=float(st.session_state.config["min_reso_score"]), step=0.5)

with st.sidebar.expander("3. 雙指標共振參數"):
    st.session_state.config["div_recent_w"] = st.number_input("近波低點比較區間 (天)", min_value=3, max_value=20, value=st.session_state.config["div_recent_w"])
    st.session_state.config["div_older_w"] = st.number_input("前波低點比較區間 (天)", min_value=10, max_value=60, value=st.session_state.config["div_older_w"])
    st.session_state.config["reso_kd_older_low"] = st.number_input("前波 KD 需低於", value=float(st.session_state.config["reso_kd_older_low"]))
    st.session_state.config["reso_kd_recent_low"] = st.number_input("近波 KD 需高於", value=float(st.session_state.config["reso_kd_recent_low"]))
    st.session_state.config["reso_price_basis"] = st.selectbox("價格比較基準", ["最低價 (Low)", "收盤價 (Close)"], index=0 if st.session_state.config["reso_price_basis"]=="最低價 (Low)" else 1)

with st.sidebar.expander("4. 扣抵與進階濾網"):
    st.session_state.config["require_cross_confirm"] = st.checkbox("嚴格要求金叉確認背離與共振", value=st.session_state.config["require_cross_confirm"])
    st.session_state.config["use_cross_position_filter"] = st.checkbox("啟用進場位置建議 (K值過高提示)", value=st.session_state.config["use_cross_position_filter"])
    st.session_state.config["use_weekly_filter"] = st.checkbox("啟用週K線共振與底背離判定", value=st.session_state.config["use_weekly_filter"])

# ==========================================
# 主頁面 Tabs
# ==========================================
st.title("📈 台股量化與基本面雙軌分析平台 v6.0")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["單檔深度解析", "全市場智慧掃描", "財報三表健檢", "幾何型態測距", "量化狀態機"])

# ------------------------------------------
# Tab 1: 單檔深度解析
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 標的查詢")
        input_symbol = st.text_input("輸入股票代號 (如 2330)", value="2330")
        if st.button("載入技術圖表"):
            st.session_state.current_ticker = input_symbol
            
    if "current_ticker" in st.session_state:
        ticker = core.resolve_ticker(st.session_state.current_ticker, stock_dict)
        df, actual_ticker = get_stock_data(ticker)
        
        if df.empty:
            st.error(f"找不到標的 {ticker} 的資料。")
        else:
            stock_name = stock_dict.get(actual_ticker, actual_ticker)
            df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['MA120'] = df['Close'].rolling(120).mean()
            df['MA240'] = df['Close'].rolling(240).mean()
            df['Volume_Lots'] = df['Volume'] / 1000
            
            st.markdown(f"### {stock_name} ({actual_ticker.split('.')[0]}) - 綜合量化技術分析")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線", increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='月線(20)', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='季線(60)', line=dict(color='purple', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA240'], name='年線(240)', line=dict(color='brown', width=2, dash='dot')), row=1, col=1)
            
            colors = ['red' if row['Close'] >= row['Open'] else 'green' for i, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume_Lots'], name="成交量(張)", marker_color=colors), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K(9)', line=dict(color='blue', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D(9)', line=dict(color='orange', width=1)), row=3, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 💡 演算法即時判定")
            eval_df = core.ScanEngine.calculate_all_phase1_scores(df.copy(), st.session_state.config)
            latest = eval_df.iloc[-1]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("底部翻轉分數", f"{latest.get('Candle_Score', 0):.2f}")
            c2.metric("VCP收斂分數", f"{latest.get('VCP_Score', 0):.2f}")
            c3.metric("雙指標共振分數", f"{latest.get('Reso_Score', 0):.2f}")
            c4.metric("長下引突破", "✅ 觸發" if latest.get('Breakout_Signal') else "❌ 無")

            ent_res = core.MA240PullbackRightSideStrategy.evaluate_ma_entanglement(df)
            st.markdown("##### 📌 均線糾結狀態")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("長均線最大差異(%)", f"{ent_res['ma_diff_pct']}%" if pd.notna(ent_res['ma_diff_pct']) else "-")
            m2.metric("長均線糾結判定", ent_res['ma_tangled'])
            m3.metric("短均線最大差異(%)", f"{ent_res['ma_short_diff_pct']}%" if pd.notna(ent_res['ma_short_diff_pct']) else "-")
            m4.metric("短均線糾結判定", ent_res['ma_short_tangled'])

# ------------------------------------------
# Tab 2: 全市場智慧掃描
# ------------------------------------------
with tab2:
    st.markdown(f"### 🚀 全市場智慧區間掃描 (目前策略：**{st.session_state.config['algo_mode']}**)")
    
    # === 恢復遺失的：指定個股與回測日期 UI ===
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        scan_target = st.radio("掃描範圍", ["全市場掃描", "自訂標的清單"], horizontal=True)
        custom_tickers = ""
        if scan_target == "自訂標的清單":
            custom_tickers = st.text_input("輸入股票代號 (多檔以逗號分隔，如 2330, 2454)", value="2330, 2454")
    
    with col_s2:
        use_backtest = st.checkbox("啟用時光機回測 (指定基準日)", value=st.session_state.config.get("use_backtest_date", False))
        st.session_state.config["use_backtest_date"] = use_backtest
        backtest_date = st.date_input("回測基準日", value=pd.to_datetime(st.session_state.config.get("backtest_date", datetime.date.today())))
        st.session_state.config["backtest_date"] = str(backtest_date)
    # ==========================================

    if st.button("🚀 開始智慧區間掃描", type="primary"):
        with st.spinner("正在執行全市場兩階段掃描與多級別運算，請稍候... (約需 1-3 分鐘)"):
            cfg = st.session_state.config
            
            # 處理自訂標的
            if scan_target == "自訂標的清單" and custom_tickers.strip():
                raw_tickers = [x.strip() for x in custom_tickers.split(",") if x.strip()]
                tickers = [core.resolve_ticker(t, stock_dict) for t in raw_tickers]
            else:
                tickers = list(stock_dict.keys())
            
            div_pairs = [(cfg["div_recent_w"], cfg["div_older_w"])] if cfg["use_single_div"] else [(5, 20), (5, 60), (20, 60)]
            max_older_w = max(pair[1] for pair in div_pairs)
            extra_buffer_days = 30 if cfg["enable_30d_backtest"] else 0
            total_needed = cfg["lookback_start"] + max_older_w + cfg["pivot_left"] + 70 + extra_buffer_days
            dl_period = "3mo" if total_needed <= 60 else "6mo" if total_needed <= 120 else "1y"
            
            # 處理 yfinance 下載參數 (支援回測基準日截斷)
            dl_kwargs = {"threads": False, "progress": False, "session": st.session_state.yf_session}
            dl_kwargs_60m = {"interval": "60m", "threads": False, "progress": False, "session": st.session_state.yf_session}
            
            if use_backtest:
                end_dt = pd.to_datetime(backtest_date) + datetime.timedelta(days=1)
                dl_kwargs["end"] = end_dt.strftime('%Y-%m-%d')
                dl_kwargs_60m["end"] = end_dt.strftime('%Y-%m-%d')
                
                start_dt = end_dt - datetime.timedelta(days=total_needed + 60) # 寬放交易日
                dl_kwargs["start"] = start_dt.strftime('%Y-%m-%d')
                
                start_60m = end_dt - datetime.timedelta(days=30)
                dl_kwargs_60m["start"] = start_60m.strftime('%Y-%m-%d')
            else:
                dl_kwargs["period"] = dl_period
                dl_kwargs_60m["period"] = "1mo"
            
            reversal_candidates = {}
            chunk_size = 40
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                status_text.text(f"Phase 1: 掃描進度 {i}/{len(tickers)}")
                progress_bar.progress(min(1.0, i / len(tickers)))
                
                try:
                    data = yf.download(chunk, **dl_kwargs)
                    data_60m = yf.download(chunk, **dl_kwargs_60m)
                    
                    for ticker in chunk:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if ticker in data.columns.get_level_values(0): df = data.xs(ticker, axis=1, level=0).dropna(how='all')
                                elif ticker in data.columns.get_level_values(1): df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                                else: df = pd.DataFrame()
                            else: df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                            if df.empty or len(df) <= cfg["lookback_start"] + 20: continue

                            df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
                            df = core.TechnicalIndicators.add_rsi(df)
                            expert_res = core.ExpertTradingStrategy.evaluate(df)

                            df['Pct_Change'] = df['Close'].pct_change() * 100
                            df['Volume_Lots'] = df['Volume'] / 1000
                            df['MA5'] = df['Close'].rolling(5).mean()
                            df['MA20'] = df['Close'].rolling(20).mean()
                            df['MA60'] = df['Close'].rolling(60).mean()
                            df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()

                            df = core.ScanEngine.calculate_all_phase1_scores(df, cfg)
                            best_combined, best_row, best_offset, best_vol_mult = -1, None, 0, 0.0

                            for offset in range(cfg["lookback_end"], cfg["lookback_start"] + 1):
                                if len(df) <= offset + 1: continue
                                t_row = df.iloc[-1 - offset]
                                prev_vol = df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset + 1 else np.nan
                                v_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                                if t_row.get('Vol_MA20', 0) >= cfg["min_vol_ma20"]:
                                    is_pass, combo = core.ScanEngine.check_algo_pass(t_row, cfg, cfg['algo_mode'])
                                    if is_pass and (combo > best_combined or best_row is None):
                                        best_combined, best_row, best_offset, best_vol_mult = combo, t_row, offset, v_mult

                            pass_60m_adv = False
                            adv_60m_data = None
                            if data_60m is not None:
                                if isinstance(data_60m.columns, pd.MultiIndex):
                                    if ticker in data_60m.columns.get_level_values(0): df_60m = data_60m.xs(ticker, axis=1, level=0).dropna(how='all')
                                    elif ticker in data_60m.columns.get_level_values(1): df_60m = data_60m.xs(ticker, axis=1, level=1).dropna(how='all')
                                    else: df_60m = pd.DataFrame()
                                else: df_60m = data_60m.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                                if not df_60m.empty and len(df_60m) >= 60:
                                    df_60m = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df_60m))
                                    df_60m['Volume_Lots'] = df_60m.get('Volume', pd.Series(0)) / 1000.0
                                    df_60m['MA20'] = df_60m['Close'].rolling(20).mean()
                                    df_60m['BB_Lower'] = df_60m['MA20'] - 2 * df_60m['Close'].rolling(20).std()
                                    
                                    adv_res_60m = core.AdvancedBottomStrategy.evaluate(df_60m)
                                    if not adv_res_60m.empty:
                                        t_date = df.index[-1 - best_offset].date() if (best_row is not None and best_offset >= 0) else df.index[-1].date()
                                        adv_target = adv_res_60m[adv_res_60m.index.tz_localize(None).date == t_date]
                                        if not adv_target.empty:
                                            trig = adv_target[(adv_target['Adv_Signal_A'] == True) | (adv_target['Adv_Signal_B'] == True)]
                                            if not trig.empty:
                                                pass_60m_adv = True
                                                adv_60m_data = trig.iloc[-1]

                            if (best_row is not None and best_combined >= 0) or pass_60m_adv:
                                if best_row is None:
                                    best_row = df.iloc[-1]
                                    best_offset = 0
                                    pv = df['Volume_Lots'].iloc[-2] if len(df) > 1 else np.nan
                                    best_vol_mult = (best_row['Volume_Lots'] / pv) if pd.notna(pv) and pv > 0 else 0.0

                                count_30d = 0
                                if cfg["enable_30d_backtest"]:
                                    for b_offset in range(best_offset + 1, min(len(df)-1, best_offset + 31)):
                                        c_row = df.iloc[-1 - b_offset]
                                        if pd.isna(c_row.get('Vol_MA20')): continue
                                        if c_row['Vol_MA20'] >= cfg["min_vol_ma20"]:
                                            c_pass, _ = core.ScanEngine.check_algo_pass(c_row, cfg, cfg['algo_mode'])
                                            if c_pass: count_30d += 1

                                last_exp = expert_res.iloc[-1 - best_offset]
                                adv_src = adv_60m_data if pass_60m_adv and adv_60m_data is not None else best_row
                                rec_status = "✅ 建議參考" if float(best_row['K']) < cfg.get("cross_position_threshold", 70) else "⚠️ 位置偏高"

                                b_info = {
                                    "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                                    "_Pass_60m_Adv": pass_60m_adv, "_Adv_60m_Data": adv_60m_data,
                                    "股票代號": ticker.split('.')[0], "股票名稱": stock_dict.get(ticker, ticker),
                                    "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                                    "近30日觸發次數": count_30d if cfg["enable_30d_backtest"] else "-",
                                    "當日收盤": round(float(best_row['Close']), 2),
                                    "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                                    "當日成交量(張)": int(best_row['Volume_Lots']),
                                    "量增倍數": round(best_vol_mult, 2),
                                    "月均量(張)": int(best_row['Vol_MA20']),
                                    "DIF": round(float(best_row['MACD']), 3),
                                    "當日KD(K值)": round(float(best_row['K']), 2),
                                    "進場位置建議": rec_status,
                                    "盤整過濾": "✅ 盤整中(建議停用指標)" if last_exp['RSI盤整'] else "趨勢盤",
                                    "建議水位": core.ExpertTradingStrategy.get_position_advice(last_exp['多空指標數']),
                                    "多週期KD": core.ExpertTradingStrategy.get_multi_tf_advice(last_exp['週KD多'], last_exp['日KD多'])
                                }
                                b_info.update(core.ScanEngine.get_all_metrics(best_row, adv_src))
                                reversal_candidates[ticker] = b_info
                        except Exception: continue
                except Exception: pass
            
            progress_bar.progress(1.0)
            status_text.text("Phase 1 完成！準備進入 Phase 2 深度複審...")
            
            reversal_list = list(reversal_candidates.values())
            
            if reversal_list:
                c_tickers = [item["_Full_Ticker"] for item in reversal_list]
                
                long_dl_kwargs = {"threads": False, "progress": False, "session": st.session_state.yf_session}
                if use_backtest:
                    end_dt = pd.to_datetime(backtest_date) + datetime.timedelta(days=1)
                    long_dl_kwargs["start"] = (end_dt - datetime.timedelta(days=750)).strftime('%Y-%m-%d')
                    long_dl_kwargs["end"] = end_dt.strftime('%Y-%m-%d')
                else:
                    long_dl_kwargs["period"] = "2y"
                    
                long_data = yf.download(c_tickers, **long_dl_kwargs)
                
                final_results = []
                for item in reversal_list:
                    try:
                        ticker, s_offset, d_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
                        pass_60m_adv, adv_60m_data = item.pop("_Pass_60m_Adv"), item.pop("_Adv_60m_Data")
                        has_d_div = False
                        
                        item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"
                        item['均線最大最小差異百分比(%)'] = "-"
                        item['均線糾結狀態'] = "-"
                        item['短均線最大最小差異百分比(%)'] = "-"
                        item['短均線糾結狀態'] = "-"

                        try:
                            if isinstance(long_data.columns, pd.MultiIndex):
                                if ticker in long_data.columns.get_level_values(0): l_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                                elif ticker in long_data.columns.get_level_values(1): l_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                                else: l_df = pd.DataFrame()
                            else: l_df = long_data.dropna(how='all') if len(c_tickers) == 1 else pd.DataFrame()

                            if not l_df.empty:
                                if s_offset > 0: l_df = l_df.iloc[:-s_offset]
                                
                                if len(l_df) >= 240:
                                    ent_res = core.MA240PullbackRightSideStrategy.evaluate_ma_entanglement(l_df)
                                    item['均線最大最小差異百分比(%)'] = ent_res['ma_diff_pct']
                                    item['均線糾結狀態'] = ent_res['ma_tangled']
                                    item['短均線最大最小差異百分比(%)'] = ent_res['ma_short_diff_pct']
                                    item['短均線糾結狀態'] = ent_res['ma_short_tangled']

                                if len(l_df) >= 120:
                                    ma120 = l_df['Close'].rolling(120).mean()
                                    item['半年線斜率(%)'] = round((ma120.iloc[-1] - ma120.iloc[-2]) / ma120.iloc[-2] * 100, 3)
                                if len(l_df) >= 240:
                                    ma240 = l_df['Close'].rolling(240).mean()
                                    item['年線斜率(%)'] = round((ma240.iloc[-1] - ma240.iloc[-2]) / ma240.iloc[-2] * 100, 3)
                                    item['年線/半年線空排'] = "✅ 是" if ma120.iloc[-1] < ma240.iloc[-1] else "❌ 否"
                                    
                                    m240_res = core.MA240PullbackRightSideStrategy.evaluate_phase2(l_df)
                                    item['*回測走揚年線'] = "✅ 是" if m240_res["is_pass"] else "❌ 否"
                                    item['*年線回測分數'] = m240_res["score"]

                                if cfg["use_weekly_filter"]:
                                    wk_res = core.WeeklyAnalysisStrategy.evaluate(l_df, div_pairs, cfg["recent_lows_cnt"], cfg["older_lows_cnt"], cfg["pivot_left"], cfg["pivot_right"], cfg["require_cross_confirm"], cfg["reso_cross_days"], 'Low' if cfg["reso_price_basis"] == "最低價 (Low)" else 'Close')
                                    item["週K共振"], item["週K背離"] = wk_res["週K共振"], wk_res["週K背離"]
                        except Exception:
                            if cfg["use_weekly_filter"]: item["週K共振"] = item["週K背離"] = "錯誤"

                        if not d_df.empty:
                            if s_offset > 0: d_df = d_df.iloc[:-s_offset]
                            for r_w, o_w in div_pairs:
                                dk, dm = False, False
                                for b in range(0, 25):
                                    if len(d_df) <= b + r_w + o_w: break
                                    t_df = d_df.iloc[:len(d_df)-b] if b > 0 else d_df
                                    if not dk: dk = core.DivergenceStrategy.check_bottom_divergence(t_df, 'Low', 'K', 'K', 'D', r_w, o_w, cfg["recent_lows_cnt"], cfg["older_lows_cnt"], cfg["pivot_left"], cfg["pivot_right"], cfg["require_cross_confirm"], cfg["reso_cross_days"])
                                    if not dm: dm = core.DivergenceStrategy.check_bottom_divergence(t_df, 'Low', 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, cfg["recent_lows_cnt"], cfg["older_lows_cnt"], cfg["pivot_left"], cfg["pivot_right"], cfg["require_cross_confirm"], cfg["reso_cross_days"])
                                    if dk and dm: break
                                r_tags = [x for x, b in zip(["KD", "MACD"], [dk, dm]) if b]
                                item[f"日K背離({r_w},{o_w})"] = "+".join(r_tags) if r_tags else "無"
                                if r_tags: has_d_div = True

                        b_tags = core.ScanEngine.generate_base_tags(item, cfg)
                        d_tag = " + 雙級別共振" if (has_d_div and pass_60m_adv) else (" + 日K背離" if has_d_div else (" + 60分K背離" if pass_60m_adv else " (無背離)"))
                        
                        if pass_60m_adv and adv_60m_data is not None:
                            s_name = "【🔥 60分K底倉突破】" if adv_60m_data.get('Adv_Signal_A') else "【🔥 60分K右肩回測】"
                            b_tags.insert(0, s_name)

                        item["演算法建議結果"] = " & ".join(b_tags) + d_tag if b_tags else "觀察標的" + d_tag
                        final_results.append(item)
                    except Exception: continue

                res_df = pd.DataFrame(final_results)
                
                b_cols = ['股票代號', '股票名稱', '觸發日期']
                if cfg["enable_30d_backtest"]: b_cols.append('近30日觸發次數')
                
                b_cols.extend([
                    '演算法建議結果', 
                    '均線最大最小差異百分比(%)', '均線糾結狀態', 
                    '短均線最大最小差異百分比(%)', '短均線糾結狀態'
                ])
                
                b_cols.extend(core.ScanEngine.get_all_metric_keys())
                b_cols.extend([
                    '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
                    '當日KD(K值)', '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', '當日收盤', '月均量(張)'
                ])
                if cfg["use_weekly_filter"]:
                    i_idx = b_cols.index('盤整過濾') + 1
                    b_cols.insert(i_idx, '週K共振')
                    b_cols.insert(i_idx + 1, '週K背離')
                if cfg["show_ma120_240_bear_align"]: b_cols.append('年線/半年線空排')
                if cfg["show_ma240_slope"]: b_cols.append('年線斜率(%)')
                if cfg["show_ma120_slope"]: b_cols.append('半年線斜率(%)')

                cols = b_cols + [c for c in res_df.columns if '背離' in c and c not in b_cols]
                res_df = res_df[[c for c in cols if c in res_df.columns]]
                
                s_keys = [k for k in ["*年線回測分數", "共振分數", "反轉分數", "VCP分數"] if k in res_df.columns]
                if s_keys:
                    res_df = res_df.sort_values(by=s_keys, ascending=[False]*len(s_keys)).reset_index(drop=True)
                res_df.index = res_df.index + 1
                
                st.session_state.scan_res_df = res_df
                st.session_state.scan_msg = f"🎉 掃描完成！本次共精選出 **{len(res_df)}** 檔標的。"
                
                # 執行 Google Drive 自動上傳
                if cfg.get("auto_upload_gdrive", False) and not res_df.empty:
                    webhook_url = ""
                    token = ""
                    try:
                        webhook_url = st.secrets.get("GAS_WEBHOOK_URL", os.environ.get("GAS_WEBHOOK_URL", ""))
                        token = st.secrets.get("GAS_SECRET_TOKEN", os.environ.get("GAS_SECRET_TOKEN", ""))
                    except Exception:
                        pass
                        
                    if webhook_url and token:
                        tw_tz = datetime.timezone(datetime.timedelta(hours=8))
                        timestamp = datetime.datetime.now(tw_tz).strftime('%Y-%m-%d %H%M')
                        try:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                res_df.to_excel(writer, index=False)
                            buffer.seek(0)
                            xlsx_bytes = buffer.read()
                            
                            file_name = f"auto_scan_all_{timestamp}.xlsx"
                            success, msg = core.GoogleDriveUploader.upload(
                                file_name=file_name,
                                webhook_url=webhook_url,
                                token=token,
                                file_bytes=xlsx_bytes
                            )
                            if success:
                                st.success(f"✅ Google Drive 上傳成功 (.xlsx)")
                            else:
                                st.warning(f"⚠️ Google Drive 上傳失敗：{msg}")
                        except ImportError:
                            st.warning("⚠️ 系統缺少 openpyxl，降級為 CSV 格式上傳。")
                            csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')
                            file_name = f"auto_scan_all_{timestamp}.csv"
                            success, msg = core.GoogleDriveUploader.upload(file_name, webhook_url, token, csv_bytes=csv_bytes)
                            if not success:
                                st.warning(f"⚠️ Google Drive 上傳失敗：{msg}")
            else:
                st.session_state.scan_res_df = pd.DataFrame()
                st.session_state.scan_msg = "💡 掃描完成，無符合條件標的。"
                
            status_text.empty()

    if st.session_state.scan_msg:
        st.success(st.session_state.scan_msg)
    if not st.session_state.scan_res_df.empty:
        st.dataframe(st.session_state.scan_res_df, use_container_width=True)

# ------------------------------------------
# Tab 3: 財報三表健檢
# ------------------------------------------
with tab3:
    st.markdown("### 📑 財報三表交叉健檢 V5 (合約負債防誤判引擎)")
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        f_input = st.text_area("輸入股票代號 (支援多檔批次，如 2330, 2454, 2382)", value="2330")
        if st.button("執行基本面健檢"):
            st.session_state.f_targets = [x.strip() for x in f_input.split(",") if x.strip()]
            
    if "f_targets" in st.session_state and st.session_state.f_targets:
        f_results = []
        with st.spinner("正在執行財報交叉診斷..."):
            for sym in st.session_state.f_targets:
                ticker = core.resolve_ticker(sym, stock_dict)
                res = core.FinancialAuditStrategy.evaluate(ticker, st.session_state.yf_session)
                if res.get("status") == "success":
                    f_results.append(res)
                else:
                    st.error(res.get("msg", f"{sym} 診斷失敗"))
                    
        if f_results:
            st.markdown("#### 📊 健檢總覽")
            summary_df = pd.DataFrame([{
                "代號": r["代號"].split('.')[0],
                "綜合評級": r["綜合評級"],
                "體質分數": r["體質分數"],
                "風險分數": r["風險分數"],
                "最新季報": r["當期季報"],
                "正向特徵": r["正向訊號"],
                "警訊數": r["警訊數"]
            } for r in f_results])
            st.dataframe(summary_df, use_container_width=True)
            
            for res in f_results:
                with st.expander(f"📖 {res['代號']} 詳細診斷報告 ({res['綜合評級']})"):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("體質護城河分數", res['體質分數'])
                    m2.metric("財務地雷風險分數", res['風險分數'])
                    m3.metric("合約負債動能", res['cl_status'])
                    
                    st.markdown("##### 📌 核心財務指標追蹤")
                    core_df = pd.DataFrame(res["core_metrics"])
                    st.table(core_df.drop(columns=["4Q趨勢"]))
                    
                    if res["risk_resonance"] and res["risk_resonance"][0] != "目前三表結構中，無重大風險共振鏈。":
                        st.error("**🚨 風險共振警訊**\n\n" + "\n".join([f"- {x}" for x in res["risk_resonance"]]))
                    if res["positives"] and res["positives"][0] != "無顯著突出之護城河優勢。":
                        st.success("**🛡️ 護城河與正向特徵**\n\n" + "\n".join([f"- {x}" for x in res["positives"]]))
                    
                    st.info("**👀 下一步追蹤建議**\n\n" + "\n".join([f"- {x}" for x in res["watch_list"]]))

# ------------------------------------------
# Tab 4: 幾何型態測距
# ------------------------------------------
with tab4:
    st.markdown("### 📐 波段幾何型態與滿足點測距")
    g_input = st.text_input("分析標的", value="2330", key="geo_input")
    if st.button("計算對稱與費氏滿足點"):
        ticker = core.resolve_ticker(g_input, stock_dict)
        df, actual_ticker = get_stock_data(ticker)
        if not df.empty:
            res_close = core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'Close', 'Close')
            res_shadow = core.GeometricPatternStrategy.calculate_geometric_patterns(df, 'High', 'Low')
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 實體收盤價模型 (過濾盤中突刺)")
                if res_close["bull"]:
                    b = res_close["bull"]
                    st.success(f"**W底突破判定**\n- 左腳: {b['trough1']:.2f} ({b['date1']})\n- 右腳: {b['trough2']:.2f} ({b['date2']})\n- 頸線: {b['neckline']:.2f}\n- 1:1 滿足點: **{b['target_1to1']:.2f}**\n- 1.618 延伸: **{b['target_1618']:.2f}**")
                else:
                    st.info("未偵測到實體 W 底型態")
            with c2:
                st.markdown("#### 影線極值模型 (涵蓋盤中踩踏)")
                if res_shadow["bull"]:
                    b = res_shadow["bull"]
                    st.success(f"**W底突破判定**\n- 左腳: {b['trough1']:.2f} ({b['date1']})\n- 右腳: {b['trough2']:.2f} ({b['date2']})\n- 頸線: {b['neckline']:.2f}\n- 1:1 滿足點: **{b['target_1to1']:.2f}**\n- 1.618 延伸: **{b['target_1618']:.2f}**")
                else:
                    st.info("未偵測到影線 W 底型態")
        else:
            st.error("查無資料")

# ------------------------------------------
# Tab 5: 量化狀態機
# ------------------------------------------
with tab5:
    st.markdown("### 🤖 量化型態狀態機 (動態波段位階過濾)")
    q_input = st.text_input("分析標的", value="2330", key="quant_input")
    if st.button("啟動狀態機演化推論"):
        ticker = core.resolve_ticker(q_input, stock_dict)
        df, actual_ticker = get_stock_data(ticker)
        if not df.empty:
            engine = core.QuantPatternEngine(core.QuantConfig())
            df_clean = engine.clean_and_validate_data(df)
            df_ind = engine.compute_indicators(df_clean)
            
            patterns = engine.analyze_pipeline(df_ind, df_ind['High'], df_ind['Low'])
            channel = engine.identify_diagonal_channel(df_ind, df_ind['High'], df_ind['Low'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 水平對稱型態 (W底/M頭)")
                w_pat = patterns.get('W')
                if w_pat:
                    st.success(f"**W底狀態**: {w_pat.state}\n- 頸線: {w_pat.neckline.price:.2f}\n- 位階: {w_pat.regime_note}\n- 防守線: {w_pat.invalidation_level:.2f}")
                else:
                    st.info("無有效 W 底型態")
                    
                m_pat = patterns.get('M')
                if m_pat:
                    st.error(f"**M頭狀態**: {m_pat.state}\n- 頸線: {m_pat.neckline.price:.2f}\n- 位階: {m_pat.regime_note}")
                else:
                    st.info("無有效 M 頭型態")
            with c2:
                st.markdown("#### 斜向趨勢軌道 (通道/楔形/三角收斂)")
                if channel:
                    st.warning(f"**型態判定**: {channel.channel_type}\n**當前狀態**: {channel.state}\n- 上軌壓力: {channel.proj_upper:.2f}\n- 下軌支撐: {channel.proj_lower:.2f}")
                else:
                    st.info("無明顯斜向軌道特徵")
        else:
            st.error("查無資料")
