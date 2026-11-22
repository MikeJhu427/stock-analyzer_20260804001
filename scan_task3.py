import os
import io
import glob
import time
import datetime
from copy import copy
import numpy as np
import pandas as pd
import strategy_core as core

CONFIG = core.DEFAULT_CONFIG.copy()
FOREIGN_FLOW_SHEET_NAME = "外資資金流向監控"
FOREIGN_FLOW_HISTORY_FILE = "foreign_fund_flow_history.xlsx"


def _calc_display_width(text) -> int:
    """估算儲存格文字顯示寬度（中文字／全形字計 2，其餘計 1）。"""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text)
    width = 0
    for ch in s:
        width += 2 if ord(ch) > 127 else 1
    return width


def _normalize_pct_ratio(val):
    """將歷史紀錄中的百分比欄位統一正規化為小數比例（例如 -0.35% -> -0.0035）。"""
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        if s in ("", "-"):
            return np.nan
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except Exception:
                return np.nan
        try:
            num = float(s)
            return num / 100.0 if abs(num) > 0.2 else num
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def _load_existing_foreign_flow_history() -> pd.DataFrame:
    """讀取既有 Excel 活頁簿中的「外資資金流向監控」歷史紀錄（若不存在則回傳空表）。"""
    candidates = []
    if os.path.exists(FOREIGN_FLOW_HISTORY_FILE):
        candidates.append(FOREIGN_FLOW_HISTORY_FILE)

    existing_scans = sorted(
        glob.glob("auto_scan_all_*.xlsx"),
        key=lambda p: os.path.getmtime(p),
        reverse=True
    )
    candidates.extend(existing_scans)

    for path in candidates:
        try:
            hist_df = pd.read_excel(path, sheet_name=FOREIGN_FLOW_SHEET_NAME, engine="openpyxl")
            if hist_df is not None and not hist_df.empty and "日期" in hist_df.columns:
                for col in core.FOREIGN_FUND_FLOW_COLUMNS:
                    if col not in hist_df.columns:
                        hist_df[col] = np.nan
                hist_df = hist_df[core.FOREIGN_FUND_FLOW_COLUMNS].copy()
                hist_df["日期"] = hist_df["日期"].astype(str).str.strip()
                hist_df = hist_df[hist_df["日期"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
                for col in ["外資單日買進(億)", "外資單日賣出(億)", "外資單日買賣超(億)", "當月累計買賣超(億)", "USD/TWD匯率"]:
                    hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")
                hist_df["匯率單日變動(%)"] = hist_df["匯率單日變動(%)"].apply(_normalize_pct_ratio)
                if not hist_df.empty:
                    return hist_df
        except Exception:
            continue

    return pd.DataFrame(columns=core.FOREIGN_FUND_FLOW_COLUMNS)


def _merge_foreign_flow_history(flow_res: dict) -> pd.DataFrame:
    """依日期去重後附加（Append & Deduplicate by 日期）保留歷史趨勢。"""
    hist_df = _load_existing_foreign_flow_history()
    new_df = core.build_foreign_fund_flow_df(flow_res, for_excel=True)

    if flow_res.get("status") != "success":
        return hist_df if not hist_df.empty else new_df

    try:
        new_df["日期"] = new_df["日期"].astype(str).str.strip()
        if hist_df.empty:
            merged_df = new_df.copy()
        else:
            merged_df = pd.concat([hist_df, new_df], ignore_index=True)
            merged_df["日期"] = merged_df["日期"].astype(str).str.strip()
            merged_df = merged_df.drop_duplicates(subset=["日期"], keep="last")
            merged_df = merged_df.sort_values(by="日期", ascending=True).reset_index(drop=True)
        return merged_df[core.FOREIGN_FUND_FLOW_COLUMNS]
    except Exception:
        return new_df


def _format_foreign_flow_sheet(writer, sheet_name: str, ref_sheet_name: str = "全市場掃描結果"):
    """維持與原本 Excel 相同字型，並套用數值格式（金額兩位小數、百分比 0.00%）與自動欄寬。"""
    try:
        ws = writer.sheets.get(sheet_name)
        if ws is None:
            return

        ref_ws = writer.sheets.get(ref_sheet_name)
        header_font = copy(ref_ws["A1"].font) if (ref_ws is not None and ref_ws.max_row >= 1) else None
        data_font = copy(ref_ws["A2"].font) if (ref_ws is not None and ref_ws.max_row >= 2) else None

        amount_cols = {
            "外資單日買進(億)",
            "外資單日賣出(億)",
            "外資單日買賣超(億)",
            "當月累計買賣超(億)"
        }
        fx_cols = {"USD/TWD匯率"}
        pct_cols = {"匯率單日變動(%)"}

        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]

        for col_idx, col_name in enumerate(headers, start=1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            max_len = _calc_display_width(col_name)

            if header_font is not None:
                ws.cell(row=1, column=col_idx).font = copy(header_font)

            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if data_font is not None:
                    cell.font = copy(data_font)

                val = cell.value
                display_str = ""
                if col_name in amount_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "#,##0.00"
                    display_str = f"{val:,.2f}"
                elif col_name in fx_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "0.0000"
                    display_str = f"{val:.4f}"
                elif col_name in pct_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "0.00%"
                    display_str = f"{val * 100:.2f}%"
                elif val is not None:
                    display_str = str(val)

                max_len = max(max_len, _calc_display_width(display_str))

            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 45)
    except Exception:
        pass


def _persist_foreign_flow_history(flow_df: pd.DataFrame):
    """將去重後的歷史紀錄同步保存於本地檔案，供後續排程延續趨勢。"""
    try:
        if flow_df is None or flow_df.empty:
            return
        valid_df = flow_df[flow_df["日期"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        if valid_df.empty:
            return
        with pd.ExcelWriter(FOREIGN_FLOW_HISTORY_FILE, engine="openpyxl") as writer:
            valid_df.to_excel(writer, sheet_name=FOREIGN_FLOW_SHEET_NAME, index=False)
            _format_foreign_flow_sheet(writer, FOREIGN_FLOW_SHEET_NAME, ref_sheet_name=FOREIGN_FLOW_SHEET_NAME)
    except Exception:
        pass


def run_full_scan():
    print("🚀 [Step 0] 執行跨市場波動率體制與避險/反彈監控...")
    yf_session = core.get_yf_session()
    try:
        vol_res = core.CrossMarketVolatilityMonitor.evaluate(yf_session)
    except Exception as e:
        vol_res = {"status": "error", "msg": f"波動率監控執行異常：{str(e)}"}

    vol_report_df = core.CrossMarketVolatilityMonitor.to_report_dataframe(vol_res)
    if vol_res.get("status") == "success":
        print(f"🛡️ 市場體制：{vol_res['regime']} | 訊號狀態：{vol_res['state_label']}")
        if vol_res.get("alert_reminders"):
            for rem in vol_res["alert_reminders"]:
                print(f"   {rem}")
        print(f"💡 操作建議：{vol_res['advice']}")
    else:
        print(f"⚠️ 跨市場波動率監控警告：{vol_res.get('msg', '資料取得失敗')}")

    print("🚀 [Step 0.5] 執行外資實質資金流向（匯率＋現貨買賣超連動）監控...")
    try:
        flow_res = core.fetch_foreign_fund_flow()
    except Exception as e:
        flow_res = {"status": "error", "msg": f"外資資金流向監控異常：{str(e)}"}

    try:
        foreign_flow_df = _merge_foreign_flow_history(flow_res)
    except Exception:
        foreign_flow_df = core.build_foreign_fund_flow_df(flow_res, for_excel=True)

    if flow_res.get("status") == "success":
        cum_val = flow_res.get("month_cum_net_yi", np.nan)
        cum_str = f"{cum_val:+.2f} 億" if pd.notna(cum_val) else "-"
        print(
            f"💱 [{flow_res['date']}] 外資單日買賣超：{flow_res['foreign_net_yi']:+.2f} 億 "
            f"(當月累計：{cum_str}) | USD/TWD：{flow_res['fx_rate']:.4f} "
            f"({flow_res['fx_change_pct']:+.2f}% / {flow_res['fx_change_jiao']:+.2f} 角) | "
            f"判定：【{flow_res['flow_signal']}】"
        )
    else:
        print(f"⚠️ 外資資金流向監控警告：{flow_res.get('msg', '資料取得失敗')}")

    print("🚀 [Step 1] 初始化：正在獲取台股上市櫃清單...")
    stock_dict = core.fetch_all_tw_stocks()
    tickers = list(stock_dict.keys())
    print(f"✅ 成功獲取 {len(tickers)} 檔標的，開始執行第一階段篩選...")

    div_pairs = [(CONFIG["div_recent_w"], CONFIG["div_older_w"])] if CONFIG["use_single_div"] else [(5, 20), (5, 60), (20, 60)]
    max_older_w = max(pair[1] for pair in div_pairs)
    extra_buffer_days = 30 if CONFIG["enable_30d_backtest"] else 0
    total_needed_days = CONFIG["lookback_start"] + max_older_w + CONFIG["pivot_left"] + 70 + extra_buffer_days
    dl_period = "3mo" if total_needed_days <= 60 else "6mo" if total_needed_days <= 120 else "1y"
    
    reversal_candidates = {}
    chunk_size = 40

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            import yfinance as yf
            data = yf.download(chunk, period=dl_period, threads=False, progress=False, session=yf_session)
            data_60m = yf.download(chunk, period="1mo", interval="60m", threads=False, progress=False, session=yf_session)
            
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(0): df = data.xs(ticker, axis=1, level=0).dropna(how='all')
                        elif ticker in data.columns.get_level_values(1): df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                        else: df = pd.DataFrame()
                    else: df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()

                    if df.empty or len(df) <= CONFIG["lookback_start"] + 20: continue

                    df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
                    df = core.TechnicalIndicators.add_rsi(df)
                    expert_res_df = core.ExpertTradingStrategy.evaluate(df)

                    df['Pct_Change'] = df['Close'].pct_change() * 100
                    df['Volume_Lots'] = df['Volume'] / 1000
                    df['MA5'] = df['Close'].rolling(5).mean()
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['MA60'] = df['Close'].rolling(60).mean()
                    df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()

                    df = core.ScanEngine.calculate_all_phase1_scores(df, CONFIG)
                    best_combined_score, best_row, best_offset, best_vol_mult = -1, None, 0, 0.0

                    for offset in range(CONFIG["lookback_end"], CONFIG["lookback_start"] + 1):
                        if len(df) <= offset + 1: continue
                        t_row = df.iloc[-1 - offset]
                        prev_vol = df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset + 1 else np.nan
                        vol_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                        if t_row.get('Vol_MA20', 0) >= CONFIG["min_vol_ma20"]:
                            is_pass, combo_score = core.ScanEngine.check_algo_pass(t_row, CONFIG, CONFIG['algo_mode'])
                            if is_pass and (combo_score > best_combined_score or best_row is None):
                                best_combined_score, best_row, best_offset, best_vol_mult = combo_score, t_row, offset, vol_mult

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
                                target_eval_date = df.index[-1 - best_offset].date() if (best_row is not None and best_offset >= 0) else df.index[-1].date()
                                adv_res_60m_target_day = adv_res_60m[adv_res_60m.index.tz_localize(None).date == target_eval_date]
                                if not adv_res_60m_target_day.empty:
                                    triggered_rows = adv_res_60m_target_day[(adv_res_60m_target_day['Adv_Signal_A'] == True) | (adv_res_60m_target_day['Adv_Signal_B'] == True)]
                                    if not triggered_rows.empty:
                                        pass_60m_adv = True
                                        adv_60m_data = triggered_rows.iloc[-1]

                    if (best_row is not None and best_combined_score >= 0) or pass_60m_adv:
                        if best_row is None:
                            best_row = df.iloc[-1]
                            best_offset = 0
                            prev_v = df['Volume_Lots'].iloc[-2] if len(df) > 1 else np.nan
                            best_vol_mult = (best_row['Volume_Lots'] / prev_v) if pd.notna(prev_v) and prev_v > 0 else 0.0

                        count_30d = 0
                        if CONFIG["enable_30d_backtest"]:
                            for b_offset in range(best_offset + 1, min(len(df)-1, best_offset + 31)):
                                c_row = df.iloc[-1 - b_offset]
                                if pd.isna(c_row.get('Vol_MA20')): continue
                                if c_row['Vol_MA20'] >= CONFIG["min_vol_ma20"]:
                                    c_pass, _ = core.ScanEngine.check_algo_pass(c_row, CONFIG, CONFIG['algo_mode'])
                                    if c_pass: count_30d += 1

                        last_expert = expert_res_df.iloc[-1 - best_offset]
                        rec_status = "✅ 建議參考" if float(best_row['K']) < CONFIG.get("cross_position_threshold", 70) else "⚠️ 位置偏高"
                        adv_src = adv_60m_data if pass_60m_adv and adv_60m_data is not None else best_row

                        base_info = {
                            "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                            "_Pass_60m_Adv": pass_60m_adv, "_Adv_60m_Data": adv_60m_data,
                            "股票代號": ticker.split('.')[0], "股票名稱": stock_dict[ticker],
                            "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                            "近30日觸發次數": count_30d if CONFIG["enable_30d_backtest"] else "-",
                            "當日收盤": round(float(best_row['Close']), 2),
                            "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                            "當日成交量(張)": int(best_row['Volume_Lots']),
                            "量增倍數": round(best_vol_mult, 2),
                            "月均量(張)": int(best_row['Vol_MA20']),
                            "DIF": round(float(best_row['MACD']), 3),
                            "當日KD(K值)": round(float(best_row['K']), 2),
                            "進場位置建議": rec_status,
                            "盤整過濾": "✅ 盤整中(建議停用指標)" if last_expert['RSI盤整'] else "趨勢盤",
                            "建議水位": core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數']),
                            "多週期KD": core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])
                        }
                        
                        dynamic_metrics = core.ScanEngine.get_all_metrics(best_row, adv_src)
                        base_info.update(dynamic_metrics)
                        
                        reversal_candidates[ticker] = base_info
                except Exception: continue
        except Exception: pass
        time.sleep(1.0)

    reversal_list = list(reversal_candidates.values())
    print(f"🎯 第一階段篩選完畢，共入選 {len(reversal_list)} 檔標的。")
    if not reversal_list: return print("💡 今日全市場無符合條件標的，任務結束。")

    print("🚀 [Step 2] 正在下載入選標的 2 年資料以運算週K、240MA年線深度評分、背離與實體波段留倉量...")
    candidate_tickers = [item["_Full_Ticker"] for item in reversal_list]
    import yfinance as yf
    long_data = yf.download(candidate_tickers, period="2y", threads=False, progress=False, session=yf_session)

    final_results = []
    for item in reversal_list:
        try:
            ticker, specific_offset, daily_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
            pass_60m_adv, adv_60m_data = item.pop("_Pass_60m_Adv"), item.pop("_Adv_60m_Data")
            has_daily_div, has_m60_div = False, False
            item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"
            item['均線最大最小差異百分比(%)'] = "-"
            item['均線糾結狀態'] = "-"
            item['短天期均線差異(%)'] = "-"
            item['短均線糾結狀態'] = "-"
            item['當沖率(%)'] = 0.0
            item['實體留倉量(張)'] = 0
            item['實體量比'] = 0.0
            item['量能結構標籤'] = "⚪ 一般量能結構"

            long_df = pd.DataFrame()
            try:
                if isinstance(long_data.columns, pd.MultiIndex):
                    if ticker in long_data.columns.get_level_values(0): long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                    elif ticker in long_data.columns.get_level_values(1): long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else: long_df = pd.DataFrame()
                else: long_df = long_data.dropna(how='all') if len(candidate_tickers) == 1 else pd.DataFrame()

                if not long_df.empty:
                    if specific_offset > 0: long_df = long_df.iloc[:-specific_offset]

                    if len(long_df) >= 240:
                        ent_res = core.MA240PullbackRightSideStrategy.evaluate_ma_entanglement(long_df)
                        item['均線最大最小差異百分比(%)'] = ent_res['ma_long_diff_pct']
                        item['均線糾結狀態'] = ent_res['ma_long_tangled']
                        item['短天期均線差異(%)'] = ent_res['ma_short_diff_pct']
                        item['短均線糾結狀態'] = ent_res['ma_short_tangled']

                    if len(long_df) >= 120:
                        ma120 = long_df['Close'].rolling(120).mean()
                        item['半年線斜率(%)'] = round((ma120.iloc[-1] - ma120.iloc[-2]) / ma120.iloc[-2] * 100, 3)
                    if len(long_df) >= 240:
                        ma240 = long_df['Close'].rolling(240).mean()
                        item['年線斜率(%)'] = round((ma240.iloc[-1] - ma240.iloc[-2]) / ma240.iloc[-2] * 100, 3)
                        item['年線/半年線空排'] = "✅ 是" if ma120.iloc[-1] < ma240.iloc[-1] else "❌ 否"
                        
                        ma240_res = core.MA240PullbackRightSideStrategy.evaluate_phase2(long_df)
                        item['*回測走揚年線'] = "✅ 是" if ma240_res["is_pass"] else "❌ 否"
                        item['*年線回測分數'] = ma240_res["score"]

                    if CONFIG["use_weekly_filter"]:
                        wk_res = core.WeeklyAnalysisStrategy.evaluate(long_df, div_pairs, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"], 'Low' if CONFIG["reso_price_basis"] == "最低價 (Low)" else 'Close')
                        item["週K共振"], item["週K背離"] = wk_res["週K共振"], wk_res["週K背離"]
            except Exception:
                if CONFIG["use_weekly_filter"]: item["週K共振"] = item["週K背離"] = "錯誤"

            if not daily_df.empty:
                if specific_offset > 0: daily_df = daily_df.iloc[:-specific_offset]
                for r_w, o_w in div_pairs:
                    d_kd, d_macd = False, False
                    for b in range(0, 25):
                        if len(daily_df) <= b + r_w + o_w: break
                        temp_df = daily_df.iloc[:len(daily_df)-b] if b > 0 else daily_df
                        if not d_kd: d_kd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'K', 'K', 'D', r_w, o_w, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"])
                        if not d_macd: d_macd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"])
                        if d_kd and d_macd: break
                    res_tags = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                    item[f"日K背離({r_w},{o_w})"] = "+".join(res_tags) if res_tags else "無"
                    if res_tags: has_daily_div = True

            # 實體波段留倉量（扣除當日沖銷）後置檢核
            vol_src_df = long_df if (long_df is not None and not long_df.empty) else daily_df
            try:
                rv_res = core.get_latest_real_volume_summary(vol_src_df.tail(60), item["股票代號"])
                item["當沖率(%)"] = rv_res["當沖率(%)"]
                item["實體留倉量(張)"] = rv_res["實體留倉量(張)"]
                item["實體量比"] = rv_res["實體量比"]
                item["量能結構標籤"] = rv_res["量能結構標籤"]
                if rv_res.get("is_fake_vol_trap", False):
                    item["進場位置建議"] = "🚫 假爆量誘多(剔除多訊)"
            except Exception:
                pass

            base_tags = core.ScanEngine.generate_base_tags(item, CONFIG)
            div_tag = " + 雙級別共振" if (has_daily_div and pass_60m_adv) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if pass_60m_adv else " (無背離)"))
            
            if pass_60m_adv and adv_60m_data is not None:
                sig_name = "【🔥 60分K底倉突破】" if adv_60m_data.get('Adv_Signal_A') else "【🔥 60分K右肩回測】"
                base_tags.insert(0, sig_name)

            item["演算法建議結果"] = " & ".join(base_tags) + div_tag if base_tags else "觀察標的" + div_tag
            final_results.append(item)
        except Exception: continue

    res_df = pd.DataFrame(final_results)
    
    base_cols = ['股票代號', '股票名稱', '觸發日期']
    if CONFIG["enable_30d_backtest"]: base_cols.append('近30日觸發次數')
    base_cols.extend([
        '演算法建議結果', '均線最大最小差異百分比(%)', '均線糾結狀態', '短天期均線差異(%)', '短均線糾結狀態'
    ])
    
    base_cols.extend(core.ScanEngine.get_all_metric_keys())
    
    base_cols.extend([
        '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
        '當沖率(%)', '實體留倉量(張)', '實體量比', '量能結構標籤',
        '當日KD(K值)', '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', '當日收盤', '月均量(張)'
    ])
    if CONFIG["use_weekly_filter"]:
        insert_idx = base_cols.index('盤整過濾') + 1
        base_cols.insert(insert_idx, '週K共振')
        base_cols.insert(insert_idx + 1, '週K背離')
    if CONFIG["show_ma120_240_bear_align"]: base_cols.append('年線/半年線空排')
    if CONFIG["show_ma240_slope"]: base_cols.append('年線斜率(%)')
    if CONFIG["show_ma120_slope"]: base_cols.append('半年線斜率(%)')

    cols = base_cols + [c for c in res_df.columns if '背離' in c and c not in base_cols]
    if not res_df.empty:
        res_df = res_df[[c for c in cols if c in res_df.columns]]
        sort_keys = [k for k in ["*年線回測分數", "共振分數", "反轉分數", "VCP分數"] if k in res_df.columns]
        if sort_keys:
            res_df = res_df.sort_values(by=sort_keys, ascending=[False]*len(sort_keys)).reset_index(drop=True)
        res_df.index = res_df.index + 1

        print("📊 掃描完成！最終報表標的預覽：")
        print(res_df[['股票代號', '股票名稱', '演算法建議結果', '均線糾結狀態', '短均線糾結狀態', '量能結構標籤']].to_string())

        # 本地同步保留外資資金流向歷史紀錄（供下次執行去重附加）
        _persist_foreign_flow_history(foreign_flow_df)

        webhook_url = os.environ.get("GAS_WEBHOOK_URL", "").strip()
        token = os.environ.get("GAS_SECRET_TOKEN", "").strip()
        
        if webhook_url and token:
            tw_tz = datetime.timezone(datetime.timedelta(hours=8))
            timestamp = datetime.datetime.now(tw_tz).strftime('%Y-%m-%d %H%M')
            file_name = f"auto_scan_all_{timestamp}.xlsx"
            
            try:
                # 強制採用 openpyxl 輸出三工作表 .xlsx（掃描結果 + 波動率體制監控 + 外資資金流向監控）
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    res_df.to_excel(writer, sheet_name="全市場掃描結果", index=False)
                    vol_report_df.to_excel(writer, sheet_name="波動率體制監控", index=False)
                    try:
                        foreign_flow_df.to_excel(writer, sheet_name=FOREIGN_FLOW_SHEET_NAME, index=False)
                        _format_foreign_flow_sheet(writer, FOREIGN_FLOW_SHEET_NAME, ref_sheet_name="全市場掃描結果")
                    except Exception as fe:
                        print(f"⚠️ 寫入「{FOREIGN_FLOW_SHEET_NAME}」工作表失敗，維持主報表匯出：{fe}")
                buffer.seek(0)
                xlsx_bytes = buffer.read()
                
                print(f"☁️ 正在將報表上傳至 Google Drive: {file_name} ...")
                
                success, msg = core.GoogleDriveUploader.upload(
                    file_name=file_name,
                    webhook_url=webhook_url,
                    token=token,
                    file_bytes=xlsx_bytes
                )
                print(f"✅ 上傳結果: {msg}")

            except ImportError:
                print("⚠️ 系統缺少 Excel 輸出套件 (openpyxl)，將自動降級為 CSV 格式進行上傳。")
                file_name_csv = f"auto_scan_all_{timestamp}.csv"
                csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')
                
                success, msg = core.GoogleDriveUploader.upload(
                    file_name=file_name_csv,
                    webhook_url=webhook_url,
                    token=token,
                    csv_bytes=csv_bytes
                )
                print(f"✅ 上傳結果: {msg}")
        else: 
            print("⚠️ 未檢測到 GAS_WEBHOOK_URL，略過上傳。")
    else:
        print("💡 第一階段入選標的，在經過第二階段條件篩選後無一達標。")

if __name__ == "__main__":
    run_full_scan()
