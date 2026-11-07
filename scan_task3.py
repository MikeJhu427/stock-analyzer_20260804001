import os
import time
import datetime
import pandas as pd
import strategy_core as core

# 使用 Core 預設參數 (可在此腳本環境覆寫)
CONFIG = core.DEFAULT_CONFIG.copy()

def run_full_scan():
    print("🚀 [Step 1] 初始化：正在獲取台股上市櫃清單...")
    yf_session = core.get_yf_session()
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

                    # 一鍵完成所有核心策略的 Phase 1 計算
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

                    count_30d = 0
                    if CONFIG["enable_30d_backtest"] and best_row is not None:
                        for b_offset in range(best_offset + 1, best_offset + 31):
                            if len(df) <= b_offset + 1: continue
                            c_row = df.iloc[-1 - b_offset]
                            if pd.isna(c_row.get('Vol_MA20')): continue
                            if c_row['Vol_MA20'] >= CONFIG["min_vol_ma20"]:
                                c_pass, _ = core.ScanEngine.check_algo_pass(c_row, CONFIG, CONFIG['algo_mode'])
                                if c_pass: count_30d += 1

                    if best_row is not None:
                        last_expert = expert_res_df.iloc[-1 - best_offset]
                        rec_status = "-"
                        if CONFIG["use_cross_position_filter"]:
                            rec_status = "✅ 建議參考" if float(best_row['K']) < CONFIG["cross_position_threshold"] else "⚠️ 位置偏高"

                        reversal_candidates[ticker] = {
                            "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                            "股票代號": ticker.split('.')[0], "股票名稱": stock_dict[ticker],
                            "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                            "近30日觸發次數": count_30d if CONFIG["enable_30d_backtest"] else "-",
                            "當日收盤": round(float(best_row['Close']), 2),
                            "今日漲跌幅(%)": round(float(best_row['Pct_Change']), 2),
                            "當日成交量(張)": int(best_row['Volume_Lots']),
                            "量增倍數": round(best_vol_mult, 2),
                            "月均量(張)": int(best_row['Vol_MA20']),
                            "DIF": round(float(best_row['MACD']), 3),
                            "反轉分數": round(float(best_row['Candle_Score']), 2),
                            "VCP分數": round(float(best_row['VCP_Score']), 2),
                            "共振分數": round(float(best_row['Reso_Score']), 2),
                            "量價雙背離": "✅ 是" if best_row.get('PriceVol_Div_Pass') else "❌ 否",
                            "長下引突破": "✅ 是" if best_row.get('Breakout_Signal') else "❌ 否",
                            "*回測走揚年線": "❌ 否", "*年線回測分數": 0.0,
                            "進場位置建議": rec_status,
                            "盤整過濾": "✅ 盤整中(建議停用指標)" if last_expert['RSI盤整'] else "趨勢盤",
                            "建議水位": core.ExpertTradingStrategy.get_position_advice(last_expert['多空指標數']),
                            "多週期KD": core.ExpertTradingStrategy.get_multi_tf_advice(last_expert['週KD多'], last_expert['日KD多'])
                        }
                except Exception: continue
        except Exception: pass
        time.sleep(1.0)

    reversal_list = list(reversal_candidates.values())
    print(f"🎯 第一階段篩選完畢，共入選 {len(reversal_list)} 檔標的。")
    if not reversal_list: return print("💡 今日全市場無符合條件標的，任務結束。")

    print("🚀 [Step 2] 正在下載入選標的 2 年資料以運算週K、240MA年線深度評分與背離...")
    candidate_tickers = [item["_Full_Ticker"] for item in reversal_list]
    import yfinance as yf
    long_data = yf.download(candidate_tickers, period="2y", threads=False, progress=False, session=yf_session)

    final_results = []
    for item in reversal_list:
        try:
            ticker, specific_offset, daily_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
            has_daily_div, has_m60_div = False, False
            item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"

            try:
                if isinstance(long_data.columns, pd.MultiIndex):
                    if ticker in long_data.columns.get_level_values(0): long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                    elif ticker in long_data.columns.get_level_values(1): long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else: long_df = pd.DataFrame()
                else: long_df = long_data.dropna(how='all') if len(candidate_tickers) == 1 else pd.DataFrame()

                if not long_df.empty:
                    if specific_offset > 0: long_df = long_df.iloc[:-specific_offset]

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

            try:
                time.sleep(0.5)
                m60_df = yf.Ticker(ticker, session=yf_session).history(interval="60m", period="730d")
                if specific_offset > 0 and not daily_df.empty:
                    target_date = daily_df.index[-1].date()
                    m60_df = m60_df[[d.date() <= target_date for d in m60_df.index]]
                if not m60_df.empty:
                    m60_df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(m60_df))
                    for r_w, o_w in div_pairs:
                        d_kd, d_macd = False, False
                        for b in range(0, 20):
                            if len(m60_df) <= b + r_w + o_w: break
                            temp_df = m60_df.iloc[:len(m60_df)-b] if b > 0 else m60_df
                            if not d_kd: d_kd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'K', 'K', 'D', r_w, o_w, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"])
                            if not d_macd: d_macd = core.DivergenceStrategy.check_bottom_divergence(temp_df, 'Low', 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, CONFIG["recent_lows_cnt"], CONFIG["older_lows_cnt"], CONFIG["pivot_left"], CONFIG["pivot_right"], CONFIG["require_cross_confirm"], CONFIG["reso_cross_days"])
                            if d_kd and d_macd: break
                        res_tags = [x for x, b in zip(["KD", "MACD"], [d_kd, d_macd]) if b]
                        item[f"60分K背離({r_w},{o_w})"] = "+".join(res_tags) if res_tags else "無"
                        if res_tags: has_m60_div = True
                else:
                    for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"
            except Exception:
                for r_w, o_w in div_pairs: item[f"60分K背離({r_w},{o_w})"] = "無資料"

            base_tags = core.ScanEngine.generate_base_tags(item, CONFIG)
            div_tag = " + 雙級別共振" if (has_daily_div and has_m60_div) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if has_m60_div else " (無背離)"))
            item["演算法建議結果"] = " & ".join(base_tags) + div_tag if base_tags else "觀察標的" + div_tag
            final_results.append(item)
        except Exception: continue

    res_df = pd.DataFrame(final_results)
    base_cols = ['股票代號', '股票名稱', '觸發日期']
    if CONFIG["enable_30d_backtest"]: base_cols.append('近30日觸發次數')
    base_cols.extend([
        '演算法建議結果', '*回測走揚年線', '*年線回測分數', '長下引突破', '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
        '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)'
    ])
    if CONFIG["use_weekly_filter"]:
        insert_idx = base_cols.index('盤整過濾') + 1
        base_cols.insert(insert_idx, '週K共振')
        base_cols.insert(insert_idx + 1, '週K背離')
    if CONFIG["show_ma120_240_bear_align"]: base_cols.append('年線/半年線空排')
    if CONFIG["show_ma240_slope"]: base_cols.append('年線斜率(%)')
    if CONFIG["show_ma120_slope"]: base_cols.append('半年線斜率(%)')

    cols = base_cols + [c for c in res_df.columns if '背離' in c and c not in base_cols]
    res_df = res_df[cols].sort_values(by=["*年線回測分數", "共振分數", "反轉分數", "VCP分數"], ascending=[False, False, False, False]).reset_index(drop=True)
    res_df.index = res_df.index + 1

    print("📊 掃描完成！最終報表標的預覽：")
    print(res_df[['股票代號', '股票名稱', '演算法建議結果', '*年線回測分數', '當日收盤', '量增倍數']].to_string())

    webhook_url = os.environ.get("GAS_WEBHOOK_URL", "").strip()
    token = os.environ.get("GAS_SECRET_TOKEN", "").strip()
    if webhook_url and token:
        file_name = f"auto_scan_all_{datetime.date.today().strftime('%Y-%m-%d')}.csv"
        csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')
        print(f"☁️ 正在將報表上傳至 Google Drive: {file_name} ...")
        success, msg = core.GoogleDriveUploader.upload(file_name, webhook_url, token, csv_bytes=csv_bytes)
        print(f"✅ 上傳結果: {msg}")
    else: print("⚠️ 未檢測到 GAS_WEBHOOK_URL，略過上傳。")

if __name__ == "__main__":
    run_full_scan()
