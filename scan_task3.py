import os
import io
import glob
import time
import datetime
from copy import copy
import numpy as np
import pandas as pd
import strategy_core as core
import ma_squeeze_strategy as msq

CONFIG = core.DEFAULT_CONFIG.copy()
MAIN_SCAN_SHEET_NAME = "全市場掃描結果"
FOREIGN_FLOW_SHEET_NAME = "外資資金流向監控"
FOREIGN_FLOW_HISTORY_FILE = "foreign_fund_flow_history.xlsx"
SAFE_HAVEN_SHEET_NAME = "空頭避風港_方案B"


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


def _build_scan_output_columns(config: dict, div_pairs: list, existing_cols=None) -> list:
    """組合主掃描報表的完整欄位順序（既有欄位＋獨立均線壓縮欄位＋背離欄位＋威科夫 5 項欄位＋8 項實體量新增欄位），確保 0 檔空表結構一致。"""
    base_cols = ['股票代號', '股票名稱', '觸發日期']
    if config.get("enable_30d_backtest", False):
        base_cols.append('近30日觸發次數')
    base_cols.extend([
        '演算法建議結果', '均線最大最小差異百分比(%)', '均線糾結狀態', '短天期均線差異(%)', '短均線糾結狀態'
    ])
    base_cols.extend(core.ScanEngine.get_all_metric_keys())
    for col in msq.MA_SQUEEZE_COLUMNS:
        if col not in base_cols:
            base_cols.append(col)
    base_cols.extend([
        '今日漲跌幅(%)', '當日成交量(張)', '量增倍數',
        '當日KD(K值)', '進場位置建議', '建議水位', '多週期KD', '盤整過濾', 'DIF', '當日收盤', '月均量(張)'
    ])
    if config.get("use_weekly_filter", False):
        insert_idx = base_cols.index('盤整過濾') + 1
        base_cols.insert(insert_idx, '週K共振')
        base_cols.insert(insert_idx + 1, '週K背離')
    if config.get("show_ma120_240_bear_align", False):
        base_cols.append('年線/半年線空排')
    if config.get("show_ma240_slope", False):
        base_cols.append('年線斜率(%)')
    if config.get("show_ma120_slope", False):
        base_cols.append('半年線斜率(%)')

    div_cols = [f"日K背離({r_w},{o_w})" for r_w, o_w in div_pairs]
    if existing_cols is not None:
        extra_div_cols = [c for c in existing_cols if '背離' in c and c not in base_cols and c not in div_cols]
        div_cols.extend(extra_div_cols)

    all_cols = base_cols + [c for c in div_cols if c not in base_cols]
    for col in core.WYCKOFF_COLUMNS:
        if col not in all_cols:
            all_cols.append(col)
    for col in core.REAL_TURNOVER_COLUMNS:
        if col not in all_cols:
            all_cols.append(col)
    return all_cols


def _format_main_scan_sheet(writer, sheet_name: str = MAIN_SCAN_SHEET_NAME):
    """為「全市場掃描結果」工作表套用威科夫、實體量與均線壓縮突破數值格式及自動欄寬調整。"""
    try:
        ws = writer.sheets.get(sheet_name)
        if ws is None:
            return

        two_dec_cols = {
            "均線壓縮極差(%)",
            "壓縮布林帶寬(%)",
            "突破量比(5MA)",
            "威科夫TR支撐價",
            "威科夫TR壓力價",
            "當沖張數",
            "當沖率(%)",
            "實體留倉量(張)",
            "實體20日均量(張)",
            "實體量比"
        }

        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        for col_idx, col_name in enumerate(headers, start=1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            max_len = _calc_display_width(col_name)

            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                val = cell.value
                display_str = ""
                if col_name in two_dec_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "#,##0.00"
                    display_str = f"{val:,.2f}"
                elif val is not None:
                    display_str = str(val)
                max_len = max(max_len, _calc_display_width(display_str))

            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 50)
    except Exception:
        pass


def _format_foreign_flow_sheet(writer, sheet_name: str, ref_sheet_name: str = MAIN_SCAN_SHEET_NAME):
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


def _format_safe_haven_sheet(writer, sheet_name: str, ref_sheet_name: str = MAIN_SCAN_SHEET_NAME):
    """為「空頭避風港_方案B」工作表套用統一字型、數值格式與自動欄寬。"""
    try:
        ws = writer.sheets.get(sheet_name)
        if ws is None:
            return

        ref_ws = writer.sheets.get(ref_sheet_name)
        header_font = copy(ref_ws["A1"].font) if (ref_ws is not None and ref_ws.max_row >= 1) else None
        data_font = copy(ref_ws["A2"].font) if (ref_ws is not None and ref_ws.max_row >= 2) else None

        two_dec_cols = {
            "最新收盤價",
            "60MA季線價",
            "10MA移動停利參考價",
            "距60MA乖離率(%)",
            "15日超額報酬RS(%)"
        }
        one_dec_cols = {"5日均量(張)"}

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
                if col_name in two_dec_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "#,##0.00"
                    display_str = f"{val:,.2f}"
                elif col_name in one_dec_cols and isinstance(val, (int, float)) and pd.notna(val):
                    cell.number_format = "#,##0.0"
                    display_str = f"{val:,.1f}"
                elif val is not None:
                    display_str = str(val)

                max_len = max(max_len, _calc_display_width(display_str))

            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 55)
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


def _run_plan_b_safe_haven(stock_dict: dict, preloaded_daily_map: dict, yf_session) -> dict:
    """
    執行方案 B（空頭避風港強勢股）盤後掃描步驟：
    1. 先檢查加權指數 (^TWII) 條件 1 與條件 2 閘門；未同時達標則印出鎖定訊息並跳過個股掃描。
    2. 若通過閘門，優先複用 Step 1 已下載日線資料執行全市場普通股掃描，印出排序結果並匯出本地 UTF-8-SIG CSV。
    """
    print("🚀 [Step 3] 執行空頭避風港強勢股（方案 B）大盤閘門檢核與盤後掃描...")
    try:
        gate_info = core.BearMarketSafeHavenStrategy.evaluate_market_gate(session=yf_session)
    except Exception as e:
        err_msg = f"方案 B 大盤閘門檢核異常：{str(e)}"
        print(f"⚠️ {err_msg}")
        return {
            "status": "error",
            "gate_passed": False,
            "message": err_msg,
            "results_df": pd.DataFrame(columns=core.SAFE_HAVEN_COLUMNS)
        }

    if gate_info.get("status") != "success":
        print(f"⚠️ {gate_info.get('message', '無法取得大盤資料，略過方案 B 掃描。')}")
        return {
            "status": "error",
            "gate_passed": False,
            "message": gate_info.get("message", "大盤閘門檢核失敗"),
            "gate_info": gate_info,
            "results_df": pd.DataFrame(columns=core.SAFE_HAVEN_COLUMNS)
        }

    print(
        f"🛡️ [^TWII {gate_info['latest_date']}] 收盤：{gate_info['latest_close']:.2f} | "
        f"60MA：{gate_info['latest_ma60']:.2f} | 近10日破季線：{gate_info['below_ma60_days_10']} 日 | "
        f"前段Low_1：{gate_info['prev_low_1']:.2f} ({gate_info['prev_low_1_date']}) | "
        f"近7日最低：{gate_info['recent_low_7d']:.2f} ({gate_info['recent_low_7d_date']})"
    )

    if not gate_info.get("gate_passed", False):
        print(f"⏸️ {core.BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG}")
        return {
            "status": "blocked",
            "gate_passed": False,
            "message": core.BearMarketSafeHavenStrategy.GATE_BLOCKED_MSG,
            "gate_info": gate_info,
            "results_df": pd.DataFrame(columns=core.SAFE_HAVEN_COLUMNS)
        }

    print("✅ 大盤同時滿足條件 1 與條件 2，啟動方案 B 普通股掃描...")
    try:
        scan_res = core.BearMarketSafeHavenStrategy.scan_market(
            stock_dict=stock_dict,
            twii_df=gate_info.get("twii_df"),
            preloaded_data=preloaded_daily_map,
            max_workers=5,
            chunk_size=30,
            request_delay=0.35,
            session=yf_session
        )
    except Exception as e:
        err_msg = f"方案 B 普通股掃描執行異常：{str(e)}"
        print(f"⚠️ {err_msg}")
        return {
            "status": "error",
            "gate_passed": True,
            "message": err_msg,
            "gate_info": gate_info,
            "results_df": pd.DataFrame(columns=core.SAFE_HAVEN_COLUMNS)
        }

    res_df = scan_res.get("results_df", pd.DataFrame(columns=core.SAFE_HAVEN_COLUMNS))
    print(f"🎯 {scan_res.get('message', '方案 B 掃描完成')}（複用日線快取：{scan_res.get('reused_count', 0)} 檔）")

    if not res_df.empty:
        print("📋 【空頭避風港強勢股（方案 B）】依 15日超額報酬RS(%) 降冪排序結果：")
        print(res_df.to_string(index=False))

    tw_tz = datetime.timezone(datetime.timedelta(hours=8))
    today_str = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d")
    csv_filename = f"plan_b_safe_haven_{today_str}.csv"
    try:
        res_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"💾 已於本地匯出方案 B 報表：{csv_filename}")
    except Exception as e:
        print(f"⚠️ 匯出本地 CSV ({csv_filename}) 失敗：{str(e)}")

    return scan_res


def export_scan_reports(
    res_df: pd.DataFrame,
    vol_report_df: pd.DataFrame,
    foreign_flow_df: pd.DataFrame,
    safe_haven_excel_df: pd.DataFrame,
    export_format: str = "xlsx"
):
    """
    統一報表匯出介面（支援 .xlsx 與 .csv，預設 .xlsx）：
    - .xlsx：輸出 4 工作表（含自動欄寬與數值格式），即使 res_df 為 0 檔仍保留完整標頭。
    - .csv：以 utf-8-sig 編碼輸出完整欄位（包含威科夫與 8 項新增實體量欄位）。
    """
    _persist_foreign_flow_history(foreign_flow_df)

    webhook_url = os.environ.get("GAS_WEBHOOK_URL", "").strip()
    token = os.environ.get("GAS_SECRET_TOKEN", "").strip()
    tw_tz = datetime.timezone(datetime.timedelta(hours=8))
    timestamp = datetime.datetime.now(tw_tz).strftime('%Y-%m-%d %H%M')
    fmt = str(export_format or "xlsx").strip().lower()

    if fmt in ("csv", "both"):
        csv_filename = f"auto_scan_all_{timestamp}.csv"
        csv_bytes = res_df.to_csv(index=False).encode("utf-8-sig")
        try:
            with open(csv_filename, "wb") as f:
                f.write(csv_bytes)
            print(f"💾 已於本地匯出 CSV 報表：{csv_filename}")
        except Exception as e:
            print(f"⚠️ 匯出本地 CSV ({csv_filename}) 失敗：{e}")

        if fmt == "csv":
            if webhook_url and token:
                print(f"☁️ 正在將 CSV 報表上傳至 Google Drive: {csv_filename} ...")
                success, msg = core.GoogleDriveUploader.upload(
                    file_name=csv_filename,
                    webhook_url=webhook_url,
                    token=token,
                    csv_bytes=csv_bytes
                )
                print(f"✅ 上傳結果: {msg}")
            else:
                print("⚠️ 未檢測到 GAS_WEBHOOK_URL，略過上傳。")
            return

    if webhook_url and token:
        file_name = f"auto_scan_all_{timestamp}.xlsx"
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                res_df.to_excel(writer, sheet_name=MAIN_SCAN_SHEET_NAME, index=False)
                _format_main_scan_sheet(writer, MAIN_SCAN_SHEET_NAME)
                vol_report_df.to_excel(writer, sheet_name="波動率體制監控", index=False)
                try:
                    foreign_flow_df.to_excel(writer, sheet_name=FOREIGN_FLOW_SHEET_NAME, index=False)
                    _format_foreign_flow_sheet(writer, FOREIGN_FLOW_SHEET_NAME, ref_sheet_name=MAIN_SCAN_SHEET_NAME)
                except Exception as fe:
                    print(f"⚠️ 寫入「{FOREIGN_FLOW_SHEET_NAME}」工作表失敗，維持主報表匯出：{fe}")
                try:
                    safe_haven_excel_df.to_excel(writer, sheet_name=SAFE_HAVEN_SHEET_NAME, index=False)
                    _format_safe_haven_sheet(writer, SAFE_HAVEN_SHEET_NAME, ref_sheet_name=MAIN_SCAN_SHEET_NAME)
                except Exception as pbe:
                    print(f"⚠️ 寫入「{SAFE_HAVEN_SHEET_NAME}」工作表失敗，維持主報表匯出：{pbe}")
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
    liquidity_min_vol_lots = float(CONFIG.get("phase1_liquidity_min_vol_ma20_lots", 300))
    print(f"✅ 成功獲取 {len(tickers)} 檔標的，開始執行第一階段篩選（流動性濾網：20日均量 > {liquidity_min_vol_lots:.0f} 張，含獨立均線壓縮一陽穿多線並行）...")

    div_pairs = [(CONFIG["div_recent_w"], CONFIG["div_older_w"])] if CONFIG["use_single_div"] else [(5, 20), (5, 60), (20, 60)]
    max_older_w = max(pair[1] for pair in div_pairs)
    extra_buffer_days = 30 if CONFIG["enable_30d_backtest"] else 0
    total_needed_days = CONFIG["lookback_start"] + max_older_w + CONFIG["pivot_left"] + 70 + extra_buffer_days
    dl_period = "3mo" if total_needed_days <= 60 else "6mo" if total_needed_days <= 120 else "1y"
    
    common_stock_set = set(core.BearMarketSafeHavenStrategy.filter_common_stocks(stock_dict).keys())
    plan_b_daily_cache = {}

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

                    if (
                        ticker in common_stock_set
                        and not df.empty
                        and len(df) >= core.BearMarketSafeHavenStrategy.MIN_REQUIRED_BARS
                        and all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])
                    ):
                        plan_b_daily_cache[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()

                    if df.empty or len(df) <= CONFIG["lookback_start"] + 20: continue

                    # 第一階段流動性過濾：yfinance Volume 單位為股，統一除以 1000 換算為張，僅保留 20 日均量 > 300 張
                    df['Volume_Lots'] = df['Volume'] / 1000.0
                    df['Vol_MA20'] = df['Volume_Lots'].rolling(20).mean()
                    latest_vol_ma20 = df['Vol_MA20'].iloc[-1]
                    if pd.isna(latest_vol_ma20) or float(latest_vol_ma20) <= liquidity_min_vol_lots:
                        continue

                    df = core.TechnicalIndicators.add_macd(core.TechnicalIndicators.add_kd(df))
                    df = core.TechnicalIndicators.add_rsi(df)
                    expert_res_df = core.ExpertTradingStrategy.evaluate(df)

                    df['Pct_Change'] = df['Close'].pct_change() * 100
                    df['MA5'] = df['Close'].rolling(5).mean()
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['MA60'] = df['Close'].rolling(60).mean()

                    df = core.ScanEngine.calculate_all_phase1_scores(df, CONFIG)

                    # 獨立並行策略：均線極致壓縮後一陽穿多線（不與既有 8 大策略混算）
                    sq_res_df = msq.MASqueezeBreakoutStrategy.evaluate(df)
                    for sq_col in sq_res_df.columns:
                        df[sq_col] = sq_res_df[sq_col]

                    best_combined_score, best_row, best_offset, best_vol_mult = -1, None, 0, 0.0
                    pass_ma_squeeze = False
                    sq_best_row, sq_best_offset, sq_best_vol_mult = None, 0, 0.0

                    for offset in range(CONFIG["lookback_end"], CONFIG["lookback_start"] + 1):
                        if len(df) <= offset + 1: continue
                        t_row = df.iloc[-1 - offset]
                        prev_vol = df['Volume_Lots'].iloc[-2 - offset] if len(df) > offset + 1 else np.nan
                        vol_mult = (t_row['Volume_Lots'] / prev_vol) if pd.notna(prev_vol) and prev_vol > 0 else 0.0

                        if t_row.get('Vol_MA20', 0) >= CONFIG["min_vol_ma20"]:
                            is_pass, combo_score = core.ScanEngine.check_algo_pass(t_row, CONFIG, CONFIG['algo_mode'])
                            if is_pass and (combo_score > best_combined_score or best_row is None):
                                best_combined_score, best_row, best_offset, best_vol_mult = combo_score, t_row, offset, vol_mult

                        if bool(t_row.get('MA_Squeeze_Pass', False)):
                            pass_ma_squeeze = True
                            if sq_best_row is None:
                                sq_best_row, sq_best_offset, sq_best_vol_mult = t_row, offset, vol_mult

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

                    if (best_row is not None and best_combined_score >= 0) or pass_60m_adv or pass_ma_squeeze:
                        if best_row is None:
                            if pass_ma_squeeze and sq_best_row is not None:
                                best_row, best_offset, best_vol_mult = sq_best_row, sq_best_offset, sq_best_vol_mult
                            else:
                                best_row = df.iloc[-1]
                                best_offset = 0
                                prev_v = df['Volume_Lots'].iloc[-2] if len(df) > 1 else np.nan
                                best_vol_mult = (best_row['Volume_Lots'] / prev_v) if pd.notna(prev_v) and prev_v > 0 else 0.0

                        count_30d = 0
                        if CONFIG["enable_30d_backtest"]:
                            for b_offset in range(best_offset + 1, min(len(df)-1, best_offset + 31)):
                                c_row = df.iloc[-1 - b_offset]
                                if pd.isna(c_row.get('Vol_MA20')): continue
                                c_pass = False
                                if c_row['Vol_MA20'] >= CONFIG["min_vol_ma20"]:
                                    c_pass, _ = core.ScanEngine.check_algo_pass(c_row, CONFIG, CONFIG['algo_mode'])
                                if c_pass or bool(c_row.get('MA_Squeeze_Pass', False)):
                                    count_30d += 1

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
                        sq_src = sq_best_row if (pass_ma_squeeze and sq_best_row is not None) else best_row
                        base_info.update(msq.MASqueezeBreakoutStrategy.get_metrics(sq_src))
                        
                        reversal_candidates[ticker] = base_info
                except Exception: continue
        except Exception: pass
        time.sleep(1.0)

    reversal_list = list(reversal_candidates.values())
    print(f"🎯 第一階段篩選完畢，共入選 {len(reversal_list)} 檔標的。")

    final_results = []
    candidate_daily_dfs = {}
    candidate_full_tickers = {}

    if reversal_list:
        print("🚀 [Step 2] 正在下載入選標的 2 年資料以運算週K、240MA年線深度評分、背離與威科夫吸籌偵測...")
        candidate_tickers = [item["_Full_Ticker"] for item in reversal_list]
        import yfinance as yf
        long_data = yf.download(candidate_tickers, period="2y", threads=False, progress=False, session=yf_session)

        for item in reversal_list:
            try:
                ticker, specific_offset, daily_df = item.pop("_Full_Ticker"), item.pop("_Offset"), item.pop("_Daily_DF")
                pass_60m_adv, adv_60m_data = item.pop("_Pass_60m_Adv"), item.pop("_Adv_60m_Data")
                has_daily_div = False
                item['年線/半年線空排'] = item['半年線斜率(%)'] = item['年線斜率(%)'] = "無資料"
                item['均線最大最小差異百分比(%)'] = "-"
                item['均線糾結狀態'] = "-"
                item['短天期均線差異(%)'] = "-"
                item['短均線糾結狀態'] = "-"
                item.update(core.WyckoffAccumulationAnalyzer.empty_result("資料不足"))

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

                        wyk_res = core.WyckoffAccumulationAnalyzer.evaluate(long_df, CONFIG)
                        item.update(wyk_res)

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

                base_tags = core.ScanEngine.generate_base_tags(item, CONFIG)
                base_tags.extend(msq.MASqueezeBreakoutStrategy.get_tags(item))
                div_tag = " + 雙級別共振" if (has_daily_div and pass_60m_adv) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if pass_60m_adv else " (無背離)"))
                
                if pass_60m_adv and adv_60m_data is not None:
                    sig_name = "【🔥 60分K底倉突破】" if adv_60m_data.get('Adv_Signal_A') else "【🔥 60分K右肩回測】"
                    base_tags.insert(0, sig_name)

                item["演算法建議結果"] = " & ".join(base_tags) + div_tag if base_tags else "觀察標的" + div_tag

                # 暫存已入選標的日線與完整代號，供後置實體波段留倉量計算使用
                candidate_daily_dfs[ticker] = daily_df
                candidate_full_tickers[item["股票代號"]] = ticker
                final_results.append(item)
            except Exception: continue

    # 僅針對已完成篩選的最終入選標的，批次抓取當沖資料並補齊 8 項實體量欄位
    if final_results:
        print(f"🚀 [Step 2.5] 針對入選的 {len(final_results)} 檔標的計算實體波段留倉量與資料同步狀態...")
        core.RealTurnoverVolumeAnalyzer.enrich_candidates(
            candidates=final_results,
            daily_df_map=candidate_daily_dfs,
            ticker_key="股票代號",
            date_key="觸發日期",
            full_ticker_map=candidate_full_tickers,
            session=yf_session,
            request_delay=0.25
        )

    plan_b_res = _run_plan_b_safe_haven(stock_dict, plan_b_daily_cache, yf_session)
    safe_haven_excel_df = core.BearMarketSafeHavenStrategy.build_excel_dataframe(plan_b_res)

    if final_results:
        res_df = pd.DataFrame(final_results)
        cols = _build_scan_output_columns(CONFIG, div_pairs, existing_cols=res_df.columns)
        for c in cols:
            if c not in res_df.columns:
                res_df[c] = np.nan
        res_df = res_df[cols]
        sort_keys = [k for k in ["*年線回測分數", "共振分數", "反轉分數", "VCP分數"] if k in res_df.columns]
        if sort_keys:
            res_df = res_df.sort_values(by=sort_keys, ascending=[False]*len(sort_keys)).reset_index(drop=True)
        res_df.index = res_df.index + 1

        print("📊 掃描完成！最終報表標的預覽：")
        preview_cols = ['股票代號', '股票名稱', '演算法建議結果', '均線壓縮一陽穿多線', '實體量是否更新至指定日期', '實體留倉量(張)', '實體量比', '實體量能判讀']
        print(res_df[[c for c in preview_cols if c in res_df.columns]].to_string())
    else:
        # 0 檔符合條件時仍保留完整原有欄位＋均線壓縮＋威科夫＋8 項新增實體量欄位的空表結構
        cols = _build_scan_output_columns(CONFIG, div_pairs)
        res_df = pd.DataFrame(columns=cols)
        print("💡 今日全市場無符合條件標的，將保留完整欄位標頭之空表結構進行報表匯出。")

    export_fmt = os.environ.get("SCAN_EXPORT_FORMAT", "xlsx").strip().lower()
    export_scan_reports(
        res_df=res_df,
        vol_report_df=vol_report_df,
        foreign_flow_df=foreign_flow_df,
        safe_haven_excel_df=safe_haven_excel_df,
        export_format=export_fmt
    )


if __name__ == "__main__":
    run_full_scan()
