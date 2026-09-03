import os
import json
import warnings
import logging
import requests
import datetime
import time
import re
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
    "use_backtest_date": False,
    "backtest_date": str(datetime.date.today())
}

def load_config():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if "profiles" not in config:
                    config["profiles"] = {}
                config["profiles"]["預設參數 (Default)"] = DEFAULT_PARAMS.copy()
                return config
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
        except Exception:
            return None

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

class IndicatorResonanceStrategy:
    @staticmethod
    def evaluate(df, recent_w=5, older_w=20, kd_older_low_th=20, kd_older_high_th=90, kd_recent_low_th=0.0, 
                 use_macd_abs=False, macd_older_low_th=0.0, macd_recent_low_th=0.0, cross_days=3,
                 require_price_higher_low=False, require_macd_cross_zero=False):
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
        
        if require_macd_cross_zero:
            macd_signal = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        else:
            # 寬鬆條件：柱狀圖谷底翻揚 (今天大於昨天，昨天小於前天) 或 已經實質金叉
            macd_turn_up = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))
            macd_signal = macd_turn_up | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))

        kd_recent_cross = kd_cross.rolling(window=cross_days, min_periods=1).max() >= 1
        macd_recent_cross = macd_signal.rolling(window=cross_days, min_periods=1).max() >= 1

        cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th) & kd_recent_cross
        
        cond_macd = (recent_macd_low > older_macd_low) & macd_recent_cross
        if use_macd_abs:
            cond_macd = cond_macd & (older_macd_low < macd_older_low_th) & (recent_macd_low > macd_recent_low_th)
            
        if require_price_higher_low:
            cond_price = recent_price_low > older_price_low
        else:
            cond_price = pd.Series(True, index=df.index)

        resonance_score = pd.Series(0.0, index=df.index)
        valid_mask = cond_kd & cond_macd & cond_price
        
        hist_momentum = (df['MACD_Hist'] - df['MACD_Hist'].shift(1)) * 100
        resonance_score[valid_mask] = 10.0 + hist_momentum[valid_mask].clip(0, 5) 
        
        return resonance_score

class DivergenceStrategy:
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
# 模組 3：財報雙軌評鑑系統 V4 (體質評分 + 地雷掃描)
# ==========================================
class FinancialAuditStrategy:
    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value): return np.nan
            return float(value)
        except Exception: return np.nan

    @staticmethod
    def _find_row(df, candidates):
        if df is None or df.empty: return None
        index_map = {str(x).strip().lower(): x for x in df.index}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in index_map: return index_map[key]
        for candidate in candidates:
            key = str(candidate).strip().lower()
            for idx in df.index:
                idx_str = str(idx).strip().lower()
                if key in idx_str: return idx
        return None

    @staticmethod
    def _get_series(df, candidates):
        row = FinancialAuditStrategy._find_row(df, candidates)
        if row is None: return pd.Series(dtype=float)
        s = df.loc[row].copy()
        return pd.Series([FinancialAuditStrategy._safe_float(x) for x in s.values], index=s.index, dtype=float)

    @staticmethod
    def _normalize_statement(df):
        if df is None or df.empty: return pd.DataFrame()
        result = df.copy()
        new_columns = []
        for col in result.columns:
            try:
                dt = pd.to_datetime(col)
                if hasattr(dt, "tz_localize"):
                    try: dt = dt.tz_localize(None)
                    except Exception: pass
                new_columns.append(dt)
            except Exception: new_columns.append(col)
        result.columns = new_columns
        try: result = result.loc[:, sorted(result.columns)]
        except Exception: pass
        result = result.loc[:, ~pd.Index(result.columns).duplicated()]
        return result

    @staticmethod
    def _growth(current, previous):
        if pd.isna(current) or pd.isna(previous) or abs(previous) < 1e-9: return np.nan
        return (current - previous) / abs(previous) * 100

    @staticmethod
    def _ratio(a, b):
        if pd.isna(a) or pd.isna(b) or abs(b) < 1e-9: return np.nan
        return a / b

    @staticmethod
    def _get_common_periods(*dfs):
        valid = [set(df.columns) for df in dfs if df is not None and not df.empty]
        if not valid: return []
        common = set.intersection(*valid)
        try: return sorted(common)
        except Exception: return list(common)

    @staticmethod
    def evaluate(ticker_symbol, session):
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            inc_stmt = FinancialAuditStrategy._normalize_statement(stock.quarterly_income_stmt)
            bal_sheet = FinancialAuditStrategy._normalize_statement(stock.quarterly_balance_sheet)
            cash_flow = FinancialAuditStrategy._normalize_statement(stock.quarterly_cashflow)

            if inc_stmt.empty and bal_sheet.empty and cash_flow.empty:
                return {"status": "error", "msg": f"無法取得 {ticker_symbol} 財報資料。"}

            revenue = FinancialAuditStrategy._get_series(inc_stmt, ["Total Revenue", "Operating Revenue"])
            net_income = FinancialAuditStrategy._get_series(inc_stmt, ["Net Income", "Net Income Common Stockholders"])
            operating_income = FinancialAuditStrategy._get_series(inc_stmt, ["Operating Income"])
            gross_profit = FinancialAuditStrategy._get_series(inc_stmt, ["Gross Profit"])
            pretax_income = FinancialAuditStrategy._get_series(inc_stmt, ["Pretax Income", "Income Before Tax"])

            cfo = FinancialAuditStrategy._get_series(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            capex = FinancialAuditStrategy._get_series(cash_flow, ["Capital Expenditure"])

            ar = FinancialAuditStrategy._get_series(bal_sheet, ["Accounts Receivable", "Net Receivables"])
            inventory = FinancialAuditStrategy._get_series(bal_sheet, ["Inventory", "Inventories"])
            ap = FinancialAuditStrategy._get_series(bal_sheet, ["Accounts Payable", "Payables And Accrued Expenses"])
            cash = FinancialAuditStrategy._get_series(bal_sheet, ["Cash And Cash Equivalents", "Cash Financial"])
            debt = FinancialAuditStrategy._get_series(bal_sheet, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])
            equity = FinancialAuditStrategy._get_series(bal_sheet, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
            current_assets = FinancialAuditStrategy._get_series(bal_sheet, ["Current Assets"])
            current_liabilities = FinancialAuditStrategy._get_series(bal_sheet, ["Current Liabilities"])

            common_periods = FinancialAuditStrategy._get_common_periods(inc_stmt, bal_sheet, cash_flow)
            if not common_periods:
                all_periods = set()
                for df in [inc_stmt, bal_sheet, cash_flow]:
                    if df is not None and not df.empty: all_periods.update(df.columns)
                if not all_periods: return {"status": "error", "msg": "無財報期間資料"}
                common_periods = sorted(all_periods)

            latest_period = common_periods[-1]
            prior_period = common_periods[-5] if len(common_periods) >= 5 else (common_periods[-2] if len(common_periods) >= 2 else None)
            comp_type = "YoY" if len(common_periods) >= 5 else "QoQ"

            def val_at(s, p): return FinancialAuditStrategy._safe_float(s.loc[p]) if not s.empty and p in s.index else np.nan

            rev_now = val_at(revenue, latest_period)
            ni_now = val_at(net_income, latest_period)
            cfo_now = val_at(cfo, latest_period)
            gp_now = val_at(gross_profit, latest_period)
            op_now = val_at(operating_income, latest_period)
            pretax_now = val_at(pretax_income, latest_period)
            ar_now = val_at(ar, latest_period)
            inv_now = val_at(inventory, latest_period)
            ap_now = val_at(ap, latest_period)
            capex_now = val_at(capex, latest_period)
            cash_now = val_at(cash, latest_period)
            debt_now = val_at(debt, latest_period) if pd.notna(val_at(debt, latest_period)) else 0.0
            equity_now = val_at(equity, latest_period)
            ca_now = val_at(current_assets, latest_period)
            cl_now = val_at(current_liabilities, latest_period)

            rev_prev = val_at(revenue, prior_period) if prior_period else np.nan
            ni_prev = val_at(net_income, prior_period) if prior_period else np.nan
            ar_prev = val_at(ar, prior_period) if prior_period else np.nan
            inv_prev = val_at(inventory, prior_period) if prior_period else np.nan
            gp_prev = val_at(gross_profit, prior_period) if prior_period else np.nan

            rev_growth = FinancialAuditStrategy._growth(rev_now, rev_prev)
            ni_growth = FinancialAuditStrategy._growth(ni_now, ni_prev)
            ar_growth = FinancialAuditStrategy._growth(ar_now, ar_prev)
            inv_growth = FinancialAuditStrategy._growth(inv_now, inv_prev)

            cogs_now = (rev_now - gp_now) if (pd.notna(rev_now) and pd.notna(gp_now) and (rev_now - gp_now) > 0) else np.nan
            cogs_prev = (rev_prev - gp_prev) if (pd.notna(rev_prev) and pd.notna(gp_prev) and (rev_prev - gp_prev) > 0) else np.nan

            dso_now = (ar_now / rev_now) * 90 if (pd.notna(rev_now) and rev_now > 0 and pd.notna(ar_now)) else np.nan
            dso_prev = (ar_prev / rev_prev) * 90 if (pd.notna(rev_prev) and rev_prev > 0 and pd.notna(ar_prev)) else np.nan
            dso_diff = dso_now - dso_prev if (pd.notna(dso_now) and pd.notna(dso_prev)) else np.nan

            if pd.notna(cogs_now) and cogs_now > 0 and pd.notna(inv_now):
                dio_now = (inv_now / cogs_now) * 90
            elif pd.notna(rev_now) and rev_now > 0 and pd.notna(inv_now):
                dio_now = (inv_now / rev_now) * 90
            else:
                dio_now = np.nan

            if pd.notna(cogs_prev) and cogs_prev > 0 and pd.notna(inv_prev):
                dio_prev = (inv_prev / cogs_prev) * 90
            elif pd.notna(rev_prev) and rev_prev > 0 and pd.notna(inv_prev):
                dio_prev = (inv_prev / rev_prev) * 90
            else:
                dio_prev = np.nan

            dio_diff = dio_now - dio_prev if (pd.notna(dio_now) and pd.notna(dio_prev)) else np.nan

            dpo_now = (ap_now / cogs_now) * 90 if (pd.notna(cogs_now) and cogs_now > 0 and pd.notna(ap_now)) else ((ap_now / rev_now) * 90 if pd.notna(rev_now) and rev_now > 0 and pd.notna(ap_now) else np.nan)
            ccc_now = (dso_now + dio_now - dpo_now) if (pd.notna(dso_now) and pd.notna(dio_now) and pd.notna(dpo_now)) else np.nan

            cfo_to_ni = FinancialAuditStrategy._ratio(cfo_now, ni_now)
            fcf_now = (cfo_now + capex_now) if (pd.notna(cfo_now) and pd.notna(capex_now) and capex_now < 0) else (cfo_now - capex_now if (pd.notna(cfo_now) and pd.notna(capex_now)) else cfo_now)

            current_ratio = FinancialAuditStrategy._ratio(ca_now, cl_now)
            debt_equity = FinancialAuditStrategy._ratio(debt_now, equity_now)
            net_cash = (cash_now - debt_now) if (pd.notna(cash_now) and pd.notna(debt_now)) else np.nan

            gm_now = FinancialAuditStrategy._ratio(gp_now, rev_now) * 100
            om_now = FinancialAuditStrategy._ratio(op_now, rev_now) * 100
            net_margin = FinancialAuditStrategy._ratio(ni_now, rev_now) * 100
            roe_annual = (ni_now / equity_now) * 4 * 100 if (pd.notna(ni_now) and pd.notna(equity_now) and equity_now > 0) else np.nan

            op_to_pretax = FinancialAuditStrategy._ratio(op_now, pretax_now) if (pd.notna(op_now) and pd.notna(pretax_now) and pretax_now > 0) else np.nan

            # ----------------------------------------------------
            # 軌道一：地雷風險檢測 (Risk Score: 0~100，越低越好)
            # ----------------------------------------------------
            risk_score = 0
            warnings = []

            if pd.notna(ni_now) and pd.notna(cfo_now):
                if ni_now > 0 and cfo_now < 0:
                    risk_score += 15
                    warnings.append("【現金流背離】單季淨利為正但營運現金流 (CFO) 為負，防範黑字倒閉或營運資金積壓。")
                elif ni_now > 0 and pd.notna(cfo_to_ni) and cfo_to_ni < 0.5:
                    risk_score += 8
                    warnings.append(f"【現金含金量偏低】CFO / 淨利比僅 {cfo_to_ni:.2f}x，獲利變現效率差。")

            if pd.notna(dso_now):
                if dso_now > 150:
                    risk_score += 15
                    warnings.append(f"【應收過長】收現天數高達 {dso_now:.0f} 天，有塞貨或呆帳疑慮。")
                elif dso_now > 120:
                    risk_score += 8
                    warnings.append(f"【應收偏高】收現天數達 {dso_now:.0f} 天，票期偏長。")
            if pd.notna(dso_diff) and dso_diff > 30:
                risk_score += 10
                warnings.append(f"【應收惡化】收現天數較去年同期增加 {dso_diff:.0f} 天。")

            if pd.notna(dio_now):
                if dio_now > 180:
                    risk_score += 15
                    warnings.append(f"【存貨滯銷】存貨週轉天數高達 {dio_now:.0f} 天，具庫存跌價重大風險。")
                elif dio_now > 120:
                    risk_score += 8
                    warnings.append(f"【庫存偏高】存貨週轉天數達 {dio_now:.0f} 天。")
            if pd.notna(dio_diff) and dio_diff > 45:
                risk_score += 10
                warnings.append(f"【庫存急升】存貨去化嚴重放緩，週轉天數較同期暴增 {dio_diff:.0f} 天。")

            if pd.notna(current_ratio) and current_ratio < 1.0:
                risk_score += 15
                warnings.append(f"【流動性緊縮】流動比率僅 {current_ratio:.2f} (< 1.0)，短期償債壓力重。")

            if pd.notna(debt_equity) and debt_equity > 1.8:
                risk_score += 10
                warnings.append(f"【槓桿偏高】負債權益比 (D/E) 達 {debt_equity:.2f}，財務結構較為脆弱。")

            risk_score = min(100, max(0, round(risk_score, 1)))

            # ----------------------------------------------------
            # 軌道二：優良體質評鑑 (Quality Score: 0~100，越高越好)
            # ----------------------------------------------------
            quality_score = 0
            positives = []

            # 1. 護城河與獲利能力 (30分)
            if pd.notna(gm_now) and gm_now >= 45:
                quality_score += 15; positives.append(f"高毛利護城河 (毛利率 {gm_now:.1f}%)")
            elif pd.notna(gm_now) and gm_now >= 25:
                quality_score += 10; positives.append(f"具備健全毛利率 (毛利率 {gm_now:.1f}%)")

            if pd.notna(om_now) and om_now >= 18:
                quality_score += 15; positives.append(f"營業利益率極佳 (營益率 {om_now:.1f}%)")
            elif pd.notna(om_now) and om_now >= 10:
                quality_score += 10; positives.append(f"營業獲利穩健 (營益率 {om_now:.1f}%)")

            # 2. 本業獲利造血與品質 (25分)
            if pd.notna(cfo_to_ni) and cfo_to_ni >= 1.2:
                quality_score += 15; positives.append(f"盈餘含金量超群 (CFO/淨利 {cfo_to_ni:.2f}x)")
            elif pd.notna(cfo_to_ni) and cfo_to_ni >= 1.0:
                quality_score += 10; positives.append("營業現金充沛 (CFO ≥ 淨利)")

            if pd.notna(fcf_now) and fcf_now > 0:
                quality_score += 10; positives.append("正向自由現金流 (FCF > 0)")

            # 3. 資本效率與本業純度 (20分)
            if pd.notna(roe_annual) and roe_annual >= 18:
                quality_score += 15; positives.append(f"高股東權益報酬 (年化 ROE 約 {roe_annual:.1f}%)")
            elif pd.notna(roe_annual) and roe_annual >= 12:
                quality_score += 10; positives.append(f"ROE 達標 (年化 ROE 約 {roe_annual:.1f}%)")

            if pd.notna(op_to_pretax) and op_to_pretax >= 0.8:
                quality_score += 5; positives.append("本業純度高 (本業獲利佔稅前淨利 ≥ 80%)")

            # 4. 資產負債表實力與週轉效率 (25分)
            if pd.notna(net_cash) and net_cash > 0:
                quality_score += 15; positives.append("實質淨現金公司 (手頭現金 > 總借款)")
            elif pd.notna(debt_equity) and debt_equity < 0.5:
                quality_score += 10; positives.append(f"低槓桿保守營運 (D/E {debt_equity:.2f})")

            if pd.notna(ccc_now) and ccc_now <= 60:
                quality_score += 10; positives.append(f"強勢供應鏈週轉效率 (CCC 僅 {ccc_now:.0f} 天)")
            elif pd.notna(ccc_now) and ccc_now <= 90:
                quality_score += 5; positives.append(f"週轉效率良好 (CCC {ccc_now:.0f} 天)")

            quality_score = min(100, max(0, round(quality_score, 1)))

            # 綜合評級判定
            if risk_score >= 40:
                conclusion = "🔴 高風險示警：存在明顯財務結構弱化或地雷特徵，建議嚴格避開。"
                overall_grade = "高危排除"
            elif risk_score < 20 and quality_score >= 70:
                conclusion = "💎 卓越績優：同時兼具強大造血力、護城河與極低財務風險，屬於頂級優質公司。"
                overall_grade = "卓越績優"
            elif risk_score < 30 and quality_score >= 45:
                conclusion = "🟢 穩健健康：財務防線扎實，具備良好營運造血能力，無明顯風險暴雷點。"
                overall_grade = "穩健健康"
            elif risk_score >= 20:
                conclusion = "🟠 觀察注意：雖無立即性重度風險，但存在部分週轉或現金流拉警報之項目。"
                overall_grade = "觀察注意"
            else:
                conclusion = "🟡 體質平庸：無財務暴雷風險，但獲利造血或競爭護城河相對普通。"
                overall_grade = "體質平庸"

            return {
                "status": "success",
                "當期季報": latest_period.strftime("%Y-%m-%d") if hasattr(latest_period, "strftime") else str(latest_period),
                "比較基準": comp_type,
                "綜合評級": overall_grade,
                "體質分數": quality_score,
                "風險分數": risk_score,
                "結論": conclusion,
                "營收成長(%)": rev_growth,
                "淨利成長(%)": ni_growth,
                "應收成長(%)": ar_growth,
                "存貨成長(%)": inv_growth,
                "稅後淨利(千)": ni_now / 1000 if pd.notna(ni_now) else np.nan,
                "營業現金流CFO(千)": cfo_now / 1000 if pd.notna(cfo_now) else np.nan,
                "自由現金流FCF(千)": fcf_now / 1000 if pd.notna(fcf_now) else np.nan,
                "CFO/淨利": cfo_to_ni,
                "DSO_now": dso_now,
                "DSO_diff": dso_diff,
                "DIO_now": dio_now,
                "DIO_diff": dio_diff,
                "DPO_now": dpo_now,
                "CCC_now": ccc_now,
                "流動比率": current_ratio,
                "Debt/Equity": debt_equity,
                "淨現金(千)": net_cash / 1000 if pd.notna(net_cash) else np.nan,
                "毛利率": gm_now,
                "營益率": om_now,
                "淨利率": net_margin,
                "年化ROE": roe_annual,
                "本業純度": op_to_pretax,
                "警訊數": len(warnings),
                "診斷明細": "\n".join(warnings) if warnings else "✅ 財報天數與槓桿指標均在安全水位。",
                "正向訊號": "\n".join(positives) if positives else "無顯著優質體質特徵"
            }
        except Exception as e:
            return {"status": "error", "msg": f"解析 {ticker_symbol} 發生錯誤：{str(e)}"}

# ==========================================
# 網路請求與全域快取
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
    stocks = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    for mode in ['2', '4']: 
        suffix = '.TW' if mode == '2' else '.TWO'
        for attempt in range(3):
            try:
                url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
                res = requests.get(url, headers=headers, timeout=15)
                res.encoding = 'big5' 
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
                if fetched_count > 500: break
                else: time.sleep(2)
            except Exception:
                time.sleep(2)
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

def resolve_ticker(symbol, stock_dict):
    code = str(symbol).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"): return code
    tw = f"{code}.TW"
    two = f"{code}.TWO"
    if tw in stock_dict: return tw
    if two in stock_dict: return two
    return tw 

# ==========================================
# Streamlit UI 主程式
# ==========================================
st.set_page_config(page_title="台股 K線型態與位階深度解析系統", layout="wide")

# 初始化跨分頁連動狀態
if "audit_mode" not in st.session_state: st.session_state.audit_mode = "單檔查詢"
if "batch_input_area" not in st.session_state: st.session_state.batch_input_area = "2330, 2454, 3231, 6147, 1909"
if "run_batch_audit" not in st.session_state: st.session_state.run_batch_audit = False

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

tab1, tab2, tab3 = st.tabs(["📊 單檔深度解析", "🚀 全市場智慧掃描 (回測/翻轉/VCP/共振/背離)", "📑 財報體質與雙軌健檢 (基本面 V4)"])

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
                df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8) * 100
                df['Std20'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + 2 * df['Std20']
                df['BB_Lower'] = df['MA20'] - 2 * df['Std20']
                
                df['VWMA20'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
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
                    cross_days=st.session_state.reso_cross_days,
                    require_price_higher_low=st.session_state.reso_price_higher_low,
                    require_macd_cross_zero=st.session_state.reso_macd_cross_zero
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
                plt.close(fig)

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
        
        st.markdown("**1. 掃描模式與演算法選擇**")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
        with col_m1:
            scan_target_mode = st.radio("掃描對象", ['全市場掃描', '指定個股測試'], index=0, horizontal=True)
        with col_m2:
            test_ticker_input = ""
            if scan_target_mode == '指定個股測試':
                test_ticker_input = st.text_input("輸入測試股票代號", value="2330", placeholder="例如: 2330").strip()
        with col_m3:
            algo_mode = st.radio("請選擇欲執行的掃描演算法", ['全部', '底部翻轉', 'VCP', '指標共振'], index=0, horizontal=True)
            
        if algo_mode == '全部' and scan_target_mode == '全市場掃描':
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
            st.number_input("KD前波低點 <", step=1.0, key="reso_kd_older_low")
            st.number_input("KD前波高點 <", step=1.0, key="reso_kd_older_high")
        with col_r2:
            st.number_input("KD近波低點 >", step=1.0, key="reso_kd_recent_low")
            st.number_input("近期金叉天數 <=", step=1, key="reso_cross_days")
        with col_r3:
            st.checkbox("啟用 MACD絕對數值濾網", key="use_macd_abs")
            st.number_input("MACD前波低 <", step=0.1, key="reso_macd_older_low", disabled=not st.session_state.use_macd_abs)
            st.number_input("MACD近波低 >", step=0.1, key="reso_macd_recent_low", help="若要抓水下翻紅，請設為負數或 0.0", disabled=not st.session_state.use_macd_abs)
        with col_r4:
            st.checkbox("嚴格要求股價底底高", key="reso_price_higher_low", help="不勾選則允許股價破底但指標背離 (典型底背離)")
            st.checkbox("嚴格要求MACD金叉(>0)", key="reso_macd_cross_zero", help="不勾選則只要求柱狀圖谷底翻揚即可")

    st.markdown("---")
    
    if st.button("🚀 開始智慧區間掃描", type="primary"):
        status_text = st.empty(); progress_bar = st.progress(0)
        status_text.text("⏳ [初始化] 正在同步台股最新代號與名稱清單，請稍候...")
        
        yf_session = get_yf_session()
        stock_dict = get_all_tw_stocks()
        
        if not stock_dict:
            status_text.empty(); progress_bar.empty(); st.error("❌ 無法取得台股清單，請檢查網路連線。")
            st.stop()
            
        # 決定掃描標的
        if scan_target_mode == "指定個股測試":
            if not test_ticker_input:
                status_text.empty(); progress_bar.empty(); st.error("❌ 請輸入測試股票代號！")
                st.stop()
            target_yf_ticker = resolve_ticker(test_ticker_input, stock_dict)
            tickers = [target_yf_ticker]
        else:
            tickers = list(stock_dict.keys())
            
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
        
        reversal_candidates = {} 
        chunk_size = 40 
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            status_text.text(f"[階段一] 正在全市場區間掃描 [{algo_mode}]：進度 {min(i+chunk_size, len(tickers))} / {len(tickers)} 檔...")
            try:
                data = yf.download(chunk, threads=False, progress=False, session=yf_session, **dl_kwargs)
                for ticker in chunk:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            if ticker in data.columns.get_level_values(0):
                                df = data.xs(ticker, axis=1, level=0).dropna(how='all')
                            elif ticker in data.columns.get_level_values(1):
                                df = data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else:
                                df = pd.DataFrame()
                        else:
                            df = data.dropna(how='all') if len(chunk) == 1 else pd.DataFrame()
                        
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
                        df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8) * 100
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
                                cross_days=st.session_state.reso_cross_days,
                                require_price_higher_low=st.session_state.reso_price_higher_low,
                                require_macd_cross_zero=st.session_state.reso_macd_cross_zero
                            )
                        else:
                            df['Reso_Score'] = pd.Series(0.0, index=df.index)
                        
                        # 儲存供稍後詳細表格顯示用
                        if scan_target_mode == "指定個股測試":
                            st.session_state.test_stock_df = df.copy()
                            st.session_state.test_ticker_name = stock_dict.get(ticker, ticker)
                        
                        best_combined_score = -1
                        best_row, best_offset = None, 0
                        force_pass = (scan_target_mode == "指定個股測試")
                        
                        for offset in range(st.session_state.lookback_end, st.session_state.lookback_start + 1):
                            if len(df) <= offset + 1: continue
                            t_row = df.iloc[-1 - offset]
                            r_score, v_score, reso_score, vol_ma = t_row['Candle_Score'], t_row['VCP_Score'], t_row['Reso_Score'], t_row['Vol_MA20']
                            
                            if vol_ma >= st.session_state.min_vol_ma20 or force_pass:
                                is_r_pass = (algo_mode in ['全部', '底部翻轉']) and (r_score >= st.session_state.min_score)
                                is_v_pass = (algo_mode in ['全部', 'VCP']) and (v_score >= st.session_state.min_vcp_score)
                                is_reso_pass = (algo_mode in ['全部', '指標共振']) and (reso_score >= st.session_state.min_reso_score)
                                
                                # 若為指定個股測試，則無條件使其通過以便檢視背離/扣抵狀態
                                if is_r_pass or is_v_pass or is_reso_pass or force_pass:
                                    combo_score = r_score + v_score + reso_score
                                    if combo_score > best_combined_score or best_row is None:
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
            
            time.sleep(1.0) 
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
                        time.sleep(0.5) 
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
                    
                    if not base_tags and scan_target_mode == "指定個股測試":
                        base_tags.append("未達標(測試)")
                        
                    div_tag = " + 雙級別共振" if (has_daily_div and has_m60_div) else (" + 日K背離" if has_daily_div else (" + 60分K背離" if has_m60_div else " (無背離)"))
                    item["演算法建議結果"] = " & ".join(base_tags) + div_tag
                    
                    final_results.append(item)
                except Exception: pass
                progress_bar.progress(min(1.0, (idx + 1) / len(reversal_list)))

            status_text.empty(); progress_bar.empty()
            
            res_df = pd.DataFrame(final_results)
            base_cols = ['股票代號', '股票名稱', '觸發日期', '演算法建議結果', '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)']
            div_cols = [c for c in res_df.columns if '背離' in c and c not in base_cols]
            kou_cols = [c for c in res_df.columns if '扣抵狀態' in c]
            cols = base_cols + div_cols + kou_cols
            
            res_df = res_df[cols].sort_values(by=["共振分數", "反轉分數", "VCP分數"], ascending=[False, False, False]).reset_index(drop=True)
            res_df.index = res_df.index + 1
            
            st.session_state.scan_res_df = res_df
            st.session_state.scan_algo_mode = algo_mode
            st.session_state.scan_target_mode_saved = scan_target_mode
            st.session_state.market_info = market_info
            
            date_str = f"({st.session_state.backtest_date_obj})" if st.session_state.use_backtest_date else ""
            if scan_target_mode == "指定個股測試":
                st.session_state.scan_msg = f"🎉 測試完成！已為您匯出個股 {date_str} 的計算結果與下方詳細參數表。"
            else:
                st.session_state.scan_msg = f"🎉 掃描完成！本次共精選出 **{len(res_df)}** 檔符合條件的標的 {date_str}。"
        else:
            status_text.empty(); progress_bar.empty()
            st.session_state.scan_res_df = pd.DataFrame()
            st.session_state.scan_algo_mode = algo_mode
            st.session_state.scan_target_mode_saved = scan_target_mode
            st.session_state.market_info = market_info
            if scan_target_mode == "指定個股測試":
                st.session_state.scan_msg = f"無法取得測試標的資料，請確認代號是否正確。"
            else:
                st.session_state.scan_msg = f"掃描完成！在指定的區間與條件下，全市場無任何符合「{algo_mode}」的標的。"

    # 確保掃描結果在點擊按鈕或重新渲染時不會消失
    if "scan_res_df" in st.session_state:
        # 重新渲染大盤結果與提示訊息
        if not st.session_state.scan_res_df.empty:
            st.success(st.session_state.scan_msg)
        else:
            st.info(st.session_state.scan_msg)
            
        market_info = st.session_state.get("market_info")
        if market_info:
            st.markdown("### 🌐 大盤位階與期貨基準動態濾網評估結果")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("加權指數收盤", market_info["加權指數收盤"])
            m_col2.metric("月線 (MA20)", market_info["月線 (MA20)"])
            m_col3.metric("季線 (MA60)", market_info["季線 (MA60)"])
            m_col4.metric("約當大台基礎", market_info["自動運算基準價值 (約當大台基礎)"])
            st.info(f"**大盤環境判定：** {market_info['大盤環境判定']}")
            st.markdown("---")
            
        res_df = st.session_state.scan_res_df
        algo_mode_saved = st.session_state.get("scan_algo_mode", "全部")
        target_mode_saved = st.session_state.get("scan_target_mode_saved", "全市場掃描")
        
        if not res_df.empty:
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載建議清單 (CSV)",
                data=csv,
                file_name=f'stock_scan_{algo_mode_saved}_results.csv',
                mime='text/csv'
            )
            
            # --- 若為指定個股測試模式，印出每日指標詳細表供參數調整參考 ---
            if target_mode_saved == '指定個股測試' and 'test_stock_df' in st.session_state:
                st.markdown("---")
                st.markdown(f"### 🛠️ 參數測試詳細指標結果 - {st.session_state.get('test_ticker_name', '')}")
                st.write("顯示回測設定區間內的每日指標計算結果，協助您觀察分數變化與調整門檻參數。")
                
                debug_df = st.session_state.test_stock_df.copy()
                if not debug_df.empty:
                    # 擷取使用者設定的回測區間 (往前多抓10天做對照)
                    start_offset = st.session_state.lookback_start + 10
                    end_offset = st.session_state.lookback_end
                    
                    start_idx = max(0, len(debug_df) - 1 - start_offset)
                    end_idx = max(1, len(debug_df) - end_offset)
                    
                    debug_df = debug_df.iloc[start_idx:end_idx].copy()
                    
                    debug_df['BB_Width(%)'] = (debug_df['BB_Upper'] - debug_df['BB_Lower']) / (debug_df['MA20'] + 1e-8) * 100
                    
                    show_cols = ['Close', 'Volume_Lots', 'Vol_MA20', 'Candle_Score', 'VCP_Score', 'Reso_Score', 'K', 'D', 'MACD', 'MACD_Hist', 'BB_Width(%)']
                    show_cols = [c for c in show_cols if c in debug_df.columns]
                    
                    disp_df = debug_df[show_cols].sort_index(ascending=False)
                    disp_df.index = disp_df.index.strftime('%Y-%m-%d')
                    
                    format_dict = {
                        'Close': "{:.2f}", 'Volume_Lots': "{:.0f}", 'Vol_MA20': "{:.0f}",
                        'Candle_Score': "{:.2f}", 'VCP_Score': "{:.2f}", 'Reso_Score': "{:.2f}",
                        'K': "{:.2f}", 'D': "{:.2f}", 'MACD': "{:.3f}", 'MACD_Hist': "{:.3f}", 'BB_Width(%)': "{:.2f}"
                    }
                    
                    st.dataframe(disp_df.style.format(format_dict), use_container_width=True)
            
            # --- 跨分頁連動功能 ---
            st.markdown("---")
            st.markdown("### 🧬 進階基本面健檢連動")
            
            def transfer_to_audit():
                tickers_str = ", ".join(res_df['股票代號'].astype(str).tolist())
                st.session_state.batch_input_area = tickers_str
                st.session_state.run_batch_audit = True
                st.session_state.audit_mode = "自選股批次掃描"
                
            st.button("🚀 將上述標的傳送至【財報雙軌健檢】(第3分頁) 進行基本面分析", type="primary", on_click=transfer_to_audit)
            
            if st.session_state.get('run_batch_audit', False):
                st.success("✅ **傳送成功！已自動載入標的。** 請手動點擊上方的 **【📑 財報體質與雙軌健檢 (基本面 V4)】** 分頁查看綜合評分（系統將自動從最佳到最差排序）。")

# ----------------------------------------------------
# 頁籤 3：財報體質與雙軌健檢 (基本面 V4)
# ----------------------------------------------------
def fmt_val(val, suffix="", is_int=False):
    """安全格式化數值，避免 NaN 報錯"""
    if pd.isna(val): return "-"
    if is_int: return f"{int(val):,}{suffix}"
    return f"{val:,.2f}{suffix}"

with tab3:
    st.write("透過財報三表（損益表、資產負債表、現金流量表）交叉勾稽，提供**「防雷風險檢驗」**與**「優良體質評鑑」**雙軌判定，挖掘具備護城河、真實造血力與安全結構的卓越企業。")
    
    # 使用 Session State 同步切換狀態
    audit_mode = st.radio("請選擇操作模式", ["單檔查詢", "自選股批次掃描"], horizontal=True, key="audit_mode")
    
    if audit_mode == "單檔查詢":
        c1, c2 = st.columns([4, 1])
        with c1: 
            audit_input = st.text_input("輸入單一股票代號", value="2330", key="audit_single").strip()
        with c2: 
            st.write(""); st.write("")
            audit_btn = st.button("健檢財報", type="primary", use_container_width=True)
            
        if audit_btn and audit_input:
            with st.spinner(f"⏳ 正在抓取 [{audit_input}] 季報資料並進行雙軌體質比對..."):
                stock_dict = get_all_tw_stocks()
                yf_ticker = resolve_ticker(audit_input, stock_dict)
                name = stock_dict.get(yf_ticker, audit_input)
                
                res = FinancialAuditStrategy.evaluate(yf_ticker, get_yf_session())
                if res["status"] == "error":
                    st.error(res["msg"])
                else:
                    st.subheader(f"📑 {name} ({audit_input}) 財報體質與風險健檢報告 (V4)")
                    st.markdown(f"**當期季報：{res['當期季報']} | 比較基準：{res['比較基準']}**")
                    
                    st.markdown(f"### 綜合評級：{res['綜合評級']}")
                    
                    c_sc1, c_sc2, c_sc3 = st.columns(3)
                    with c_sc1:
                        st.metric("🏆 體質良好評分 (滿分100)", f"{res['體質分數']} 分")
                    with c_sc2:
                        st.metric("🛡️ 地雷風險評分 (越低越好)", f"{res['風險分數']} 分")
                    with c_sc3:
                        st.metric("⚠️ 警訊項目數", f"{res['警訊數']} 項")
                    
                    st.info(f"**判定結論：** {res['結論']}")
                    
                    if res["警訊數"] > 0:
                        st.error(f"⚠️ 發現 {res['警訊數']} 項地雷/警示訊號：\n\n" + res["診斷明細"])
                    if res["正向訊號"] != "無顯著優質體質特徵":
                        st.success(f"🌟 優良體質護城河特徵：\n\n" + res["正向訊號"])
                        
                    st.markdown("---")
                    st.markdown("### 📊 獲利能力與護城河指標")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("營收 YoY", fmt_val(res['營收成長(%)'], "%"))
                    col_b.metric("毛利率", fmt_val(res['毛利率'], "%"))
                    col_c.metric("營業利益率", fmt_val(res['營益率'], "%"))
                    col_d.metric("稅後淨利率", fmt_val(res['淨利率'], "%"))
                    
                    col_e, col_f, col_g, col_h = st.columns(4)
                    col_e.metric("稅後淨利 (千)", fmt_val(res['稅後淨利(千)'], is_int=True))
                    col_f.metric("營業現金流 CFO (千)", fmt_val(res['營業現金流CFO(千)'], is_int=True))
                    col_g.metric("自由現金流 FCF (千)", fmt_val(res['自由現金流FCF(千)'], is_int=True))
                    col_h.metric("CFO / 淨利比", fmt_val(res['CFO/淨利'], " 倍"))
                    
                    st.markdown("### 🔄 營運效率與週轉天數 (DIO 已校正為銷貨成本計算)")
                    col_i, col_j, col_k, col_l = st.columns(4)
                    col_i.metric("應收收現天數 (DSO)", fmt_val(res['DSO_now'], " 天"))
                    col_j.metric("存貨週轉天數 (DIO)", fmt_val(res['DIO_now'], " 天"))
                    col_k.metric("應付週轉天數 (DPO)", fmt_val(res['DPO_now'], " 天"))
                    col_l.metric("現金轉換週期 (CCC)", fmt_val(res['CCC_now'], " 天"))
                    
                    st.markdown("### 🛡️ 資本效率與償債防禦結構")
                    col_m, col_n, col_o, col_p = st.columns(4)
                    col_m.metric("年化 ROE (估算)", fmt_val(res['年化ROE'], "%"))
                    col_n.metric("流動比率", fmt_val(res['流動比率']))
                    col_o.metric("負債權益比 (D/E)", fmt_val(res['Debt/Equity']))
                    col_p.metric("實質淨現金 (千)", fmt_val(res['淨現金(千)'], is_int=True))

    else:
        st.write("請貼上你想健檢的股票清單（可使用逗號、空白、或換行分隔），系統將產出包含**「體質評分」**與**「地雷風險」**的綜合比較清單。")
        
        # 接收來自第 2 分頁的標的字串
        batch_input = st.text_area("輸入自選股清單", height=100, key="batch_input_area")
        batch_btn = st.button("執行批次雙軌健檢", type="primary")
        
        # 如果使用者點擊按鈕，或是從第 2 分頁傳過來的自動執行指令被觸發
        if batch_btn or st.session_state.get('run_batch_audit', False):
            
            # 若為自動觸發，執行一次後便關閉開關，避免無限迴圈
            if st.session_state.get('run_batch_audit', False):
                st.session_state.run_batch_audit = False
                
            raw_tickers = re.split(r'[,\s\n]+', batch_input.strip())
            valid_tickers = [t.strip() for t in raw_tickers if t.strip()]
            
            if not valid_tickers:
                st.warning("⚠️ 找不到有效的股票代號。")
            else:
                stock_dict = get_all_tw_stocks()
                session = get_yf_session()
                
                batch_results = []
                status_text = st.empty()
                p_bar = st.progress(0)
                
                for idx, t_input in enumerate(valid_tickers):
                    status_text.text(f"⏳ 正在分析 {t_input} ... ({idx+1}/{len(valid_tickers)})")
                    yf_t = resolve_ticker(t_input, stock_dict)
                    name = stock_dict.get(yf_t, t_input)
                    
                    res = FinancialAuditStrategy.evaluate(yf_t, session)
                    if res["status"] == "success":
                        batch_results.append({
                            "代號": t_input,
                            "名稱": name,
                            "當期季報": res["當期季報"],
                            "綜合評級": res["綜合評級"],
                            "體質分數": res["體質分數"],
                            "風險分數": res["風險分數"],
                            "營收YoY(%)": fmt_val(res['營收成長(%)']),
                            "毛利率(%)": fmt_val(res['毛利率']),
                            "營益率(%)": fmt_val(res['營益率']),
                            "CFO/淨利(倍)": fmt_val(res['CFO/淨利']),
                            "CCC週期(天)": fmt_val(res['CCC_now']),
                            "收現天數(DSO)": fmt_val(res['DSO_now']),
                            "週轉天數(DIO)": fmt_val(res['DIO_now']),
                            "警訊數": res["警訊數"],
                            "護城河特徵": res["正向訊號"].replace("\n", " | "),
                            "警示明細": res["診斷明細"].replace("\n", " | ")
                        })
                    time.sleep(0.5) 
                    p_bar.progress(min(1.0, (idx + 1) / len(valid_tickers)))
                    
                status_text.empty()
                p_bar.empty()
                
                if batch_results:
                    st.success(f"🎉 批次健檢完成！共成功分析 {len(batch_results)} 檔標的。")
                    batch_df = pd.DataFrame(batch_results)
                    
                    batch_df = batch_df.sort_values(by=["體質分數", "風險分數"], ascending=[False, True]).reset_index(drop=True)
                    batch_df.index = batch_df.index + 1
                    
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下載雙軌財報健檢報告 (CSV)",
                        data=csv,
                        file_name=f'financial_audit_V4_{datetime.date.today()}.csv',
                        mime='text/csv'
                    )
                else:
                    st.error("❌ 清單內的所有標的皆無法取得有效財報數據，請確認代號正確性或 Yahoo 資料庫狀態。")
