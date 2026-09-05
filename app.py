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
# 輸出欄位顯示設定 (程式內部維護，不與 UI 參數混在一起)
# ==========================================
OUTPUT_COLUMN_CONFIG = {
    "show_ma120_240_bear_align": True,  # 顯示 年線/半年線空排 (120MA < 240MA)
    "show_ma240_slope": True,           # 顯示 年線斜率(%)
    "show_ma120_slope": True            # 顯示 半年線斜率(%)
}

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
    "reso_price_basis": "最低價 (Low)",
    "reso_macd_wave_logic": False,
    "require_cross_confirm": True,
    "use_cross_position_filter": True,
    "cross_position_threshold": 70.0,
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
                 require_price_higher_low=False, require_macd_cross_zero=False,
                 reso_price_basis="最低價 (Low)", use_macd_wave_logic=False, require_cross_confirm=True):
        if len(df) < recent_w + older_w:
            return pd.Series(0.0, index=df.index)

        price_col = 'Low' if reso_price_basis == "最低價 (Low)" else 'Close'

        recent_k_low = df['K'].rolling(window=recent_w, min_periods=1).min()
        older_k_low = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        older_k_high = df['K'].shift(recent_w).rolling(window=older_w, min_periods=1).max()
        
        if use_macd_wave_logic:
            sign = np.sign(df['MACD_Hist'])
            sign = sign.replace(0, np.nan).ffill().fillna(1)
            block_id = (sign != sign.shift(1)).cumsum().rename('block_id')
            
            neg_mask = sign < 0
            
            if not neg_mask.any():
                recent_macd_low = pd.Series(np.nan, index=df.index)
                older_macd_low = pd.Series(np.nan, index=df.index)
            else:
                block_mins = df['MACD_Hist'].groupby(block_id).transform('min')
                recent_val = pd.Series(np.nan, index=df.index)
                recent_val.loc[neg_mask] = block_mins[neg_mask]
                recent_val = recent_val.ffill()
                
                unique_neg_blocks = df.loc[neg_mask, 'MACD_Hist'].groupby(block_id).min().reset_index()
                unique_neg_blocks['prev_min'] = unique_neg_blocks['MACD_Hist'].shift(1)
                prev_min_dict = dict(zip(unique_neg_blocks['block_id'], unique_neg_blocks['prev_min']))
                
                older_val = pd.Series(np.nan, index=df.index)
                older_val.loc[neg_mask] = block_id[neg_mask].map(prev_min_dict)
                older_val = older_val.ffill()
                
                recent_macd_low = recent_val
                older_macd_low = older_val
        else:
            recent_macd_low = df['MACD_Hist'].rolling(window=recent_w, min_periods=1).min()
            older_macd_low = df['MACD_Hist'].shift(recent_w).rolling(window=older_w, min_periods=1).min()
        
        recent_price_low = df[price_col].rolling(window=recent_w, min_periods=1).min()
        older_price_low = df[price_col].shift(recent_w).rolling(window=older_w, min_periods=1).min()

        kd_cross = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        
        if require_macd_cross_zero:
            macd_signal = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        else:
            macd_turn_up = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))
            macd_signal = macd_turn_up | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))

        kd_recent_cross = kd_cross.rolling(window=cross_days, min_periods=1).max() >= 1
        macd_recent_cross = macd_signal.rolling(window=cross_days, min_periods=1).max() >= 1

        if require_cross_confirm:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th) & kd_recent_cross
            cond_macd = (recent_macd_low > older_macd_low) & macd_recent_cross
        else:
            cond_kd = (older_k_low < kd_older_low_th) & (older_k_high < kd_older_high_th) & (recent_k_low > kd_recent_low_th)
            cond_macd = (recent_macd_low > older_macd_low)

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
        df, price_col='Low', ind_col='K', cross_col1='K', cross_col2='D',
        recent_w=5, older_w=20, recent_lows_cnt=0, older_lows_cnt=0,
        pivot_left=0, pivot_right=0, require_cross_confirm=True, cross_days=3
    ):
        if len(df) < recent_w + older_w: 
            return False
            
        if require_cross_confirm:
            cross = (df[cross_col1] > df[cross_col2]) & (df[cross_col1].shift(1) <= df[cross_col2].shift(1))
            if not cross.iloc[-cross_days:].any():
                return False

        recent_start = len(df) - recent_w
        older_start = recent_start - older_w
        older_end = recent_start
        
        prices = df[price_col].values
        inds = df[ind_col].values
        
        if recent_lows_cnt == 0 and older_lows_cnt == 0 and pivot_left == 0 and pivot_right == 0:
            r_p_min = np.min(prices[recent_start:])
            o_p_min = np.min(prices[older_start:older_end])
            r_i_min = np.min(inds[recent_start:])
            o_i_min = np.min(inds[older_start:older_end])
            
            return (r_p_min <= o_p_min * 1.02) and (r_i_min > o_i_min)

        def get_valid_pivots_iloc(start_loc, end_loc):
            pivots = []
            for i_loc in range(start_loc, end_loc):
                s, e = max(0, i_loc - pivot_left), min(len(prices), i_loc + pivot_right + 1)
                if prices[i_loc] == np.min(prices[s:e]): pivots.append(i_loc)
            return pivots

        r_pivots = get_valid_pivots_iloc(recent_start, len(df))
        o_pivots = get_valid_pivots_iloc(older_start, older_end)
        
        if recent_lows_cnt > 0: r_pivots = sorted(r_pivots, key=lambda x: prices[x])[:recent_lows_cnt]
        if older_lows_cnt > 0: o_pivots = sorted(o_pivots, key=lambda x: prices[x])[:older_lows_cnt]
        
        if not r_pivots: r_pivots = [recent_start + np.argmin(prices[recent_start:])]
        if not o_pivots: o_pivots = [older_start + np.argmin(prices[older_start:older_end])]
        
        for r_idx in r_pivots:
            for o_idx in o_pivots:
                r_i_val = np.min(inds[max(recent_start, r_idx-2) : min(len(inds), r_idx+3)])
                o_i_val = np.min(inds[max(older_start, o_idx-2) : min(older_end, o_idx+3)])
                if prices[r_idx] <= prices[o_idx] * 1.02 and r_i_val > o_i_val:
                    return True
        return False

# ==========================================
# 模組 3：三表交叉分析與異常診斷引擎 (V5 升級版)
# ==========================================
class FinancialAuditStrategy:
    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value) or value is None: return np.nan
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
                if hasattr(dt, "tz_localize") and dt.tz is not None:
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
    def evaluate(ticker_symbol, session):
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            inc_stmt = FinancialAuditStrategy._normalize_statement(stock.quarterly_income_stmt)
            bal_sheet = FinancialAuditStrategy._normalize_statement(stock.quarterly_balance_sheet)
            cash_flow = FinancialAuditStrategy._normalize_statement(stock.quarterly_cashflow)

            if inc_stmt.empty and bal_sheet.empty and cash_flow.empty:
                return {"status": "error", "msg": f"無法取得 {ticker_symbol} 財報資料。"}

            # Alias Mapping
            rev_aliases = ["Total Revenue", "Operating Revenue", "Revenue"]
            ni_aliases = ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
            op_aliases = ["Operating Income", "Operating Profit"]
            gp_aliases = ["Gross Profit"]
            cogs_aliases = ["Cost Of Revenue", "Cost of Goods Sold", "Cost Of Goods And Services Sold", "Operating Cost"]
            cfo_aliases = ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]
            ar_aliases = ["Accounts Receivable", "Net Receivables", "Receivables"]
            inv_aliases = ["Inventory", "Inventories"]
            ap_aliases = ["Accounts Payable", "Payables And Accrued Expenses", "Payables"]
            cl_aliases = ["Contract Liabilities", "Contract Liability", "Contract Liabilities Current", 
                          "Current Contract Liabilities", "ContractWithCustomerLiabilityCurrent", 
                          "ContractWithCustomerLiabilitiesCurrent", "ContractWithCustomerLiability", 
                          "ContractWithCustomerLiabilities", "Deferred Revenue", "Deferred Revenue Current", 
                          "Current Deferred Revenues", "Deferred Revenues", "Advances From Customers", "Unearned Revenue"]
            cash_aliases = ["Cash And Cash Equivalents", "Cash Financial", "Cash"]
            debt_aliases = ["Total Debt", "Long Term Debt", "Short Long Term Debt", "Current Debt"]
            asset_aliases = ["Total Assets"]
            liab_aliases = ["Total Liabilities"]
            equity_aliases = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]

            revenue = FinancialAuditStrategy._get_series(inc_stmt, rev_aliases)
            net_income = FinancialAuditStrategy._get_series(inc_stmt, ni_aliases)
            operating_income = FinancialAuditStrategy._get_series(inc_stmt, op_aliases)
            gross_profit = FinancialAuditStrategy._get_series(inc_stmt, gp_aliases)
            cogs = FinancialAuditStrategy._get_series(inc_stmt, cogs_aliases)
            
            cfo = FinancialAuditStrategy._get_series(cash_flow, cfo_aliases)
            
            ar = FinancialAuditStrategy._get_series(bal_sheet, ar_aliases)
            inventory = FinancialAuditStrategy._get_series(bal_sheet, inv_aliases)
            ap = FinancialAuditStrategy._get_series(bal_sheet, ap_aliases)
            contract_liab = FinancialAuditStrategy._get_series(bal_sheet, cl_aliases)
            cash = FinancialAuditStrategy._get_series(bal_sheet, cash_aliases)
            debt = FinancialAuditStrategy._get_series(bal_sheet, debt_aliases)
            t_assets = FinancialAuditStrategy._get_series(bal_sheet, asset_aliases)
            t_liab = FinancialAuditStrategy._get_series(bal_sheet, liab_aliases)
            equity = FinancialAuditStrategy._get_series(bal_sheet, equity_aliases)

            # Get Common Periods (Up to 8 quarters)
            valid_dfs = [df for df in [inc_stmt, bal_sheet, cash_flow] if not df.empty]
            all_periods = set()
            for df in valid_dfs: all_periods.update(df.columns)
            if not all_periods: return {"status": "error", "msg": "無財報期間資料"}
            
            sorted_periods = sorted(list(all_periods))
            periods_8q = sorted_periods[-8:] if len(sorted_periods) >= 8 else sorted_periods
            periods_4q = sorted_periods[-4:] if len(sorted_periods) >= 4 else sorted_periods
            
            latest = sorted_periods[-1]
            q1_ago = sorted_periods[-2] if len(sorted_periods) >= 2 else None
            q4_ago = sorted_periods[-5] if len(sorted_periods) >= 5 else None

            def val_at(s, p): return FinancialAuditStrategy._safe_float(s.loc[p]) if not s.empty and p in s.index else np.nan
            def get_trend(s, periods): return [val_at(s, p) for p in periods]
            
            # --- Extract Latest Values ---
            v_rev = val_at(revenue, latest)
            v_ni = val_at(net_income, latest)
            v_cfo = val_at(cfo, latest)
            v_gp = val_at(gross_profit, latest)
            v_op = val_at(operating_income, latest)
            v_cogs = val_at(cogs, latest)
            v_ar = val_at(ar, latest)
            v_inv = val_at(inventory, latest)
            v_ap = val_at(ap, latest)
            v_cl = val_at(contract_liab, latest)
            v_cash = val_at(cash, latest)
            v_debt = val_at(debt, latest) if pd.notna(val_at(debt, latest)) else 0.0
            
            # --- Extract Trends (4Q) ---
            t4_rev = get_trend(revenue, periods_4q)
            t4_ni = get_trend(net_income, periods_4q)
            t4_cfo = get_trend(cfo, periods_4q)
            t4_ar = get_trend(ar, periods_4q)
            t4_inv = get_trend(inventory, periods_4q)
            t4_cl = get_trend(contract_liab, periods_4q)
            
            # --- Extract Trends (8Q) ---
            t8_cl = get_trend(contract_liab, periods_8q)

            # --- Calculate Core Metrics ---
            rev_qoq = FinancialAuditStrategy._growth(v_rev, val_at(revenue, q1_ago))
            rev_yoy = FinancialAuditStrategy._growth(v_rev, val_at(revenue, q4_ago))
            ni_qoq = FinancialAuditStrategy._growth(v_ni, val_at(net_income, q1_ago))
            ni_yoy = FinancialAuditStrategy._growth(v_ni, val_at(net_income, q4_ago))
            cfo_qoq = FinancialAuditStrategy._growth(v_cfo, val_at(cfo, q1_ago))
            cfo_yoy = FinancialAuditStrategy._growth(v_cfo, val_at(cfo, q4_ago))
            
            cl_qoq = FinancialAuditStrategy._growth(v_cl, val_at(contract_liab, q1_ago))
            cl_yoy = FinancialAuditStrategy._growth(v_cl, val_at(contract_liab, q4_ago))
            
            ar_qoq = FinancialAuditStrategy._growth(v_ar, val_at(ar, q1_ago))
            inv_qoq = FinancialAuditStrategy._growth(v_inv, val_at(inventory, q1_ago))
            
            gm_latest = FinancialAuditStrategy._ratio(v_gp, v_rev) * 100
            om_latest = FinancialAuditStrategy._ratio(v_op, v_rev) * 100
            
            cfo_ni_ratio = FinancialAuditStrategy._ratio(v_cfo, v_ni)
            cl_rev_ratio = FinancialAuditStrategy._ratio(v_cl, v_rev) * 100
            
            dso = (v_ar / v_rev) * 90 if (pd.notna(v_rev) and v_rev > 0 and pd.notna(v_ar)) else np.nan
            if pd.notna(v_cogs) and v_cogs > 0 and pd.notna(v_inv):
                inv_days = (v_inv / v_cogs) * 90
            elif pd.notna(v_rev) and v_rev > 0 and pd.notna(v_inv):
                inv_days = (v_inv / v_rev) * 90
            else:
                inv_days = np.nan

            # ==========================================
            # 交叉診斷與風險共振分析引擎
            # ==========================================
            cross_analysis = []
            risk_resonance = []
            positives = []
            watch_list = []
            
            has_cl = not contract_liab.empty and not pd.isna(v_cl)
            
            # --- 1. 合約負債判定邏輯 (Contract Liabilities Logic) ---
            cl_status = "無資料"
            if has_cl:
                cl_4q_trend = sum([1 for i in range(1, len(t4_cl)) if pd.notna(t4_cl[i]) and pd.notna(t4_cl[i-1]) and t4_cl[i] > t4_cl[i-1]])
                if pd.notna(cl_yoy) and cl_yoy > 10 and pd.notna(rev_yoy) and rev_yoy > 5 and v_cfo > 0:
                    cl_status = "🟢 正向 (客戶預付增加且轉化為營收與現金)"
                    cross_analysis.append("【合約負債 → 營收/現金】合約負債年增且帶動營收與 CFO 成長，顯示訂單能見度高且確實收到現金。")
                    positives.append("客戶預付款/合約負債帶動營運資金正向循環。")
                elif pd.notna(cl_yoy) and cl_yoy > 5:
                    if pd.notna(rev_yoy) and rev_yoy <= 0:
                        cl_status = "🟡 觀察 (負債增加但營收尚未發酵)"
                        cross_analysis.append("【合約負債 → 營收】合約負債增加，但後續營收尚未明顯增長，需觀察下季遞延收入認列狀況。")
                        watch_list.append("合約負債是否能順利轉化為實際營收。")
                    else:
                        cl_status = "🟢 正向 (訂單/預付款穩健增加)"
                        cross_analysis.append("【合約負債】客戶預付款與遞延收入穩健增加，具備未來營收認列潛力。")
                elif pd.notna(cl_yoy) and cl_yoy < -15 and pd.notna(rev_yoy) and rev_yoy < 0:
                    cl_status = "🔴 可能異常 (訂單動能衰退)"
                    cross_analysis.append("【合約負債 → 營收】合約負債與營收同步衰退，可能代表整體市場需求與客戶下單動能減弱。")
                    risk_resonance.append("【訂單動能萎縮】合約負債大幅下降伴隨營收衰退，未來業績缺乏支撐。")
                else:
                    cl_status = "⚪ 中性波動"
            else:
                cl_status = "⚪ 合約負債資料不足，未納入此項評分"

            # --- 2. 存貨與合約負債/營收共振判定 (Inventory Risk Mitigation) ---
            if pd.notna(inv_qoq) and inv_qoq > 15:
                if has_cl and pd.notna(cl_qoq) and cl_qoq > 10 and pd.notna(rev_qoq) and rev_qoq > 5:
                    cross_analysis.append("【存貨 ↔ 合約負債/營收】存貨雖大增，但合約負債與營收同步成長，屬於「強烈需求帶動之健康備貨」，大幅抵銷存貨滯銷風險。")
                    positives.append("存貨增加具備合約負債與營收成長作為實質支撐。")
                elif pd.notna(rev_qoq) and rev_qoq < 0 and v_cfo < 0:
                    cross_analysis.append("【存貨 ↔ 營收/CFO】營收衰退且 CFO 轉負下，存貨卻大幅上升，營運資金遭嚴重積壓。")
                    risk_resonance.append("【存貨去化危機】營收與現金流轉弱，但存貨持續堆積，具備庫存跌價與資金斷鏈風險。")
                    watch_list.append("密切追蹤存貨週轉天數是否持續攀升、是否有打呆帳風險。")
                else:
                    cross_analysis.append("【存貨 ↔ 營收】存貨急升但缺乏相應幅度的合約負債或營收支撐，需謹慎觀察去化能力。")
                    watch_list.append("存貨去化速度與後續營收是否跟上。")

            # --- 3. 獲利轉現金能力 (NI to CFO) ---
            if pd.notna(v_ni) and v_ni > 0 and pd.notna(v_cfo) and v_cfo < 0:
                cfo_neg_count = sum([1 for val in t4_cfo if pd.notna(val) and val < 0])
                if cfo_neg_count >= 2:
                    cross_analysis.append("【獲利 ↔ CFO】連續多季帳面淨利為正，但營業現金流持續為負，獲利品質嚴重堪慮。")
                    risk_resonance.append("【黑字倒閉風險】長期獲利無法轉化為現金，公司靠吃老本或借款維持營運資金，屬重大財務警訊。")
                    watch_list.append("CFO 必須盡快轉正，否則面臨流動性風險。")
                else:
                    cross_analysis.append("【獲利 ↔ CFO】本季淨利為正但 CFO 為負，可能因短期營運資金墊高（如應收或存貨增加）所致。")
                    watch_list.append("確認下季 CFO 是否能回穩轉正。")
            elif pd.notna(v_ni) and v_ni > 0 and pd.notna(v_cfo) and v_cfo > v_ni * 1.2:
                cross_analysis.append("【獲利 ↔ CFO】CFO 顯著大於稅後淨利，獲利含金量極高。")
                positives.append("獲利轉現金能力極強，盈餘品質優良。")

            # --- 4. 應收帳款風險 (AR to Revenue) ---
            if pd.notna(ar_qoq) and ar_qoq > 20 and pd.notna(rev_qoq) and rev_qoq < 5:
                cross_analysis.append("【營收 ↔ 應收帳款】營收成長停滯，但應收帳款卻大幅增加，有塞貨或放寬授信條件的疑慮。")
                risk_resonance.append("【應收帳款惡化】資金卡在客戶端，可能衍生呆帳風險並吃掉公司現金。")
                watch_list.append("應收帳款天數是否失控攀升。")
            
            # --- 5. 總結評分與等級 ---
            quality_score = 50
            risk_score = 20
            
            # 加分項
            if pd.notna(gm_latest) and gm_latest > 30: quality_score += 10
            if pd.notna(om_latest) and om_latest > 10: quality_score += 15
            if has_cl and "🟢 正向" in cl_status: quality_score += 15; risk_score -= 10
            if pd.notna(cfo_ni_ratio) and cfo_ni_ratio > 1.0: quality_score += 15; risk_score -= 10
            
            # 扣分項
            if pd.notna(inv_days) and inv_days > 150: risk_score += 20; quality_score -= 10
            if pd.notna(dso) and dso > 120: risk_score += 15; quality_score -= 5
            if pd.notna(v_cfo) and v_cfo < 0: risk_score += 25
            if pd.notna(v_cash) and pd.notna(v_debt) and v_debt > v_cash * 1.5: risk_score += 15
            
            # Risk Resonance Adjustments
            if any("黑字倒閉風險" in r for r in risk_resonance): risk_score += 30
            if any("存貨去化危機" in r for r in risk_resonance): risk_score += 20
            if any("抵銷存貨滯銷風險" in r for r in cross_analysis): risk_score = max(10, risk_score - 20)

            quality_score = min(100, max(0, int(quality_score)))
            risk_score = min(100, max(0, int(risk_score)))

            if risk_score >= 60:
                overall_grade = "🔴 高度風險 (地雷特徵明顯)"
            elif risk_score >= 40:
                overall_grade = "🟠 觀察警戒 (財務結構存在弱點)"
            elif quality_score >= 75 and risk_score < 30:
                overall_grade = "💎 卓越績優 (護城河深厚，營運資金健康)"
            else:
                overall_grade = "🟡 穩健平庸 (無重大風險但動能一般)"

            # ==========================================
            # 建立回傳 Dict (相容 Batch Mode 且支援 Single Mode 顯示)
            # ==========================================
            def fmt(v, suffix=""): return f"{v:,.2f}{suffix}" if pd.notna(v) else "-"
            
            # Core Table Generation Data
            core_metrics = [
                {"指標": "營業收入 (Revenue)", "最新": fmt(v_rev), "QoQ": fmt(rev_qoq, "%"), "YoY": fmt(rev_yoy, "%"), "4Q趨勢": t4_rev},
                {"指標": "稅後淨利 (Net Income)", "最新": fmt(v_ni), "QoQ": fmt(ni_qoq, "%"), "YoY": fmt(ni_yoy, "%"), "4Q趨勢": t4_ni},
                {"指標": "營業現金流 (CFO)", "最新": fmt(v_cfo), "QoQ": fmt(cfo_qoq, "%"), "YoY": fmt(cfo_yoy, "%"), "4Q趨勢": t4_cfo},
                {"指標": "應收帳款 (AR)", "最新": fmt(v_ar), "QoQ": fmt(ar_qoq, "%"), "YoY": "-", "4Q趨勢": t4_ar},
                {"指標": "存貨 (Inventory)", "最新": fmt(v_inv), "QoQ": fmt(inv_qoq, "%"), "YoY": "-", "4Q趨勢": t4_inv},
                {"指標": "合約負債 (Contract Liab)", "最新": fmt(v_cl), "QoQ": fmt(cl_qoq, "%"), "YoY": fmt(cl_yoy, "%"), "4Q趨勢": t4_cl},
                {"指標": "毛利率 (Gross Margin)", "最新": fmt(gm_latest, "%"), "QoQ": "-", "YoY": "-", "4Q趨勢": []},
                {"指標": "營益率 (Operating Margin)", "最新": fmt(om_latest, "%"), "QoQ": "-", "YoY": "-", "4Q趨勢": []},
                {"指標": "存貨天數 (Inv Days)", "最新": fmt(inv_days, " 天"), "QoQ": "-", "YoY": "-", "4Q趨勢": []},
                {"指標": "收現天數 (DSO)", "最新": fmt(dso, " 天"), "QoQ": "-", "YoY": "-", "4Q趨勢": []},
            ]

            if not cross_analysis: cross_analysis.append("未偵測到明顯的三表交叉異常或正向特徵。")
            if not risk_resonance: risk_resonance.append("目前三表結構中，無重大風險共振鏈。")
            if not positives: positives.append("無顯著突出之護城河或營運資金優勢。")
            if not watch_list: watch_list.append("持續觀察基本面變化與毛利率維持能力。")

            return {
                "status": "success",
                "代號": ticker_symbol,
                "當期季報": latest.strftime("%Y-%m-%d") if hasattr(latest, "strftime") else str(latest),
                "比較基準": "最近 4~8 季綜合評估",
                "綜合評級": overall_grade,
                "體質分數": quality_score,
                "風險分數": risk_score,
                
                # Metrics for UI
                "core_metrics": core_metrics,
                
                # CL specific
                "cl_latest": fmt(v_cl),
                "cl_qoq": fmt(cl_qoq, "%"),
                "cl_yoy": fmt(cl_yoy, "%"),
                "cl_rev_ratio": fmt(cl_rev_ratio, "%"),
                "cl_status": cl_status,
                "cl_8q": t8_cl,
                
                # Narratives
                "cross_analysis": cross_analysis,
                "risk_resonance": risk_resonance,
                "positives": positives,
                "watch_list": watch_list,
                
                # For Batch Mode fallback compatibility
                "營收成長(%)": rev_qoq,
                "毛利率": gm_latest,
                "營益率": om_latest,
                "CFO/淨利": cfo_ni_ratio,
                "CCC_now": inv_days + dso if pd.notna(inv_days) and pd.notna(dso) else np.nan,
                "DSO_now": dso,
                "DIO_now": inv_days,
                "警訊數": len(risk_resonance) if risk_resonance[0] != "目前三表結構中，無重大風險共振鏈。" else 0,
                "正向訊號": " | ".join(positives),
                "診斷明細": " | ".join(risk_resonance)
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

tab1, tab2, tab3 = st.tabs(["📊 單檔深度解析", "🚀 全市場智慧掃描 (回測/翻轉/VCP/共振/背離)", "📑 財報三表地雷掃描／財報體質與風險分析"])

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
                    require_macd_cross_zero=st.session_state.reso_macd_cross_zero,
                    reso_price_basis=st.session_state.reso_price_basis,
                    use_macd_wave_logic=st.session_state.reso_macd_wave_logic,
                    require_cross_confirm=st.session_state.require_cross_confirm
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
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns([1, 1, 1, 1, 1.2])
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
        with col_r5:
            st.radio("價格比較基準", ["最低價 (Low)", "收盤價 (Close)"], key="reso_price_basis")
            st.checkbox("MACD零軸動能波段判定", key="reso_macd_wave_logic", help="開啟後，MACD自動以0軸分界波段，無懼靜態天數限制精確比對。")
            
        st.markdown("**6. 金叉確認與進場位置建議過濾**")
        col_x1, col_x2, col_x3 = st.columns(3)
        with col_x1:
            st.checkbox("嚴格要求金叉確認背離與共振", key="require_cross_confirm", help="啟用後，發生背離且在設定天數內發生指標金叉，才判定為有效。")
        with col_x2:
            st.checkbox("啟用進場位置建議 (K值過高提示)", key="use_cross_position_filter", help="依據觸發時的 K 值給予參考建議。")
        with col_x3:
            st.number_input("建議參考的 K值上限 <", step=1.0, key="cross_position_threshold", disabled=not st.session_state.use_cross_position_filter)

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
                        
                        # 計算近期金叉判定狀態 (供輸出呈現)
                        df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
                        if st.session_state.reso_macd_cross_zero:
                            df['MACD_Cross'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
                        else:
                            macd_turn_up = (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) & (df['MACD_Hist'].shift(1) <= df['MACD_Hist'].shift(2))
                            df['MACD_Cross'] = macd_turn_up | ((df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0))
                            
                        df['KD_Cross_Recent'] = df['KD_Cross'].rolling(window=st.session_state.reso_cross_days, min_periods=1).max() >= 1
                        df['MACD_Cross_Recent'] = df['MACD_Cross'].rolling(window=st.session_state.reso_cross_days, min_periods=1).max() >= 1
                        
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
                                require_macd_cross_zero=st.session_state.reso_macd_cross_zero,
                                reso_price_basis=st.session_state.reso_price_basis,
                                use_macd_wave_logic=st.session_state.reso_macd_wave_logic,
                                require_cross_confirm=st.session_state.require_cross_confirm
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
                            
                            # 評估進場位置建議
                            rec_status = "-"
                            if st.session_state.use_cross_position_filter:
                                k_val = float(best_row['K'])
                                if k_val < st.session_state.cross_position_threshold:
                                    rec_status = "✅ 建議參考"
                                else:
                                    rec_status = "⚠️ 位置偏高"

                            reversal_candidates[ticker] = {
                                "_Full_Ticker": ticker, "_Offset": best_offset, "_Daily_DF": df.copy(),
                                "股票代號": clean_ticker, "股票名稱": stock_dict[ticker],
                                "觸發日期": best_row.name.strftime('%Y-%m-%d'),
                                "當日收盤": round(float(best_row['Close']), 2),
                                "月均量(張)": int(best_row['Vol_MA20']),
                                "DIF": round(float(best_row['MACD']), 3),
                                "KD金叉判定": "✅ 是" if best_row['KD_Cross_Recent'] else "❌ 否",
                                "MACD金叉判定": "✅ 是" if best_row['MACD_Cross_Recent'] else "❌ 否",
                                "反轉分數": round(float(best_row['Candle_Score']), 2),
                                "VCP分數": round(float(best_row['VCP_Score']), 2),
                                "共振分數": round(float(best_row['Reso_Score']), 2),
                                "進場位置建議": rec_status
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
            candidate_tickers = [item["_Full_Ticker"] for item in reversal_list]
            progress_bar.progress(0)
            status_text.text(f"[階段二] 正在下載初步標的長天期資料以計算年線/半年線...")
            
            long_dl_kwargs = {"period": "2y"}
            if st.session_state.use_backtest_date:
                start_long = end_dt - datetime.timedelta(days=1000) # 確保有超過240天
                long_dl_kwargs = {"start": start_long.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
            
            long_data = yf.download(candidate_tickers, threads=False, progress=False, session=yf_session, **long_dl_kwargs)
            
            status_text.text(f"[階段二] 正在分析 {len(reversal_list)} 檔入選標的之多級別背離特徵與扣抵判定...")
            
            final_results = []
            price_col_for_div = 'Low' if st.session_state.reso_price_basis == "最低價 (Low)" else 'Close'
            req_cross = st.session_state.require_cross_confirm
            c_days = st.session_state.reso_cross_days
            
            for idx, item in enumerate(reversal_list):
                try:
                    ticker = item.pop("_Full_Ticker")
                    specific_offset = item.pop("_Offset")
                    daily_df = item.pop("_Daily_DF")
                    has_daily_div, has_m60_div = False, False
                    rl_cnt, ol_cnt = st.session_state.recent_lows_cnt, st.session_state.older_lows_cnt
                    p_left, p_right = st.session_state.pivot_left, st.session_state.pivot_right
                    
                    # === 年線與半年線計算 ===
                    item['年線/半年線空排'] = "無資料"
                    item['半年線斜率(%)'] = "無資料"
                    item['年線斜率(%)'] = "無資料"
                    
                    try:
                        if isinstance(long_data.columns, pd.MultiIndex):
                            if ticker in long_data.columns.get_level_values(0):
                                long_df = long_data.xs(ticker, axis=1, level=0).dropna(how='all')
                            elif ticker in long_data.columns.get_level_values(1):
                                long_df = long_data.xs(ticker, axis=1, level=1).dropna(how='all')
                            else:
                                long_df = pd.DataFrame()
                        else:
                            long_df = long_data.dropna(how='all') if len(candidate_tickers) == 1 else pd.DataFrame()
                            
                        if not long_df.empty:
                            if st.session_state.use_backtest_date:
                                target_dt = pd.to_datetime(st.session_state.backtest_date_obj)
                                long_df = long_df[long_df.index.tz_localize(None).normalize() <= target_dt]
                                
                            if specific_offset > 0:
                                long_df = long_df.iloc[:-specific_offset]
                                
                            if len(long_df) >= 120:
                                ma120 = long_df['Close'].rolling(120).mean()
                                c_ma120 = ma120.iloc[-1]
                                p_ma120 = ma120.iloc[-2]
                                item['半年線斜率(%)'] = round((c_ma120 - p_ma120) / p_ma120 * 100, 3)
                                
                            if len(long_df) >= 240:
                                ma240 = long_df['Close'].rolling(240).mean()
                                c_ma240 = ma240.iloc[-1]
                                p_ma240 = ma240.iloc[-2]
                                item['年線斜率(%)'] = round((c_ma240 - p_ma240) / p_ma240 * 100, 3)
                                
                                is_bear = c_ma120 < c_ma240
                                item['年線/半年線空排'] = "✅ 是" if is_bear else "❌ 否"
                    except Exception as e:
                        pass
                    # ==========================
                    
                    if not daily_df.empty:
                        for n in kou_di_periods:
                            if len(daily_df) >= n:
                                curr_p = daily_df['Close'].iloc[-1]
                                drop_p = daily_df['Close'].iloc[-n]
                                item[f"扣抵狀態({n}MA)"] = "✅ 扣低" if curr_p > drop_p else "❌ 扣高"
                            else:
                                item[f"扣抵狀態({n}MA)"] = "無資料"
                        
                        for r_w, o_w in div_pairs:
                            d_kd = DivergenceStrategy.check_bottom_divergence(
                                daily_df, price_col_for_div, 'K', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                            )
                            d_macd = DivergenceStrategy.check_bottom_divergence(
                                daily_df, price_col_for_div, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                            )
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
                                m_kd = DivergenceStrategy.check_bottom_divergence(
                                    m60_df, price_col_for_div, 'K', 'K', 'D', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                                )
                                m_macd = DivergenceStrategy.check_bottom_divergence(
                                    m60_df, price_col_for_div, 'MACD_Hist', 'MACD', 'MACD_Signal', r_w, o_w, rl_cnt, ol_cnt, p_left, p_right, req_cross, c_days
                                )
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
            # 在基礎欄位中加入新的 DIF、KD金叉判定、MACD金叉判定
            base_cols = ['股票代號', '股票名稱', '觸發日期', '演算法建議結果', '進場位置建議', 'DIF', 'KD金叉判定', 'MACD金叉判定', '反轉分數', 'VCP分數', '共振分數', '當日收盤', '月均量(張)']
            
            # 動態加入新增的 MA 欄位
            if OUTPUT_COLUMN_CONFIG.get("show_ma120_240_bear_align", True):
                base_cols.append('年線/半年線空排')
            if OUTPUT_COLUMN_CONFIG.get("show_ma240_slope", True):
                base_cols.append('年線斜率(%)')
            if OUTPUT_COLUMN_CONFIG.get("show_ma120_slope", True):
                base_cols.append('半年線斜率(%)')
                
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
            st.markdown("### 📋 自訂輸出欄位與順序")
            # 提供額外功能讓使用者調整順序，預設為固定好的 res_df.columns
            all_columns = res_df.columns.tolist()
            selected_cols = st.multiselect(
                "您可以新增/移除欄位，或依序點選、拖曳標籤來改變表格顯示順序：",
                options=all_columns,
                default=all_columns,
                key="custom_column_order"
            )
            
            # 若使用者將欄位全清空，則顯示空表格以免報錯
            display_df = res_df[selected_cols] if selected_cols else pd.DataFrame()
            
            if not display_df.empty:
                st.dataframe(display_df, use_container_width=True)
                
                csv = display_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下載建議清單 (CSV)",
                    data=csv,
                    file_name=f'stock_scan_{algo_mode_saved}_results.csv',
                    mime='text/csv'
                )
            else:
                st.warning("⚠️ 請至少選擇一個欄位來顯示資料。")
            
            # --- 若為指定個股測試模式，印出每日指標詳細表供參數調整參考 ---
            if target_mode_saved == '指定個股測試' and 'test_stock_df' in st.session_state:
                st.markdown("---")
                st.markdown(f"### 🛠️ 參數測試詳細指標結果 - {st.session_state.get('test_ticker_name', '')}")
                st.write("顯示回測設定區間內的每日指標計算結果，協助您觀察分數變化與調整門檻參數。")
                
                debug_df = st.session_state.test_stock_df.copy()
                if not debug_df.empty:
                    # 擷取使用者設定的回測區間 (往前多抓前波與近波範圍天數做對照)
                    if st.session_state.use_single_div:
                        max_older_w = st.session_state.div_older_w
                    else:
                        max_older_w = max(20, 60) # 預設的三組比對最大天數為60
                        
                    start_offset = st.session_state.lookback_start + max_older_w + st.session_state.div_recent_w
                    end_offset = st.session_state.lookback_end
                    
                    start_idx = max(0, len(debug_df) - 1 - start_offset)
                    end_idx = max(1, len(debug_df) - end_offset)
                    
                    debug_df = debug_df.iloc[start_idx:end_idx].copy()
                    
                    debug_df['BB_Width(%)'] = (debug_df['BB_Upper'] - debug_df['BB_Lower']) / (debug_df['MA20'] + 1e-8) * 100
                    
                    show_cols = ['Close', 'Low', 'Volume_Lots', 'Vol_MA20', 'Candle_Score', 'VCP_Score', 'Reso_Score', 'K', 'D', 'MACD', 'MACD_Hist', 'BB_Width(%)']
                    show_cols = [c for c in show_cols if c in debug_df.columns]
                    
                    disp_df = debug_df[show_cols].sort_index(ascending=False)
                    disp_df.index = disp_df.index.strftime('%Y-%m-%d')
                    
                    format_dict = {
                        'Close': "{:.2f}", 'Low': "{:.2f}", 'Volume_Lots': "{:.0f}", 'Vol_MA20': "{:.0f}",
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
                st.success("✅ **傳送成功！已自動載入標的。** 請手動點擊上方的 **【📑 財報三表地雷掃描／財報體質與風險分析】** 分頁查看綜合評分（系統將自動從最佳到最差排序）。")

# ----------------------------------------------------
# 頁籤 3：財報三表地雷掃描／財報體質與風險分析 (V5 升級版)
# ----------------------------------------------------
def render_trend_sparkline(trend_list):
    """將趨勢陣列轉換為簡單的視覺化文字"""
    if not trend_list or len(trend_list) < 2: return "資料不足"
    valid_vals = [v for v in trend_list if pd.notna(v)]
    if len(valid_vals) < 2: return "資料不足"
    
    res = []
    for i in range(1, len(valid_vals)):
        if valid_vals[i] > valid_vals[i-1]: res.append("↗")
        elif valid_vals[i] < valid_vals[i-1]: res.append("↘")
        else: res.append("→")
    return " ".join(res[-3:])  # 只顯示最近三次的變化符號

with tab3:
    st.write("透過財報三表（損益表、資產負債表、現金流量表）交叉勾稽，提供**「三表交叉異常診斷」**與**「合約負債防誤判引擎」**，挖掘具備真實造血力與安全結構的卓越企業。")
    
    # 使用 Session State 同步切換狀態
    audit_mode = st.radio("請選擇操作模式", ["單檔查詢", "自選股批次掃描"], horizontal=True, key="audit_mode")
    
    if audit_mode == "單檔查詢":
        c1, c2 = st.columns([4, 1])
        with c1: 
            audit_input = st.text_input("輸入單一股票代號", value="2495", key="audit_single").strip()
        with c2: 
            st.write(""); st.write("")
            audit_btn = st.button("健檢財報", type="primary", use_container_width=True)
            
        if audit_btn and audit_input:
            with st.spinner(f"⏳ 正在抓取 [{audit_input}] 季報資料並進行三表交叉診斷..."):
                stock_dict = get_all_tw_stocks()
                yf_ticker = resolve_ticker(audit_input, stock_dict)
                name = stock_dict.get(yf_ticker, audit_input)
                
                res = FinancialAuditStrategy.evaluate(yf_ticker, get_yf_session())
                if res["status"] == "error":
                    st.error(res["msg"])
                else:
                    st.subheader(f"📑 {name} ({audit_input}) 財報三表體質與風險診斷報告")
                    st.markdown(f"**當期季報：{res['當期季報']} | 分析期間：{res['比較基準']}**")
                    
                    st.markdown("---")
                    st.markdown("### 【一、總體診斷】")
                    st.markdown(f"#### 綜合評級：{res['綜合評級']}")
                    
                    c_sc1, c_sc2 = st.columns(2)
                    with c_sc1:
                        st.metric("🏆 體質綜合評分 (滿分100)", f"{res['體質分數']} 分")
                    with c_sc2:
                        st.metric("🛡️ 財務風險評分 (越低越好)", f"{res['風險分數']} 分")
                    
                    st.markdown("---")
                    st.markdown("### 【二、三表核心指標】")
                    
                    # 處理 4Q Trend 成視覺化文字
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
                    for text in res["cross_analysis"]:
                        st.markdown(f"- {text}")
                        
                    st.markdown("---")
                    st.markdown("### 【五、風險共振】")
                    for text in res["risk_resonance"]:
                        if "無重大風險" in text:
                            st.success(f"- {text}")
                        else:
                            st.error(f"- 🚨 {text}")
                            
                    st.markdown("---")
                    st.markdown("### 【六、正向特徵】")
                    for text in res["positives"]:
                        if "無顯著突出" in text:
                            st.warning(f"- {text}")
                        else:
                            st.success(f"- 🌟 {text}")
                            
                    st.markdown("---")
                    st.markdown("### 【七、下一季追蹤事項】")
                    for idx, text in enumerate(res["watch_list"]):
                        st.markdown(f"{idx+1}. {text}")

    else:
        st.write("請貼上你想健檢的股票清單（可使用逗號、空白、或換行分隔），系統將產出包含**「體質評分」**與**「財務風險」**的綜合比較清單。")
        
        batch_input = st.text_area("輸入自選股清單", height=100, key="batch_input_area")
        batch_btn = st.button("執行批次交叉健檢", type="primary")
        
        if batch_btn or st.session_state.get('run_batch_audit', False):
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
                            "合約負債判定": res["cl_status"].split(" ")[0] if " " in res["cl_status"] else res["cl_status"],
                            "營收YoY(%)": f"{res['營收成長(%)']:.2f}%" if pd.notna(res['營收成長(%)']) else "-",
                            "毛利率(%)": f"{res['毛利率']:.2f}%" if pd.notna(res['毛利率']) else "-",
                            "營益率(%)": f"{res['營益率']:.2f}%" if pd.notna(res['營益率']) else "-",
                            "CFO/淨利(倍)": f"{res['CFO/淨利']:.2f}" if pd.notna(res['CFO/淨利']) else "-",
                            "CCC週期(天)": f"{res['CCC_now']:.0f}" if pd.notna(res['CCC_now']) else "-",
                            "收現天數(DSO)": f"{res['DSO_now']:.0f}" if pd.notna(res['DSO_now']) else "-",
                            "週轉天數(DIO)": f"{res['DIO_now']:.0f}" if pd.notna(res['DIO_now']) else "-",
                            "警訊數": res["警訊數"],
                            "護城河特徵": res["正向訊號"],
                            "警示明細": res["診斷明細"]
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
                        label="📥 下載三表交叉財報健檢報告 (CSV)",
                        data=csv,
                        file_name=f'financial_cross_audit_{datetime.date.today()}.csv',
                        mime='text/csv'
                    )
                else:
                    st.error("❌ 清單內的所有標的皆無法取得有效財報數據，請確認代號正確性或 Yahoo 資料庫狀態。")
