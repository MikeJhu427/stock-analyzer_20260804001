import numpy as np
import pandas as pd
from typing import Dict, Optional
import strategy_core as core

MA_SQUEEZE_TAG = "均線壓縮一陽穿多線"

MA_SQUEEZE_COLUMNS = [
    "均線壓縮一陽穿多線",
    "均線壓縮極差(%)",
    "壓縮布林帶寬(%)",
    "突破量比(5MA)"
]

MA_SQUEEZE_DEFAULT_PARAMS = {
    "ma_squeeze_window": 5,
    "ma_max_spread_pct": 6.0,
    "bbw_max_pct": 12.0,
    "bbw_lookback": 60,
    "bbw_low_quantile": 0.50,
    "anti_choppy_window": 10,
    "max_ma20_crosses": 2,
    "max_kd_crosses": 2,
    "min_body_pct": 3.5,
    "max_upper_shadow_pct": 1.5,
    "min_open_below_mas": 2,
    "min_vol_mult_5ma": 1.8,
    "min_vol_lots": 1500.0,
    "kd_low_lookback": 15,
    "kd_div_lookback": 20,
    "kd_high_zone_th": 70.0,
    "macd_prior_lookback": 25,
    "macd_prior_min_hist": -0.15,
    "macd_cur_min_hist": -0.10,
    "macd_cur_max_hist": 0.25
}


class MASqueezeBreakoutStrategy:
    """
    獨立右側波段選股策略：「零模糊、均線極致壓縮後一陽穿多線」（一票否決制）。
    不參與既有 ALGO_REGISTRY 分數計算，獨立於第一階段平行判定。
    """

    @staticmethod
    def _merge_params(params: Optional[Dict] = None) -> Dict:
        cfg = MA_SQUEEZE_DEFAULT_PARAMS.copy()
        if isinstance(params, dict):
            for k in cfg.keys():
                if k in params:
                    cfg[k] = params[k]
        return cfg

    @staticmethod
    def _count_series_crosses(diff_values: np.ndarray) -> int:
        """統計序列在零軸上下的有效穿越次數（忽略剛好等於 0 的平盤點）。"""
        signs = np.sign(diff_values)
        non_zero = signs[signs != 0]
        if len(non_zero) < 2:
            return 0
        return int(np.sum(non_zero[1:] != non_zero[:-1]))

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Volume_Lots" not in out.columns:
            out["Volume_Lots"] = pd.to_numeric(out["Volume"], errors="coerce") / 1000.0
        if "K" not in out.columns or "D" not in out.columns:
            out = core.TechnicalIndicators.add_kd(out, n=9, m1=3, m2=3)
        if "MACD" not in out.columns or "MACD_Signal" not in out.columns or "MACD_Hist" not in out.columns:
            out = core.TechnicalIndicators.add_macd(out, fast=12, slow=26, signal=9)

        close = out["Close"].astype(float)
        out["SQ_MA5"] = close.rolling(5).mean()
        out["SQ_MA10"] = close.rolling(10).mean()
        out["SQ_MA20"] = close.rolling(20).mean()
        out["SQ_MA60"] = close.rolling(60).mean()
        out["SQ_MA120"] = close.rolling(120).mean()

        std20 = close.rolling(20).std()
        bb_upper = out["SQ_MA20"] + 2.0 * std20
        bb_lower = out["SQ_MA20"] - 2.0 * std20
        out["SQ_BBW_Pct"] = (bb_upper - bb_lower) / (out["SQ_MA20"] + 1e-8) * 100.0
        out["SQ_Vol_MA5"] = out["Volume_Lots"].astype(float).rolling(5).mean()

        return out

    @staticmethod
    def evaluate(df: Optional[pd.DataFrame], params: Optional[Dict] = None) -> pd.DataFrame:
        """
        逐日檢核 5 項硬性條件（一票否決制），回傳與輸入 df 等長之結果 DataFrame：
        - MA_Squeeze_Pass (bool)
        - MA_Squeeze_Spread_Pct (float)
        - MA_Squeeze_BBW_Pct (float)
        - MA_Squeeze_Vol_Mult5 (float)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "MA_Squeeze_Pass",
                "MA_Squeeze_Spread_Pct",
                "MA_Squeeze_BBW_Pct",
                "MA_Squeeze_Vol_Mult5"
            ])

        res = pd.DataFrame(index=df.index)
        res["MA_Squeeze_Pass"] = False
        res["MA_Squeeze_Spread_Pct"] = np.nan
        res["MA_Squeeze_BBW_Pct"] = np.nan
        res["MA_Squeeze_Vol_Mult5"] = np.nan

        if len(df) < 65:
            return res

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in required_cols):
            return res

        cfg = MASqueezeBreakoutStrategy._merge_params(params)
        work = MASqueezeBreakoutStrategy._prepare_dataframe(df)

        opens = work["Open"].astype(float).values
        highs = work["High"].astype(float).values
        closes = work["Close"].astype(float).values
        vols = work["Volume_Lots"].astype(float).values
        vol_ma5 = work["SQ_Vol_MA5"].astype(float).values

        ma5 = work["SQ_MA5"].astype(float).values
        ma10 = work["SQ_MA10"].astype(float).values
        ma20 = work["SQ_MA20"].astype(float).values
        ma60 = work["SQ_MA60"].astype(float).values
        ma120 = work["SQ_MA120"].astype(float).values
        bbw = work["SQ_BBW_Pct"].astype(float).values

        k_vals = work["K"].astype(float).values
        d_vals = work["D"].astype(float).values
        macd_hist = work["MACD_Hist"].astype(float).values

        sq_win = int(cfg["ma_squeeze_window"])
        max_spread_pct = float(cfg["ma_max_spread_pct"])
        bbw_max_pct = float(cfg["bbw_max_pct"])
        bbw_lookback = int(cfg["bbw_lookback"])
        bbw_low_q = float(cfg["bbw_low_quantile"])

        chop_win = int(cfg["anti_choppy_window"])
        max_ma20_crosses = int(cfg["max_ma20_crosses"])
        max_kd_crosses = int(cfg["max_kd_crosses"])

        min_body_pct = float(cfg["min_body_pct"])
        max_upper_shadow_pct = float(cfg["max_upper_shadow_pct"])
        min_open_below = int(cfg["min_open_below_mas"])
        min_vol_mult_5ma = float(cfg["min_vol_mult_5ma"])
        min_vol_lots = float(cfg["min_vol_lots"])

        kd_low_lb = int(cfg["kd_low_lookback"])
        kd_div_lb = int(cfg["kd_div_lookback"])
        kd_high_th = float(cfg["kd_high_zone_th"])

        macd_prior_lb = int(cfg["macd_prior_lookback"])
        macd_prior_min = float(cfg["macd_prior_min_hist"])
        macd_cur_min = float(cfg["macd_cur_min_hist"])
        macd_cur_max = float(cfg["macd_cur_max_hist"])

        pass_arr = np.zeros(len(work), dtype=bool)
        spread_arr = np.full(len(work), np.nan, dtype=float)
        bbw_arr = np.full(len(work), np.nan, dtype=float)
        vol_mult_arr = np.full(len(work), np.nan, dtype=float)

        start_idx = max(60, chop_win, macd_prior_lb, kd_div_lb)

        for i in range(start_idx, len(work)):
            o_i = opens[i]
            h_i = highs[i]
            c_i = closes[i]
            v_i = vols[i]
            v_ma5_i = vol_ma5[i]

            if any(np.isnan([o_i, h_i, c_i, v_i, v_ma5_i])) or o_i <= 0 or c_i <= 0 or v_ma5_i <= 0:
                continue

            # 判斷當日與過去 5 日是否具備完整 120MA；充足則納入 5 條均線，否則使用 4 條均線
            use_ma120 = not np.any(np.isnan(ma120[i - sq_win + 1:i + 1]))
            if use_ma120:
                ma_matrix = np.vstack([
                    ma5[i - sq_win + 1:i + 1],
                    ma10[i - sq_win + 1:i + 1],
                    ma20[i - sq_win + 1:i + 1],
                    ma60[i - sq_win + 1:i + 1],
                    ma120[i - sq_win + 1:i + 1]
                ])
                cur_mas = np.array([ma5[i], ma10[i], ma20[i], ma60[i], ma120[i]], dtype=float)
            else:
                ma_matrix = np.vstack([
                    ma5[i - sq_win + 1:i + 1],
                    ma10[i - sq_win + 1:i + 1],
                    ma20[i - sq_win + 1:i + 1],
                    ma60[i - sq_win + 1:i + 1]
                ])
                cur_mas = np.array([ma5[i], ma10[i], ma20[i], ma60[i]], dtype=float)

            if np.any(np.isnan(ma_matrix)) or np.any(ma_matrix <= 0):
                continue

            # 計算各項即時紀錄值（供報表呈現）
            ma_min_5d = np.min(ma_matrix, axis=0)
            ma_max_5d = np.max(ma_matrix, axis=0)
            spread_5d_pct = (ma_max_5d - ma_min_5d) / ma_min_5d * 100.0
            max_spread_5d = float(np.max(spread_5d_pct))
            cur_bbw = float(bbw[i]) if not np.isnan(bbw[i]) else np.nan
            cur_vol_mult = float(v_i / v_ma5_i)

            spread_arr[i] = round(max_spread_5d, 2)
            bbw_arr[i] = round(cur_bbw, 2) if not np.isnan(cur_bbw) else np.nan
            vol_mult_arr[i] = round(cur_vol_mult, 2)

            # ------------------------------------------
            # 條件 1：均線高度壓縮糾結 + 布林帶寬近 60 日低檔壓縮 (< 12%)
            # ------------------------------------------
            if max_spread_5d > max_spread_pct:
                continue

            if np.isnan(cur_bbw) or cur_bbw >= bbw_max_pct:
                continue

            bbw_hist_start = max(0, i - bbw_lookback + 1)
            bbw_hist = bbw[bbw_hist_start:i + 1]
            bbw_hist = bbw_hist[~np.isnan(bbw_hist)]
            if len(bbw_hist) < 20:
                continue
            if cur_bbw > float(np.quantile(bbw_hist, bbw_low_q)):
                continue

            # ------------------------------------------
            # 條件 2：排除「連續穿插」鋸齒雜訊區
            # ------------------------------------------
            chop_start = i - chop_win + 1
            ma20_diff = closes[chop_start:i + 1] - ma20[chop_start:i + 1]
            if np.any(np.isnan(ma20_diff)):
                continue
            if MASqueezeBreakoutStrategy._count_series_crosses(ma20_diff) > max_ma20_crosses:
                continue

            kd_diff = k_vals[chop_start:i + 1] - d_vals[chop_start:i + 1]
            if np.any(np.isnan(kd_diff)):
                continue
            if MASqueezeBreakoutStrategy._count_series_crosses(kd_diff) > max_kd_crosses:
                continue

            # ------------------------------------------
            # 條件 3：一陽穿多線實心長紅突破 + 量能與流動性
            # ------------------------------------------
            body_pct = (c_i - o_i) / o_i * 100.0
            upper_shadow_pct = (h_i - c_i) / c_i * 100.0
            if body_pct < min_body_pct or upper_shadow_pct > max_upper_shadow_pct:
                continue

            open_below_cnt = int(np.sum(o_i < cur_mas))
            close_above_all = bool(np.all(c_i > cur_mas))
            if open_below_cnt < min_open_below or not close_above_all:
                continue

            if cur_vol_mult <= min_vol_mult_5ma or v_i <= min_vol_lots:
                continue

            # ------------------------------------------
            # 條件 4：雙指標低檔同步翻多共振（KD 平滑上升無頂背離 + MACD 綠柱收斂或剛翻紅）
            # ------------------------------------------
            if np.isnan(k_vals[i]) or np.isnan(d_vals[i]) or np.isnan(k_vals[i - 1]):
                continue
            if not (k_vals[i] > d_vals[i] and k_vals[i] >= k_vals[i - 1]):
                continue

            # KD 前波低點至今呈平滑上升弧度
            kd_seg = k_vals[max(0, i - kd_low_lb + 1):i + 1]
            if np.any(np.isnan(kd_seg)) or len(kd_seg) < 5:
                continue
            min_k_pos = int(np.argmin(kd_seg))
            if min_k_pos == len(kd_seg) - 1 or k_vals[i] <= float(kd_seg[min_k_pos]):
                continue
            rise_seg = kd_seg[min_k_pos:]
            if len(rise_seg) >= 3:
                down_steps = int(np.sum(np.diff(rise_seg) < -2.0))
                if down_steps > 1:
                    continue

            # 檢查 KD 無高檔頂背離（價格創高但 K 值未創高且前高處於高檔區）
            div_start = max(0, i - kd_div_lb + 1)
            mid_pt = div_start + (i - div_start + 1) // 2
            older_p_max = float(np.max(highs[div_start:mid_pt]))
            recent_p_max = float(np.max(highs[mid_pt:i + 1]))
            older_k_max = float(np.max(k_vals[div_start:mid_pt]))
            recent_k_max = float(np.max(k_vals[mid_pt:i + 1]))
            if older_k_max >= kd_high_th and recent_p_max > older_p_max and recent_k_max < older_k_max:
                continue

            # MACD 前波明顯綠柱修正 (< -0.15) 且當日收斂接近零軸 (>= -0.10) 或剛翻紅 (<= 0.25)
            cur_hist = macd_hist[i]
            prev_hist = macd_hist[i - 1]
            prior_hists = macd_hist[max(0, i - macd_prior_lb):i]
            if np.isnan(cur_hist) or np.isnan(prev_hist) or np.all(np.isnan(prior_hists)):
                continue
            if float(np.nanmin(prior_hists)) >= macd_prior_min:
                continue
            if not (macd_cur_min <= cur_hist <= macd_cur_max):
                continue
            if cur_hist <= prev_hist:
                continue

            pass_arr[i] = True

        res["MA_Squeeze_Pass"] = pass_arr
        res["MA_Squeeze_Spread_Pct"] = spread_arr
        res["MA_Squeeze_BBW_Pct"] = bbw_arr
        res["MA_Squeeze_Vol_Mult5"] = vol_mult_arr
        return res

    @staticmethod
    def get_metrics(row) -> Dict:
        passed = bool(row.get("MA_Squeeze_Pass", False))
        spread = row.get("MA_Squeeze_Spread_Pct", np.nan)
        bbw = row.get("MA_Squeeze_BBW_Pct", np.nan)
        vol_m = row.get("MA_Squeeze_Vol_Mult5", np.nan)
        return {
            "均線壓縮一陽穿多線": "✅ 是" if passed else "❌ 否",
            "均線壓縮極差(%)": round(float(spread), 2) if pd.notna(spread) else np.nan,
            "壓縮布林帶寬(%)": round(float(bbw), 2) if pd.notna(bbw) else np.nan,
            "突破量比(5MA)": round(float(vol_m), 2) if pd.notna(vol_m) else np.nan
        }

    @staticmethod
    def get_tags(item: Dict) -> list:
        if item.get("均線壓縮一陽穿多線") == "✅ 是":
            return [MA_SQUEEZE_TAG]
        return []
