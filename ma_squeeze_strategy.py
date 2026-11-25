import numpy as np
import pandas as pd
from typing import Dict, Optional
import strategy_core as core

MA_SQUEEZE_TAG = "均線壓縮一陽穿多線"

MA_SQUEEZE_COLUMNS = [
    "均線壓縮一陽穿多線",
    "均線壓縮診斷",
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
    "ma20_cross_deadband_pct": 0.5,
    "max_ma20_crosses": 2,
    "max_kd_crosses": 2,
    "min_body_pct": 3.5,
    "max_upper_shadow_pct": 1.5,
    "min_open_below_mas": 2,
    "prev_close_above_ma_max_pct": 1.5,
    "open_above_ma_max_pct": 2.0,
    "min_vol_mult_5ma": 1.8,
    "min_vol_lots": 1500.0,
    "kd_low_lookback": 5,
    "kd_div_lookback": 20,
    "kd_high_zone_th": 70.0,
    "macd_prior_lookback": 25,
    "macd_prior_min_hist": -0.15,
    "macd_prior_min_pct": -0.25,
    "macd_cur_min_hist": -0.10,
    "macd_cur_max_pct": 1.50
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
    def _count_series_crosses(diff_values: np.ndarray, deadband: float = 0.0) -> int:
        """統計序列在零軸上下的有效穿越次數（絕對值 <= deadband 視為貼線忽略）。"""
        if deadband > 0.0:
            signs = np.where(diff_values > deadband, 1, np.where(diff_values < -deadband, -1, 0))
        else:
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
        逐日檢核硬性條件（一票否決制），回傳與輸入 df 等長之結果 DataFrame：
        - MA_Squeeze_Pass (bool)
        - MA_Squeeze_Diag (str，記錄通過狀態或第一個未過條件與實際數值)
        - MA_Squeeze_Spread_Pct (float，取突破前 5 日壓縮極差)
        - MA_Squeeze_BBW_Pct (float，取突破前一日布林帶寬)
        - MA_Squeeze_Vol_Mult5 (float，當日量對前一日 5 日均量倍數)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "MA_Squeeze_Pass",
                "MA_Squeeze_Diag",
                "MA_Squeeze_Spread_Pct",
                "MA_Squeeze_BBW_Pct",
                "MA_Squeeze_Vol_Mult5"
            ])

        res = pd.DataFrame(index=df.index)
        res["MA_Squeeze_Pass"] = False
        res["MA_Squeeze_Diag"] = "資料不足(<66日)"
        res["MA_Squeeze_Spread_Pct"] = np.nan
        res["MA_Squeeze_BBW_Pct"] = np.nan
        res["MA_Squeeze_Vol_Mult5"] = np.nan

        if len(df) < 66:
            return res

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in required_cols):
            res["MA_Squeeze_Diag"] = "缺少OHLCV欄位"
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
        ma20_deadband_pct = float(cfg["ma20_cross_deadband_pct"])
        max_ma20_crosses = int(cfg["max_ma20_crosses"])
        max_kd_crosses = int(cfg["max_kd_crosses"])

        min_body_pct = float(cfg["min_body_pct"])
        max_upper_shadow_pct = float(cfg["max_upper_shadow_pct"])
        min_open_below = int(cfg["min_open_below_mas"])
        prev_c_above_pct = float(cfg["prev_close_above_ma_max_pct"])
        open_above_pct = float(cfg["open_above_ma_max_pct"])
        min_vol_mult_5ma = float(cfg["min_vol_mult_5ma"])
        min_vol_lots = float(cfg["min_vol_lots"])

        kd_low_lb = int(cfg["kd_low_lookback"])
        kd_div_lb = int(cfg["kd_div_lookback"])
        kd_high_th = float(cfg["kd_high_zone_th"])

        macd_prior_lb = int(cfg["macd_prior_lookback"])
        macd_prior_min = float(cfg["macd_prior_min_hist"])
        macd_prior_min_pct = float(cfg["macd_prior_min_pct"])
        macd_cur_min = float(cfg["macd_cur_min_hist"])
        macd_cur_max_pct = float(cfg["macd_cur_max_pct"])

        pass_arr = np.zeros(len(work), dtype=bool)
        diag_arr = np.full(len(work), "樣本不足", dtype=object)
        spread_arr = np.full(len(work), np.nan, dtype=float)
        bbw_arr = np.full(len(work), np.nan, dtype=float)
        vol_mult_arr = np.full(len(work), np.nan, dtype=float)

        start_idx = max(60 + sq_win, chop_win + 1, macd_prior_lb + 1, kd_div_lb + 1)

        for i in range(start_idx, len(work)):
            o_i = opens[i]
            h_i = highs[i]
            c_i = closes[i]
            v_i = vols[i]
            prev_c = closes[i - 1]
            v_ma5_prev = vol_ma5[i - 1]

            if any(np.isnan([o_i, h_i, c_i, v_i, prev_c, v_ma5_prev])) or o_i <= 0 or c_i <= 0 or prev_c <= 0 or v_ma5_prev <= 0:
                diag_arr[i] = "價量缺值或為零"
                continue

            # 突破前 5 個交易日（i-5 ~ i-1）檢核均線是否具備完整 120MA
            pre_start = i - sq_win
            use_ma120 = (not np.any(np.isnan(ma120[pre_start:i + 1])))
            if use_ma120:
                ma_matrix_pre = np.vstack([
                    ma5[pre_start:i],
                    ma10[pre_start:i],
                    ma20[pre_start:i],
                    ma60[pre_start:i],
                    ma120[pre_start:i]
                ])
                prev_mas = np.array([ma5[i - 1], ma10[i - 1], ma20[i - 1], ma60[i - 1], ma120[i - 1]], dtype=float)
                cur_mas = np.array([ma5[i], ma10[i], ma20[i], ma60[i], ma120[i]], dtype=float)
            else:
                ma_matrix_pre = np.vstack([
                    ma5[pre_start:i],
                    ma10[pre_start:i],
                    ma20[pre_start:i],
                    ma60[pre_start:i]
                ])
                prev_mas = np.array([ma5[i - 1], ma10[i - 1], ma20[i - 1], ma60[i - 1]], dtype=float)
                cur_mas = np.array([ma5[i], ma10[i], ma20[i], ma60[i]], dtype=float)

            if np.any(np.isnan(ma_matrix_pre)) or np.any(ma_matrix_pre <= 0) or np.any(np.isnan(cur_mas)):
                diag_arr[i] = "均線缺值"
                continue

            # 紀錄突破前壓縮數值與當日量比（供報表呈現）
            ma_min_5d = np.min(ma_matrix_pre, axis=0)
            ma_max_5d = np.max(ma_matrix_pre, axis=0)
            spread_5d_pct = (ma_max_5d - ma_min_5d) / ma_min_5d * 100.0
            max_spread_5d = float(np.max(spread_5d_pct))
            pre_bbw = float(bbw[i - 1]) if not np.isnan(bbw[i - 1]) else np.nan
            cur_vol_mult = float(v_i / v_ma5_prev)

            spread_arr[i] = round(max_spread_5d, 2)
            bbw_arr[i] = round(pre_bbw, 2) if not np.isnan(pre_bbw) else np.nan
            vol_mult_arr[i] = round(cur_vol_mult, 2)

            # ------------------------------------------
            # 條件 1：均線高度壓縮糾結 + 布林帶寬近 60 日低檔壓縮（皆檢核突破前一日 i-1）
            # ------------------------------------------
            if max_spread_5d > max_spread_pct:
                diag_arr[i] = f"未過1:均線極差{max_spread_5d:.2f}%>{max_spread_pct:.1f}%"
                continue

            if np.isnan(pre_bbw):
                diag_arr[i] = "未過1:前日BBW缺值"
                continue

            if pre_bbw >= bbw_max_pct:
                diag_arr[i] = f"未過1:前日BBW({pre_bbw:.2f}%)>={bbw_max_pct:.1f}%"
                continue

            bbw_hist_start = max(0, i - bbw_lookback)
            bbw_hist = bbw[bbw_hist_start:i]
            bbw_hist = bbw_hist[~np.isnan(bbw_hist)]
            if len(bbw_hist) < 20:
                diag_arr[i] = "未過1:BBW歷史不足20日"
                continue

            bbw_q_val = float(np.quantile(bbw_hist, bbw_low_q))
            if pre_bbw > bbw_q_val:
                diag_arr[i] = f"未過1:前日BBW({pre_bbw:.2f}%)>60日分位({bbw_q_val:.2f}%)"
                continue

            # ------------------------------------------
            # 條件 2：排除「連續穿插」鋸齒雜訊區（檢核突破前 10 日 i-10 ~ i-1，20MA 含 ±0.5% 貼線死區）
            # ------------------------------------------
            chop_start = i - chop_win
            ma20_seg = ma20[chop_start:i]
            if np.any(np.isnan(ma20_seg)) or np.any(ma20_seg <= 0):
                diag_arr[i] = "未過2:20MA缺值"
                continue

            ma20_diff_pct = (closes[chop_start:i] - ma20_seg) / ma20_seg * 100.0
            ma20_crosses = MASqueezeBreakoutStrategy._count_series_crosses(ma20_diff_pct, deadband=ma20_deadband_pct)
            if ma20_crosses > max_ma20_crosses:
                diag_arr[i] = f"未過2:20MA穿刺{ma20_crosses}次>{max_ma20_crosses}次"
                continue

            kd_diff = k_vals[chop_start:i] - d_vals[chop_start:i]
            if np.any(np.isnan(kd_diff)):
                diag_arr[i] = "未過2:KD缺值"
                continue

            kd_crosses = MASqueezeBreakoutStrategy._count_series_crosses(kd_diff, deadband=0.0)
            if kd_crosses > max_kd_crosses:
                diag_arr[i] = f"未過2:KD交叉{kd_crosses}次>{max_kd_crosses}次"
                continue

            # ------------------------------------------
            # 條件 3：一陽穿多線實心長紅突破（支援原定義與貼線點火）+ 量能與流動性
            # ------------------------------------------
            body_pct = (c_i - o_i) / o_i * 100.0
            upper_shadow_pct = (h_i - c_i) / c_i * 100.0
            if body_pct < min_body_pct:
                diag_arr[i] = f"未過3:實體{body_pct:.2f}%<{min_body_pct:.1f}%"
                continue

            if upper_shadow_pct > max_upper_shadow_pct:
                diag_arr[i] = f"未過3:上影{upper_shadow_pct:.2f}%>{max_upper_shadow_pct:.1f}%"
                continue

            prev_ma_max = float(np.max(prev_mas))
            cur_ma_max = float(np.max(cur_mas))
            close_above_all = bool(np.all(c_i > cur_mas))
            if not close_above_all:
                diag_arr[i] = f"未過3:收盤{c_i:.2f}未站上全部均線(頂{cur_ma_max:.2f})"
                continue

            open_below_cnt = int(np.sum(o_i < cur_mas))
            cond_classic_pierce = open_below_cnt >= min_open_below
            cond_tight_ignition = (
                prev_c <= prev_ma_max * (1.0 + prev_c_above_pct / 100.0)
                and o_i <= cur_ma_max * (1.0 + open_above_pct / 100.0)
            )
            if not (cond_classic_pierce or cond_tight_ignition):
                prev_gap_pct = (prev_c / prev_ma_max - 1.0) * 100.0
                open_gap_pct = (o_i / cur_ma_max - 1.0) * 100.0
                diag_arr[i] = f"未過3:起漲位階(開低{open_below_cnt}線/前收{prev_gap_pct:+.2f}%/開{open_gap_pct:+.2f}%)"
                continue

            if cur_vol_mult <= min_vol_mult_5ma:
                diag_arr[i] = f"未過3:量比{cur_vol_mult:.2f}x<={min_vol_mult_5ma:.1f}x"
                continue

            if v_i <= min_vol_lots:
                diag_arr[i] = f"未過3:成交量{v_i:.0f}張<={min_vol_lots:.0f}張"
                continue

            # ------------------------------------------
            # 條件 4：雙指標低檔同步翻多共振（KD 近 5 日低點平滑上升無頂背離 + MACD 綠柱收斂翻揚）
            # ------------------------------------------
            if np.isnan(k_vals[i]) or np.isnan(d_vals[i]) or np.isnan(k_vals[i - 1]):
                diag_arr[i] = "未過4:KD缺值"
                continue

            if not (k_vals[i] > d_vals[i] and k_vals[i] >= k_vals[i - 1]):
                diag_arr[i] = f"未過4:KD未翻多(K={k_vals[i]:.1f},D={d_vals[i]:.1f},前K={k_vals[i-1]:.1f})"
                continue

            kd_seg = k_vals[max(0, i - kd_low_lb + 1):i + 1]
            if np.any(np.isnan(kd_seg)) or len(kd_seg) < 5:
                diag_arr[i] = "未過4:KD樣本不足"
                continue

            min_k_pos = int(np.argmin(kd_seg))
            if min_k_pos == len(kd_seg) - 1 or k_vals[i] <= float(kd_seg[min_k_pos]):
                diag_arr[i] = f"未過4:KD非低點上升(K={k_vals[i]:.1f})"
                continue

            rise_seg = kd_seg[min_k_pos:]
            if len(rise_seg) >= 3:
                down_steps = int(np.sum(np.diff(rise_seg) < -2.0))
                if down_steps > 1:
                    diag_arr[i] = f"未過4:KD上升不平滑(回折{down_steps}次)"
                    continue

            div_start = max(0, i - kd_div_lb + 1)
            mid_pt = div_start + (i - div_start + 1) // 2
            older_p_max = float(np.max(highs[div_start:mid_pt]))
            recent_p_max = float(np.max(highs[mid_pt:i + 1]))
            older_k_max = float(np.max(k_vals[div_start:mid_pt]))
            recent_k_max = float(np.max(k_vals[mid_pt:i + 1]))
            if older_k_max >= kd_high_th and recent_p_max > older_p_max and recent_k_max < older_k_max:
                diag_arr[i] = f"未過4:KD頂背離(前K高{older_k_max:.1f},近K高{recent_k_max:.1f})"
                continue

            cur_hist = macd_hist[i]
            prev_hist = macd_hist[i - 1]
            prior_hists = macd_hist[max(0, i - macd_prior_lb):i]
            prior_closes = closes[max(0, i - macd_prior_lb):i]
            if np.isnan(cur_hist) or np.isnan(prev_hist) or np.all(np.isnan(prior_hists)):
                diag_arr[i] = "未過4:MACD缺值"
                continue

            prior_min_hist = float(np.nanmin(prior_hists))
            prior_hist_pct = prior_hists / (prior_closes + 1e-8) * 100.0
            prior_min_pct = float(np.nanmin(prior_hist_pct))
            if not (prior_min_hist < macd_prior_min or prior_min_pct < macd_prior_min_pct):
                diag_arr[i] = f"未過4:MACD前波綠柱不足(最深{prior_min_hist:.3f}/{prior_min_pct:.2f}%)"
                continue

            cur_hist_pct = float(cur_hist / c_i * 100.0)
            if cur_hist < macd_cur_min or cur_hist_pct > macd_cur_max_pct:
                diag_arr[i] = f"未過4:MACD柱體越界({cur_hist:.3f}/{cur_hist_pct:.2f}%)"
                continue

            if cur_hist <= prev_hist:
                diag_arr[i] = f"未過4:MACD未翻揚({prev_hist:.3f}->{cur_hist:.3f})"
                continue

            pass_arr[i] = True
            diag_arr[i] = "✅ 全部通過"

        res["MA_Squeeze_Pass"] = pass_arr
        res["MA_Squeeze_Diag"] = diag_arr
        res["MA_Squeeze_Spread_Pct"] = spread_arr
        res["MA_Squeeze_BBW_Pct"] = bbw_arr
        res["MA_Squeeze_Vol_Mult5"] = vol_mult_arr
        return res

    @staticmethod
    def get_metrics(row) -> Dict:
        passed = bool(row.get("MA_Squeeze_Pass", False))
        diag = row.get("MA_Squeeze_Diag", "-")
        spread = row.get("MA_Squeeze_Spread_Pct", np.nan)
        bbw = row.get("MA_Squeeze_BBW_Pct", np.nan)
        vol_m = row.get("MA_Squeeze_Vol_Mult5", np.nan)
        return {
            "均線壓縮一陽穿多線": "✅ 是" if passed else "❌ 否",
            "均線壓縮診斷": str(diag) if pd.notna(diag) else "-",
            "均線壓縮極差(%)": round(float(spread), 2) if pd.notna(spread) else np.nan,
            "壓縮布林帶寬(%)": round(float(bbw), 2) if pd.notna(bbw) else np.nan,
            "突破量比(5MA)": round(float(vol_m), 2) if pd.notna(vol_m) else np.nan
        }

    @staticmethod
    def get_tags(item: Dict) -> list:
        if item.get("均線壓縮一陽穿多線") == "✅ 是":
            return [MA_SQUEEZE_TAG]
        return []
