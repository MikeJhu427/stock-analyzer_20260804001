# 📈 台股量化與基本面雙軌分析平台 
**(Taiwan Stock Quant & Fundamental Analysis Platform)**

基於 Python 與 Streamlit 開發的全方位台股技術面量化與機構級基本面分析系統。本系統採用**核心邏輯抽離（Single Source of Truth）**的三層模組化架構，打破傳統單一技術指標或財報數據的盲點，結合**左側抄底**、**右側突破**、**雙指標共振**、**兩階段回測走揚年線**等技術面量化演算法，並導入**幾何型態測距**、**量化狀態機**與**機構級財報三表交叉診斷引擎（V5）**，支援本機互動式儀表板與雲端定時無人值守排程掃描。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動介面邏輯 100% 同步，系統將所有領域計算模型統一抽離至核心庫：

```text
├── strategy_core.py       # 核心運算庫 (指標計算、選股策略、財報診斷、幾何狀態機、雲端上傳)
├── app.py                 # 程式B：Streamlit 互動式視覺化分析平台 (Web UI)
├── scan_task3.py          # 程式A：全市場自動化批次掃描引擎 (CLI / GitHub Actions / Cron)
├── params_config.json     # 參數設定檔 (本機自動保存自訂策略參數組)
└── requirements.txt       # 依賴套件清單
```

- **`strategy_core.py` (核心領域層)**：集中維護技術指標、多週期背離、大盤濾網、兩階段年線篩選、量化狀態機、三表財報審計及 Google Apps Script (GAS) 上傳邏輯。未來擴充策略僅需修改此核心檔案。
- **`app.py` (展示層 - Web UI)**：提供直覺的五大操作分頁、即時圖表視覺化、自選股批次傳送與參數配置介面。
- **`scan_task3.py` (排程層 - CLI)**：全市場兩階段快速初篩與長天期深度複審，支援輸出 CSV 並自動透過 Webhook 上傳至 Google Drive 備份。

---

## ✨ 核心特色與功能模組

### 📊 Tab 1: 單檔深度解析 (Technical Analysis)
針對單一個股進行深入的技術與籌碼結構分析：
- **三大演算法計分**：自動判定底部翻轉（左側）、VCP 收斂（右側）與雙指標共振分數。
- **主力防守線追蹤**：動態鎖定近 60 根 K 線的「最大量防守價」，跌破即發出警訊。
- **動能與極端訊號日誌**：採用動能標準差（Momentum Force $\pm 1.5\sigma$）抓取異常爆量長紅/長黑，並條列近期買賣訊號歷程。
- **VWMA 籌碼加權線**：結合均量與加權均價，判定短線實質解套賣壓與支撐。
- **老手實戰交易濾網**：整合 RSI 盤整區間過濾、日/週 KD 多週期趨勢共振判定與建議持股水位。

### 🚀 Tab 2: 全市場智慧掃描 (Market Scanner)
全自動掃描台股 1,700+ 檔上市櫃標的，挖掘潛力飆股：
- **可擴充多策略引擎**：支援選擇「全部」、「底部翻轉」、「VCP」、「指標共振」、「量價雙背離」、「長下引突破」與新增的「*回測走揚年線」等演算法。
- **兩階段（Two-Phase）極速初篩與深度複審**：
  - *Phase 1（初篩）*：利用短天期資料快速過濾打底未破、均線收斂之標的，大幅降低資料傳輸負載。
  - *Phase 2（複審）*：針對入選名單下載 2 年長天期資料，深度驗證 240MA 斜率、年線支撐回測力道、週K共振與日/60分K雙級別背離。
- **近 30 日觸發統計**：針對入選標的自動回溯統計過去 30 天內的訊號發動次數。
- **時光機回測與設定檔管理**：允許指定「歷史基準日」進行事後驗證；支援隨時自訂、儲存與切換專屬參數組合（`params_config.json`）。
- **無縫跨頁連動與雲端上傳**：掃描結果支援 CSV 下載，可一鍵傳送至基本面模組進行批次健檢，或自動 Webhook 上傳至 Google Drive。

### 📑 Tab 3: 財報三表交叉健檢 V5 (Fundamental Audit)
超越傳統「紅燈數量＝地雷」的死板邏輯，建立財報三表動態防誤判機制：
- **合約負債 (Contract Liabilities) 防誤判引擎**：自動抓取並交叉比對營收與現金流。精準辨識「存貨增加」究竟是滯銷地雷，還是大單滿載的「健康備貨」，避免將擴張期成長股誤判為財務地雷。
- **雙軌評分系統**：同時計算「體質護城河分數 (0-100)」與「財務地雷風險分數 (0-100)」。
- **長天期趨勢追蹤**：涵蓋最新季度、4 季短中期趨勢，以及 8 季歷史常態軌跡。
- **七大專業診斷面板**：自動生成「風險共振提示」、「正向營運資金循環」及「下一季追蹤事項」。
- **支援批次掃描**：可直接貼上自選股代號清單，一鍵產生全清單評分與警訊比較總表。

### 📐 Tab 4: 幾何型態滿足點測距 (Geometric Pattern Projection)
利用極值演算法自動辨識波段轉折，計算對稱型態之目標價：
- **雙重型態模型**：
  - *模式一（實體收盤價模型）*：過濾盤中突刺雜訊，錨定主力與法人實體成本。
  - *模式二（影線極值模型）*：涵蓋盤中斷頭恐慌踩踏與插針極端價。
- **等幅與費氏延伸測距**：自動辨識 W 底突破與 M 頭跌破，計算 1:1 等幅滿足位與 1.618 費氏延伸目標價。
- **SPC 統計管制與 Z-Score**：利用動能標準差判定當前價格是否落入極端超買/超跌常態外區間。

### 🤖 Tab 5: 量化型態狀態機 (Quant Pattern State Machine)
結合狀態機概念，解決傳統形態學之滯後與誤判問題：
- **波段位階濾網 (Regime Filter)**：導入近 60 日百分位統計，自動排除「高檔橫盤誤判 W 底」與「低檔頭頭高誤判 M 頭」。
- **斜向趨勢軌道與收斂楔形引擎**：自動繪製動態上下軌趨勢線，辨識上升/下降楔形、通道與對稱三角收斂，並提供動態支撐與壓力價。
- **狀態機即時演化**：嚴格追蹤 `FORMING` ➔ `NECKLINE_TEST` ➔ `DAY1_BREAK_WARNING`（首日破線即時預警）➔ `BREAKOUT_CONFIRMED`（次日有效確認）➔ `RETEST` / `BREAKOUT_FAILED`。
- **盤中即時模擬**：支援手動輸入盤中急殺/急拉即時價，在收盤前搶先模擬破線狀態與實質風報比（Risk/Reward Ratio）。

---

## 🛠️ 環境需求與安裝指引

### 1. 系統需求
- **Python**: 3.9 或以上版本
- **作業系統**: Windows / macOS / Linux

### 2. 安裝套件
請將以下內容存為 `requirements.txt`：
```text
streamlit>=1.30.0
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
beautifulsoup4>=4.12.0
requests>=2.31.0
urllib3>=2.0.0
```

執行安裝指令：
```bash
pip install -r requirements.txt
```

---

## 🚀 執行方式

### 1. 啟動 Web 視覺化分析平台 (Streamlit)
```bash
streamlit run app.py
```
啟動後於瀏覽器開啟 `http://localhost:8501` 即可操作。

### 2. 執行全市場無人值守掃描 (CLI 批次腳本)
適合設定於伺服器排程（如 Crontab 或 GitHub Actions）每日盤後自動執行：
```bash
python scan_task3.py
```

---

## 🔒 雲端自動備份設定 (選填)

若需將每日掃描產生的 CSV 報表自動上傳至 Google Drive，請配置 Google Apps Script (GAS) Webhook：

1. **Streamlit 應用 (`app.py`)**：在 `.streamlit/secrets.toml` 中加入：
   ```toml
   GAS_WEBHOOK_URL = "https://script.google.com/macros/s/你的GAS發佈ID/exec"
   GAS_SECRET_TOKEN = "你的自訂安全性Token"
   ```
2. **批次掃描腳本 (`scan_task3.py`)**：設定作業系統環境變數：
   ```bash
   export GAS_WEBHOOK_URL="https://script.google.com/macros/s/你的GAS發佈ID/exec"
   export GAS_SECRET_TOKEN="你的自訂安全性Token"
   ```