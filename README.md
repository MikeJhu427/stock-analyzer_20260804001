# 📈 台股量化與基本面雙軌分析平台
**Taiwan Stock Quant & Fundamental Analysis Platform**

基於 Python 與 Streamlit 開發的全方位台股技術面量化、跨市場波動率風險監控與機構級基本面分析系統。本系統採用「核心邏輯抽離（Single Source of Truth）」的三層模組化架構，維持高度鬆散耦合，確保全市場自動化掃描與 Web 互動介面邏輯保持 100% 同步。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動邏輯同步，系統將所有領域計算模型統一抽離至核心庫，並以三層架構進行分工：

```text
├── strategy_core.py       # 核心運算庫：跨市場波動率監控、指標計算、選股策略、財報診斷、幾何狀態機、三級備援爬蟲、雲端上傳
├── app.py                 # Streamlit 互動式視覺化分析平台（Web UI，共 6 大分頁）
├── scan_task3.py          # 全市場自動化批次掃描引擎（CLI / GitHub Actions / Cron，雙 Sheet Excel 輸出）
├── params_config.json     # 自訂策略參數設定檔（本機自動保存）
├── requirements.txt       # 依賴套件清單
└── README.md              # 專案說明文件
```

### 1) `strategy_core.py`（核心領域層）
集中維護以下能力：
- 跨市場波動率體制與避險／反彈監控（`CrossMarketVolatilityMonitor`，純 `yfinance` 資料源，相容海外機房部署）
- 技術指標計算（KD、MACD、RSI）
- 多週期背離分析與大盤濾網
- 鬆散耦合選股註冊中心（`ALGO_REGISTRY` / `ScanEngine`，含最新雙底正背離模組）
- 財報三表交叉健檢 V5（含合約負債防誤判引擎）
- 幾何型態測距與量化型態狀態機
- 三級備援爬蟲與 JSON + Base64 雲端上傳整合

### 2) `app.py`（展示層 / Web UI）
提供直覺式操作介面，包含：
- 六大功能分頁（以「跨市場波動率監控」為預設首頁，其餘功能依序遞延）
- 即時圖表視覺化與手動重整快取控制
- 自選股批次傳送與 Excel / CSV 下載
- 參數配置管理、快取機制與模組熱重載（Hot Reload）

### 3) `scan_task3.py`（排程層 / CLI）
負責每日自動化盤後任務，包含：
- Step 0：先執行跨市場波動率體制與避險／反彈監控並輸出終端機日誌
- Step 1 & 2：全市場兩階段快速初篩與長天期深度複審（含 60 分 K 輔助）
- 報表輸出：當日有達標標的時，產生雙工作表 `.xlsx`（Sheet 1：`全市場掃描結果`、Sheet 2：`波動率體制監控`）並透過 Webhook 自動備份至 Google Drive（若無達標標的則結束任務不上傳；缺少 `openpyxl` 時自動降級為 `.csv`）

---

## ✨ 核心特色與功能模組

### 🛡️ Tab 1: 跨市場波動率監控（Cross-Market Volatility Regime Monitor）
專為 GitHub Actions 與 Streamlit Cloud 等海外機房 IP 設計，**完全不爬取易遭 Cloudflare / HTTP 403 封鎖之台灣期交所與玩股網（不抓取 VIXTWN）**，全程採用純 `yfinance`（`auto_adjust=False, progress=False`，處理 MultiIndex 並統一轉換為台北時間 UTC+8）。

- **監控標的與無痛替代指標**：
  - 美股恐慌與尾部風險：`^VIX`、`^VXN`、`^VVIX`（計算 5MA、20MA、日漲跌幅、20 日布林上軌、20MA 乖離率，以及科技股風險溢價 $\text{VXN} - \text{VIX}$）。
  - 台股本地波動替代指標：以 `^TWII` 同步計算 **20 日年化已實現波動率（HV20）** 與 **ATR14 百分比（$\text{ATR14} / \text{Close}$）**，並計算 5MA、20MA 及過去 60 日滾動百分位數。
- **兩大體制切換與四狀態訊號**：
  - **【戰時恐慌警戒區】**（$\text{VIX} > 25$ 或 $\text{VXN} > 30$ 或突破 20 日布林上軌）：
    - **狀態 A（止跌反彈確認）**：$\text{VIX} < \text{VIX}_{5\text{MA}}$ 且 $\text{VXN} < \text{VXN}_{5\text{MA}}$，並自動偵測「恐慌高點下移背離」。
    - **狀態 B（尚未止血警戒）**：$\text{VIX}$ 或 $\text{VXN}$ 仍壓在 5MA 之上，特別標記 $\text{VXN} > 5\text{MA}$ 且 $\text{VXN} - \text{VIX}$ 利差擴大之高危情境，警示忽略個股左側底背離。
  - **【平時常態震盪區】**（$\text{VIX} \le 25$ 且 $\text{VXN} \le 30$ 且未破布林上軌）：
    - **狀態 C（大跌前避險預警）**：符合「$\text{VVIX} > 95$ 且 $\text{VVIX} > \text{VVIX}_{20\text{MA}}$」或「台股 `HV20` / `ATR14%` 任一自 60 日低分位（$< 20\%$）向上突破 20MA，且 `^TWII` 跌破 10MA 或 20MA」即觸發，並明確提醒實際觸發指標來源。
    - **狀態 D（結構正常）**：無系統性風險，關閉波動率雜訊干擾，回歸個股均線與籌碼操作。
- **互動視覺化**：提供狀態燈號看板、關鍵警示提醒、8 組指標卡片、完整明細表與近 3 個月 4 聯走勢圖。

### 📊 Tab 2: 單檔深度解析（Technical Analysis）
針對單一個股進行深入的技術與籌碼結構分析，包含：
- 三大演算法計分（左側底部翻轉、右側 VCP 收斂、雙指標共振分數）與 🚨 **量化雙底正背離模組**（自動計算底倉突破、右肩回測加碼點、1 段／3 段目標與破底停損防線）。
- 主力防守線追蹤：動態鎖定近 60 根 K 線的「最大量防守價」。
- 動能與極端訊號日誌：利用動能標準差（Momentum Force $\pm 1.5\sigma$）捕捉異常爆量長紅／長黑。
- VWMA 籌碼加權線與老手實戰交易濾網（RSI 盤整區間、日／週 KD 多週期共振、建議持股水位）。

### 🚀 Tab 3: 全市場智慧掃描（Market Scanner）
全自動掃描台股上市櫃與 ETF 標的，具備三級備援容錯機制，發掘潛力標的。

- **支援多策略引擎（`ALGO_REGISTRY`）**：
  - 全部、底部翻轉、VCP、指標共振、量價雙背離、長下引突破、*回測走揚年線、低位長下引量縮突破、雙底正背離與右肩回測。
- **兩階段（Two-Phase）掃描與多級別運算**：
  - Phase 1：快速初篩，並可同步啟用 60 分 K 輔助掃描捕捉盤中雙底突破與右肩回測。
  - Phase 2：下載 2 年長天期資料，驗證 120MA／240MA 斜率與空排、年線支撐回測能力、週 K 共振、日／60 分 K 雙級別背離，並分別判定長天期（60/120/240MA）與短天期（5/10/20MA）均線糾結狀態。
- **其他功能**：近 30 日觸發統計、時光機回測（指定歷史基準日）、設定檔管理（`params_config.json`）、Excel / CSV 匯出、一鍵傳送至基本面模組、Google Drive 自動備份。

### 📑 Tab 4: 財報三表地雷掃描（Fundamental Audit V5）
超越傳統「紅燈數量 = 地雷」的固定邏輯，建立動態防誤判機制：
- **合約負債（Contract Liabilities）防誤判引擎**：交叉比對存貨、營收與營業現金流，區分滯銷地雷與健康接單備料。
- **雙軌評分系統**：體質綜合分數（0–100）與財務風險分數（0–100）。
- **長天期追蹤與批次健檢**：涵蓋最新季度、4 季短中期與 8 季歷史軌跡，提供七大診斷面板與自選股批次比較總表（支援 Excel / CSV 下載）。

### 📐 Tab 5: 幾何型態滿足點測距（Geometric Pattern Projection）
利用極值演算法自動辨識波段轉折，提供「模式一：實體收盤價模型」與「模式二：影線極值模型」之 W 底／M 頭判定，同步計算 1:1 等幅滿足位、1.618 費氏延伸目標價及 SPC 統計管制（Z-Score）。

### 🤖 Tab 6: 量化型態狀態機（Quant Pattern State Machine）
結合 60 日波段位階濾網、斜向趨勢軌道／收斂楔形引擎與即時狀態演化（`FORMING`、`NECKLINE_TEST`、`DAY1_BREAK_WARNING`、`BREAKOUT_CONFIRMED`、`RETEST`、`BREAKOUT_FAILED`、`INVALIDATED`），支援盤中即時價模擬與實質風險報酬比試算。

---

## 🛠️ 環境需求與安裝指引

### 1) 系統需求
- Python：3.9 以上
- 作業系統：Windows / macOS / Linux

### 2) 安裝套件
確認 `requirements.txt` 包含以下依賴套件：

```txt
streamlit
yfinance
pandas
numpy
matplotlib
beautifulsoup4
requests
scipy
html5lib
lxml
openpyxl
plotly
```

執行安裝：

```bash
pip install -r requirements.txt
```

---

## 🚀 執行方式

### 1) 啟動 Web 視覺化分析平台（Streamlit）
```bash
streamlit run app.py
```

啟動後，請在瀏覽器開啟 `http://localhost:8501`，預設將直接載入「🛡️ 跨市場波動率監控」首頁。

### 2) 執行全市場無人值守掃描（CLI 批次腳本）
適合設定於伺服器排程（如 Crontab 或 GitHub Actions）每日盤後自動執行：

```bash
python scan_task3.py
```

---

## 🔒 雲端自動備份設定（選填）

若需將每日掃描產生的雙工作表 Excel 報表（`.xlsx`）自動上傳至 Google Drive，請設定 Google Apps Script（GAS）Webhook。

### Streamlit 應用（`app.py`）
在 `.streamlit/secrets.toml` 中加入：

```toml
GAS_WEBHOOK_URL = "https://script.google.com/macros/s/你的GAS發佈ID/exec"
GAS_SECRET_TOKEN = "你的自訂安全性Token"
```

### 批次掃描腳本（`scan_task3.py`）
設定系統環境變數：

```bash
export GAS_WEBHOOK_URL="https://script.google.com/macros/s/你的GAS發佈ID/exec"
export GAS_SECRET_TOKEN="你的自訂安全性Token"
```

---

## 📌 總結

本平台整合了：
- 跨市場波動率體制與避險／反彈監控（純 `yfinance`、無封鎖風險）
- 技術面多策略量化掃描（含 60 分 K 雙底與多級別背離）
- 財報三表交叉診斷與合約負債防誤判引擎
- 幾何滿足點測距與量化型態狀態機
- 雙工作表自動化排程與 JSON + Base64 雲端無損備份