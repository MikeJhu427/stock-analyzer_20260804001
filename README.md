# 📈 台股量化與基本面雙軌分析平台
**Taiwan Stock Quant & Fundamental Analysis Platform**

基於 Python 與 Streamlit 開發的全方位台股技術面量化、總經波動率與外資資金流向監控、空頭避風港選股，以及機構級基本面分析系統。本系統採用「核心邏輯抽離（Single Source of Truth）」的三層模組化架構，維持高度鬆散耦合，確保全市場自動化掃描與 Web 互動介面邏輯保持 100% 同步。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動邏輯同步，系統將所有領域計算模型統一抽離至核心庫，並以三層架構進行分工：

```text
├── strategy_core.py                   # 核心運算庫：波動率體制監控、外資資金流向、方案B空頭避風港、指標計算、選股策略、財報診斷、幾何狀態機、三級備援爬蟲、雲端上傳
├── app.py                             # Streamlit 互動式視覺化分析平台（Web UI，共 7 大分頁）
├── scan_task3.py                      # 全市場自動化批次掃描引擎（CLI / GitHub Actions / Cron，4 工作表 Excel 與方案 B CSV 輸出）
├── params_config.json                 # 自訂策略參數設定檔（本機自動保存）
├── foreign_fund_flow_history.xlsx     # 外資資金流向歷史紀錄（自動依日期去重附加）
├── plan_b_safe_haven_YYYY-MM-DD.csv   # 方案 B 空頭避風港盤後選股報表（大盤閘門通過時產生，UTF-8-SIG 編碼）
├── requirements.txt                   # 依賴套件清單
└── README.md                          # 專案說明文件
```

### 1) `strategy_core.py`（核心領域層）
集中維護以下能力：
- **跨市場波動率體制與避險／反彈監控**（`CrossMarketVolatilityMonitor`，純 `yfinance` 資料源，相容海外機房部署）
- **外資實質資金流向監控**（`fetch_foreign_fund_flow`／`build_foreign_fund_flow_df`，串接證交所 `BFI82U` 與 `TWD=X` 匯率連動判讀）
- **空頭避風港強勢股引擎（方案 B）**（`BearMarketSafeHavenStrategy`，大盤雙波回測閘門、普通股過濾、`ThreadPoolExecutor` 多執行緒批次掃描與快取複用）
- **技術指標與選股註冊中心**（KD、MACD、RSI、多週期背離、大盤濾網、`ALGO_REGISTRY`／`ScanEngine` 8 大策略含雙底正背離模組）
- **財報三表交叉健檢 V5**（含合約負債防誤判引擎）
- **幾何型態測距與量化型態狀態機**（`GeometricPatternStrategy`、`QuantPatternEngine`）
- **基礎設施**：三級備援爬蟲（`fetch_all_tw_stocks`）與 JSON + Base64 雲端上傳整合（`GoogleDriveUploader`）

### 2) `app.py`（展示層 / Web UI）
提供直覺式操作介面，包含：
- **七大功能分頁**（以「跨市場波動率監控」為預設首頁，尾端獨立新增「🛡️ 空頭避風港 (方案B)」分頁）
- 即時圖表視覺化、大盤閘門即時看板與手動重整快取控制
- 自選股批次傳送、Excel 與 `UTF-8-SIG` CSV 報表下載
- 參數配置管理、記憶體快取機制與模組熱重載（Hot Reload）

### 3) `scan_task3.py`（排程層 / CLI）
負責每日自動化盤後任務，執行管線包含：
- **Step 0**：執行跨市場波動率體制與避險／反彈監控並輸出終端機日誌。
- **Step 0.5**：執行外資實質資金流向監控，並與本地 `foreign_fund_flow_history.xlsx` 依日期去重合併保留趨勢。
- **Step 1 & 2**：全市場兩階段快速初篩（同步快取上市櫃普通股日線供方案 B 複用）與長天期深度複審（含 60 分 K 輔助）。
- **Step 3**：執行方案 B 空頭避風港大盤閘門檢核；若大盤通過條件 1 與條件 2，優先複用 Step 1 快取執行全市場普通股掃描並匯出本地 `plan_b_safe_haven_YYYY-MM-DD.csv`（即使 Step 1 無達標標的仍會執行方案 B）。
- **報表輸出**：當日主掃描有達標標的時，產生 **4 工作表 `.xlsx`**（Sheet 1：`全市場掃描結果`、Sheet 2：`波動率體制監控`、Sheet 3：`外資資金流向監控`、Sheet 4：`空頭避風港_方案B`）並透過 Webhook 自動備份至 Google Drive（缺少 `openpyxl` 時自動降級為 `.csv`）。

---

## ✨ 核心特色與功能模組

### 🛡️ Tab 1: 跨市場波動率監控（含外資實質資金流向連動）
專為 GitHub Actions 與 Streamlit Cloud 等海外機房 IP 設計，波動率監控全程採用純 `yfinance`（`auto_adjust=False, progress=False`，處理 MultiIndex 並統一轉換為台北時間 UTC+8），並整合外資匯率與現貨買賣超連動判讀。

- **跨市場波動率體制與四狀態訊號**：
  - 監控 `^VIX`、`^VXN`、`^VVIX` 與 `^TWII`（同步計算 20 日年化已實現波動率 `HV20` 與 `ATR14%` 作為台股本地波動替代指標，取代易遭封鎖之 VIXTWN）。
  - **【戰時恐慌警戒區】**（$\text{VIX} > 25$ 或 $\text{VXN} > 30$ 或突破 20 日布林上軌）：區分 **狀態 A（止跌反彈確認）** 與 **狀態 B（尚未止血警戒）**，自動偵測恐慌高點下移背離與科技股風險溢價擴大警示。
  - **【平時常態震盪區】**（$\text{VIX} \le 25$ 且 $\text{VXN} \le 30$ 且未破布林上軌）：區分 **狀態 C（大跌前避險預警，由 VVIX 或台股 HV20／ATR14% 低分位突破觸發）** 與 **狀態 D（結構正常）**。
- **💱 外資資金流向監控（匯率＋現貨買賣超連動）**：
  - 自動抓取台灣證交所 `BFI82U`「外資及陸資（不含外資自營商）」單日買進、賣出、買賣超與當月累計買賣超（週末與休市自動回退），並連動 `yfinance` 的 `TWD=X` 計算單日匯率漲跌幅與變動角數。
  - 自動輸出四大流向判定：`實質撤資匯出警報`、`帳上換股停泊（未匯出）`、`實質資金匯入偏多`、`中性常態波動`，並整合本地歷史紀錄表單檢視。

### 📊 Tab 2: 單檔深度解析（Technical Analysis）
針對單一個股進行深入的技術與籌碼結構分析，包含三大演算法計分（底部翻轉、VCP 收斂、雙指標共振）、🚨 **量化雙底正背離模組**、近 60 日最大量主力防守線、動能標準差（$\pm 1.5\sigma$）極端訊號日誌、VWMA 籌碼加權線與老手實戰交易濾網（RSI 盤整、日／週 KD 共振與建議水位）。

### 🚀 Tab 3: 全市場智慧掃描（Market Scanner）
全自動掃描台股上市櫃與 ETF 標的，具備三級備援容錯機制，透過 `ALGO_REGISTRY` 支援 8 大策略引擎（底部翻轉、VCP、指標共振、量價雙背離、長下引突破、*回測走揚年線、低位長下引量縮突破、雙底正背離與右肩回測）。提供兩階段掃描、60 分 K 輔助掃描、長短天期雙軌均線糾結判定、近 30 日觸發統計、時光機回測、設定檔管理（`params_config.json`）、Excel／CSV 匯出、一鍵傳送至基本面模組與 Google Drive 自動備份。

### 📑 Tab 4: 財報三表地雷掃描（Fundamental Audit V5）
透過損益表、資產負債表與現金流量表交叉勾稽，內建**合約負債（Contract Liabilities）防誤判引擎**，精準區分滯銷地雷與健康接單備料。提供體質綜合分數（0–100）與財務風險分數（0–100）雙軌評分，涵蓋最新季度、4 季短中期與 8 季歷史軌跡，支援單檔七大診斷面板與自選股批次健檢下載。

### 📐 Tab 5: 幾何型態滿足點測距（Geometric Pattern Projection）
自動辨識波段轉折極值，提供「模式一：實體收盤價模型」與「模式二：影線極值模型」之 W 底／M 頭判定，同步計算 1:1 等幅滿足位、1.618 費氏延伸目標價與 SPC 統計管制（Z-Score）。

### 🤖 Tab 6: 量化型態狀態機（Quant Pattern State Machine）
結合 60 日波段位階濾網、斜向趨勢軌道／收斂楔形引擎與即時狀態演化（`FORMING`、`NECKLINE_TEST`、`DAY1_BREAK_WARNING`、`BREAKOUT_CONFIRMED`、`RETEST`、`BREAKOUT_FAILED`、`INVALIDATED`），支援盤中即時價模擬與實質風險報酬比試算。

### 🛡️ Tab 7: 空頭避風港 (方案B)（Bear Market Safe Haven Scanner）
專為大盤跌破季線並進入第二波回測壓力測試期打造的盤後強勢股掃描器，嚴格過濾上市（`.TW`）與上櫃（`.TWO`）4 碼普通股（自動排除 ETF、權證、ETN 與 TDR）。

- **大盤雙重閘門即時看板（`^TWII`）**：
  - **條件 1（時間閘門）**：加權指數最新收盤低於 `60MA`，且過去 10 個交易日至少 7 日收盤位於 `60MA` 之下。
  - **條件 2（雙波壓力測試）**：近 7 個交易日最低點再次回測至前段（第 $-25$ 至 $-8$ 交易日）波段低點 $\text{Low}_1 \times 1.01$ 以內，或直接破底創新低。
  - 若大盤未同時滿足條件 1 與條件 2，直接顯示鎖定提示並停止個股下載，避免於第一波無差別恐慌期過早進場。
- **個股三大嚴格篩選條件**：
  - **條件 3（守穩季線）**：過去 15 個交易日每日收盤皆大於當日 `60MA`，且最新 `60MA` 維持走平或上揚（$\ge$ 10 個交易日前 `60MA` $\times 0.99$）。
  - **條件 4（底背離與強勢超額報酬）**：近 7 日低點高於前段低點至少 $+2\%$，且近 15 日對大盤超額報酬 $\text{RS} > +8\%$。
  - **條件 5（流動性濾網）**：5 日均量 $> 1,500$ 張（$1,500,000$ 股）。
- **訊號分級與輸出**：
  - `🚀 突破點火（右側試單訊號）`：收盤突破近 10 日（不含當日）最高價，且當日量 $>$ 前一日 5 日均量 $\times 1.3$。
  - `🛡️ 通過雙波測試（抗跌蓄勢觀察中）`：符合全部 5 項核心條件但尚未帶量突破。
  - 結果依 `15日超額報酬RS(%)` 由高到低排序，輸出標準 9 欄表格並提供含當日日期之 `UTF-8-SIG` 編碼 CSV 下載。

---

## 🛠️ 環境需求與安裝指引

### 1) 系統需求
- Python：3.9 以上
- 作業系統：Windows / macOS / Linux

### 2) 安裝套件
確認 `requirements.txt` 包含以下依賴套件（多執行緒採用 Python 標準庫 `concurrent.futures`，不需額外安裝）：

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

啟動後，請在瀏覽器開啟 `http://localhost:8501`，預設將直接載入「🛡️ 跨市場波動率監控」首頁，可透過上方分頁列快速切換至其餘 6 項分析功能。

### 2) 執行全市場無人值守掃描（CLI 批次腳本）
適合設定於伺服器排程（如 Crontab 或 GitHub Actions）每日盤後自動執行：

```bash
python scan_task3.py
```

---

## 🔒 雲端自動備份設定（選填）

若需將每日掃描產生的 4 工作表 Excel 報表（`.xlsx`）自動上傳至 Google Drive，請設定 Google Apps Script（GAS）Webhook。

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
- 外資實質資金流向監控（證交所 `BFI82U` 現貨買賣超連動 `USD/TWD` 匯率與本地歷史追蹤）
- 空頭避風港強勢股盤後掃描器（方案 B：大盤雙波回測閘門 + 季線抗跌與 RS 強勢篩選）
- 技術面多策略量化掃描（含 60 分 K 雙底、雙軌均線糾結與多級別背離）
- 財報三表交叉診斷與合約負債防誤判引擎
- 幾何滿足點測距與量化型態狀態機
- 四工作表自動化排程、本地 CSV 匯出與 JSON + Base64 雲端無損備份